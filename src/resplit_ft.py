"""
Пересборка FT-сплита с ГРУППИРОВКОЙ по соединению.

Группа = (phase_compositions, lattice_a округлённая) - серия спектров одного
образца/соединения. Вся группа целиком попадает в train или val.
Стратификация: по источнику и наличию голов (как раньше), но на уровне групп.

Также чинит zero-shot/pretrain-сплит? Нет: пре-трейн-сплит уже резал по формуле
соединения (98/2), он честный. Пересобираем только FT.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

ft = pd.read_parquet(BASE / "data" / "clean" / "ft_pool_combined.parquet")
ft["conn_key"] = (
    ft["phase_compositions"].astype(str) + "|" + ft["lattice_a"].round(2).astype(str)
)

# одна строка на группу с агрегированными признаками стратификации
g = (
    ft.groupby("conn_key")
    .agg(
        n=("sample_id", "size"),
        role=("dataset_role", "first"),
        m_lat=("head_mask_lattice", "max"),
        m_sg=("head_mask_spacegroup", "max"),
        m_sys=("head_mask_crystal_system", "max"),
        m_el=("head_mask_elements", "max"),
    )
    .reset_index()
)
g["stratum"] = (
    g["role"].astype(str) + "|"
    + g["m_lat"].astype(int).astype(str)
    + g["m_sg"].astype(int).astype(str)
    + g["m_sys"].astype(int).astype(str)
    + g["m_el"].astype(int).astype(str)
)
print("групп:", len(g), "| страт:", g['stratum'].nunique())

# ручной стратифицированный сплит групп (5% на вал из каждого страта, где их >= 2)

rng = np.random.default_rng(42)
va_keys = []

for stratum, sub in g.groupby("stratum"):
    keys = sub["conn_key"].to_numpy()
    if len(keys) < 2:
        continue  # единичная группа остаётся в трейне

    k = max(1, int(round(len(keys) * 0.05)))
    va_keys.extend(rng.choice(keys, size=k, replace=False).tolist())

va_keys = set(va_keys)

ft["ft_split"] = np.where(ft["conn_key"].isin(set(va_keys)), "val", "train")
ft[["sample_id", "ft_split"]].to_parquet(BASE / "outputs" / "splits_ft.parquet", index=False)

# контроль
leak = ft.groupby("conn_key")["ft_split"].nunique()
assert (leak == 1).all(), "остались разрезанные группы!"

n_val = int((ft["ft_split"] == "val").sum())
sg_val = int(((ft["ft_split"] == "val") & ft["spacegroup_number"].notna()).sum())
lat_val = int(((ft["ft_split"] == "val") & ft["head_mask_lattice"].astype(bool)).sum())

print(f"\nновый сплит: train {len(ft)-n_val} / val {n_val}")
print("val: SG-строк", sg_val, "| lattice-строк", lat_val)
print("\nval по источникам:")
print(ft[ft["ft_split"] == "val"].groupby("dataset_role").size().to_string())
print("\nутечка групп: 0 (проверено ассертом)")
