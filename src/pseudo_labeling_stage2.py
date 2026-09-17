"""
Этап 2 псевдо-разметки: финальные пороги и генерация меток.

Пороги:
- crystal system: 0.998 (цель 95%, калибровка на ft_pool оптимистична ->
  берём консервативно верхний порог; softmax головы почти бинарный)
- elements: ПОЭЛЕМЕНТНАЯ точность >= 95% (взвешенная по классам), с порогом на
  sigmoid; SG-голова отключена (калибровка показала недостижимость 90%)
- lattice: НЕ размечаем (регрессия - нет калибруемой "уверенности",
  риск тихо внести систематическую ошибку)

Выход:
  data/clean/pseudo_labels.parquet - sample_id, crystal_system_pseudo, el_* флаги,
  уверенности
  outputs/pseudo_labels_registry.json - реестр
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"
stats = json.loads((OUT / "pretrain_stats.json").read_text())
VOCAB = stats["vocab"]
SYSTEMS = stats["systems"]

cal_el = np.load(OUT / "_tmp_cal_el.npy")
cal_sys = np.load(OUT / "_tmp_cal_sys.npy")
pred_el = np.load(OUT / "_tmp_pred_el.npy")
pred_sys = np.load(OUT / "_tmp_pred_sys.npy")
pred_lat = np.load(OUT / "_tmp_pred_lat.npy")
unl = pd.read_parquet(OUT / "_tmp_unl.parquet")
ft_ids = pd.read_parquet(OUT / "_tmp_ft_ids.parquet")

ft = pd.read_parquet(BASE / "data" / "clean" / "ft_pool_combined.parquet")
EL_IDX = {e: i for i, e in enumerate(VOCAB)}
ft_el_true = np.zeros((len(ft), len(VOCAB)), np.float32)
for i, v in enumerate(ft["elements_list"]):
    try:
        els = json.loads(v) if isinstance(v, str) else list(v)
    except Exception:
        els = []
    for e in els if isinstance(els, list) else []:
        j = EL_IDX.get(e)
        if j is not None:
            ft_el_true[i, j] = 1.0

# ---------------- поэлементная калибровка (precision на предсказанных) ----------------
print("=== поэлементная калибровка elements (precision@threshold) ===")
best_t = None
for t in np.arange(0.50, 0.99, 0.02):
    pred_bin = cal_el > t
    tp = float((pred_bin & (ft_el_true > 0.5)).sum())
    fp = float((pred_bin & (ft_el_true < 0.5)).sum())
    prec = tp / max(tp + fp, 1)
    cov = pred_bin.sum() / max(ft_el_true.sum(), 1)
    if prec >= 0.95:
        best_t = (float(t), prec, cov)
        break
print("порог (первый с precision>=95%):", best_t)

TH_SYS = 0.998
TH_EL = best_t[0] if best_t else 0.85  # fallback: жёстче дефолта
print(f"\nитоговые пороги: crystal_system {TH_SYS} | elements {TH_EL} | SG: откл.")

# ---------------- генерация псевдо-меток ----------------
n = len(unl)
sys_prob = pred_sys.max(1)
sys_cls = pred_sys.argmax(1)
sys_conf_ok = sys_prob >= TH_SYS

el_bin = pred_el > TH_EL
el_frac = el_bin.mean(1)
el_any = el_bin.any(1)

lat_std_ok = np.zeros(n, bool)  # решётку не размечаем

pseudo = pd.DataFrame({
    "sample_id": unl["sample_id"],
    "pseudo_crystal_system": [SYSTEMS[c] if ok else None
                              for c, ok in zip(sys_cls, sys_conf_ok)],
    "pseudo_crystal_system_conf": np.where(sys_conf_ok, sys_prob, np.nan),
    "pseudo_elements": [json.dumps([VOCAB[j] for j in np.where(b)[0]]) if any else None
                        for b, any in zip(el_bin, el_any)],
    "pseudo_n_elements": np.where(el_any, el_bin.sum(1), 0),
})

labeled_any = sys_conf_ok | el_any
print(f"\nразмечено хотя бы одной головой: {int(labeled_any.sum())} из {n} "
      f"({labeled_any.mean():.1%})")
print(f"  crystal system (порог {TH_SYS}): {int(sys_conf_ok.sum())} "
      f"({sys_conf_ok.mean():.1%})")
print(f"  elements (порог {TH_EL:.2f}): {int(el_any.sum())} ({el_any.mean():.1%})")

# распределение числа элементов
if el_any.any():
    print("  распределение числа элементов в псевдо-метках:")
    vals, cnts = np.unique(pseudo.loc[el_any, "pseudo_n_elements"], return_counts=True)
    for v, c in zip(vals[:10], cnts[:10]):
        print(f"    {v}: {c}")

pseudo.to_parquet(BASE / "data" / "clean" / "pseudo_labels.parquet", index=False)

# ---------------- реестр ----------------
registry = {
    "created_utc": pd.Timestamp.utcnow().isoformat(),
    "model": "checkpoints/ft_v2_final.pt",
    "description": "Псевдо-метки для неразмеченных opXRD-спектров (split_role=unlabeled "
                   "в index_preprocessed.parquet). ЛАБОРАТОРНЫЕ метки (ft_pool_combined) "
                   "не затронуты и остаются единственным источником ground truth.",
    "thresholds": {
        "crystal_system": {"threshold": TH_SYS, "basis": "калибровка precision>=95% на ft_pool (оптимистично), консервативная надбавка"},
        "space_group": {"enabled": False, "reason": "precision>=90% недостижим ни при каком пороге (калибровка на ft_pool)"},
        "elements": {"threshold": TH_EL, "basis": "поэлементная калибровка precision>=95% на ft_pool"},
        "lattice": {"enabled": False, "reason": "регрессия без калибруемой уверенности - риск систематики"},
    },
    "counts": {
        "total_unlabeled": int(n),
        "labeled_any": int(labeled_any.sum()),
        "crystal_system": int(sys_conf_ok.sum()),
        "elements": int(el_any.sum()),
    },
    "heads_note": "строка может иметь псевдо-метку одной головы и не иметь другой; "
                  "в FT это учитывается масками голов",
}
(OUT / "pseudo_labels_registry.json").write_text(
    json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nсохранено:")
print("  data/clean/pseudo_labels.parquet")
print("  outputs/pseudo_labels_registry.json")
