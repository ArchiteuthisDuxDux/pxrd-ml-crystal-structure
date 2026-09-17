"""
Финализация синтетики: склейка шардов, валидация схемы, итоговый паркет.

Запуск:
    python synth_finalize.py
"""

import sys

import numpy as np
import pandas as pd

import synth_config as cfg


def main():
    shards = sorted(cfg.SHARDS_DIR.glob("*_part_*.parquet"))
    if not shards:
        print("Шардов нет.")
        return 1

    frames = []
    for s in shards:
        df = pd.read_parquet(s)
        frames.append(df)
        print(f"  {s.name}: {len(df)} строк")
    full = pd.concat(frames, ignore_index=True)

    # дедуп на всякий случай

    before = len(full)
    full = full.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)
    if len(full) != before:
        print(f"Удалено дублей sample_id: {before - len(full)}")

    # --- валидация схемы против оригиналов --------------------------

    ref = pd.read_parquet(
        cfg.PIPELINE_ROOT / "data" / "source" / "df_opxrd_summary_final_clean.parquet"
    )
    missing = [c for c in cfg.SCHEMA_COLUMNS if c not in full.columns]
    extra = [c for c in full.columns if c not in cfg.SCHEMA_COLUMNS]
    order_ok = list(ref.columns) == cfg.SCHEMA_COLUMNS
    cols_ok = list(full.columns) == cfg.SCHEMA_COLUMNS

    print()
    print("=" * 70)
    print("ВАЛИДАЦИЯ СХЕМЫ")
    print("=" * 70)
    print(f"колонок в синтетике: {len(full.columns)} (в оригинале: {len(ref.columns)})")
    print(f"порядок колонок совпадает с оригиналом: {cols_ok and order_ok}")

    if missing:
        print("ОТСУТСТВУЮТ:", missing)

    if extra:
        print("ЛИШНИЕ:", extra)

    if not cols_ok:
        full = full[cfg.SCHEMA_COLUMNS]
        print("Колонки приведены к эталонному порядку.")

    # --- статистика ---------------------------------------------------
    print()
    print("=" * 70)
    print("СТАТИСТИКА СИНТЕТИКИ")
    print("=" * 70)
    print(f"Всего строк: {len(full)}")
    print("\nПо источникам:")
    print(full["dataset_role"].value_counts().to_string())

    print("\nДоля с spacegroup:", f"{full['has_spacegroup'].mean():.1%}")
    wl1 = pd.to_numeric(full["primary_wavelength"], errors="coerce")
    print("λ1 топ-8:")
    print(wl1.round(4).value_counts().head(8).to_string())

    in_window = (
        (wl1.notna())
        & (full[ "lattice_a"].notna())
    )
    print(f"\nС решёткой: {int(full['lattice_a'].notna().sum())}")
    print(f"crystallite_size_nm: медиана {full['crystallite_size_nm'].median():.1f} нм")

    # --- проверка нескольких npy ---------------------------------------

    print("\nПроверка .npy:")
    ok = 0
    for _, r in full.sample(min(5, len(full)), random_state=0).iterrows():
        p = cfg.RAW_SPECTRA_DIR.parent / r["raw_spectrum_path"]
        arr = np.load(p)
        print(f"  {p.name}: shape={arr.shape}, x=[{arr[:,0].min():.2f},{arr[:,0].max():.2f}], "
              f"I_max={arr[:,1].max():.0f}")
        ok += 1

    assert ok > 0

    full.to_parquet(cfg.FINAL_PARQUET, index=False)
    print(f"\nСохранено: {cfg.FINAL_PARQUET}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
