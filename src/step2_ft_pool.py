"""
Шаг 2. Формирование fine-tuning пула.

В FT-пул попадают только строки, у которых размечена
ХОТЯ БЫ ОДНА целевая голова:
    lattice / spacegroup / crystal_system / elements.

Для каждой строки добавляются маски голов (1 - метка указана,
0 - нет) и маска углового окна in_angle_window_0_90.

phase_count / phase_fraction игнорируются (не целевые головы).
"""

import sys

import pandas as pd

import config as cfg
from utils_spectra import is_nonempty_sequence, normalize_object_columns


def head_masks(df):
    masks = pd.DataFrame(index=df.index)

    masks[cfg.HEAD_MASK_COLUMNS["lattice"]] = (
        df[cfg.LATTICE_COLUMNS].notna().all(axis=1).astype("int8")
    )

    masks[cfg.HEAD_MASK_COLUMNS["spacegroup"]] = (
        df["spacegroup_number"].notna().astype("int8")
    )

    masks[cfg.HEAD_MASK_COLUMNS["crystal_system"]] = (
        df["crystal_system"].notna().astype("int8")
    )

    masks[cfg.HEAD_MASK_COLUMNS["elements"]] = (
        df["elements_list"].apply(is_nonempty_sequence).astype("int8")
    )

    return masks


def main():
    frames = []

    for name, path, out_path in [
        ("rruff", cfg.RRUFF_CLEAN_PARQUET, cfg.FT_POOL_RRUFF_PARQUET),
        ("opxrd", cfg.OPXRD_CLEAN_PARQUET, cfg.FT_POOL_OPXRD_PARQUET),
    ]:
        print("=" * 70)
        print(f"FT-пул: {name}")
        print("=" * 70)

        df = pd.read_parquet(path)
        masks = head_masks(df)

        df = pd.concat([df, masks], axis=1)

        mask_cols = list(cfg.HEAD_MASK_COLUMNS.values())
        df["ft_row"] = (df[mask_cols].sum(axis=1) > 0).astype("int8")

        pool = df[df["ft_row"] == 1].copy()

        print(f"Строк всего:            {len(df)}")
        print(f"Строк в FT-пуле:        {len(pool)} ({len(pool)/len(df):.1%})")
        for head, col in cfg.HEAD_MASK_COLUMNS.items():
            print(f"  {head:<15} размечено: {int(pool[col].sum())}")

        # Пересечение масок углового окна и голов
        outside = pool[pool[cfg.ANGLE_MASK_COLUMN] == 0]
        print(
            f"\nFT-строк ВНЕ окна [0, 90]: "
            f"{len(outside)} ({len(outside)/len(pool):.1%}) - при обучении "
            f"дополнительно гейтятся маской {cfg.ANGLE_MASK_COLUMN}"
        )
        if len(outside):
            for head, col in cfg.HEAD_MASK_COLUMNS.items():
                n_out = int(outside[col].sum())
                if n_out:
                    print(f"  {head:<15} вне окна: {n_out}")

        pool = normalize_object_columns(pool)

        pool.to_parquet(out_path, index=False)
        frames.append(pool)
        print()

    combined = pd.concat(frames, ignore_index=True)
    combined = normalize_object_columns(combined)
    combined.to_parquet(cfg.FT_POOL_COMBINED_PARQUET, index=False)

    print("=" * 70)
    print("СВОДКА ПО ИСТОЧНИКАМ FT-ПУЛА")
    print("=" * 70)
    counts = combined["dataset_role"].value_counts()
    for role, n in counts.items():
        print(f"{role}: {n} ({n/len(combined):.1%})")
    print(f"Итого: {len(combined)}")

    print("\nСохранены:")
    print(f"  {cfg.FT_POOL_RRUFF_PARQUET}")
    print(f"  {cfg.FT_POOL_OPXRD_PARQUET}")
    print(f"  {cfg.FT_POOL_COMBINED_PARQUET}")


if __name__ == "__main__":
    sys.exit(main())
