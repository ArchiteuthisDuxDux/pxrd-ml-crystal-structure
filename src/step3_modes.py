"""
Шаг 3. Режимы работы генератора (modes) и их распределения.

Режим = (lambda_primary, lambda_secondary, x_min, x_max, step):

    x_min, x_max  - округляются до целого градуса
                    (89.99 и 90.00 -> один режим);
    длины волн    - кластеризация с допуском 5e-4 А
                    (варианты записи одной линии сливаются,
                    дублет Kalpha1/Kalpha2 не сливается);
    шаг           - кластеризация с допуском max(2e-4 abs, 1e-3 rel).

Нормализация двойни: если указаны обе линии, короткая считается
Kalpha1 (primary, интенсивность 1.0), длинная - Kalpha2 (~0.5).
Это чинит 12,624 строки opXRD, где в исходнике они перепутаны;
важно и для q-space: иначе слабая линия попадает не на свой q.

Иерархия сэмплирования для генератора:
    1) источник: rruff <-> opxrd с вероятностью 50/50;
    2) внутри источника: режим пропорционально частоте встречаемости.
"""

import sys

import numpy as np
import pandas as pd

import config as cfg
from utils_spectra import cluster_values


def prepare_frame(df):

    """Добавляем нормализованные компоненты режима."""

    out = df.copy()

    # --- длины волн -------------------------------------------------

    wl1 = pd.to_numeric(out["primary_wavelength"], errors="coerce")
    wl2 = pd.to_numeric(out["secondary_wavelength"], errors="coerce")

    # вторичная линия = 0 - мусор, превращаем в NaN

    n_zero = int((wl2 == 0).sum())
    wl2 = wl2.replace(0.0, np.nan)

    lam_short = np.minimum(wl1, wl2)
    lam_long = np.maximum(wl1, wl2)

    out["lambda_1"] = np.where(wl2.notna(), lam_short, wl1)
    out["lambda_2"] = np.where(wl2.notna(), lam_long, np.nan)

    n_swapped = int((wl2.notna() & (wl1 > wl2)).sum())

    # Кластеризация вариантов записи (представитель - самое частое значение)

    wl1_counts = out["lambda_1"].value_counts().to_dict()
    map1 = cluster_values(sorted(out["lambda_1"].dropna().unique()),
                          atol=cfg.WAVELENGTH_TOL, weights=wl1_counts)
    out["lambda_1"] = out["lambda_1"].map(map1)

    known2 = out["lambda_2"].dropna()
    map2 = cluster_values(sorted(known2.unique()),
                          atol=cfg.WAVELENGTH_TOL,
                          weights=out["lambda_2"].value_counts().to_dict())
    out.loc[known2.index, "lambda_2"] = known2.map(map2)

    # --- углы: целые градусы ----------------------------------------

    out["x_min_mode"] = np.round(out["x_min"]).astype("int64")
    out["x_max_mode"] = np.round(out["x_max"]).astype("int64")

    # --- шаг ---------------------------------------------------------

    steps = out["step"].dropna().round(7)
    map_s = cluster_values(sorted(steps.unique()),
                           atol=cfg.STEP_ATOL, rtol=cfg.STEP_RTOL,
                           weights=steps.value_counts().to_dict())
    out["step_mode"] = out["step"].round(7).map(map_s)

    return out, n_swapped, n_zero


MODE_KEY_COLUMNS = ["lambda_1", "lambda_2", "x_min_mode", "x_max_mode", "step_mode"]


def build_modes(df, ft_mask_series, name):
    modes = (
        df.groupby(MODE_KEY_COLUMNS, dropna=False)
        .size()
        .reset_index(name="count_all")
    )
    modes = modes.sort_values(
        MODE_KEY_COLUMNS, na_position="last"
    ).reset_index(drop=True)

    # статистика по FT-пулу
    ft = df[ft_mask_series]
    ft_counts = (
        ft.groupby(MODE_KEY_COLUMNS, dropna=False)
        .size()
        .reset_index(name="count_ft")
    )
    modes = modes.merge(ft_counts, on=MODE_KEY_COLUMNS, how="left")
    modes["count_ft"] = modes["count_ft"].fillna(0).astype("int64")

    total = len(df)
    modes["fraction_all"] = modes["count_all"] / total
    modes = modes.sort_values(["count_all", "fraction_all"], ascending=False).reset_index(drop=True)
    modes["cumulative_fraction_all"] = modes["fraction_all"].cumsum()

    ft_total = int(ft_mask_series.sum())
    modes["fraction_ft"] = modes["count_ft"] / ft_total if ft_total else 0.0

    modes["mode_id"] = np.arange(len(modes))

    # округление для читаемого CSV

    for c in ["lambda_1", "lambda_2", "step_mode"]:
        modes[c] = modes[c].round(6)

    fname = cfg.MODES_OPXRD_CSV if name == "opxrd" else cfg.MODES_RRUFF_CSV
    modes.to_csv(fname, index=False)
    return modes


def coverage_report(modes, name):
    print(f"\n{name}: покрытие топ-k режимов")
    print(f"{'k':>6} {'all':>10} {'ft_pool':>10}")
    for k in [1, 3, 5, 10, 20, 50, 100, 200, 500]:
        if k > len(modes):
            break
        a = modes["cumulative_fraction_all"].iloc[k - 1]
        f = modes.loc[: k - 1, "fraction_ft"].sum()
        print(f"{k:>6} {a:>10.2%} {f:>10.2%}")


def main():
    ft_combined = pd.read_parquet(cfg.FT_POOL_COMBINED_PARQUET)
    ft_ids = set(ft_combined["sample_id"])

    results = {}

    for name, path in [
        ("rruff", cfg.RRUFF_CLEAN_PARQUET),
        ("opxrd", cfg.OPXRD_CLEAN_PARQUET),
    ]:
        print("=" * 70)
        print(f"РЕЖИМЫ: {name}")
        print("=" * 70)

        df = pd.read_parquet(path)
        df, n_swapped, n_zero = prepare_frame(df)

        if name == "opxrd":
            print(f"Двойня с перепутанными линиями исправлена: {n_swapped} строк")
            print(f"Строк с secondary=0 (считаются без двойни): {n_zero}")

        ft_mask = df["sample_id"].isin(ft_ids)

        modes = build_modes(df, ft_mask, name)

        print(f"\nСтрок: {len(df)} | уникальных режимов: {len(modes)}")
        print(f"Режимов с неизвестной lambda (NaN): "
              f"{int(modes['lambda_1'].isna().sum())} "
              f"(доля {modes.loc[modes['lambda_1'].isna(), 'fraction_all'].sum():.2%})")

        show_cols = [
            "mode_id", "lambda_1", "lambda_2",
            "x_min_mode", "x_max_mode", "step_mode",
            "count_all", "fraction_all", "cumulative_fraction_all", "count_ft",
        ]
        print("\nTOP-15:")
        print(modes[show_cols].head(15).to_string(index=False))

        coverage_report(modes, name)
        results[name] = modes
        print()

    # Итоговая схема сэмплирования генератора

    print("=" * 70)
    print("СХЕМА СЭМПЛИРОВАНИЯ РЕЖИМОВ ДЛЯ ГЕНЕРАТОРА")
    print("=" * 70)
    for src, w in cfg.SOURCE_SAMPLE_WEIGHTS.items():
        m = results[src]
        print(
            f"P(источник={src}) = {w:.0%}; далее режим ~ "
            f"'fraction_all' ({len(m)} режимов, Top-50 покрывает "
            f"{m['cumulative_fraction_all'].iloc[min(49, len(m)-1)]:.1%})"
        )
    print("\nCSV режимов:", cfg.MODES_RRUFF_CSV.name, ",", cfg.MODES_OPXRD_CSV.name)


if __name__ == "__main__":
    sys.exit(main())
