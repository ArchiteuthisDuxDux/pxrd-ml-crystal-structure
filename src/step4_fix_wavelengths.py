"""
Шаг 4. Нормализация длин волн (исправление перепутанных primary/secondary).

Физическая конвенция: primary = Kalpha1 (короче, сильнее),
secondary = Kalpha2 (длиннее, ~0.5 интенсивности).

Что делается для rruff/opxrd clean и всех ft_pool:
    - если обе линии указаны и primary > secondary -> поменять местами;
    - secondary == 0 -> NaN (дублета нет);
    - синтетика уже согласована (lambda_1 <= lambda_2 по построению) - проверяется.

Идемпотентно: повторный запуск ничего не меняет.
"""

import sys

import numpy as np
import pandas as pd

import config as cfg


def normalize_wavelengths(df, name):
    wl1 = pd.to_numeric(df["primary_wavelength"], errors="coerce")
    wl2 = pd.to_numeric(df["secondary_wavelength"], errors="coerce")

    n_zero = int((wl2 == 0).sum())
    wl2 = wl2.replace(0.0, np.nan)

    both = wl1.notna() & wl2.notna()
    n_swap = int((both & (wl1 > wl2)).sum())

    lam_short = np.minimum(wl1, wl2)
    lam_long = np.maximum(wl1, wl2)

    out = df.copy()
    out["primary_wavelength"] = np.where(both, lam_short, wl1)
    out["secondary_wavelength"] = np.where(both, lam_long, wl2)

    # контроль согласованности после правки
    both2 = out["primary_wavelength"].notna() & out["secondary_wavelength"].notna()
    assert not (both2 & (out["primary_wavelength"] > out["secondary_wavelength"])).any()

    print(
        f"{name}: дублетов {int(both2.sum())}, исправлено swap {n_swap}, "
        f"secondary=0 -> NaN {n_zero}, без primary {int(out['primary_wavelength'].isna().sum())}"
    )
    return out


def main():
    targets = [
        ("df_rruff_clean", cfg.RRUFF_CLEAN_PARQUET),
        ("df_opxrd_clean", cfg.OPXRD_CLEAN_PARQUET),
        ("ft_pool_rruff", cfg.FT_POOL_RRUFF_PARQUET),
        ("ft_pool_opxrd", cfg.FT_POOL_OPXRD_PARQUET),
        ("ft_pool_combined", cfg.FT_POOL_COMBINED_PARQUET),
    ]

    for name, path in targets:
        df = pd.read_parquet(path)
        out = normalize_wavelengths(df, name)
        out.to_parquet(path, index=False)

    # синтетика: только проверка (кратчайшая должна быть primary)
    import synth_config as scfg

    if scfg.FINAL_PARQUET.exists():
        syn = pd.read_parquet(scfg.FINAL_PARQUET)
        w1 = pd.to_numeric(syn["primary_wavelength"], errors="coerce")
        w2 = pd.to_numeric(syn["secondary_wavelength"], errors="coerce")
        both = w1.notna() & w2.notna()
        n_bad = int((both & (w1 > w2)).sum())
        print(f"синтетика: дублетов {int(both.sum())}, нарушений порядка {n_bad} (ожидался 0)")
        assert n_bad == 0
    else:
        print("синтетика ещё не создана; её wavelength-пары будут проверены после генерации")
    print("Готово.")


if __name__ == "__main__":
    sys.exit(main())
