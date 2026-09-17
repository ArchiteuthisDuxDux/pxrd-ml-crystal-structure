"""Проверка: где в реальных данных есть/нет длин волн и какие при этом размечены головы."""

import sys

import numpy as np
import pandas as pd

import config as cfg


def wl_table(df, name):
    wl1 = pd.to_numeric(df["primary_wavelength"], errors="coerce")
    wl2 = pd.to_numeric(df["secondary_wavelength"], errors="coerce")

    both = int((wl1.notna() & wl2.notna()).sum())
    only1 = int((wl1.notna() & wl2.isna()).sum())
    only2 = int((wl1.isna() & wl2.notna()).sum())          # только второстепенная!
    none = int((wl1.isna() & wl2.isna()).sum())

    print(f"\n--- {name}: {len(df)} строк ---")
    print(f"обе линии (λ1+λ2):        {both}")
    print(f"только основная (λ1):     {only1}")
    print(f"только второстепенная:    {only2}")
    print(f"ни одной:                 {none}")

    # строки совсем без λ - что у них размечено
    m_none = df[wl1.isna() & wl2.isna()]
    if len(m_none):
        lat = int(m_none[cfg.LATTICE_COLUMNS].notna().all(axis=1).sum())
        sg = int(m_none["spacegroup_number"].notna().sum())
        cs = int(m_none["crystal_system"].notna().sum())
        el = int(m_none["elements_list"].apply(
            lambda v: isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0).sum())
        any_head = int(((m_none[cfg.LATTICE_COLUMNS].notna().all(axis=1)) |
                        m_none["spacegroup_number"].notna() |
                        m_none["crystal_system"].notna() |
                        m_none["elements_list"].apply(
                            lambda v: isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0)).sum())
        print(f"  среди них размечено: lattice={lat}, sg={sg}, system={cs}, elements={el}, "
              f"хотя бы одна голова={any_head} из {len(m_none)}")
        print(f"  по источникам: {m_none['dataset_role'].value_counts().to_dict()}")

    # только второстепенная - примеры
    m_only2 = df[wl1.isna() & wl2.notna()]
    if len(m_only2):
        print(f"  'только второстепенная' примеры λ2: "
              f"{pd.to_numeric(m_only2['secondary_wavelength'], errors='coerce').round(4).value_counts().head(5).to_dict()}")


def main():
    for path, name in [
        (cfg.RRUFF_CLEAN_PARQUET, "RRUFF (весь чистый)"),
        (cfg.OPXRD_CLEAN_PARQUET, "opXRD (весь чистый)"),
        (cfg.FT_POOL_COMBINED_PARQUET, "FT-пул combined"),
    ]:
        df = pd.read_parquet(path)
        wl_table(df, name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
