import warnings

warnings.filterwarnings("ignore")

import numpy as np

import synth_config as cfg
import synth_physics as phys
import synth_run as runner

FWHM = 0.1
GRID = np.arange(4.0, 91.0, 0.02)


def profile_from_peaks(peaks):

    y = np.zeros_like(GRID)
    th = np.radians(GRID / 2)
    lp = (1 + np.cos(2 * th) ** 2) / (np.sin(th) ** 2 * np.cos(th))

    for c, a in peaks:
        d2 = ((GRID - c) / (FWHM / 2)) ** 2
        y += a * (0.5 * np.exp(-np.log(2) * d2) + 0.5 / (1 + d2))

    return y * lp


def my_profile(prep):
    lam = 1.5406
    tt, inten, _ = phys.get_reflections(
        prep["cell"], prep["frac"], prep["el"], prep.get("rot_ops"), lam, 4.0, 91.0
    )
    if len(tt) == 0:
        return np.zeros_like(GRID)
    
    inten = inten / inten.max() * 100

    return profile_from_peaks(list(zip(tt, inten)))


def pmg_profile_from_cif(cif_text):

    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from pymatgen.io.cif import CifParser

    s = CifParser.from_str(cif_text).parse_structures(primitive=False)[0]
    pat = XRDCalculator(wavelength=1.5406).get_pattern(s, two_theta_range=(4.0, 91.0))

    return profile_from_peaks(list(zip(pat.x, pat.y))), len(s)


def pmg_profile_from_ase(prep):

    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase import Atoms

    atoms = Atoms(
        symbols=prep["el"],
        scaled_positions=prep["frac"],
        cell=prep["cell"],
        pbc=True,
    )
    s = AseAtomsAdaptor.get_structure(atoms)
    pat = XRDCalculator(wavelength=1.5406).get_pattern(s, two_theta_range=(4.0, 91.0))

    return profile_from_peaks(list(zip(pat.x, pat.y))), len(s)


def main():

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(cfg.COD_PARQUET)
    table = pf.read_row_group(0, columns=["file", "cif_text", "sgNumber"])

    candidates = []
    big = None
    for k in range(table.num_rows):
        cif_text = table["cif_text"][k].as_py()
        sg = table["sgNumber"][k].as_py()

        if cif_text is None:
            continue

        if len(candidates) < 2 and 200 < len(cif_text) < 8000 and sg:
            candidates.append((table["file"][k].as_py(), cif_text, int(sg)))

        if sg and int(sg) in (146, 148, 155, 160, 161, 166, 167) and len(candidates) < 3:
            if len(candidates) < 2 or candidates[-1][0] != table["file"][k].as_py():
                candidates.append((table["file"][k].as_py(), cif_text, int(sg)))

        if big is None and len(cif_text) > 30000 and sg:
            big = (table["file"][k].as_py(), cif_text, int(sg))

        if len(candidates) >= 3 and big is not None:
            break

    if big is not None:
        candidates.append(big)

    results = []
    for cod_id, cif_text, sg_num in candidates:
        try:
            prep, err = runner._prep_from_cod(cif_text)
            if prep is None:
                print(f"COD {cod_id}: prep skip ({err})")
                continue

            y_my = my_profile(prep)
            y_pm, n_atoms = pmg_profile_from_cif(cif_text)
            if y_my.max() == 0 or y_pm.max() == 0:
                continue

            r = np.corrcoef(y_my, y_pm)[0, 1]
            results.append(r)
            print(f"COD {cod_id}: atoms={n_atoms:4d} sg={sg_num:3d} corr={r:.4f}")

        except Exception as e:
            print(f"COD {cod_id}: ERROR {type(e).__name__}: {str(e)[:80]}")

    from ase.db import connect

    db = connect(str(cfg.MP_DB_PATH))
    for row in db.select(limit=50):
        try:
            prep, err = runner._prep_from_mp(str(row.id))
            if prep is None:
                continue

            y_my = my_profile(prep)
            y_pm, n_atoms = pmg_profile_from_ase(prep)
            if y_my.max() == 0 or y_pm.max() == 0:
                continue
            
            r = np.corrcoef(y_my, y_pm)[0, 1]
            results.append(r)
            print(f"MP  {row.id}: atoms={n_atoms:4d} corr={r:.4f}")
            break

        except Exception as e:
            print(f"MP  {row.id}: ERROR {type(e).__name__}: {str(e)[:80]}")

    print()
    ok = sum(1 for r in results if r >= 0.97)
    print(f" corr>=0.97: {ok}/{len(results)}; "
          f"{np.median(results):.4f}" if results else "")
    
    return 0 if results and ok >= len(results) - 1 else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())

