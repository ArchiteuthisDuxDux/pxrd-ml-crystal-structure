"""
Воркер пре-расчёта space group для crystalDB через spglib.

Используется ТОЛЬКО из precompute_mp_sg.py в изолированном субпроцессе,
потому что spglib на некоторых структурах падает сегфолтом.

Аргументы: start end  (индексы в data/mp_ids.txt)
Печать: "<id> <sg|E>" по строке на структуру, flush после каждой.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def main():
    start, end = int(sys.argv[1]), int(sys.argv[2])

    import numpy as np
    import gemmi
    import spglib
    from ase.db import connect

    from synth_physics import lattice_matrix_from_params

    project_root = Path(__file__).resolve().parents[1]

    with open(project_root / "data" / "mp_ids.txt", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]

    db = connect(str(project_root / "data" / "MP.db"))

    for idx in range(start, min(end, len(ids))):
        sid = ids[idx]
        try:
            atoms = db.get(id=int(sid)).toatoms()
            cell = tuple(float(v) for v in atoms.cell.cellpar())
            frac = atoms.get_scaled_positions() % 1.0
            numbers = [gemmi.Element(e).atomic_number for e in atoms.get_chemical_symbols()]
            matrix = lattice_matrix_from_params(*cell)
            ds = spglib.get_symmetry_dataset((matrix, frac, numbers), symprec=1e-3)
            sg = int(ds.number) if 1 <= int(ds.number) <= 230 else -1
        except Exception:
            sg = -2
        print(f"{sid} {sg}", flush=True)


if __name__ == "__main__":
    main()
