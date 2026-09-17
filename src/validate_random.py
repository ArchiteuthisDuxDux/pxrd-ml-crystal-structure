import random
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import synth_config as cfg
import synth_run as runner
from validate_engine import (
    my_profile,
    pmg_profile_from_ase,
    pmg_profile_from_cif,
)


def sample_cod_candidates(rng, n_groups=15, per_group=20):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(cfg.COD_PARQUET)
    rg_all = list(range(pf.num_row_groups))
    rng.shuffle(rg_all)

    candidates = []
    for rg in rg_all[:n_groups]:
        t = pf.read_row_group(rg, columns=["file", "cif_text", "sgNumber"])
        idxs = rng.choice(t.num_rows, size=min(per_group, t.num_rows), replace=False)

        for i in idxs:
            cif = t["cif_text"][int(i)].as_py()
            sg = t["sgNumber"][int(i)].as_py()

            if isinstance(cif, str) and len(cif) > 100 and sg is not None and not pd.isna(sg):
                candidates.append((str(t["file"][int(i)].as_py()), cif, int(float(sg))))

        if len(candidates) >= n_groups * per_group:
            break

    return candidates


def main():
    n_cod = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    n_mp = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    rng = np.random.default_rng(42)
    random.seed(42)

    corrs = []          # (corr, id, natoms)
    skipped = {}
    pmg_fail = 0

    # ---------------- COD ----------------
    candidates = sample_cod_candidates(rng)
    rng.shuffle(candidates)
    print(f"COD: {len(candidates)}")

    t0 = time.time()
    done = 0
    for sid, cif_text, sg_num in candidates:
        if done >= n_cod:
            break
        try:
            prep, err = runner._prep_from_cod(cif_text)
        except Exception as e:
            skipped[f"prep_{type(e).__name__}"] = skipped.get(f"prep_{type(e).__name__}", 0) + 1
            continue
        if prep is None:
            skipped[err] = skipped.get(err, 0) + 1
            continue

        try:
            y_my = my_profile(prep)
            y_pm, n_atoms = pmg_profile_from_cif(cif_text)
        except Exception as e:
            pmg_fail += 1
            skipped[f"pmg_{type(e).__name__}"] = skipped.get(f"pmg_{type(e).__name__}", 0) + 1
            continue

        if y_my.max() == 0 or y_pm.max() == 0:
            skipped["no_reflections"] = skipped.get("no_reflections", 0) + 1
            continue

        r = float(np.corrcoef(y_my, y_pm)[0, 1])
        corrs.append((r, f"COD_{sid}", n_atoms))
        done += 1
        if done % 20 == 0:
            arr = np.array([c for c, _, _ in corrs])
            print(f"  COD {done}/{n_cod} | corr {np.median(arr):.4f} | "
                  f"{time.time()-t0:.0f} c", flush=True)

    # ---------------- crystalDB ----------------
    import pandas as pd

    df_sg = pd.read_csv(cfg.PIPELINE_ROOT / "data" / "mp_sg.csv", dtype={"mp_id": str})
    ok_ids = df_sg.loc[df_sg["status"] == "ok", "mp_id"].tolist()
    mp_sample = random.sample(ok_ids, min(n_mp, len(ok_ids)))

    from ase.db import connect

    db = connect(str(cfg.MP_DB_PATH))
    done_mp = 0
    for sid in mp_sample:
        if done_mp >= n_mp:
            break
        try:
            prep, err = runner._prep_from_mp(sid)
            if prep is None:
                skipped[f"mp_{err}"] = skipped.get(f"mp_{err}", 0) + 1
                continue
            y_my = my_profile(prep)
            y_pm, n_atoms = pmg_profile_from_ase(prep)
            if y_my.max() == 0 or y_pm.max() == 0:
                skipped["no_reflections"] = skipped.get("no_reflections", 0) + 1
                continue
            r = float(np.corrcoef(y_my, y_pm)[0, 1])
            corrs.append((r, f"crystalDB_{sid}", n_atoms))
            done_mp += 1
        except Exception as e:
            skipped[f"mp_{type(e).__name__}"] = skipped.get(f"mp_{type(e).__name__}", 0) + 1

    arr = np.array([c for c, _, _ in corrs])
    print()

    print(f"{len(arr)} (COD {sum(1 for c in corrs if c[1].startswith('COD'))}, "
          f"crystalDB {sum(1 for c in corrs if c[1].startswith('crystalDB'))})")

    if len(arr):
        print(f"corr: {np.median(arr):.4f} |  {arr.mean():.4f} | "
              f"{arr.min():.4f} | p05 {np.percentile(arr, 5):.4f}")
        print(f"corr>=0.99: {np.mean(arr >= 0.99):.1%}")
        print(f"corr>=0.97: {np.mean(arr >= 0.97):.1%}")
        print(f"corr>=0.95: {np.mean(arr >= 0.95):.1%}")
        print(f"corr>=0.90: {np.mean(arr >= 0.90):.1%}")
        print()
        print("5:")

        for r, sid, n_atoms in sorted(corrs)[:5]:
            print(f"  {sid}: corr={r:.4f} (atoms={n_atoms})")

    if skipped:

        for k, v in sorted(skipped.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")

    gate = len(arr) > 0 and np.mean(arr >= 0.97) >= 0.90 and np.median(arr) >= 0.97

    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())

