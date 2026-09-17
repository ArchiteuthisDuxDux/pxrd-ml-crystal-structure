"""
Оркестратор генерации синтетики из COD и crystalDB

Запуск (из корня DS_XRD_project):
    python synth_run.py --source both --limit 150        # пилот
    python synth_run.py --source cod    --workers 10     # полный прогон
    python synth_run.py --source crystaldb

    - COD читается построчно;
      строки с непустым duplicateof пропускаются;
    - CIF парсится gemmi, битые/кривые спокойно пропускаются;
    - движок отражений - собственный numpy (см. synth_physics), валидирован
      против pymatgen;
    - возобновляемость: успешные id дописываются в outputs/synth_done_*.txt;
    - результаты - шардами в data/clean/synth_shards/, финальный паркет
      собирает synth_finalize.py;
    - .npy пишутся в data/interim/raw: COD_<id>.npy, crystalDB_<id>.npy.
"""

import argparse
from pathlib import Path
import json
import os
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pandas as pd

import synth_config as cfg
import synth_physics as phys

# Воркер

_MP_CONN = None


def _init_worker():
    warnings.filterwarnings("ignore")
    global _MP_CONN


def _extract_rot_ops(small):

    """
    Вращательные части операций симметрии из CIF как список int-матриц 3x3.
    R[:, i] = op.apply_to_hkl(e_i). Пиклится, в отличие от gemmi.Op.
    """

    import gemmi

    ops = []
    symops = getattr(small, "symops", None) or []
    for s in symops:
        try:
            op = s if isinstance(s, gemmi.Op) else gemmi.Op(str(s))

        except Exception:
            continue

        R = np.zeros((3, 3), dtype=np.int64)

        for i, e in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
            v = op.apply_to_hkl(e)
            R[:, i] = (int(v[0]), int(v[1]), int(v[2]))
        ops.append(R)

    return ops if ops else None


def _prep_from_cod(cif_text):

    """CIF -> dict(cell, frac, el, rot_ops). Битые -> (None, причина)."""

    import gemmi

    doc = gemmi.cif.read_string(cif_text)
    block = doc.sole_block()
    small = gemmi.make_small_structure_from_block(block)

    sites = small.get_all_unit_cell_sites()
    if not sites:
        return None, "no_sites"

    cell_params = (
        float(small.cell.a), float(small.cell.b), float(small.cell.c),
        float(small.cell.alpha), float(small.cell.beta), float(small.cell.gamma),
    )
    if min(cell_params[:3]) <= 0:
        return None, "bad_cell"

    frac = np.empty((len(sites), 3))
    els = []
    for i, st in enumerate(sites):
        if st.occ < 0.99:
            return None, "disordered"
        
        frac[i] = (st.fract.x, st.fract.y, st.fract.z)
        name = st.element.name

        if not name or name == "X":
            return None, "unknown_element"
        
        els.append(name)

    frac %= 1.0
    return {
        "cell": cell_params,
        "frac": frac,
        "el": els,
        "rot_ops": _extract_rot_ops(small),
    }, None


def _prep_from_mp(sid):
    global _MP_CONN
    from ase.db import connect

    if _MP_CONN is None:
        _MP_CONN = connect(str(cfg.MP_DB_PATH))
    row = _MP_CONN.get(id=int(sid))
    atoms = row.toatoms()

    if len(atoms) == 0:
        return None, "empty_atoms"

    cell_params = tuple(float(v) for v in atoms.cell.cellpar())

    if min(cell_params[:3]) <= 0:
        return None, "bad_cell"

    frac = atoms.get_scaled_positions() % 1.0
    els = atoms.get_chemical_symbols()

    return {"cell": cell_params, "frac": frac, "el": els, "rot_ops": None}, None


def _resolve_spacegroup(declared):

    """
    Только заявленный номер SG - ИСКЛЮЧИТЕЛЬНО как метка для головы
    (COD - колонка sgNumber, crystalDB - пре-расчёт data/mp_sg.csv).
    В физике перечисления номер не используется: операции симметрии
    берутся из самого CIF (см. _extract_rot_ops).
    """

    if declared is not None and not pd.isna(declared):
        try:
            n = int(declared)
            if 1 <= n <= 230:
                return n
            
        except Exception:
            pass

    return None


def _reduced_formula(el_list):
    try:
        from pymatgen.core import Composition

        return Composition("".join(el_list)).reduced_formula
    
    except Exception:
        from collections import Counter
        from math import gcd

        cnt = Counter(el_list)
        g = 0
        for v in cnt.values():
            g = gcd(g, v)
        parts = [
            f"{e}{'' if v // g == 1 else v // g}" for e, v in sorted(cnt.items())
        ]

        return "".join(parts)


def _process_task(task):

    """
    task = dict(source, sid, cif_text|None, sg_declared)
    Возвращает (sid, status, reason, row_dict | None, pt | None).
    """

    source = task["source"]
    sid = task["sid"]
    seed = phys.sample_seed(source, sid)

    try:
        if source == "cod":
            prep, err = _prep_from_cod(task["cif_text"])

        else:
            prep, err = _prep_from_mp(sid)

        if prep is None:
            return sid, "skipped", err, None, None

        if len(prep["el"]) > cfg.MAX_ATOMS:
            return sid, "skipped", "too_many_atoms", None, None

        sg_num = _resolve_spacegroup(task.get("sg_declared"))

        mode = phys.sample_mode(np.random.default_rng(seed))
        grid, y_counts, meta, peaks_arr, amp_scale = phys.generate_spectrum(
            prep["cell"], prep["frac"], prep["el"], prep["rot_ops"],
            mode, seed ^ 0x9E3779B9
        )

        prefix = "COD" if source == "cod" else "crystalDB"
        fname = f"{prefix}_{sid}.npy"
        np.save(cfg.RAW_SPECTRA_DIR / fname, np.column_stack([grid, y_counts]))

        abc = list(prep["cell"][:3])
        angles = list(prep["cell"][3:])
        lat_row = abc + angles

        elements = sorted(set(prep["el"]))
        formula = _reduced_formula(prep["el"])

        row = {
            "sample_id": f"{'cod' if source == 'cod' else 'crystaldb'}_{sid}",
            "source": f"{prefix}_{sid}",
            "dataset_role": source,
            "raw_spectrum_path": f"raw/{fname}",
            "primary_wavelength": mode["lambda_1"],
            "secondary_wavelength": mode["lambda_2"],
            "lattice_a": lat_row[0], "lattice_b": lat_row[1], "lattice_c": lat_row[2],
            "alpha": lat_row[3], "beta": lat_row[4], "gamma": lat_row[5],
            "spacegroup_number": sg_num,
            "crystal_system": cfg.sg_number_to_crystal_system(sg_num),
            "elements": elements,
            "phase_count": 1,
            "phase_fraction": [1.0],
            "has_lattice": True,
            "has_spacegroup": sg_num is not None,
            "has_elements": True,
            "has_phase_count": True,
            "has_phase_fraction": True,
            "phase_compositions": [formula],
            "spacegroups_all": [sg_num] if sg_num else None,
            "lattices_all": [lat_row],
            "is_single_phase": True,
            "is_simulated": True,
            "crystallite_size_nm": meta["crystallite_size_nm"],
            "temp_K": None,
            "elements_list": elements,
            "elements_json": json.dumps(elements),
        }
        pt = (peaks_arr, amp_scale, meta["zero_shift"], meta["displacement"])

        return sid, "ok", None, row, pt

    except Exception as e:
        return sid, "skipped", f"error:{type(e).__name__}:{str(e)[:120]}", None, None


# Генераторы задач

def iter_cod_tasks(done, limit):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(cfg.COD_PARQUET)
    taken = 0

    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=cfg.COD_COLUMNS)
        df = table.to_pandas()

        if cfg.SKIP_COD_DUPLICATES:
            df = df[df["duplicateof"].isna()]

        for r in df.itertuples(index=False):
            sid = str(int(r.file))

            if sid in done:
                continue
            cif_text = r.cif_text

            if not isinstance(cif_text, str) or len(cif_text) < 50:
                continue

            yield {
                "source": "cod",
                "sid": sid,
                "cif_text": cif_text,
                "sg_declared": getattr(r, "sgNumber", None),
            }
            taken += 1

            if limit and taken >= limit:
                return


def iter_mp_tasks(done, limit):
    from ase.db import connect

    # пре-расчитанные space group (precompute_mp_sg.py)

    sg_map = {}
    sg_csv = cfg.PIPELINE_ROOT / "data" / "mp_sg.csv"

    if sg_csv.exists():
        df_sg = pd.read_csv(sg_csv, dtype={"mp_id": str})
        sg_map = dict(zip(df_sg["mp_id"], df_sg["sg"]))
        print(f"[crystaldb] sg-таблица: {len(sg_map)} записей", flush=True)

    db = connect(str(cfg.MP_DB_PATH))
    taken = 0

    for row in db.select():
        sid = str(row.id)

        if sid in done:
            continue

        sg = sg_map.get(sid)
        sg_decl = int(sg) if sg is not None and sg >= 1 else None
        yield {
            "source": "crystaldb",
            "sid": sid,
            "cif_text": None,
            "sg_declared": sg_decl,
        }
        taken += 1
        if limit and taken >= limit:
            return


# Основной цикл

def run(source, limit, workers):
    cfg.SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RAW_SPECTRA_DIR.mkdir(parents=True, exist_ok=True)

    done_path = cfg.DONE_LISTS[source]
    done = set()
    if done_path.exists():
        done = set(done_path.read_text(encoding="utf-8").split())
    print(f"[{source}] уже готово (resume): {len(done)}", flush=True)

    state = {
        "rows": [],
        "pt": [],
        "shard_idx": len(list(cfg.SHARDS_DIR.glob(f"{source}_part_*.parquet"))),
        "ok": 0,
        "skip": 0,
        "reasons": {},
    }
    t0 = time.time()

    cfg.PEAKTABLE_DIR.mkdir(parents=True, exist_ok=True)

    def flush_shard():
        if not state["rows"]:
            return
        
        out = cfg.SHARDS_DIR / f"{source}_part_{state['shard_idx']:05d}.parquet"
        pd.DataFrame(state["rows"], columns=cfg.SCHEMA_COLUMNS).to_parquet(out, index=False)

        # таблицы пиков: для перерендера дисторшнов без пересчёта физики

        pts = state["pt"]
        peaks = (np.concatenate([p[0] for p in pts])
                 if any(len(p[0]) for p in pts) else np.zeros((0, 4), np.float32))
        offsets = np.concatenate([[0], np.cumsum([len(p[0]) for p in pts])]).astype(np.int64)
        np.savez_compressed(
            cfg.PEAKTABLE_DIR / f"{source}_pt_{state['shard_idx']:05d}.npz",
            peaks=peaks.astype(np.float32),
            offsets=offsets,
            amp_scale=np.array([p[1] for p in pts], dtype=np.float32),
            zero_shift=np.array([p[2] for p in pts], dtype=np.float32),
            displacement=np.array([p[3] for p in pts], dtype=np.float32),
            sample_ids=np.array([r["source"] for r in state["rows"]]),
        )

        state["rows"] = []
        state["pt"] = []
        state["shard_idx"] += 1

    done_file = open(done_path, "a", encoding="utf-8")
    skip_log = open(cfg.OUTPUTS_DIR / f"synth_skipped_{source}.log", "a", encoding="utf-8")

    def handle_result(res):
        sid, status, reason, row, pt = res

        if status == "ok":
            state["rows"].append(row)
            state["pt"].append(pt)
            done_file.write(sid + "\n")
            if len(state["rows"]) >= 400:
                flush_shard()
            state["ok"] += 1

        else:
            state["skip"] += 1
            key = reason.split(":")[0]
            state["reasons"][key] = state["reasons"].get(key, 0) + 1
            skip_log.write(f"{sid}\t{reason}\n")

        total = state["ok"] + state["skip"]

        if total % 100 == 0:
            dt = time.time() - t0
            print(
                f"[{source}] обработано {total} | ок {state['ok']} | "
                f"пропуск {state['skip']} | {dt/max(total,1):.3f} с/шт | "
                f"elapsed {dt/60:.1f} мин",
                flush=True,
            )

    if source == "cod":
        tasks = iter_cod_tasks(done, limit)

    else:
        tasks = iter_mp_tasks(done, limit)

    max_inflight = workers * 4
    inflight = set()
    task_iter = iter(tasks)

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:

        def submit_more():
            for t in task_iter:
                inflight.add(ex.submit(_process_task, t))
                if len(inflight) >= max_inflight:

                    return

        submit_more()
        while inflight:
            finished, _ = wait(inflight, return_when=FIRST_COMPLETED)

            for f in finished:
                inflight.discard(f)
                handle_result(f.result())
            submit_more()

    flush_shard()
    done_file.close()
    skip_log.close()

    dt = time.time() - t0
    print("=" * 70)
    print(f"[{source}] ГОТОВО за {dt/60:.1f} мин")
    print(f"  успешно: {state['ok']} | пропущено: {state['skip']}")
    if state["reasons"]:
        print(f"  причины пропусков: {state['reasons']}")
    print(f"  шарды: {cfg.SHARDS_DIR}\\{source}_part_*.parquet")

    # манифест генерации: полная прослеживаемость параметров

    import hashlib
    import json as _json
    from datetime import datetime, timezone

    cfg_hash = hashlib.md5(
        Path(__file__).with_name("synth_config.py").read_bytes()
    ).hexdigest()[:12]
    manifest = {
        "generation_version": cfg.GENERATION_VERSION,
        "source": source,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "synth_config_md5": cfg_hash,
        "n_ok": state["ok"],
        "n_skip": state["skip"],
        "skip_reasons": state["reasons"],
        "params": {
            "counts_log_range": cfg.COUNTS_LOG_RANGE,
            "cheb_base_range": cfg.CHEB_BASE_RANGE,
            "caglioti_w_range": cfg.CAGLIOTI_W_RANGE,
            "crystallite_logmu": cfg.CRYSTALLITE_SIZE_LOGMU,
            "strain_sigma": cfg.STRAIN_SIGMA,
            "halo": cfg.HALO,
            "kalpha2_weight": cfg.KALPHA2_WEIGHT,
        },
    }
    mpath = cfg.OUTPUTS_DIR / f"synth_manifest_{source}.json"
    mpath.write_text(_json.dumps(manifest, indent=2, ensure_ascii=False),
                     encoding="utf-8")
    print(f"  манифест: {mpath.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["cod", "crystaldb", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="максимум новых задач на источник")
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 4))
    args = ap.parse_args()

    sources = ["cod", "crystaldb"] if args.source == "both" else [args.source]

    for src in sources:
        run(src, args.limit, args.workers)


if __name__ == "__main__":
    main()
