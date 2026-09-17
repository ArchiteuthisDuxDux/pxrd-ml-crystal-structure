"""
Шаг 5 (этап B). Единый препроцессинг ВСЕХ спектров: синтетика + RRUFF + opXRD.

Для каждого спектра:
    1. загрузка .npy, сортировка по x, чистка NaN;
    2. валидная область = точки внутри окна [0, 90] градусов;
    3. нормализация: sqrt(I) / max по валидной области
       (sqrt - variance-stabilizing для Пуассона, выравнивает динамический
       диапазон и делает шум примерно гомоскедастичным);
    4. PCHIP-интерполяция на общий грид linspace(0, 90, 4096);
       вне валидной области интенсивность = 0;
    5. пер-точечная маска: 1 внутри [x_first_valid, x_last_valid].

Выход:
    data/preprocessed/X_intensity.f16      fp16 (N, 4096)
    data/preprocessed/M_mask.u8            uint8 (N, 4096)
    data/preprocessed/index_preprocessed.parquet
        sample_id, dataset_role, source, row_idx, lambda_1, lambda_2,
        x_min_valid, x_max_valid, head-маски, split_role
        (split_role: pretrain - синтетика, ft - есть целевая голова,
         unlabeled - реальный спектр без меток).

Запуск: python step5_preprocess_grid.py [--workers 16]
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

import config as cfg
from utils_spectra import is_nonempty_sequence

# Сборка единой таблицы задач

def head_masks(df):
    m = pd.DataFrame(index=df.index)
    m["head_mask_lattice"] = (
        df[cfg.LATTICE_COLUMNS].notna().all(axis=1).astype("int8")
    )
    m["head_mask_spacegroup"] = df["spacegroup_number"].notna().astype("int8")
    m["head_mask_crystal_system"] = df["crystal_system"].notna().astype("int8")
    m["head_mask_elements"] = (
        df["elements_list"].apply(is_nonempty_sequence).astype("int8")
    )
    return m


def build_task_table():
    import synth_config as scfg

    frames = []

    syn = pd.read_parquet(scfg.FINAL_PARQUET)
    syn = syn[["sample_id", "dataset_role", "source", "raw_spectrum_path",
               "primary_wavelength", "secondary_wavelength"] +
              cfg.LATTICE_COLUMNS +
              ["spacegroup_number", "crystal_system", "elements_list"]]
    frames.append(syn.assign(split_role="pretrain"))

    for path in [cfg.RRUFF_CLEAN_PARQUET, cfg.OPXRD_CLEAN_PARQUET]:
        df = pd.read_parquet(path)
        df = df[["sample_id", "dataset_role", "source", "raw_spectrum_path",
                 "primary_wavelength", "secondary_wavelength"] +
                cfg.LATTICE_COLUMNS +
                ["spacegroup_number", "crystal_system", "elements_list"]]
        frames.append(df.assign(split_role="real"))

    full = pd.concat(frames, ignore_index=True)
    masks = head_masks(full)
    full = pd.concat([full, masks], axis=1)

    mask_cols = list(cfg.HEAD_MASK_COLUMNS.values())
    full.loc[full["split_role"] == "real", "split_role"] = np.where(
        full.loc[full["split_role"] == "real", mask_cols].sum(axis=1) > 0,
        "ft", "unlabeled",
    )

    return full.reset_index(drop=True)

# Обработка одного спектра

GRID = np.linspace(cfg.GRID_LO, cfg.GRID_HI, cfg.GRID_N)

def retry_failed(workers):
    """Дочитывает строки с preprocess_failed=1 в существующие меммапы."""
    index = pd.read_parquet(cfg.INDEX_PARQUET)
    failed_idx = np.where(index["preprocess_failed"].to_numpy() == 1)[0]
    print(f"К: {len(failed_idx)} строк")
    if len(failed_idx) == 0:
        return

    X = np.memmap(cfg.X_INTENSITY_PATH, dtype=np.float16, mode="r+",
                  shape=(len(index), cfg.GRID_N))
    M = np.memmap(cfg.MASK_PATH, dtype=np.uint8, mode="r+",
                  shape=(len(index), cfg.GRID_N))

    base = cfg.RAW_SPECTRA_DIR.parent

    def reconstruct_path(row):
        role = row["dataset_role"]
        if role == "rruff":
            return base / "raw" / f"{row['source']}.npy"
        
        if role == "opxrd":
            return base / "raw" / f"{row['sample_id']}.npy"
        
        if role == "cod":
            return base / "raw" / f"COD_{row['sample_id'].split('_', 1)[1]}.npy"
        
        if role == "crystaldb":
            return base / "raw" / f"crystalDB_{row['sample_id'].split('_', 1)[1]}.npy"
        
        raise ValueError(role)

    x_lo = index["x_min_valid"].to_numpy(copy=True)
    x_hi = index["x_max_valid"].to_numpy(copy=True)
    n_native = index["n_native_points"].to_numpy(copy=True)

    still = []
    t0 = time.time()
    failed_paths = [str(reconstruct_path(index.iloc[i])) for i in failed_idx]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, res in zip(failed_idx, ex.map(process_one, failed_paths)):
            if res is None:
                still.append(i)
                continue

            yi, mk, lo, hi, npts = res
            X[i] = yi
            M[i] = mk
            x_lo[i], x_hi[i], n_native[i] = lo, hi, npts

    X.flush()
    M.flush()

    for i in still:
        M[i] = 0

    M.flush()

    index["x_min_valid"] = x_lo
    index["x_max_valid"] = x_hi
    index["n_native_points"] = n_native
    index["preprocess_failed"] = 0
    index.loc[still, "preprocess_failed"] = 1
    index.to_parquet(cfg.INDEX_PARQUET, index=False)

    print(f"Прочтено: {len(failed_idx) - len(still)} | осталось битых: {len(still)} "
          f"| {time.time()-t0:.0f} c")

    # контроль: по источникам

    print(index.groupby("dataset_role")["ошибка препроцессинга"].sum().to_string())


def process_one(npy_path):
    try:
        arr = np.load(npy_path, allow_pickle=False)

    except Exception:
        return None

    if arr.ndim != 2 or len(arr) < 2:
        return None

    # исправление форматом (N, 2) и (2, N)

    if arr.shape[1] == 2 and arr.shape[0] != 2:
        x = np.asarray(arr[:, 0], dtype=np.float64)
        y = np.asarray(arr[:, 1], dtype=np.float64)

    elif arr.shape[0] == 2:
        x = np.asarray(arr[0, :], dtype=np.float64)
        y = np.asarray(arr[1, :], dtype=np.float64)

    else:
        return None

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return None

    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]

    # дубликаты глюки RRUFF - оставляем первое вхождение

    keep = np.concatenate([[True], np.diff(x) > 0])
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return None

    y = np.clip(y, 0.0, None)

    valid = (x >= cfg.GRID_LO) & (x <= cfg.GRID_HI)

    if valid.sum() < 2:
        return None

    xv, yv = x[valid], y[valid]

    # sqrt-нормировка по максимуму валидной области

    ys = np.sqrt(yv)
    scale = ys.max()
    if scale > 0:
        yn = ys / scale

    else:
        yn = np.zeros_like(ys)

    interp = PchipInterpolator(xv, yn, extrapolate=False)
    yi = interp(GRID)
    yi = np.nan_to_num(yi, nan=0.0)
    yi = np.clip(yi, 0.0, 1.0)

    mask = ((GRID >= xv[0]) & (GRID <= xv[-1])).astype(np.uint8)
    yi = yi * mask  # вне валидной области - ноль

    return (
        yi.astype(np.float16),
        mask,
        float(xv[0]),
        float(xv[-1]),
        int(len(x)),
    )

# Основной цикл

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--retry-failed", action="store_true",
                    help="дочитать только строки с preprocess_failed=1 "
                         "(меммапы и индекс должны существовать)")
    args = ap.parse_args()

    cfg.PREPROC_DIR.mkdir(parents=True, exist_ok=True)

    if args.retry_failed:
        retry_failed(args.workers)
        return 0

    tasks = build_task_table()
    n = len(tasks)
    print(f"Спектров к обработке: {n}")
    print(tasks["split_role"].value_counts().to_string())

    X = np.memmap(cfg.X_INTENSITY_PATH, dtype=np.float16, mode="w+",
                  shape=(n, cfg.GRID_N))
    M = np.memmap(cfg.MASK_PATH, dtype=np.uint8, mode="w+",
                  shape=(n, cfg.GRID_N))

    base = cfg.RAW_SPECTRA_DIR.parent
    paths = [str(base / p) for p in tasks["raw_spectrum_path"]]

    x_lo = np.full(n, np.nan)
    x_hi = np.full(n, np.nan)
    n_native = np.full(n, -1, dtype=np.int64)
    failed = []

    t0 = time.time()
    done = 0
    CHUNK = 20000

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for c0 in range(0, n, CHUNK):
            c1 = min(c0 + CHUNK, n)

            for i, res in zip(range(c0, c1),
                              ex.map(process_one, paths[c0:c1])):
                
                if res is None:
                    failed.append(i)
                    continue

                yi, mk, lo, hi, npts = res
                X[i] = yi
                M[i] = mk
                x_lo[i], x_hi[i], n_native[i] = lo, hi, npts
            done = c1
            dt = time.time() - t0
            print(f"  {done}/{n} | {dt:.0f} c | {dt/done*1000:.2f} мс/спектр",
                  flush=True)

    X.flush()
    M.flush()

    # провалы
    for i in failed:
        M[i] = 0
    M.flush()

    # Индекс

    index = tasks[[
        "sample_id", "dataset_role", "source", "split_role",
        "primary_wavelength", "secondary_wavelength",
        "head_mask_lattice", "head_mask_spacegroup",
        "head_mask_crystal_system", "head_mask_elements",
    ]].copy()
    index["row_idx"] = np.arange(n, dtype=np.int64)
    index["x_min_valid"] = x_lo
    index["x_max_valid"] = x_hi
    index["n_native_points"] = n_native
    index["preprocess_failed"] = 0
    index.loc[failed, "preprocess_failed"] = 1
    index = index.rename(columns={
        "primary_wavelength": "lambda_1",
        "secondary_wavelength": "lambda_2",
    })
    index.to_parquet(cfg.INDEX_PARQUET, index=False)

    # Валидация

    print()
    print("=" * 70)
    print("ВАЛИДАЦИЯ ЭТАПА B")
    print("=" * 70)
    print(f"Провалов обработки: {len(failed)}")
    fully_masked = int((np.asarray(M).sum(axis=1) == 0).sum())
    print(f"Полностью замаскированных строк: {fully_masked}")

    rng = np.random.default_rng(0)
    ok_cnt = 0
    for i in rng.choice(np.where(n_native > 0)[0], size=5, replace=False):
        row_x = np.asarray(X[i], dtype=np.float32)
        row_m = np.asarray(M[i], dtype=bool)
        g = GRID[row_m]
        inside = row_x[row_m]
        print(f"  row {i} ({index['sample_id'].iloc[i]}): "
              f"маска [{g[0]:.2f}, {g[-1]:.2f}] | "
              f"max внутри={inside.max():.3f} | вне={row_x[~row_m].max():.3f}")
        ok_cnt += 1

    print()
    print("λ1 NaN по источникам:")
    print(index.assign(no_l1=index["lambda_1"].isna())
          .groupby("dataset_role")["no_l1"].sum().to_string())
    print(f"\nСохранено: {cfg.X_INTENSITY_PATH.name}, {cfg.MASK_PATH.name}, "
          f"{cfg.INDEX_PARQUET.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
