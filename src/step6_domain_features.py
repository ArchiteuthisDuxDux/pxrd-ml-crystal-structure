"""
Шаг 6. Извлечение именованных физических признаков для всех спектров
из препроцессированных меммапов (X_intensity, M_mask).

Все признаки считаются ТОЛЬКО внутри валидной маски, на
sqrt-нормированной интенсивности - одинаковый трансформ для обоих доменов.

Выход: data/preprocessed/features_preprocessed.parquet
       (row_idx + ~25 признаков; NaN = признак не определён, например
        пиков не нашлось)

Запуск: python step6_domain_features.py [--workers 16]
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

import config as cfg

GRID = np.linspace(cfg.GRID_LO, cfg.GRID_HI, cfg.GRID_N)

FEATURE_NAMES = [
    "p05", "p10", "p25", "p50", "p75", "p90",
    "mean_i", "std_i",
    "skew_i", "kurt_i",
    "frac_gt_01", "frac_gt_03",
    "total_variation",
    "roughness",            # MAD вторых разностей в плоских местах
    "hf_noise",             # std высокочастотной компоненты
    "max_slope",
    "bg_p10",               # уровень фона (p10 сглаженного)
    "bg_curvature",         # p10(начало) - p10(конец)
    "gini",
    "n_peaks",              # на 10 градусов
    "peak_density_ratio",   # пики / ожидание от случайного шума
    "median_fwhm",
    "p90_fwhm",
    "fwhm_theta_corr",      # тренд ширины с углом
    "halo_bump",
    "mask_frac",
]


def _gini(x):
    x = np.sort(x)
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return np.nan
    
    cum = np.cumsum(x)

    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def compute_features(y, m):
    """y - sqrt-нормированная интенсивность (4096,), m - маска bool."""
    idx = np.where(m)[0]

    if len(idx) < 20:
        return {k: np.nan for k in FEATURE_NAMES}

    yi = y[idx].astype(np.float32)
    gi = GRID[idx]

    feats = {}
    qs = np.percentile(yi, [5, 10, 25, 50, 75, 90])
    (feats["p05"], feats["p10"], feats["p25"],
     feats["p50"], feats["p75"], feats["p90"]) = qs
    feats["mean_i"] = float(yi.mean())
    feats["std_i"] = float(yi.std())
    feats["skew_i"] = float(skew(yi))
    feats["kurt_i"] = float(kurtosis(yi))
    feats["frac_gt_01"] = float((yi > 0.1).mean())
    feats["frac_gt_03"] = float((yi > 0.3).mean())
    feats["total_variation"] = float(np.abs(np.diff(yi)).mean())
    feats["gini"] = _gini(yi)
    feats["mask_frac"] = float(len(idx) / cfg.GRID_N)

    # --- шум: высокочастотная компонента и шероховатость в плоских местах

    d2 = np.abs(yi[2:] - 2 * yi[1:-1] + yi[:-2])
    slope1 = np.abs(np.diff(yi, 2))
    flat = slope1 <= np.quantile(slope1, 0.5)  # наименее наклонные участки

    if flat.sum() > 10:
        med = np.median(yi)
        feats["roughness"] = float(
            1.4826 * np.median(d2[flat]) / max(med, 1e-6)
        )
    else:
        feats["roughness"] = np.nan

    # высокочастотный шум: y минус сглаженное (окно ~0.5 град)

    w = max(3, int(round(0.5 / (GRID[1] - GRID[0])) // 2 * 2 + 1))
    kernel = np.ones(w) / w
    smooth = np.convolve(yi, kernel, mode="same")
    feats["hf_noise"] = float((yi - smooth).std())
    feats["max_slope"] = float(np.abs(np.gradient(yi)).max())

    # --- фон

    feats["bg_p10"] = float(np.quantile(smooth, 0.10))
    third = max(3, len(smooth) // 3)
    feats["bg_curvature"] = float(
        np.quantile(smooth[:third], 0.10) - np.quantile(smooth[-third:], 0.10)
    )

    # --- пики

    prom = 0.05

    # width=(None, None) просит scipy вычислить ширины найденных пиков,
    # не отбрасывая пики по ширине. По умолчанию это FWHM (rel_height=0.5).

    peaks, props = find_peaks(
        yi,
        prominence=prom,
        distance=3,
        width=(None, None),
    )
    n_peaks = len(peaks)
    deg = gi[-1] - gi[0]
    feats["n_peaks"] = float(n_peaks / max(deg, 1) * 10)  # на 10 градусов

    # ожидание случайных максимумов от шума (для плотности сверх шума)

    exp_noise = max(len(yi) * 0.5 / (np.pi * np.sqrt(np.mean(np.abs(np.diff(yi)) ** 2) + 1e-12)), 1)
    feats["peak_density_ratio"] = float(n_peaks / exp_noise)

    if n_peaks >= 3:
        widths = props["widths"] * (GRID[1] - GRID[0])  # в градусах
        feats["median_fwhm"] = float(np.median(widths))
        feats["p90_fwhm"] = float(np.quantile(widths, 0.9))

        if np.std(gi[peaks]) > 1:
            feats["fwhm_theta_corr"] = float(np.corrcoef(gi[peaks], widths)[0, 1])

        else:
            feats["fwhm_theta_corr"] = np.nan

    else:
        feats["median_fwhm"] = np.nan
        feats["p90_fwhm"] = np.nan
        feats["fwhm_theta_corr"] = np.nan

    # --- аморфное гало: широкий горб на сильном сглаживании

    w2 = max(3, int(round(4.0 / (GRID[1] - GRID[0])) // 2 * 2 + 1))
    very_smooth = np.convolve(yi, np.ones(w2) / w2, mode="same")
    feats["halo_bump"] = float(
        (very_smooth.max() - np.quantile(very_smooth, 0.10))
        / max(np.median(very_smooth), 1e-6)
    )

    return feats


def process_one(args):
    i, path = args

    try:
        arr = np.load(path, allow_pickle=False)

    except Exception:
        return i, None
    
    if arr.ndim != 2 or len(arr) < 2:
        return i, None

    if arr.shape[1] == 2 and arr.shape[0] != 2:
        y = arr[:, 1]
        x = arr[:, 0]

    elif arr.shape[0] == 2:
        y = arr[1]
        x = arr[0]

    else:
        return i, None

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    if len(x) < 2:
        return i, None
    
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    keep = np.concatenate([[True], np.diff(x) > 0])
    x, y = x[keep], y[keep]
    y = np.clip(y, 0, None)

    valid = (x >= cfg.GRID_LO) & (x <= cfg.GRID_HI)

    if valid.sum() < 2:
        return i, None
    
    xv, yv = x[valid], y[valid]
    ys = np.sqrt(yv)
    sc = ys.max()
    yn = ys / sc if sc > 0 else np.zeros_like(ys)

    # на общий грид (как в этапе B)

    from scipy.interpolate import PchipInterpolator

    yi = PchipInterpolator(xv, yn, extrapolate=False)(GRID)
    yi = np.nan_to_num(yi, nan=0.0)
    yi = np.clip(yi, 0, 1)
    m = (GRID >= xv[0]) & (GRID <= xv[-1])
    yi = yi * m

    return i, compute_features(yi.astype(np.float32), m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    index = pd.read_parquet(cfg.INDEX_PARQUET)
    n = len(index)
    print(f"Спектров: {n}")

    base = cfg.RAW_SPECTRA_DIR.parent

    def path_of(row):

        # itertuples() возвращает namedtuple, поэтому поля доступны как
        # атрибуты, а не через строковые индексы DataFrame.

        role = row.dataset_role

        if role == "rruff":
            return base / "raw" / f"{row.source}.npy"
        
        if role == "opxrd":
            return base / "raw" / f"{row.sample_id}.npy"
        
        if role == "cod":
            return base / "raw" / f"COD_{row.sample_id.split('_', 1)[1]}.npy"
        
        return base / "raw" / f"crystalDB_{row.sample_id.split('_', 1)[1]}.npy"

    paths = [str(path_of(r)) for r in index.itertuples(index=False)]

    out = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    t0 = time.time()

    done = 0
    CHUNK = 20000

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for c0 in range(0, n, CHUNK):
            c1 = min(c0 + CHUNK, n)
            for i, feats in ex.map(process_one,
                                   zip(range(c0, c1), paths[c0:c1])):
                
                if feats is not None:
                    out[i] = [feats[k] for k in FEATURE_NAMES]
            done = c1

            print(f"  {done}/{n} | {time.time()-t0:.0f} c", flush=True)

    df = pd.DataFrame(out, columns=FEATURE_NAMES)
    df.insert(0, "row_idx", np.arange(n, dtype=np.int64))

    out_path = cfg.PREPROC_DIR / "features_preprocessed.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Сохранено: {out_path}")

    print("\nСводка по ключевым признакам (медианы):")
    key = ["roughness", "hf_noise", "median_fwhm", "bg_p10", "n_peaks",
           "halo_bump", "mask_frac"]
    
    print(df[key].median().to_string())


if __name__ == "__main__":
    sys.exit(main())
