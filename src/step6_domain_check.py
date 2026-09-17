"""
Шаг 6. Доменный классификатор: синтетика vs реальность.

Идея: если простой классификатор на ФИЗИЧЕСКИ НАЗВАННЫХ признаках
не может отличить синтетику от реальности - домены совпадают и
пре-трейн безопасен. Если может - важности признаков показывают,
какие именно ручки генератора подкрутить.

Сравнение честное:
    - парами по длине волны: Cu(1.541) synth vs RRUFF+opXRD,
      синхротрон 1.240 synth vs opXRD, 1.208 synth vs opXRD;
    - два режима: (a) все спектры + признак покрытия маски,
                  (b) только спектры, полностью покрывающие окно [10,70],
                      признаки считаются только внутри окна - чистое
                      сравнение спектральной физики без геометрии масок.

Запуск: python step6_domain_check.py
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, peak_widths
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import config as cfg

GRID = np.linspace(cfg.GRID_LO, cfg.GRID_HI, cfg.GRID_N)
STEP = GRID[1] - GRID[0]

FEATURE_NAMES = [
    "coverage",
    "noise_roughness", "noise_lag1_autocorr", "peak_sharpness", "total_variation",
    "bg_p05", "bg_p25", "int_median", "int_p95",
    "int_skew", "int_kurtosis",
    "frac_gt_01", "frac_gt_03", "frac_gt_05", "gini", "sparsity",
    "peak_density", "fwhm_median", "fwhm_lo", "fwhm_hi", "fwhm_trend",
    "halo_strength", "doublet_frac", "dyn_range",
]

WIN_LO, WIN_HI = 10.0, 70.0
WIN_MASK = (GRID >= WIN_LO) & (GRID <= WIN_HI)
WIN_POINTS = int(WIN_MASK.sum())
MIN_WIN_COVER = 0.95


def gini(x):
    x = np.sort(x)
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def extract_features(y, m):

    """y - интенсивности (4096,), m - маска (4096,) bool. Окно [10,70]."""

    w = m & WIN_MASK

    if w.sum() < 100:
        return None

    yw = y[w].astype(np.float32)
    gw = GRID[w]
    level = np.median(yw) + 1e-6

    feats = {"coverage": float(m.mean())}

    # --- шум: высокочастотная компонента ---------------------------

    d1 = np.diff(yw)
    mad = np.median(np.abs(d1 - np.median(d1)))
    feats["noise_roughness"] = float(1.4826 * mad / level)
    feats["noise_lag1_autocorr"] = float(np.corrcoef(d1[:-1], d1[1:])[0, 1]) if len(d1) > 10 else 0.0
    feats["peak_sharpness"] = float(np.abs(d1).max() / level)
    feats["total_variation"] = float(np.abs(d1).sum() / (yw.sum() + 1e-6))

    # --- распределение интенсивности --------------------------------

    qs = np.percentile(yw, [5, 25, 50, 95])
    feats["bg_p05"], feats["bg_p25"], feats["int_median"], feats["int_p95"] = [
        float(v / level) for v in qs
    ]
    z = (yw - yw.mean()) / (yw.std() + 1e-6)
    feats["int_skew"] = float(np.mean(z**3))
    feats["int_kurtosis"] = float(np.mean(z**4))

    feats["frac_gt_01"] = float((yw > 0.1).mean())
    feats["frac_gt_03"] = float((yw > 0.3).mean())
    feats["frac_gt_05"] = float((yw > 0.5).mean())
    feats["gini"] = gini(yw)
    feats["sparsity"] = float((yw < 0.01).mean())

    # --- пики --------------------------------------------------------

    smooth = uniform_filter1d(yw, size=7)
    noise_sigma = 1.4826 * mad + 1e-6
    peaks, props = find_peaks(smooth, prominence=5 * noise_sigma, distance=3)
    n_peaks = len(peaks)
    feats["peak_density"] = float(n_peaks / (gw[-1] - gw[0] + 1e-6))

    if n_peaks > 0:
        widths_pts = peak_widths(smooth, peaks, rel_height=0.5)[0]
        pos = gw[peaks]
        fwhm = widths_pts * STEP * (len(yw) / w.sum())  # поправка на окно
        keep = (fwhm > 0.005) & (fwhm < 5.0)
        fwhm, pos = fwhm[keep], pos[keep]
        feats["fwhm_median"] = float(np.median(fwhm)) if len(fwhm) else np.nan
        lo_m = pos < 40
        hi_m = pos >= 40
        feats["fwhm_lo"] = float(np.median(fwhm[lo_m])) if lo_m.sum() >= 3 else np.nan
        feats["fwhm_hi"] = float(np.median(fwhm[hi_m])) if hi_m.sum() >= 3 else np.nan

        if lo_m.sum() >= 3 and hi_m.sum() >= 3:
            feats["fwhm_trend"] = float(feats["fwhm_hi"] - feats["fwhm_lo"])

        else:
            feats["fwhm_trend"] = np.nan

        # дублет: соседний пик в пределах 0.05-0.4 град при 2th>40
        # pos уже отфильтрован той же маской keep, что и fwhm.
        # Поэтому hi_m относится именно к pos, а не к исходному gw[peaks].

        hi_pos = pos[hi_m]
        if len(hi_pos) > 1:
            dd = np.diff(np.sort(hi_pos))
            feats["doublet_frac"] = float(((dd > 0.05) & (dd < 0.4)).mean())

        else:
            feats["doublet_frac"] = 0.0

    else:
        feats["fwhm_median"] = np.nan
        feats["fwhm_lo"] = np.nan
        feats["fwhm_hi"] = np.nan
        feats["fwhm_trend"] = np.nan
        feats["doublet_frac"] = 0.0

    # --- фон и гало ---------------------------------------------------

    bg = uniform_filter1d(yw, size=201)
    feats["halo_strength"] = float((bg.max() - np.percentile(bg, 10)) / level)

    feats["dyn_range"] = float(np.percentile(yw, 99.9) / (qs[0] + 1e-6))

    return feats


def build_dataset(index, X, M, role_filter, lam_lo, lam_hi, cap, rng):

    """Выборка строк с λ в диапазоне; возвращает признаки + метаданные."""

    df = index[
        (index["lambda_1"] >= lam_lo) & (index["lambda_1"] <= lam_hi)
    ]
    if role_filter == "synth":
        df = df[df["dataset_role"].isin(["cod", "crystaldb"])]

    elif role_filter == "real":
        df = df[df["dataset_role"].isin(["rruff", "opxrd"])]

    else:
        df = df[df["dataset_role"] == role_filter]

    if len(df) > cap:
        df = df.sample(cap, random_state=7)

    rows, metas = [], []

    for row_idx in df["row_idx"].to_numpy():
        y = X[row_idx].astype(np.float32)
        m = M[row_idx].astype(bool)
        f = extract_features(y, m)

        if f is None:
            continue

        f["win_cover"] = float((m & WIN_MASK).sum() / WIN_POINTS)
        rows.append(f)
        metas.append(row_idx)

    out = pd.DataFrame(rows, index=metas)
    out.index.name = "row_idx"
    return out


def main():

    t0 = time.time()
    index = pd.read_parquet(cfg.INDEX_PARQUET)
    X = np.memmap(cfg.X_INTENSITY_PATH, dtype=np.float16, mode="r",
                  shape=(len(index), cfg.GRID_N))
    M = np.memmap(cfg.MASK_PATH, dtype=np.uint8, mode="r",
                  shape=(len(index), cfg.GRID_N))

    PAIRS = [
        ("Cu 1.541", "synth", 1.5395, 1.5425, "real", 1.5395, 1.5425, 12000),
        ("Syncro 1.240", "synth", 1.2390, 1.2406, "opxrd", 1.2390, 1.2406, 12000),
        ("Syncro 1.208", "synth", 1.2075, 1.2090, "opxrd", 1.2075, 1.2090, 8000),
    ]

    all_reports = []

    for pair_name, s_role, s_lo, s_hi, r_role, r_lo, r_hi, cap in PAIRS:
        print("=" * 78)
        print(f"ПАРА: {pair_name}")
        print("=" * 78)

        ds_s = build_dataset(index, X, M, s_role, s_lo, s_hi, cap, None)
        ds_r = build_dataset(index, X, M, r_role, r_lo, r_hi, cap, None)

        print(f"  синтетика: {len(ds_s)} | реальность: {len(ds_r)} "
              f"| {time.time()-t0:.0f} c", flush=True)

        for mode in ["all", "clean_window"]:
            if mode == "all":
                a, b = ds_s.copy(), ds_r.copy()

            else:
                a = ds_s[ds_s["win_cover"] >= MIN_WIN_COVER].copy()
                b = ds_r[ds_r["win_cover"] >= MIN_WIN_COVER].copy()

            if len(a) < 100 or len(b) < 100:
                print(f"  [{mode}] мало данных: {len(a)} vs {len(b)} - пропуск")

                continue

            a["label"] = 1
            b["label"] = 0
            data = pd.concat([a, b], ignore_index=True)
            data = data.dropna(axis=1, how="all")
            feat_cols = [c for c in data.columns if c not in ("label",)]
            Xd = data[feat_cols].astype(np.float64)
            Xd = Xd.fillna(Xd.median(numeric_only=True))
            yd = data["label"].to_numpy()

            Xtr, Xte, ytr, yte = train_test_split(
                Xd, yd, test_size=0.25, random_state=0, stratify=yd
            )
            clf = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.1, max_depth=None,
                random_state=0,
            )
            clf.fit(Xtr, ytr)
            auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])

            # перестановочная важность

            from sklearn.inspection import permutation_importance

            pi = permutation_importance(
                clf, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=0
            )
            imp = pd.Series(pi.importances_mean, index=feat_cols).sort_values(ascending=False)

            # KS и направление по каждому признаку

            from scipy.stats import ks_2samp

            ks_rows = []

            for c in feat_cols:
                ks = ks_2samp(a[c].dropna(), b[c].dropna())
                ks_rows.append((c, ks.statistic, float(a[c].median()), float(b[c].median())))

            ks_df = pd.DataFrame(ks_rows, columns=["feature", "ks", "median_synth", "median_real"]
                                 ).sort_values("ks", ascending=False)

            print(f"\n  [{mode}] n_synth={len(a)} n_real={len(b)}  AUC = {auc:.3f}")
            print("  Топ-8 важностей (permutation, drop AUC):")

            for c, v in imp.head(8).items():
                ks_row = ks_df[ks_df.feature == c].iloc[0]

                print(f"    {c:22s} dAUC={v:+.4f} | KS={ks_row.ks:.3f} | "
                      f"медиана synth={ks_row.median_synth:.4g} real={ks_row.median_real:.4g}")

            all_reports.append({
                "pair": pair_name, "mode": mode, "auc": auc, "n_synth": len(a),
                "n_real": len(b),
                "top_features": "; ".join(imp.head(5).index),
            })

    print()
    print("=" * 78)
    print("СВОДКА")
    print("=" * 78)

    rep = pd.DataFrame(all_reports)

    print(rep.to_string(index=False))
    
    rep.to_csv(cfg.OUTPUTS_DIR / "domain_check_report.csv", index=False)

    print()
    print("Интерпретация AUC: <0.65 - домены близки (ок); 0.65-0.8 - умеренная "
          "щель (подкрутить ручки); >0.8 - существенная щель.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
