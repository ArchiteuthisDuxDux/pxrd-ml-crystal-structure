"""
Именованные физические признаки одного спектра (для доменной диагностики).

Все признаки считаются на ФИКСИРОВАННОМ окне [10, 70] градусов
(честное сравнение доменов без геометрии масок) и на
sqrt-нормированной интенсивности из этапа B.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, peak_widths
from scipy.stats import kurtosis, skew

FEATURE_NAMES = [
    "noise_pt", "roughness_tv", "autocorr_1", "autocorr_3", "snr",
    "bg_p10", "bg_curv", "skew", "kurtosis", "gini", "iqr", "frac_low",
    "n_peaks_per_deg", "fwhm_median", "fwhm_p25", "fwhm_p75", "fwhm_trend",
    "mean_prom_rel", "peak_h_mean", "peak_h_p25", "max_slope",
    "halo_height", "halo_frac", "doublet_frac", "p95_p50", "n_strong_peaks",
]


def gini(x):
    x = np.sort(x)
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def compute_features(y, i0, i1, step):
    """y - полная строка 4096 (sqrt-норм.); окно [i0:i1] полностью в маске."""
    yw = y[i0:i1].astype(np.float64)
    n = len(yw)
    out = {k: np.nan for k in FEATURE_NAMES}

    # --- шум: робастная оценка по первым разностям ---
    d1 = np.diff(yw)
    med = np.median(d1)
    mad = np.median(np.abs(d1 - med))
    noise_pt = 1.4826 * mad / np.sqrt(2) + 1e-9
    out["noise_pt"] = noise_pt
    out["roughness_tv"] = float(np.mean(np.abs(d1)))

    # автокорреляции (чувствительны к характеру шума)
    def ac(k):
        a, b = yw[:-k], yw[k:]
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    out["autocorr_1"] = ac(1)
    out["autocorr_3"] = ac(3)

    # --- фон ---
    ys = uniform_filter1d(yw, 9)  # лёгкое сглаживание ~0.2 град
    bg_p10 = float(np.percentile(yw, 10))
    out["bg_p10"] = bg_p10
    q = n // 4
    out["bg_curv"] = float(np.percentile(yw[:q], 10) - np.percentile(yw[-q:], 10))

    # --- распределение интенсивности ---
    out["skew"] = float(skew(yw))
    out["kurtosis"] = float(kurtosis(yw))
    out["gini"] = gini(yw)
    p25, p50, p75, p95 = np.percentile(yw, [25, 50, 75, 95])
    out["iqr"] = float(p75 - p25)
    out["frac_low"] = float((yw < bg_p10 * 1.3).mean())
    out["p95_p50"] = float((p95 - p50) / (p50 + 1e-6))
    out["snr"] = float((np.percentile(ys, 99.5) - bg_p10) / noise_pt)
    out["max_slope"] = float(np.max(np.abs(np.gradient(yw))))

    # --- пики ---
    prom_thr = max(6.0 * noise_pt, 0.01)
    peaks, props = find_peaks(ys, prominence=prom_thr, distance=5)
    n_pk = len(peaks)
    out["n_peaks_per_deg"] = n_pk / (n * step)
    out["n_strong_peaks"] = int((props["prominences"] / max(ys.max(), 1e-6) > 0.15).sum())

    if n_pk > 0:
        widths_pts = peak_widths(ys, peaks, rel_height=0.5)[0]
        widths_deg = widths_pts * step
        theta_pk = (peaks + i0) * step  # примерные 2theta вершин

        out["fwhm_median"] = float(np.median(widths_deg))
        out["fwhm_p25"] = float(np.percentile(widths_deg, 25))
        out["fwhm_p75"] = float(np.percentile(widths_deg, 75))
        hi = theta_pk >= 40
        if hi.sum() >= 2 and (~hi).sum() >= 2:
            out["fwhm_trend"] = float(np.median(widths_deg[hi]) - np.median(widths_deg[~hi]))
        out["mean_prom_rel"] = float(np.mean(props["prominences"]) / max(ys.max(), 1e-6))
        h = ys[peaks] / max(ys.max(), 1e-6)
        out["peak_h_mean"] = float(np.mean(h))
        out["peak_h_p25"] = float(np.percentile(h, 25))

        # --- дублет: у сильных пиков (2theta>50) плечо справа в 0.10-0.45 град ---
        strong = (h > 0.25) & (theta_pk > 50)
        if strong.sum() > 0:
            cnt = 0
            for p_idx in np.where(strong)[0]:
                c, h_main = peaks[p_idx], ys[peaks[p_idx]]
                lo = c + int(0.10 / step)
                hi_i = min(c + int(0.45 / step) + 1, n)
                if hi_i - lo < 3:
                    continue
                seg = ys[lo:hi_i]
                h2 = seg.max()
                if 0.25 * h_main <= h2 <= 0.75 * h_main:
                    cnt += 1
            out["doublet_frac"] = cnt / int(strong.sum())

    # --- аморфное гало ---
    heavy = uniform_filter1d(yw, 181)  # ~4 градуса
    base = float(np.median(heavy))
    hh = float(heavy.max() - base)
    out["halo_height"] = hh
    out["halo_frac"] = hh / max(float(heavy.max()), 1e-6)

    return out
