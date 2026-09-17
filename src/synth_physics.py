"""
Физика синтетического порошкового дифракционного паттерна. v2

Движок отражений - собственный векторизованный numpy-расчёт:
    F(hkl) = sum_sites occ_j * f_j(stol2) * exp(2*pi*i * hkl . x_j) * DW_j
    I(hkl) = multiplicity(hkl) * |F|^2 * LP(theta)

Форм-факторы - IT92 (gemmi), множественность - размер орбиты hkl под
точечными операциями группы с объединением Фридля. Перечисление hkl -
gemmi.make_miller_array (уникальные по симметрии, с отсечкой
систематических погасаний). Для P1 (crystalDB) - все hkl, mult=1.

Валидировано против pymatgen XRDCalculator (позиции 1-в-1,
корреляция профилей > 0.97), при этом на больших ячейках на порядки быстрее.

Остальное - как в v1: дублет Kalpha2, March-Dollase, джиттер,
Кальоти+Шеррер+деформация, псевдо-Фойгт с интегрированием по бинам,
сдвиги, чебышевский фон, аморфное гало, Пуассон, насыщение, спайки.
"""

import math

import numpy as np

import synth_config as cfg


# Решётка

def lattice_matrix_from_params(a, b, c, alpha, beta, gamma):
    """Декартовы векторы (строки) прямых векторов решётки."""
    al, be, ga = np.radians([alpha, beta, gamma])
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    vc = np.array([cx, cy, cz])
    return np.vstack([va, vb, vc])


def hkl_cartesian_directions(lattice_matrix, hkls):
    recip = np.linalg.inv(lattice_matrix).T
    vecs = np.asarray(hkls, dtype=np.float64).reshape(-1, 3) @ recip
    if len(vecs) == 0:
        return np.zeros((0, 3))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


# Отражения

_STOL2_GRID = np.linspace(0.0, 1.5, 601)  # (sin theta / lambda)^2


def _form_factor_table(el_name):
    """f(stol2) по сетке _STOL2_GRID для элемента (IT92, gemmi)."""
    import gemmi

    el = gemmi.Element(el_name)
    try:
        coef = el.it92 if el.it92 is not None else gemmi.IT92_get_exact(el, 1992)
        vals = np.array([coef.calculate_sf(float(s)) for s in _STOL2_GRID])
    except Exception:
        vals = np.zeros_like(_STOL2_GRID)
    vals[0] = vals[1] if len(vals) > 1 and vals[0] == 0 else vals[0]
    return vals


def enumerate_reflections(cell_params, rot_ops, lam, tt_lo, tt_hi):

    """
    (hkl_list, mult_list, tt, stol2) для диапазона углов.

    rot_ops - список целочисленных матриц поворота 3x3 (вращательные части
    операций симметрии ИЗ САМОГО CIF, в его собственной сетке). Это делает
    перечисление независимым от origin choice / нестандартных настроек групп.
    rot_ops=None -> только тождественная операция (P1-структуры, crystalDB).

    Перечисление: полная полусфера Фриделя (P1), группировка в орбиты,
    множественность = размер орбиты.
    """

    import gemmi

    a, b, c, alpha, beta, gamma = cell_params
    cell = gemmi.UnitCell(a, b, c, alpha, beta, gamma)

    th_max = np.radians(min(tt_hi, 179.9) / 2)
    th_min = np.radians(max(tt_lo, 0.05) / 2)
    d_min = lam / (2 * np.sin(th_max))

    if rot_ops is None:
        rot_ops = [np.eye(3, dtype=np.int64)]
    rot_ops = [np.asarray(R, dtype=np.int64) for R in rot_ops]

    hkls_all = gemmi.make_miller_array(cell, gemmi.SpaceGroup("P 1"), dmin=float(d_min))

    hkls_ok, mults = [], []
    visited = set()
    for hkl in hkls_all:
        h0 = (int(hkl[0]), int(hkl[1]), int(hkl[2]))
        if h0 in visited:
            continue

        orbit = set()

        for R in rot_ops:
            v = R @ np.array(h0)
            orbit.add((int(v[0]), int(v[1]), int(v[2])))

        orbit |= {(-h, -k, -l) for (h, k, l) in orbit}
        visited |= orbit

        d = cell.calculate_d(h0)

        if d_min <= d:
            hkls_ok.append(h0)
            mults.append(len(orbit))

    if not hkls_ok:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0)

    H = np.array(hkls_ok, dtype=np.float64)
    recip = np.linalg.inv(lattice_matrix_from_params(*cell_params)).T
    g2 = np.sum((H @ recip) ** 2, axis=1)  # 1/d^2
    d = 1.0 / np.sqrt(g2)
    s_val = lam / (2 * d)
    ok = (s_val < 1.0) & (d <= lam / (2 * np.sin(th_min)))

    H, mults, d = H[ok], np.array(mults)[ok], d[ok]
    tt = np.degrees(2 * np.arcsin(lam / (2 * d)))
    stol2 = (lam / (2 * d)) ** 2

    return H, np.asarray(mults, dtype=np.float64), tt, stol2


def structure_factors(H, stol2, sites, el_tables, el_index, b_by_el):

    """
    |F|^2 для батча отражений.

    sites: (N,3) frac координаты; el_index: (N,) индексы элементов;
    el_tables: dict el -> f(stol2) на сетке; b_by_el: dict el -> B.
    """

    n_sites = len(sites)
    f2 = np.zeros(len(H))

    chunk = max(1, int(4e6 / max(n_sites, 1)))
    for i0 in range(0, len(H), chunk):
        Hc = H[i0:i0 + chunk]
        sc = stol2[i0:i0 + chunk]

        # f(s) для каждого элемента в этом батче
        f_by_el = np.vstack([
            np.interp(sc, _STOL2_GRID, el_tables[el]) for el in el_tables
        ])  # (n_el, batch)
        dw_by_el = np.exp(-np.array([b_by_el[el] for el in el_tables])[:, None] * sc[None, :])

        fm = (f_by_el * dw_by_el).T[:, el_index]  # (batch, n_sites)

        phase = np.exp(2j * np.pi * (Hc @ sites.T))
        F = np.sum(fm * phase, axis=1)
        f2[i0:i0 + chunk] = np.abs(F) ** 2

    return f2


def get_reflections(cell_params, sites, el_names, rot_ops, lam, tt_lo, tt_hi,
                    b_iso_range=(0.5, 3.0), rng=None):
    
    """
    Полный расчёт отражений для одной длины волны.
    sites: (N,3) frac; el_names: список из N символов элементов;
    rot_ops: вращательные части операций симметрии (см. enumerate_reflections).
    Возвращает (tt, I, hkl) c I в произвольных единицах.
    """

    H, mult, tt, stol2 = enumerate_reflections(cell_params, rot_ops, lam, tt_lo, tt_hi)
    if len(H) == 0:
        return tt, np.zeros(0), []

    uniq_els = sorted(set(el_names))
    el_tables = {e: _form_factor_table(e) for e in uniq_els}
    el_index = np.array([uniq_els.index(e) for e in el_names])

    if rng is not None:
        b_by_el = {e: float(rng.uniform(*b_iso_range)) for e in uniq_els}

    else:
        b_by_el = {e: 1.5 for e in uniq_els}

    frac = np.asarray(sites, dtype=np.float64)
    f2 = structure_factors(H, stol2, frac, el_tables, el_index, b_by_el)

    th = np.radians(tt / 2)
    lp = (1 + np.cos(2 * th) ** 2) / (np.sin(th) ** 2 * np.cos(th))

    I = mult * f2 * lp

    return tt, I, [tuple(int(v) for v in h) for h in H]


# Утилиты профиля (как в v1)

def pseudo_voigt(x, center, fwhm, eta):
    d = (x - center) / (fwhm / 2.0)
    lorentz = 1.0 / (1.0 + d * d)
    gauss = np.exp(-math.log(2.0) * d * d)

    return eta * lorentz + (1.0 - eta) * gauss


def march_dollase_factor(cos_alpha, r):
    ca2 = cos_alpha * cos_alpha

    return (r * r * ca2 + (1.0 - ca2) / r) ** (-1.5)


# Режимы

_MODES_CACHE = {}


def _modes_cache(family):
    if family not in _MODES_CACHE:
        import pandas as pd
        m = pd.read_csv(cfg.MODES_CSV[family])
        m = m[m["lambda_1"].notna()].reset_index(drop=True)
        _MODES_CACHE[family] = m

    return _MODES_CACHE[family]


def sample_mode(rng):
    families = list(cfg.SOURCE_SAMPLE_WEIGHTS.keys())
    weights = np.array([cfg.SOURCE_SAMPLE_WEIGHTS[f] for f in families])
    family = families[int(rng.choice(len(families), p=weights / weights.sum()))]

    modes = _modes_cache(family)
    p = modes["fraction_all"].to_numpy()
    p = p / p.sum()
    row = modes.iloc[int(rng.choice(len(modes), p=p))]

    import pandas as pd
    lam2 = row["lambda_2"]
    return {
        "family": family,
        "lambda_1": float(row["lambda_1"]),
        "lambda_2": float(lam2) if pd.notna(lam2) else None,
        "x_min": int(row["x_min_mode"]),
        "x_max": int(row["x_max_mode"]),
        "step": float(row["step_mode"]),
    }


def sample_seed(source, sid):
    import zlib
    return zlib.crc32(f"{source}:{sid}".encode())


# Профиль пиков на сетке прибора

def build_profile(grid, peaks, step, rng):
    n_bins = len(grid)

    if not peaks:
        return np.zeros(n_bins)

    fwhm_min = min(p[2] for p in peaks)
    n_sub = int(np.clip(np.ceil(step / max(fwhm_min / 4.0, 1e-4)), 1, 40))

    lo = grid.min() - step * 0.5
    hi = grid.max() + step * 0.5
    fine_x = np.arange(lo, hi + step / n_sub * 0.5, step / n_sub)
    fine_y = np.zeros_like(fine_x)

    for center, amp, fwhm, eta in peaks:
        half_win = 25.0 * fwhm
        i0 = np.searchsorted(fine_x, center - half_win)
        i1 = np.searchsorted(fine_x, center + half_win)

        if i1 <= i0:
            continue

        seg = fine_x[i0:i1]
        fine_y[i0:i1] += amp * pseudo_voigt(seg, center, fwhm, eta)

    idx = np.searchsorted(fine_x, grid, side="right")
    starts = np.concatenate([[0], idx[:-1]])
    sums = np.add.reduceat(fine_y, np.clip(starts, 0, len(fine_y) - 1))
    counts = np.diff(np.concatenate([starts, [len(fine_y)]]))
    counts[counts == 0] = 1
    profile = sums / counts

    return np.clip(profile, 0.0, None)


# Полная генерация одного спектра

def generate_spectrum(cell_params, sites, el_names, rot_ops, mode, seed):

    """
    (cell, sites frac, элементы, rot_ops) -> (grid, y_counts, meta).
    """

    rng = np.random.default_rng(seed)

    lam1 = mode["lambda_1"]
    lam2 = mode["lambda_2"]
    x_min, x_max, step = mode["x_min"], mode["x_max"], mode["step"]

    if step <= 0 or not np.isfinite(step):
        raise ValueError("bad step in mode")

    grid = np.arange(x_min, x_max + step * 0.25, step)

    if len(grid) < 5:
        raise ValueError("grid too short")

    margin = 1.5
    tt_lo = float(grid.min() - margin)
    tt_hi = float(grid.max() + margin)

    size_logmu = (cfg.CRYSTALLITE_SIZE_LOGMU[mode["family"]]
                  if isinstance(cfg.CRYSTALLITE_SIZE_LOGMU, dict)
                  else cfg.CRYSTALLITE_SIZE_LOGMU)
    size_nm = float(np.clip(
        rng.lognormal(size_logmu, cfg.CRYSTALLITE_SIZE_LOGSIGMA),
        cfg.CRYSTALLITE_SIZE_MIN_NM, cfg.CRYSTALLITE_SIZE_MAX_NM,
    ))
    size_angstrom = size_nm * 10.0

    w_range = (cfg.CAGLIOTI_W_RANGE[mode["family"]]
               if isinstance(cfg.CAGLIOTI_W_RANGE, dict)
               else cfg.CAGLIOTI_W_RANGE)
    caglioti_w = rng.uniform(*w_range)
    caglioti_u = rng.uniform(*cfg.CAGLIOTI_U_RANGE)
    caglioti_v = rng.uniform(*cfg.CAGLIOTI_V_RANGE)
    strain = min(abs(rng.normal(0.0, cfg.STRAIN_SIGMA)), cfg.STRAIN_MAX)
    eta_pv = rng.uniform(*cfg.PV_ETA_RANGE)

    use_texture = rng.random() < cfg.TEXTURE_PROB
    texture_r = float(rng.uniform(*cfg.TEXTURE_R_RANGE)) if use_texture else 1.0
    tex_axis = rng.normal(size=3)
    tex_axis /= np.linalg.norm(tex_axis)

    zero_shift = rng.uniform(-cfg.ZERO_SHIFT_MAX_DEG, cfg.ZERO_SHIFT_MAX_DEG)
    displacement = rng.uniform(-cfg.DISPLACEMENT_FRAC_MAX, cfg.DISPLACEMENT_FRAC_MAX)

    lattice_matrix = lattice_matrix_from_params(*cell_params)

    all_peaks = []
    wavelengths = [(lam1, 1.0)]

    if lam2 is not None and lam2 > 0:
        wavelengths.append((lam2, cfg.KALPHA2_WEIGHT))

    for lam, w_lam in wavelengths:
        tt, inten, hkls = get_reflections(
            cell_params, sites, el_names, rot_ops, lam, tt_lo, tt_hi, rng=rng
        )
        if len(tt) == 0:
            continue

        keep = inten >= cfg.PEAK_MIN_REL_INTENSITY * inten.max()
        tt, inten = tt[keep], inten[keep]
        hkls = [h for h, k in zip(hkls, keep) if k]

        if len(tt) > cfg.PEAK_MAX_COUNT:
            order = np.argsort(-inten)[: cfg.PEAK_MAX_COUNT]
            tt, inten = tt[order], inten[order]
            hkls = [hkls[i] for i in order]

        dirs = hkl_cartesian_directions(lattice_matrix, [h for h in hkls if h is not None])
        cos_alpha = np.ones(len(hkls))

        if len(dirs):
            idx_valid = [i for i, h in enumerate(hkls) if h is not None]
            cos_alpha[idx_valid] = dirs @ tex_axis

        th_half = np.radians(tt / 2.0)
        tan_th = np.tan(th_half)
        cos_th = np.cos(th_half)

        fwhm_inst2 = np.clip(
            caglioti_u * tan_th**2 + caglioti_v * tan_th + caglioti_w, 1e-6, None
        )
        fwhm_size_deg = np.degrees(cfg.SCHERRER_K * lam / (size_angstrom * cos_th))
        fwhm_strain_deg = np.degrees(4.0 * strain * tan_th)
        fwhm = np.clip(
            np.sqrt(fwhm_inst2 + fwhm_size_deg**2 + fwhm_strain_deg**2),
            cfg.FWHM_MIN_DEG, cfg.FWHM_MAX_DEG,
        )

        jitter = rng.lognormal(0.0, cfg.PEAK_JITTER_SIGMA, len(tt))
        md = march_dollase_factor(cos_alpha, texture_r) if use_texture else 1.0
        amps = inten * w_lam * jitter * md

        # центры БЕЗ сдвига (сдвиг применяется при рендере - так таблица
        # пиков остаётся пригодной для перерендера с новыми сдвигами)
        
        for c, a, f in zip(tt, amps, fwhm):
            if a <= 0:
                continue
            all_peaks.append((float(c), float(a), float(f), eta_pv))

    all_peaks = all_peaks[: cfg.PEAK_MAX_COUNT]

    def shift_scalar(c):
        return c + zero_shift - displacement * math.cos(math.radians(c / 2.0))

    render_peaks = [
        (shift_scalar(c), a, f, e) for (c, a, f, e) in all_peaks
    ]

    y = build_profile(grid, render_peaks, step, rng)

    amp_scale = max(y.max(), 1e-6)
    t = np.linspace(-1.0, 1.0, len(grid))
    cheb = np.polynomial.chebyshev.chebval(
        t,
        [
            rng.uniform(*cfg.CHEB_BASE_RANGE[mode["family"]]),
            rng.uniform(*cfg.CHEB_C1_RANGE),
            rng.uniform(*cfg.CHEB_HIGH_RANGE),
            rng.uniform(*cfg.CHEB_HIGH_RANGE),
            rng.uniform(*cfg.CHEB_HIGH_RANGE),
            rng.uniform(*cfg.CHEB_HIGH_RANGE),
        ],
    ) * amp_scale
    y = y + np.clip(cheb, 0.0, None)

    halo_cfg = (cfg.HALO[mode["family"]] if isinstance(cfg.HALO, dict)
                else {"prob": cfg.HALO_PROB, "height": cfg.HALO_HEIGHT_RANGE})
    
    if rng.random() < halo_cfg["prob"]:
        halo_c = rng.uniform(*cfg.HALO_CENTER_RANGE)
        halo_s = rng.uniform(*cfg.HALO_SIGMA_RANGE)
        halo_h = rng.uniform(*halo_cfg["height"]) * amp_scale
        y = y + halo_h * np.exp(-0.5 * ((grid - halo_c) / halo_s) ** 2)

    y_max = y.max()

    if y_max <= 0:
        y = np.full(len(grid), 1e-6)
        y_max = 1e-6

    counts_range = cfg.COUNTS_LOG_RANGE[mode["family"]]
    n_counts = 10 ** rng.uniform(*counts_range)
    scale = n_counts / y_max
    y_counts = rng.poisson(y * scale).astype(np.float64)

    if rng.random() < cfg.SATURATION_PROB:
        sat = n_counts * rng.uniform(*cfg.SATURATION_LEVEL_RANGE)
        y_counts = np.minimum(y_counts, sat)

    if rng.random() < cfg.ZINGER_PROB:
        for _ in range(int(rng.integers(*cfg.ZINGER_COUNT_RANGE))):
            pos = int(rng.integers(0, len(y_counts)))
            y_counts[pos] += n_counts * rng.uniform(*cfg.ZINGER_HEIGHT_RANGE)

    meta = {
        "crystallite_size_nm": size_nm,
        "texture_r": texture_r,
        "zero_shift": zero_shift,
        "displacement": displacement,
        "strain": strain,
        "n_peaks": len(all_peaks),
        "n_points": len(grid),
        "counts_level": float(n_counts),
    }

    peaks_arr = np.array(all_peaks, dtype=np.float32).reshape(-1, 4)

    return (
        grid.astype(np.float64),
        y_counts.astype(np.float32),
        meta,
        peaks_arr,
        float(amp_scale),
    )
