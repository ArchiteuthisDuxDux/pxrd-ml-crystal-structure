"""Утилиты для работы со спектрами: загрузка, статистика сетки, хэш файла."""

import hashlib
from pathlib import Path

import numpy as np


def is_nonempty_sequence(v):
    """True для непустого list/tuple/np.ndarray (parquet возвращает ndarray)."""
    return isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0


def md5_of_file(path):
    """MD5 по байтам файла (для поиска 1-в-1 дубликатов спектров)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_spectrum(path):
    """
    Загружает .npy спектр и возвращает (x, y) как float64,
    отсортированные по возрастанию x, только конечные значения.

    Поддерживает форматы (N, 2) и (2, N).
    Возвращает (None, None), если файл битый или точек < 2.
    """
    path = Path(path)

    if not path.exists():
        return None, None

    try:
        arr = np.load(path, allow_pickle=False)
    except Exception:
        return None, None

    if arr.ndim != 2:
        return None, None

    if arr.shape[1] == 2:
        x, y = arr[:, 0], arr[:, 1]
    elif arr.shape[0] == 2:
        x, y = arr[0, :], arr[1, :]
    else:
        return None, None

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < 2:
        return None, None

    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def grid_stats(x):
    """
    Статистика сетки 2theta отсортированного спектра.

    step считается как медиана положительных конечных разностей -
    устойчиво к редким скачкам/нулевым шагам в исходных данных.
    """
    n = len(x)
    dx = np.diff(x)
    dx_valid = dx[np.isfinite(dx) & (dx > 0)]

    return {
        "n_points": int(n),
        "x_min": float(x[0]),
        "x_max": float(x[-1]),
        "step": float(np.median(dx_valid)) if len(dx_valid) else np.nan,
    }


def in_angle_window(x_min, x_max, w_min, w_max):
    """Сетка целиком внутри окна [w_min, w_max] -> True."""
    if x_min is None or x_max is None or (isinstance(x_min, float) and np.isnan(x_min)):
        return False
    return bool(x_min >= w_min and x_max <= w_max)


def cluster_values(values, atol=0.0, rtol=0.0, weights=None):
    """
    Жадная кластеризация одномерных значений с допуском.

    Значения сортируются; текущее значение присоединяется к кластеру,
    если |v - center| <= max(atol, rtol * |center|).

    weights: опциональный dict value -> вес (число строк с этим значением).
             Представитель кластера - значение с максимальным суммарным
             весом (самое частое записанное значение), при равенстве -
             медиана кандидатов. Без weights - медиана кластера.

    Возвращает dict: value -> representative.
    """
    values = sorted(values)
    assignment = {}
    members = []
    center = None

    def flush():
        # Пустой вход допустим: например, если у набора данных все длины
        # волн неизвестны. В этом случае сопоставлять просто нечего.
        if not members:
            return
        if weights is None:
            rep = float(np.median(members))
        else:
            scored = [(weights.get(v, 1), v) for v in members]
            best_w = max(w for w, _ in scored)
            cands = [v for w, v in scored if w == best_w]
            rep = float(cands[0]) if len(cands) == 1 else float(np.median(cands))
        for v in members:
            assignment[v] = rep

    for v in values:
        tol = max(atol, rtol * abs(center)) if center is not None else None
        if center is None or abs(v - center) <= tol:
            members.append(v)
            center = float(np.mean(members))
        else:
            flush()
            members = [v]
            center = v

    flush()
    return assignment


def _clean_value(v):
    """ndarray -> список, tuple -> список, float NaN -> None."""
    import math

    if isinstance(v, np.ndarray):
        return [_clean_value(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_clean_value(x) for x in v]
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def normalize_object_columns(df):
    """
    Приводит object-колонки к виду, безопасному для parquet:
    ndarray -> list, NaN -> None. Нужно после concat таблиц
    с разными вариантами записи списочных колонок.
    """
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(_clean_value)
    return out
