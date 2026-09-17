"""
Шаг 1. Очистка исходных summary-паркетов.

Что делается:

RRUFF:
    - удаляются строки-дубликаты по ##RRUFFID;
      спектр ПОСЛЕДНЕГО из дублей, поэтому keep='last');
    - проверяется существование *.npy, строки без файла удаляются;
    - пересчитываются x_min / x_max / step / n_points из самих файлов.

opXRD:
    - удаляются строки с отсутствующим .npy;
    - удаляются 1-в-1 дубликаты спектров;
    - spacegroup_number приводится к Int64;
    - dataset_role заполняется ('opxrd'), пути нормализуются к '/'.

Общее:
    - добавляется маска in_angle_window_0_90:
          1 - сетка спектра целиком внутри [0, 90] градусов,
          0 - выходит за окно (такие строки при FT дополнительно маскируются);
    - sample_id -> строка.
"""

import hashlib
import io
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import config as cfg
from utils_spectra import is_nonempty_sequence

WORKERS = 16
_lock = threading.Lock()

# Все порошковые RRUFF-паттерны в исходной выгрузке сняты на Cu Kalpha.
# Эти значения использовались в исходном проекте до построения режимов работы.

RRUFF_PRIMARY_WAVELENGTH = 1.54056
RRUFF_SECONDARY_WAVELENGTH = 1.54439


def _parse_json_list(v):

    """JSON-строку вида '["PbI2", ...]' -> список; иначе None."""

    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    
    if isinstance(v, str):
        try:
            parsed = json.loads(v)

        except Exception:
            return None
        
        return parsed if isinstance(parsed, list) else None
    
    return None


def _parse_float_list(v):

    """Список чисел из JSON; элементы '1.0' (строки) приводятся к float."""

    lst = _parse_json_list(v)

    if lst is None:
        return None
    
    out = []
    for x in lst:
        try:
            out.append(float(x))

        except (TypeError, ValueError):
            return None
        
    return out


def resolve_raw_path(rel_path):

    """raw\\xxx.npy или raw/xxx.npy -> абсолютный путь."""

    rel = str(rel_path).replace("\\", "/")

    return cfg.RAW_SPECTRA_DIR.parent / rel


def _process_one(path):
    
    """
    Один проход по файлу: MD5 байтов + статистика сетки x.
    Возвращает (md5 | None, n_points, x_min, x_max, step).
    """

    try:
        with open(path, "rb") as f:
            data = f.read()

        md5 = hashlib.md5(data).hexdigest()
        arr = np.load(io.BytesIO(data), allow_pickle=False)

    except Exception:
        return None, None, None, None, None

    if arr.ndim != 2:
        return None, None, None, None, None

    if arr.shape[1] == 2:
        x = arr[:, 0]

    elif arr.shape[0] == 2:
        x = arr[0, :]

    else:
        return None, None, None, None, None

    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return None, None, None, None, None

    x = np.sort(x)

    dx = np.diff(x)
    dx_valid = dx[np.isfinite(dx) & (dx > 0)]
    step = float(np.median(dx_valid)) if len(dx_valid) else np.nan

    return md5, int(len(x)), float(x[0]), float(x[-1]), step


def enrich_with_grid(df, source_name):

    """
    Многопоточно считает md5 и статистику сетки для всех строк.
    Строки с отсутствующим/битым файлом удаляются.
    """

    counter = {"n": 0}

    def tracked(path):
        res = _process_one(path)

        with _lock:
            counter["n"] += 1

            if counter["n"] % 5000 == 0:
                print(f"  {source_name}: {counter['n']}/{len(df)}", flush=True)

        return res

    paths = [resolve_raw_path(p) for p in df["raw_spectrum_path"]]

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(tracked, paths))

    cols = list(zip(*results))
    out = df.copy()
    out["spectrum_md5"] = cols[0]
    out["n_points"] = pd.array(cols[1], dtype="Int64")
    out["x_min"] = np.asarray(cols[2], dtype=np.float64)
    out["x_max"] = np.asarray(cols[3], dtype=np.float64)
    out["step"] = np.asarray(cols[4], dtype=np.float64)

    valid = out["spectrum_md5"].notna()
    dropped = int((~valid).sum())
    print(f"  {source_name}: удалено строк без валидного файла: {dropped}", flush=True)

    out = out[valid].copy()

    out[cfg.ANGLE_MASK_COLUMN] = (
        (out["x_min"] >= cfg.ANGLE_WINDOW_MIN) & (out["x_max"] <= cfg.ANGLE_WINDOW_MAX)
    ).astype("int8")

    return out


def clean_rruff():
    print("=" * 70)
    print("RRUFF")
    print("=" * 70)

    df = pd.read_parquet(cfg.RRUFF_SOURCE_PARQUET)
    print(f"Исходно: {len(df)} строк")

    # В сыром JSON длины волн не записаны, поэтому восстанавливаем известный
    # режим RRUFF явно.

    df["primary_wavelength"] = RRUFF_PRIMARY_WAVELENGTH
    df["secondary_wavelength"] = RRUFF_SECONDARY_WAVELENGTH

    dup_mask = df.duplicated(subset=["source"], keep="last")
    print(f"Дубликатов source (keep=last): {int(dup_mask.sum())} удаляются")
    df = df[~dup_mask].copy()

    df = enrich_with_grid(df, "rruff")

    df["sample_id"] = df["sample_id"].astype(str)
    df["dataset_role"] = "rruff"
    df["source"] = df["source"].astype(str)
    df["spacegroup_number"] = pd.to_numeric(
        df["spacegroup_number"], errors="coerce"
    ).astype("Int64")

    # Схема совместима с opxrd: доля фазы одной фазы -> [1.0]

    df["phase_fraction"] = [
        [float(v)] if pd.notna(v) else None for v in df["phase_fraction"]
    ]

    return df


def clean_opxrd():
    print("=" * 70)
    print("opXRD")
    print("=" * 70)

    df = pd.read_parquet(cfg.OPXRD_SOURCE_PARQUET)
    print(f"Исходно: {len(df)} строк")

    df = enrich_with_grid(df, "opxrd")
    before = len(df)

    dup_mask = df.duplicated(subset=["spectrum_md5"], keep="first")
    df = df[~dup_mask].copy()
    print(f"1-в-1 дубликатов спектра (по MD5) удалено: {before - len(df)}")

    df["sample_id"] = df["sample_id"].astype(str)
    df["dataset_role"] = "opxrd"
    df["raw_spectrum_path"] = (
        df["raw_spectrum_path"].astype(str).str.replace("\\", "/", regex=False)
    )

    sg = pd.to_numeric(df["spacegroup_number"], errors="coerce")
    df["spacegroup_number"] = sg.astype("Int64")

    # Гармонизация схемы с RRUFF:
    # - 'elements' в opxrd это СТРОКА состава первой фазы ("PbI2"),
    #   в rruff - список элементов. Переименовываем, чтобы не смешивать.
    # - 'phase_fraction' в opxrd - JSON-строка списка -> настоящий список.

    df = df.rename(columns={"elements": "elements_source_string"})
    df["phase_fraction"] = df["phase_fraction"].apply(_parse_float_list)
    for col in ["lattices_all", "spacegroups_all", "phase_compositions"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_json_list)

    return df


def report(name, df):
    print()
    print(f"--- Отчет {name}: {len(df)} строк ---")
    print(
        f"{cfg.ANGLE_MASK_COLUMN}=1 : "
        f"{int(df[cfg.ANGLE_MASK_COLUMN].sum())} ({df[cfg.ANGLE_MASK_COLUMN].mean():.1%})"
    )
    print(
        f"x_min: [{df['x_min'].min():.3f}, {df['x_min'].max():.3f}] | "
        f"x_max: [{df['x_max'].min():.3f}, {df['x_max'].max():.3f}]"
    )
    lat = int(df[cfg.LATTICE_COLUMNS].notna().all(axis=1).sum())
    sg = int(df["spacegroup_number"].notna().sum())
    cs = int(df["crystal_system"].notna().sum())
    el = int(df["elements_list"].apply(is_nonempty_sequence).sum())
    print(f"lattice: {lat} | spacegroup: {sg} | crystal_system: {cs} | elements: {el}")
    print(f"Дубликатов путей к файлам: {int(df['raw_spectrum_path'].duplicated().sum())}")


def main():
    df_rruff = clean_rruff()
    df_opxrd = clean_opxrd()

    report("RRUFF clean", df_rruff)
    report("opXRD clean", df_opxrd)

    df_rruff.to_parquet(cfg.RRUFF_CLEAN_PARQUET, index=False)
    df_opxrd.to_parquet(cfg.OPXRD_CLEAN_PARQUET, index=False)
    print()
    print(f"Сохранено: {cfg.RRUFF_CLEAN_PARQUET.name}, {cfg.OPXRD_CLEAN_PARQUET.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
