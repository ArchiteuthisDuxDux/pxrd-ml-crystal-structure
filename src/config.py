"""
Конфигурация пайплайна обучения голов.

Целевые головы (в порядке приоритета разработки):
    1. lattice parameters  (a, b, c, alpha, beta, gamma)
    2. space group         (номер пространственной группы)
    3. crystal system      (7 классов)
    4. element presence    (multi-hot по элементам)

phase_count / phase_fraction - НЕ целевые головы, игнорируются.
"""

from pathlib import Path

from project_paths import PROJECT_ROOT, RAW_SPECTRA_DIR

# ============================================================
# Пути
# ============================================================

PIPELINE_ROOT = PROJECT_ROOT

DATA_DIR = PIPELINE_ROOT / "data"
SOURCE_DIR = DATA_DIR / "source"
CLEAN_DIR = DATA_DIR / "clean"

OUTPUTS_DIR = PIPELINE_ROOT / "outputs"

# Все .npy-спектры создаются заново внутри проекта. Значения
# raw_spectrum_path имеют вид ``raw/<name>.npy``, поэтому потребители берут
# ``RAW_SPECTRA_DIR.parent / raw_spectrum_path``.
PROCESSED_DIR = RAW_SPECTRA_DIR.parent

OPXRD_SOURCE_PARQUET = SOURCE_DIR / "df_opxrd_summary_final_clean.parquet"
RRUFF_SOURCE_PARQUET = SOURCE_DIR / "df_rruff_summary_final_clean.parquet"

OPXRD_CLEAN_PARQUET = CLEAN_DIR / "df_opxrd_clean.parquet"
RRUFF_CLEAN_PARQUET = CLEAN_DIR / "df_rruff_clean.parquet"

FT_POOL_OPXRD_PARQUET = CLEAN_DIR / "ft_pool_opxrd.parquet"
FT_POOL_RRUFF_PARQUET = CLEAN_DIR / "ft_pool_rruff.parquet"
FT_POOL_COMBINED_PARQUET = CLEAN_DIR / "ft_pool_combined.parquet"

MODES_OPXRD_CSV = OUTPUTS_DIR / "modes_opxrd.csv"
MODES_RRUFF_CSV = OUTPUTS_DIR / "modes_rruff.csv"

# ============================================================
# Этап B: единая сетка и препроцессинг всех спектров
# ============================================================

GRID_LO = 0.0
GRID_HI = 90.0
GRID_N = 4096

PREPROC_DIR = DATA_DIR / "preprocessed"
X_INTENSITY_PATH = PREPROC_DIR / "X_intensity.f16"   # fp16, (N, GRID_N)
MASK_PATH = PREPROC_DIR / "M_mask.u8"                # uint8, (N, GRID_N)
INDEX_PARQUET = PREPROC_DIR / "index_preprocessed.parquet"

# ============================================================
# Окно углов (2 theta), в котором лежат "стандартные" эксперименты
#
# Маска: 1 - сетка спектра целиком внутри окна (данные есть),
#        0 - сетка выходит за пределы окна (нужно маскировать).
# ============================================================

ANGLE_WINDOW_MIN = 0.0
ANGLE_WINDOW_MAX = 90.0

ANGLE_MASK_COLUMN = "in_angle_window_0_90"

# ============================================================
# Целевые головы и их группы колонок
# ============================================================

LATTICE_COLUMNS = ["lattice_a", "lattice_b", "lattice_c", "alpha", "beta", "gamma"]

TARGET_HEADS = [
    "lattice",
    "spacegroup",
    "crystal_system",
    "elements",
]

# Колонки-маски: 1 - метка реально указана, 0 - нет
HEAD_MASK_COLUMNS = {
    "lattice": "head_mask_lattice",
    "spacegroup": "head_mask_spacegroup",
    "crystal_system": "head_mask_crystal_system",
    "elements": "head_mask_elements",
}

# ============================================================
# Кластеризация значений режимов
#
# Задача: слить значения, различающиеся только точностью записи,
# но НЕ слить физически разные (например Kalpha1/Kalpha2).
# Используется жадная кластеризация отсортированных значений:
# соседние значения объединяются, пока попадают в допуск.
# ============================================================

# Длина волны: допуск по абсолютной разнице, ангстрем.
# Cu Kalpha1 = 1.54056, Kalpha2 = 1.54439 -> дельта 0.00383,
# то есть допуск 5e-4 надежно разделяет дублет,
# но сливает варианты записи одной линии:
#   1.54050 / 1.54056 / 1.54060 / 1.540598 -> один кластер 1.54056
WAVELENGTH_TOL = 5e-4

# Шаг сетки: комбинированный допуск (abs ИЛИ rel, что больше).
#   0.067838 / 0.067851 / 0.067858 -> один кластер (~3e-4 rel)
#   0.067858 vs 0.069378 (rel ~0.023)     -> разные кластеры
STEP_ATOL = 2e-4
STEP_RTOL = 1e-3

# Углы сетки округляются до целого градуса (решение заказчика):
# x_max = 89.99 и 90.0 -> один режим.
ANGLE_ROUND_DECIMALS = 0

# ============================================================
# Иерархия сэмплирования режимов для генератора синтетики
# ============================================================

SOURCE_SAMPLE_WEIGHTS = {
    "rruff": 0.5,
    "opxrd": 0.5,
}
