"""
Конфигурация генератора синтетики из COD и crystalDB (MP.db).

Схема выходного паркета - точная копия схемы df_opxrd_summary_final_clean /
df_rruff_summary_final_clean (31 колонка, тот же порядок).

Режимы съемки сэмплируются иерархически (как в эксперименте):
    1) семья режимов: rruff <-> opxrd с вероятностью 50/50;
    2) внутри семьи: режим пропорционально fraction_all из
       outputs/modes_rruff.csv / modes_opxrd.csv.
"""

from pathlib import Path

from project_paths import PROJECT_ROOT, RAW_SOURCES, RAW_SPECTRA_DIR

PIPELINE_ROOT = PROJECT_ROOT

# ============================================================
# Источники структур
# ============================================================

# COD: большой паркет, читается ПОСТРОЧНО (row group за row group),
# вытягиваются только нужные колонки.
COD_PARQUET = RAW_SOURCES["cod_parquet"]
COD_COLUMNS = ["file", "cif_text", "sgNumber", "duplicateof"]

# crystalDB: база Materials Project в формате ASE DB
# (рабочая копия, восстановленная из MP.db.gz - оригинал в crystalDB/ пуст)
MP_DB_PATH = PIPELINE_ROOT / "data" / "MP.db"

# Куда пишутся спектры - общая папка со всеми сырыми графиками
# Synthetic .npy files are generated beside the rebuilt real spectra.

# Таблицы режимов из шага 3
MODES_CSV = {
    "rruff": PIPELINE_ROOT / "outputs" / "modes_rruff.csv",
    "opxrd": PIPELINE_ROOT / "outputs" / "modes_opxrd.csv",
}

# Выходы
SHARDS_DIR = PIPELINE_ROOT / "data" / "clean" / "synth_shards"
OUTPUTS_DIR = PIPELINE_ROOT / "outputs"
FINAL_PARQUET = PIPELINE_ROOT / "data" / "clean" / "df_synth_summary_final_clean.parquet"

DONE_LISTS = {
    "cod": OUTPUTS_DIR / "synth_done_cod.txt",
    "crystaldb": OUTPUTS_DIR / "synth_done_crystaldb.txt",
}

# ============================================================
# Схема выходного паркета (идентична двум исходным)
# ============================================================

SCHEMA_COLUMNS = [
    "sample_id", "source", "dataset_role", "raw_spectrum_path",
    "primary_wavelength", "secondary_wavelength",
    "lattice_a", "lattice_b", "lattice_c", "alpha", "beta", "gamma",
    "spacegroup_number", "crystal_system", "elements",
    "phase_count", "phase_fraction",
    "has_lattice", "has_spacegroup", "has_elements",
    "has_phase_count", "has_phase_fraction",
    "phase_compositions", "spacegroups_all", "lattices_all",
    "is_single_phase", "is_simulated",
    "crystallite_size_nm", "temp_K",
    "elements_list", "elements_json",
]

# ============================================================
# Отбор структур
# ============================================================

GENERATION_VERSION = "v4"   # зашивается в манифест генерации
# v4 = ФИНАЛЬНАЯ калибровочная итерация: все "декор"-ручки по-семейные.
# После v4 генератор замораживается.

MAX_ATOMS = 500          # больше - пропуск (слишком тяжело/патологично)
SKIP_COD_DUPLICATES = True   # duplicateof не пустой -> пропуск

# ============================================================
# Иерархия сэмплирования режимов
# ============================================================

SOURCE_SAMPLE_WEIGHTS = {"rruff": 0.5, "opxrd": 0.5}

# ============================================================
# Физика профиля: уширение
# ============================================================

SCHERRER_K = 0.9

# Размер кристаллитов, нм: логнормаль, ПО-СЕМЕЙНЫМ
# v4: domain_diag - у RRUFF пики уже (крупные кристаллы ~80 нм),
#     у opXRD шире (~30 нм)
CRYSTALLITE_SIZE_LOGMU = {
    "rruff": 4.381,      # ~80 нм
    "opxrd": 3.401,      # ~30 нм
}
CRYSTALLITE_SIZE_LOGSIGMA = 0.7
CRYSTALLITE_SIZE_MIN_NM = 10.0
CRYSTALLITE_SIZE_MAX_NM = 200.0

# Инструментальное разрешение (Кальоти), FWHM^2 = U tan^2 + V tan + W [deg^2]
# v4: ПО-СЕМЕЙНЫМ (RRUFF - узкая база, opXRD - широкая)
CAGLIOTI_W_RANGE = {
    "rruff": (4.0e-4, 3.6e-3),
    "opxrd": (1.2e-3, 8.0e-3),
}
CAGLIOTI_U_RANGE = (0.0, 4.0e-3)
CAGLIOTI_V_RANGE = (-2.0e-3, 4.0e-3)

# Микродеформация: eps ~ |N(0, sigma)|, клип 3e-3 -> FWHM = 4 eps tan(theta)
STRAIN_SIGMA = 1.2e-3
STRAIN_MAX = 3.0e-3

FWHM_MIN_DEG = 0.03
FWHM_MAX_DEG = 2.0

# Псевдо-Фойгт: доля лоренцевской компоненты
PV_ETA_RANGE = (0.3, 0.7)

# ============================================================
# Физика профиля: интенсивности
# ============================================================

# Джиттер индивидуальных отражений (логнормаль)
PEAK_JITTER_SIGMA = 0.25
PEAK_MIN_REL_INTENSITY = 0.05   # отсечка от максимума (=100 в pymatgen)
PEAK_MAX_COUNT = 4000

# March-Dollase: вероятность текстурированного образца и диапазон r
TEXTURE_PROB = 0.28
TEXTURE_R_RANGE = (0.75, 1.35)

# Дублет Kalpha2
KALPHA2_WEIGHT = 0.5

# ============================================================
# Фон и артефакты
# ============================================================

# Шевбашевский фон: амплитуды относительно максимума профиля, ПО-СЕМЕЙНЫМ
# v4: opXRD база снижена (перелёт 0.375 против цели 0.213)
CHEB_BASE_RANGE = {
    "rruff": (0.003, 0.04),
    "opxrd": (0.04, 0.18),
}
# v3: наклон/кривизна сужены - у реального фона наклон почти нулевой
CHEB_C1_RANGE = (-0.03, 0.03)
CHEB_HIGH_RANGE = (-0.02, 0.02)     # c2..c5 общий масштаб

# Аморфное гало, ПО-СЕМЕЙНЫМ
# v4: у очищенных RRUFF гало почти нет; у opXRD - умеренное
HALO = {
    "rruff": {"prob": 0.10, "height": (0.01, 0.05)},
    "opxrd": {"prob": 0.18, "height": (0.02, 0.08)},
}
HALO_CENTER_RANGE = (12.0, 32.0)    # градусы 2theta
HALO_SIGMA_RANGE = (4.0, 9.0)

# Сдвиги
ZERO_SHIFT_MAX_DEG = 0.04           # константный zero shift, +/-U(0, max)
DISPLACEMENT_FRAC_MAX = 0.06        # смещение высоты образца:
                                    # delta = -disp * cos(theta), disp <= max

# Счёты и шум (Пуассон): уровень счётов в максимуме профиля.
# v4: rruff ещё шумнее (шум оставался в 2.2 раза ниже реального)
COUNTS_LOG_RANGE = {
    "rruff": (1.9, 3.1),     # ~80..1250 отсчётов
    "opxrd": (3.0, 4.0),     # ~1000..10000 отсчётов
}

SATURATION_PROB = 0.12
SATURATION_LEVEL_RANGE = (0.85, 0.99)

ZINGER_PROB = 0.15
ZINGER_COUNT_RANGE = (1, 3)
ZINGER_HEIGHT_RANGE = (0.3, 1.2)

# Таблицы пиков (для перерендера дисторшнов без касания физики)
PEAKTABLE_DIR = PIPELINE_ROOT / "data" / "synth_peaktables"


def sg_number_to_crystal_system(sg):
    """Детерминированное отображение номера space group (1-230) в систему."""
    if sg is None or (isinstance(sg, float) and sg != sg) or sg < 1 or sg > 230:
        return None
    if sg <= 2:
        return "triclinic"
    if sg <= 15:
        return "monoclinic"
    if sg <= 74:
        return "orthorhombic"
    if sg <= 142:
        return "tetragonal"
    if sg <= 167:
        return "trigonal"
    if sg <= 194:
        return "hexagonal"
    return "cubic"
