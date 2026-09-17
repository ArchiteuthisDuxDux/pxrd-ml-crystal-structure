"""Конфигурация генератора синтетики."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, RAW_SOURCES, RAW_SPECTRA_DIR

PIPELINE_ROOT = PROJECT_ROOT

MODES_RRUFF_CSV = PIPELINE_ROOT / "outputs" / "modes_rruff.csv"
MODES_OPXRD_CSV = PIPELINE_ROOT / "outputs" / "modes_opxrd.csv"

COD_FULL_PARQUET = RAW_SOURCES["cod_parquet"]

MP_DB_PATH = PIPELINE_ROOT / "data" / "MP.db"

# Generated spectra live in the rebuilt project.

PARTIALS_DIR = PIPELINE_ROOT / "data" / "synthetic" / "partials"
SYNTH_PARQUET = PIPELINE_ROOT / "data" / "clean" / "df_synthetic_summary.parquet"

# Иерархия сэмплирования режимов: сначала источник 50/50,
# потом режим внутри источника пропорционально fraction_all.
SOURCE_WEIGHTS = {"rruff": 0.5, "opxrd": 0.5}

# ------------------------------------------------------------
# Физика профиля
# ------------------------------------------------------------

# Caglioti: FWHM^2 = U tan^2(th) + V tan(th) + W  (deg^2)
CAGLIOTI_U_RANGE = (0.0, 0.004)
CAGLIOTI_V_MEAN_STD = (0.010, 0.005)
CAGLIOTI_W_RANGE = (0.005, 0.030)

# Scherrer: размер кристаллитов, логнормальное распределение, нм
SIZE_LOGMEAN_STD = (4.79, 0.70)   # ln(120 nm), sigma
SIZE_CLIP_NM = (20.0, 600.0)
SCHERRER_K = 0.9

# Микродеформация (доля); чаще почти нулевая
STRAIN_PROB = 0.35
STRAIN_ABS_MEAN = 0.0008

# Доля лоренцевской компоненты pseudo-Voigt
ETA_RANGE = (0.25, 0.75)

# Интенсивность Kalpha2 относительно Kalpha1
KA2_WEIGHT = 0.5

# Джиттер индивидуальных отражений (недомоделированные эффекты)
JITTER_LOGSTD = 0.15
JITTER_CLIP = (0.4, 2.5)

# March-Dollase текстура
TEXTURE_PROB = 0.25
TEXTURE_R_RANGE = (0.75, 1.30)

# Смещения пиков, градусы
ZERO_SHIFT_STD = 0.02
DISPLACEMENT_STD = 0.03

# ------------------------------------------------------------
# Фон и счёты
# ------------------------------------------------------------

BG_CHEB_DEGREE_CHOICES = (2, 3, 4, 5)
BG_FRACTION_RANGE = (0.01, 0.12)

HALO_PROB = 0.30
HALO_CENTER_RANGE = (15.0, 35.0)   # deg 2theta
HALO_SIGMA_RANGE = (4.0, 12.0)
HALO_AMP_RANGE = (0.05, 0.25)

COUNTS_LOG_MEAN_STD = (8.99, 1.0)  # ln(8000), sigma -> медиана ~8000 отсчётов
COUNTS_CLIP = (400.0, 300000.0)

# ------------------------------------------------------------
# Ограничения и защита
# ------------------------------------------------------------

MAX_HKL_CANDIDATES = 3_000_000   # верхняя граница перебора hkl
MAX_FAMILIES = 150_000           # максимум уникальных семейств отражений
MAX_ATOMS_EXPANDED = 5000        # слишком большие структуры пропускаем
D_MAX_CLAMP = 80.0               # ангстрем

SUBBIN_INTEGRATION = 5           # усреднение профиля по под-точкам бина

VALID_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr",
}

CRYSTAL_SYSTEM_BY_SG = (
    [("triclinic", 1, 2), ("monoclinic", 3, 15), ("orthorhombic", 16, 74),
     ("tetragonal", 75, 142), ("trigonal", 143, 167), ("hexagonal", 168, 194),
     ("cubic", 195, 230)]
)


def crystal_system_from_sg(sg):
    """Номер пространственной группы -> кристаллическая система."""
    if not sg or sg < 1 or sg > 230:
        return None
    for name, lo, hi in CRYSTAL_SYSTEM_BY_SG:
        if lo <= sg <= hi:
            return name
    return None


FINAL_COLUMNS = [
    "sample_id", "source", "dataset_role", "raw_spectrum_path",
    "primary_wavelength", "secondary_wavelength",
    "lattice_a", "lattice_b", "lattice_c", "alpha", "beta", "gamma",
    "spacegroup_number", "crystal_system", "elements",
    "phase_count", "phase_fraction",
    "has_lattice", "has_spacegroup", "has_elements",
    "has_phase_count", "has_phase_fraction",
    "phase_compositions", "spacegroups_all", "lattices_all",
    "is_single_phase", "is_simulated", "crystallite_size_nm", "temp_K",
    "elements_list", "elements_json",
]
