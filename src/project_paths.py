"""Single source of truth for project and raw-data paths.

Only ``config/raw_sources.json`` may point outside this project. Every derived
artifact is written below ``PROJECT_ROOT``.
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DIR = DATA_DIR / "source"
CLEAN_DIR = DATA_DIR / "clean"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
INTERIM_DIR = DATA_DIR / "interim"
RAW_SPECTRA_DIR = INTERIM_DIR / "raw"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
LOGS_DIR = REPORTS_DIR / "logs"


def load_raw_sources() -> dict[str, Path]:
    
    """Load and resolve the five read-only inputs of the rebuilt project."""

    values = json.loads((CONFIG_DIR / "raw_sources.json").read_text(encoding="utf-8"))
    return {name: Path(value).expanduser().resolve() for name, value in values.items()}


RAW_SOURCES = load_raw_sources()


def ensure_project_directories() -> None:

    """Create every directory used for generated artifacts."""

    for path in (
        SOURCE_DIR,
        CLEAN_DIR,
        PREPROCESSED_DIR,
        RAW_SPECTRA_DIR,
        CHECKPOINTS_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        LOGS_DIR,
        DATA_DIR / "synth_peaktables",
    ):
        path.mkdir(parents=True, exist_ok=True)


def validate_raw_sources() -> dict[str, dict[str, object]]:
    """Return a compact validation report and fail on missing raw inputs."""
    report: dict[str, dict[str, object]] = {}
    missing: list[str] = []

    for name, path in RAW_SOURCES.items():
        exists = path.exists()
        report[name] = {
            "path": str(path),
            "exists": exists,
            "kind": "directory" if path.is_dir() else "file",
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }

        if not exists:
            missing.append(f"{name}: {path}")

    if missing:
        raise FileNotFoundError("Missing raw sources:\n" + "\n".join(missing))
    
    return report
