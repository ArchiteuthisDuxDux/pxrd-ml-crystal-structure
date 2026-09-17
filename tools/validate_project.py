"""Fast structural validation; does not generate data or train models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
SRC_DIR = ROOT / "src"

EXPECTED_NOTEBOOKS = [f"{i:02d}_" for i in range(25)]
EXPECTED_MODELS = set(json.loads(
    (ROOT / "config" / "expected_checkpoints.json").read_text(encoding="utf-8")
))


def validate_notebooks() -> str:
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    assert len(notebooks) == 25, f"Expected 25 notebooks, found {len(notebooks)}"
    for prefix, path in zip(EXPECTED_NOTEBOOKS, notebooks):
        assert path.name.startswith(prefix), f"Broken order: {path.name}"
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert notebook.cells and notebook.cells[0].cell_type == "markdown"
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{path.name}:cell-{index}", "exec")
    return f"notebooks: {len(notebooks)} valid"


def validate_sources() -> str:
    scripts = sorted(SRC_DIR.rglob("*.py"))
    for script in scripts:
        compile(script.read_text(encoding="utf-8-sig"), str(script), "exec")
    return f"python sources: {len(scripts)} valid"


def validate_external_references() -> str:
    offenders: list[str] = []
    for folder in (NOTEBOOK_DIR, SRC_DIR):
        for path in folder.rglob("*"):
            if path.suffix.lower() not in {".py", ".ipynb", ".md"}:
                continue
            if path.suffix.lower() == ".ipynb":
                # Проверяем только исполняемый/текстовый контент. Вывод ячеек
                # может законно печатать пути к настроенным сырым источникам.
                notebook = nbformat.read(path, as_version=4)
                text = "\n".join(cell.source for cell in notebook.cells)
            else:
                text = path.read_text(encoding="utf-8")
            if "D:/Users/user/Desktop/DS XRD" in text or "D:\\\\Users\\\\user\\\\Desktop\\\\DS XRD" in text:
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, "Old-project paths outside raw_sources.json: " + ", ".join(offenders)
    return "external references: raw_sources.json only"


def validate_model_names() -> str:
    # Dynamic f-strings in the original training notebooks construct several
    # names from experiment prefixes. The explicit manifest is the submission
    # contract and is checked for uniqueness and the required final models.
    assert len(EXPECTED_MODELS) == 14
    assert all(name.endswith(".pt") for name in EXPECTED_MODELS)
    required = {
        "pretrain_v2_full_best.pt",
        "ft_rruff_only_with_rruff_sg_final.pt",
        "ft_opxrd_only_final.pt",
        "ft_combined_no_replay_control_with_rruff_sg_final.pt",
        "ft_single_phase_only_with_rruff_sg_final.pt",
    }
    assert required.issubset(EXPECTED_MODELS)
    return f"checkpoint manifest: {len(EXPECTED_MODELS)} names preserved"


def main() -> int:
    checks = [
        validate_notebooks(),
        validate_sources(),
        validate_external_references(),
        validate_model_names(),
    ]
    print("PROJECT_STRUCTURE_OK")
    for check in checks:
        print(" -", check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
