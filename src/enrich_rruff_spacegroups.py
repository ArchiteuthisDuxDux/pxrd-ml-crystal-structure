"""Build conservative space-group labels for RRUFF powder XRD spectra.

The RRUFF JSON contains spectrum-level RRUFF IDs and mineral names, while the
IMA export contains mineral names and one or more Hermann--Mauguin symbols.
The current IMA export has no populated RRUFF IDs, so names are used as a
fallback.  A label is considered usable only when every parsed symbol is
resolved and all symbols point to the same International Tables number.

The original JSON is never modified.  The output is a compact CSV that can be
joined to the RRUFF parquet by RRUFF ID, plus a JSON summary of match quality.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from project_paths import PROJECT_ROOT, RAW_SOURCES

try:
    import gemmi
except ImportError as exc:  # pragma: no cover - depends on the active env
    raise SystemExit(
        "gemmi is required to convert Hermann-Mauguin symbols to numbers. "
        "Install it with: python -m pip install gemmi"
    ) from exc


ROOT = PROJECT_ROOT
DEFAULT_JSON = RAW_SOURCES["rruff_json"]
DEFAULT_IMA = RAW_SOURCES["rruff_ima_csv"]
DEFAULT_OUTPUT = ROOT / "data" / "source" / "rruff_spacegroup_mapping.csv"
DEFAULT_SUMMARY = ROOT / "outputs" / "rruff_spacegroup_mapping_summary.json"

SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def normalize_name(value: object) -> str:
    """Normalize spelling without using fuzzy matching."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", "", text)


def split_rruff_ids(value: object) -> list[str]:
    return [
        token.upper()
        for token in re.findall(r"R\d{5,}", str(value or ""), flags=re.I)
    ]


def split_symbols(raw: str) -> list[str]:
    """Split IMA alternatives; '/' is retained because it belongs to symbols."""
    raw = str(raw or "").strip()
    if not raw:
        return []

    parts: list[str] = []
    for chunk in re.split(r"\s*[|;,]\s*", raw):
        chunk = chunk.strip()
        if not chunk:
            continue

        # Parentheses occasionally contain an alternative setting.  Keep the
        # main symbol and the parenthesized symbol as separate candidates.
        match = re.fullmatch(r"([^()]+?)\s*\(([^()]+)\)", chunk)
        if match:
            parts.extend([match.group(1).strip(), match.group(2).strip()])
        else:
            parts.append(chunk)

    return list(dict.fromkeys(parts))


def resolve_symbol(symbol: str) -> int | None:
    text = unicodedata.normalize("NFKC", symbol).translate(SUBSCRIPT_TRANSLATION)
    text = text.replace("−", "-").replace("–", "-").strip().strip(".?")

    variants = [
        text,
        text.replace("_", ""),
        re.sub(r"\s+", "", text.replace("_", "")),
    ]
    for candidate in dict.fromkeys(v for v in variants if v):
        group = gemmi.find_spacegroup_by_name(candidate)
        if group is not None:
            return int(group.number)
    return None


def crystal_system_from_sg(number: int) -> str:
    if number <= 2:
        return "triclinic"
    if number <= 15:
        return "monoclinic"
    if number <= 74:
        return "orthorhombic"
    if number <= 142:
        return "tetragonal"
    if number <= 167:
        return "trigonal"
    if number <= 194:
        return "hexagonal"
    return "cubic"


def parse_rruff_crystal_system(value: object) -> str:
    match = re.search(
        r"crystal\s+system\s*:\s*([A-Za-z]+)", str(value or ""), flags=re.I
    )
    if match is None:
        return ""
    system = match.group(1).lower()
    # RRUFF uses the older descriptive word "rhombohedral" for part of the
    # trigonal system.  The model and SG number ranges use "trigonal".
    return "trigonal" if system == "rhombohedral" else system


def crystal_systems_compatible(rruff_system: str, sg_system: str) -> bool:
    if not rruff_system:
        return True
    if rruff_system == sg_system:
        return True
    # RRUFF often calls the hexagonal-axis setting of a trigonal structure
    # "hexagonal" (a=b, gamma=120).  The SG classification remains trigonal.
    return rruff_system == "hexagonal" and sg_system == "trigonal"


def build_ima_indexes(rows: list[dict[str, str]]):
    by_id: dict[str, dict[str, str]] = {}
    by_exact: dict[str, tuple[dict[str, str], str]] = {}
    by_normalized: dict[str, tuple[dict[str, str], str]] = {}
    normalized_owners: dict[str, set[int]] = defaultdict(set)

    for row_number, row in enumerate(rows):
        for rid in split_rruff_ids(row.get("RRUFF IDs")):
            by_id[rid] = row

        for column in ("Mineral Name", "Mineral Name (plain)"):
            name = str(row.get(column) or "").strip()
            if not name:
                continue
            by_exact.setdefault(name.casefold(), (row, column))
            key = normalize_name(name)
            if key:
                normalized_owners[key].add(row_number)
                by_normalized.setdefault(key, (row, column))

    collisions = {key for key, owners in normalized_owners.items() if len(owners) > 1}
    for key in collisions:
        by_normalized.pop(key, None)
    return by_id, by_exact, by_normalized, collisions


def match_ima_row(
    rruff_id: str,
    mineral_name: str,
    by_id: dict[str, dict[str, str]],
    by_exact: dict[str, tuple[dict[str, str], str]],
    by_normalized: dict[str, tuple[dict[str, str], str]],
):
    if rruff_id.upper() in by_id:
        return by_id[rruff_id.upper()], "rruff_id"

    exact = by_exact.get(mineral_name.strip().casefold())
    if exact is not None:
        row, column = exact
        return row, "exact_name" if column == "Mineral Name" else "exact_plain_name"

    normalized = by_normalized.get(normalize_name(mineral_name))
    if normalized is not None:
        row, column = normalized
        method = "normalized_name" if column == "Mineral Name" else "normalized_plain_name"
        return row, method
    return None, "unmatched"


def classify_spacegroup(raw: str, rruff_crystal_system: str):
    symbols = split_symbols(raw)
    resolved = [(symbol, resolve_symbol(symbol)) for symbol in symbols]
    unresolved = [symbol for symbol, number in resolved if number is None]
    numbers = sorted({number for _, number in resolved if number is not None})

    compatible_numbers = [
        number
        for number in numbers
        if crystal_systems_compatible(
            rruff_crystal_system, crystal_system_from_sg(number)
        )
    ]

    if not symbols:
        status = "matched_no_spacegroup"
    elif unresolved:
        status = "matched_unresolved_symbol"
    elif rruff_crystal_system and not compatible_numbers:
        status = "matched_crystal_system_conflict"
    elif len(compatible_numbers) == 1:
        status = (
            "usable_after_crystal_system_filter"
            if len(numbers) > 1
            else "usable_unique_spacegroup"
        )
    elif len(compatible_numbers) > 1:
        status = "matched_ambiguous_spacegroup"
    elif len(numbers) == 1:
        status = "usable_unique_spacegroup"
    elif len(numbers) > 1:
        status = "matched_ambiguous_spacegroup"
    else:
        status = "matched_unresolved_symbol"

    usable_number = (
        compatible_numbers[0]
        if status
        in {"usable_unique_spacegroup", "usable_after_crystal_system_filter"}
        else None
    )
    return symbols, numbers, compatible_numbers, unresolved, usable_number, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rruff-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--ima-csv", type=Path, default=DEFAULT_IMA)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.rruff_json.open("r", encoding="utf-8") as handle:
        spectra = json.load(handle)
    with args.ima_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        ima_rows = list(csv.DictReader(handle))

    by_id, by_exact, by_normalized, collisions = build_ima_indexes(ima_rows)
    output_rows: list[dict[str, object]] = []
    spectra_by_id: dict[str, list[dict[str, object]]] = defaultdict(list)
    for spectrum in spectra:
        spectra_by_id[str(spectrum.get("##RRUFFID") or "").strip()].append(spectrum)

    for rruff_id, id_spectra in spectra_by_id.items():
        spectrum = id_spectra[0]
        mineral_name = str(spectrum.get("##NAMES") or "").strip()
        rruff_crystal_system = parse_rruff_crystal_system(
            spectrum.get("##CELL PARAMETERS")
        )
        ima_row, match_method = match_ima_row(
            rruff_id, mineral_name, by_id, by_exact, by_normalized
        )

        if ima_row is None:
            symbols, numbers, compatible, unresolved, usable_number = [], [], [], [], None
            status = "unmatched_name"
            ima_name = ""
            raw = ""
        else:
            ima_name = str(ima_row.get("Mineral Name") or "").strip()
            raw = str(ima_row.get("Space Groups") or "").strip()
            symbols, numbers, compatible, unresolved, usable_number, status = (
                classify_spacegroup(raw, rruff_crystal_system)
            )

        duplicate_names = {
            str(item.get("##NAMES") or "").strip() for item in id_spectra
        }
        if len(duplicate_names) > 1:
            usable_number = None
            status = "duplicate_rruff_id_name_conflict"

        output_rows.append(
            {
                "rruff_id": rruff_id,
                "mineral_name": mineral_name,
                "json_record_count": len(id_spectra),
                "ima_mineral_name": ima_name,
                "match_method": match_method,
                "rruff_crystal_system": rruff_crystal_system,
                "spacegroup_raw": raw,
                "candidate_symbols": "|".join(symbols),
                "candidate_numbers": "|".join(map(str, numbers)),
                "crystal_system_compatible_numbers": "|".join(map(str, compatible)),
                "unresolved_symbols": "|".join(unresolved),
                "spacegroup_number": usable_number if usable_number is not None else "",
                "crystal_system": (
                    crystal_system_from_sg(usable_number)
                    if usable_number is not None
                    else ""
                ),
                "usable_label": int(usable_number is not None),
                "status": status,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    status_counts = Counter(str(row["status"]) for row in output_rows)
    method_counts = Counter(str(row["match_method"]) for row in output_rows)
    usable_ids = {row["rruff_id"] for row in output_rows if row["usable_label"]}
    unresolved_counts = Counter(
        symbol
        for row in output_rows
        for symbol in str(row["unresolved_symbols"]).split("|")
        if symbol
    )
    summary = {
        "rruff_json_records": len(spectra),
        "rruff_unique_ids": len(spectra_by_id),
        "mapping_rows": len(output_rows),
        "rruff_unique_mineral_names": len(
            {str(row.get("##NAMES") or "") for row in spectra}
        ),
        "ima_records": len(ima_rows),
        "ima_rows_with_rruff_ids": sum(bool(split_rruff_ids(r.get("RRUFF IDs"))) for r in ima_rows),
        "normalized_name_collisions": len(collisions),
        "match_method_counts": dict(sorted(method_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "usable_json_records": sum(
            int(row["usable_label"]) * int(row["json_record_count"])
            for row in output_rows
        ),
        "usable_unique_rruff_ids": len(usable_ids),
        "usable_fraction": round(
            len(usable_ids) / len(output_rows), 6
        ),
        "unresolved_symbols": dict(unresolved_counts.most_common()),
    }
    with args.summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nMapping: {args.output_csv}")
    print(f"Summary: {args.summary_json}")


if __name__ == "__main__":
    main()
