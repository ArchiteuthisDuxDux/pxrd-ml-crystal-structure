"""Create new FT parquet files with conservative RRUFF SG labels.

The existing parquet files are read-only inputs.  New files receive the
``with_rruff_sg`` suffix, so the historical experiments remain reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
RRUFF_POOL = BASE / "data" / "clean" / "ft_pool_rruff.parquet"
OPXRD_POOL = BASE / "data" / "clean" / "ft_pool_opxrd.parquet"
MAPPING = BASE / "data" / "source" / "rruff_spacegroup_mapping.csv"
RRUFF_OUTPUT = BASE / "data" / "clean" / "ft_pool_rruff_with_rruff_sg.parquet"
COMBINED_OUTPUT = BASE / "data" / "clean" / "ft_pool_combined_with_rruff_sg.parquet"
SUMMARY_OUTPUT = BASE / "outputs" / "rruff_sg_pool_build_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rruff-pool", type=Path, default=RRUFF_POOL)
    parser.add_argument("--opxrd-pool", type=Path, default=OPXRD_POOL)
    parser.add_argument("--mapping", type=Path, default=MAPPING)
    parser.add_argument("--rruff-output", type=Path, default=RRUFF_OUTPUT)
    parser.add_argument("--combined-output", type=Path, default=COMBINED_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_OUTPUT)
    parser.add_argument(
        "--force", action="store_true", help="replace only the suffixed output files"
    )
    return parser.parse_args()


def extract_rruff_ids(frame: pd.DataFrame) -> pd.Series:
    result = frame["source"].astype(str).str.extract(r"(R\d{5,})", expand=False)
    if "raw_spectrum_path" in frame:
        fallback = (
            frame["raw_spectrum_path"]
            .astype(str)
            .str.extract(r"(R\d{5,})", expand=False)
        )
        result = result.fillna(fallback)
    return result.str.upper()


def validate_destination(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {path}. Use --force to rebuild the suffixed file."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    for destination in (args.rruff_output, args.combined_output, args.summary_output):
        validate_destination(destination, args.force)

    mapping = pd.read_csv(args.mapping, dtype={"rruff_id": "string"})
    if mapping["rruff_id"].isna().any() or mapping["rruff_id"].duplicated().any():
        raise ValueError("The SG mapping must contain one non-empty row per RRUFF ID")

    rruff = pd.read_parquet(args.rruff_pool).copy()
    opxrd = pd.read_parquet(args.opxrd_pool).copy()
    rruff["rruff_id"] = extract_rruff_ids(rruff)
    if rruff["rruff_id"].isna().any():
        bad = rruff.loc[rruff["rruff_id"].isna(), ["source", "raw_spectrum_path"]]
        raise ValueError(f"Could not extract RRUFF IDs:\n{bad.head().to_string(index=False)}")
    if rruff["rruff_id"].duplicated().any():
        raise ValueError("RRUFF FT pool contains duplicate RRUFF IDs")

    keep = [
        "rruff_id",
        "mineral_name",
        "ima_mineral_name",
        "match_method",
        "spacegroup_raw",
        "candidate_numbers",
        "spacegroup_number",
        "crystal_system",
        "usable_label",
        "status",
    ]
    labels = mapping[keep].rename(
        columns={
            "mineral_name": "rruff_mineral_name",
            "ima_mineral_name": "ima_mineral_name_sg",
            "match_method": "spacegroup_match_method",
            "spacegroup_raw": "spacegroup_symbol_ima",
            "candidate_numbers": "spacegroup_candidate_numbers",
            "spacegroup_number": "spacegroup_number_ima",
            "crystal_system": "crystal_system_from_sg",
            "usable_label": "spacegroup_ima_usable",
            "status": "spacegroup_match_status",
        }
    )
    enriched = rruff.merge(labels, on="rruff_id", how="left", validate="one_to_one")
    if enriched["rruff_mineral_name"].isna().any():
        missing = enriched.loc[enriched["rruff_mineral_name"].isna(), "rruff_id"]
        raise ValueError(f"RRUFF IDs absent from mapping: {missing.head().tolist()}")

    old_sg = pd.to_numeric(enriched["spacegroup_number"], errors="coerce")
    new_sg = pd.to_numeric(enriched["spacegroup_number_ima"], errors="coerce")
    conflict = old_sg.notna() & new_sg.notna() & old_sg.ne(new_sg)
    if conflict.any():
        raise ValueError("Existing and IMA SG labels conflict; refusing to overwrite")

    enriched["spacegroup_number_original"] = old_sg.astype("Int64")
    enriched["crystal_system_original"] = enriched["crystal_system"]
    enriched["spacegroup_number"] = new_sg.combine_first(old_sg).astype("Int64")

    # Keep the crystal-system and SG heads internally consistent.  The raw
    # RRUFF wording remains available in crystal_system_original.
    use_ima = new_sg.notna()
    enriched.loc[use_ima, "crystal_system"] = enriched.loc[
        use_ima, "crystal_system_from_sg"
    ]
    enriched["head_mask_spacegroup"] = (
        enriched["spacegroup_number"].notna().astype("int8")
    )
    enriched["head_mask_crystal_system"] = (
        enriched["crystal_system"].notna().astype("int8")
    )
    enriched["spacegroup_label_source"] = "IMA_Export_202691_000115"

    combined = pd.concat([enriched, opxrd], ignore_index=True, sort=False)
    enriched.to_parquet(args.rruff_output, index=False)
    combined.to_parquet(args.combined_output, index=False)

    usable = int(enriched["spacegroup_number"].notna().sum())
    summary = {
        "rruff_rows": len(enriched),
        "opxrd_rows": len(opxrd),
        "combined_rows": len(combined),
        "rruff_spacegroup_labels": usable,
        "rruff_spacegroup_fraction": round(usable / len(enriched), 6),
        "rruff_minerals": int(enriched["rruff_mineral_name"].nunique()),
        "rruff_represented_spacegroups": int(enriched["spacegroup_number"].nunique()),
        "rruff_output": str(args.rruff_output),
        "combined_output": str(args.combined_output),
    }
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
