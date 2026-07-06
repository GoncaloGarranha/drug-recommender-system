from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from kg_constructor.src.utils.paths import build_output_paths, resolve_relative_path


SALT_OR_FORM_WORDS = {
    "acid", "anhydrous", "base", "calcium", "chloride", "citrate", "dihydrate",
    "disodium", "fumarate", "hemihydrate", "hcl", "hydrobromide", "hydrochloride",
    "hydrate", "maleate", "monohydrate", "phosphate", "potassium", "sodium",
    "succinate", "sulfate", "sulphate", "tartrate", "trihydrate", "xinafoate",
}

EXCIPIENT_HINTS = {
    "cellulose", "starch", "silicon dioxide", "silica", "magnesium stearate",
    "titanium dioxide", "water", "lactose", "gelatin", "hypromellose",
    "povidone", "crospovidone", "croscarmellose", "talc", "polyethylene glycol",
    "propylene glycol", "glycerin", "shellac", "ferric oxide", "fd&c", "d&c",
    "sodium lauryl sulfate", "polysorbate", "sucrose", "sorbitol",
}


def normalize_name_for_matching(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def canonical_for_salt_matching(value: Any) -> str:
    tokens = normalize_name_for_matching(value).split()
    tokens = [tok for tok in tokens if tok not in SALT_OR_FORM_WORDS]
    return " ".join(tokens)


def is_combination_name(value: str) -> bool:
    norm = normalize_name_for_matching(value)
    return " and " in f" {norm} " or "/" in value or "+" in value or "," in value


def extract_identifier(row: Dict[str, Any]) -> str:
    for key in ("chebi_id", "drugbank_id", "code"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return ""


def identifier_set(row: Dict[str, Any]) -> Set[str]:
    return {
        str(row[key]).strip().lower()
        for key in ("chebi_id", "drugbank_id", "code")
        if row.get(key)
    }


def looks_like_excipient(name: str) -> bool:
    norm = normalize_name_for_matching(name)
    return any(hint in norm for hint in EXCIPIENT_HINTS)


def add_result(
    results: List[Dict[str, str]],
    seen: Set[Tuple[str, str, str]],
    *,
    source: str,
    file_name: str,
    drug_name: str,
    drug_reference_id: str,
    ingredient: Dict[str, Any],
    mapping_method: str,
    confidence: str,
) -> None:
    key = (
        drug_name,
        extract_identifier(ingredient),
        normalize_name_for_matching(ingredient.get("name")),
    )

    if key in seen:
        return

    seen.add(key)

    results.append(
        {
            "source": source,
            "file": file_name,
            "drug_name": drug_name,
            "drug_reference_id": drug_reference_id,
            "active_ingredient_name": str(ingredient.get("name") or "").strip(),
            "active_ingredient_id": extract_identifier(ingredient),
            "mapping_method": mapping_method,
            "confidence": confidence,
        }
    )


def infer_active_ingredients(data: Dict[str, Any], file_name: str) -> List[Dict[str, str]]:
    source = str(data.get("source") or data.get("organization") or "DailyMed/MedJsonify")

    drugs = data.get("drug", [])
    ingredients = data.get("ingredients", [])

    if not isinstance(drugs, list):
        drugs = []
    if not isinstance(ingredients, list):
        ingredients = []

    drug_entries = [d for d in drugs if isinstance(d, dict)]
    ingredient_entries = [i for i in ingredients if isinstance(i, dict)]

    product_name = str(
        data.get("name") or " ".join(str(d.get("name", "")) for d in drug_entries)
    ).strip()

    product_can = canonical_for_salt_matching(product_name)
    product_is_combination = is_combination_name(product_name)

    results: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()

    # Rule 1: exact identifier match.
    for drug in drug_entries:
        drug_name = str(drug.get("name") or data.get("name") or "").strip()
        drug_reference_id = extract_identifier(drug)
        drug_reference_ids = identifier_set(drug)

        for ingredient in ingredient_entries:
            if drug_reference_ids and (drug_reference_ids & identifier_set(ingredient)):
                add_result(
                    results,
                    seen,
                    source=source,
                    file_name=file_name,
                    drug_name=drug_name,
                    drug_reference_id=drug_reference_id,
                    ingredient=ingredient,
                    mapping_method="id_match",
                    confidence="high",
                )

    # Rule 2: exact normalized name match.
    for drug in drug_entries:
        drug_name = str(drug.get("name") or data.get("name") or "").strip()
        drug_reference_id = extract_identifier(drug)
        drug_norm = normalize_name_for_matching(drug_name)

        for ingredient in ingredient_entries:
            if drug_norm and drug_norm == normalize_name_for_matching(ingredient.get("name")):
                add_result(
                    results,
                    seen,
                    source=source,
                    file_name=file_name,
                    drug_name=drug_name,
                    drug_reference_id=drug_reference_id,
                    ingredient=ingredient,
                    mapping_method="name_match",
                    confidence="high",
                )

    # Rule 3: salt/form matching.
    for drug in drug_entries:
        drug_name = str(drug.get("name") or data.get("name") or "").strip()
        drug_reference_id = extract_identifier(drug)
        drug_can = canonical_for_salt_matching(drug_name)

        if not drug_can:
            continue

        for ingredient in ingredient_entries:
            ingredient_name = str(ingredient.get("name") or "")
            ingredient_can = canonical_for_salt_matching(ingredient_name)

            if (
                drug_can
                and ingredient_can
                and drug_can == ingredient_can
                and not looks_like_excipient(ingredient_name)
            ):
                add_result(
                    results,
                    seen,
                    source=source,
                    file_name=file_name,
                    drug_name=drug_name,
                    drug_reference_id=drug_reference_id,
                    ingredient=ingredient,
                    mapping_method="salt_or_form_name_match",
                    confidence="medium",
                )

    # Rule 4: combination products.
    # Important: this rule runs even if previous rules already found one active ingredient.
    if product_is_combination and product_can:
        for drug in drug_entries or [{"name": product_name}]:
            drug_name = str(drug.get("name") or product_name).strip()
            drug_reference_id = extract_identifier(drug)

            for ingredient in ingredient_entries:
                ingredient_name = str(ingredient.get("name") or "")
                ingredient_can = canonical_for_salt_matching(ingredient_name)

                if not ingredient_can:
                    continue

                if looks_like_excipient(ingredient_name):
                    continue

                if ingredient_can in product_can:
                    add_result(
                        results,
                        seen,
                        source=source,
                        file_name=file_name,
                        drug_name=drug_name,
                        drug_reference_id=drug_reference_id,
                        ingredient=ingredient,
                        mapping_method="combination_text_match",
                        confidence="medium",
                    )

    return results


def build_active_ingredients(input_dir: Path, output_csv: Path, summary_json: Path) -> None:
    rows: List[Dict[str, str]] = []
    summary = Counter()
    files_total = 0
    files_with_rows = 0

    for path in sorted(input_dir.rglob("*.json")):
        files_total += 1

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            summary["json_read_error"] += 1
            continue

        inferred = infer_active_ingredients(data, str(path))

        if inferred:
            files_with_rows += 1
            rows.extend(inferred)

            for row in inferred:
                summary[f"method:{row['mapping_method']}"] += 1
                summary[f"confidence:{row['confidence']}"] += 1
        else:
            summary["files_without_active_ingredient"] += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source",
        "file",
        "drug_name",
        "drug_reference_id",
        "active_ingredient_name",
        "active_ingredient_id",
        "mapping_method",
        "confidence",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_payload = {
        "input_dir": str(input_dir),
        "output_csv": str(output_csv),
        "files_total": files_total,
        "files_with_active_ingredient": files_with_rows,
        "rows_total": len(rows),
        "counts": dict(summary),
    }

    summary_json.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved active ingredients: {output_csv}")
    print(f"Saved summary: {summary_json}")
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


def run(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    step_cfg = cfg.get("active_ingredients", {}) if isinstance(cfg, dict) else {}
    out = build_output_paths(paths.output_dir)

    input_subdir = step_cfg.get("input_subdir", "xml_files")
    input_dir = resolve_relative_path(paths.input_dir, input_subdir)
    output_csv = resolve_relative_path(out.root, step_cfg.get("output_csv", "interim/active_ingredients_raw.csv"))
    summary_json = resolve_relative_path(out.root, step_cfg.get("summary_json", "interim/active_ingredients_summary.json"))

    if not force and output_csv.exists() and summary_json.exists():
        return

    build_active_ingredients(input_dir=input_dir, output_csv=output_csv, summary_json=summary_json)
