import os
import re
import csv
import json
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

# ----------------------------
# PATH CONFIGURATION
# ----------------------------
INPUT_DIRS = [
    "kg_constructor/data/input/csv_files",
    "kg_constructor/data/input/txt_files",
    "kg_constructor/data/input/xml_files",
]
OUTPUT_DIR = "kg_constructor/data/output"
DOID_MAP_PATH = "kg_constructor/data/processed/disease_map.json"  # produced by build_disease_map.py

TRIPLES_OUT = os.path.join(OUTPUT_DIR, "triples.csv")
ENTITY_NAMES_OUT = os.path.join(OUTPUT_DIR, "entity_names.csv")

PREDICATE = "indicated_for"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Matches DOID identifiers like "DOID:5575" or "DOID_5575"
DOID_REGEX = re.compile(r"\bDOID[:_]\d+\b", re.IGNORECASE)

# Optional blacklist to remove generic diseases
BLACKLIST_DOIDS = {
    "DOID:4",
    "DOID:7",
    "DOID:14566",
    "DOID:150",
    "DOID:50117",   # note: normalizer will convert leading zeros away
}


# ----------------------------
# UTILITIES
# ----------------------------
def normalize_doid(raw: str) -> Optional[str]:
    """
    Normalizes DOID identifiers to canonical format "DOID:<int>".
    Accepts variants such as "DOID_5575", "doid:5575", "DOID:5575".
    """
    if not raw:
        return None
    s = str(raw).strip()
    m = re.search(r"DOID[:_](\d+)", s, flags=re.IGNORECASE)
    if not m:
        return None
    return f"DOID:{int(m.group(1))}"


def normalize_text_id(s: str) -> str:
    """
    Minimal normalization for text-based identifiers.
    Avoid aggressive cleaning to prevent collapsing distinct clinical entities.
    """
    return str(s).strip()


def load_doid_map() -> Dict[str, str]:
    """
    Loads mapping DOID -> human-readable label.
    Expected format: {"DOID:5575": "Asthma", ...}
    """
    if not os.path.exists(DOID_MAP_PATH):
        return {}
    with open(DOID_MAP_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # Normalize keys to canonical DOID format
    norm: Dict[str, str] = {}
    for k, v in mapping.items():
        nk = normalize_doid(k)
        if nk:
            norm[nk] = v
    return norm


def safe_add_triple(triples_set: set, h: Optional[str], r: str, t: Optional[str]) -> None:
    """
    Adds a triple only if head and tail are valid.
    This prevents invalid edges (e.g., missing drug_id from XML/TXT fallback).
    """
    if h and t and r:
        triples_set.add((h, r, t))


def get_drug_id_and_label_from_record(record: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Robust drug extraction for JSON-like records.

    Strategy:
    - Prefer canonical identifiers when available (chebi_id/rxnorm_id/atc_code).
    - Fallback to name-based ID (uppercased) when canonical IDs are missing.

    Returns:
    - drug_id: stable identifier for the node
    - drug_label: display label for entity_names.csv
    """
    drug_block = record.get("drug", {})
    if not isinstance(drug_block, dict):
        drug_block = {}

    name = (drug_block.get("name") or record.get("name") or "").strip()

    canonical = (
        drug_block.get("chebi_id")
        or drug_block.get("rxnorm_id")
        or drug_block.get("atc_code")
        or record.get("chebi_id")
        or record.get("rxnorm_id")
        or record.get("atc_code")
    )

    if canonical:
        drug_id = normalize_text_id(canonical)
        drug_label = name if name else drug_id
        return drug_id, drug_label

    if not name:
        return None, None

    drug_id = normalize_text_id(name.upper())
    drug_label = name
    return drug_id, drug_label


def extract_doids_from_json_record(record: dict) -> List[str]:
    """
    Extracts DOIDs from a JSON-like record.

    Primary path:
    - record["indications"]["doid_entities"] list items containing "doid_id" (or similar).

    Fallback:
    - Scan likely text fields for DOID patterns.
    """
    doids: List[str] = []

    indications = record.get("indications", {})
    if isinstance(indications, dict):
        ents = indications.get("doid_entities", [])
        if isinstance(ents, list):
            for e in ents:
                if isinstance(e, dict):
                    did = e.get("doid_id") or e.get("id") or e.get("doid")
                    nd = normalize_doid(did) if did else None
                    if nd:
                        doids.append(nd)
                elif isinstance(e, str):
                    nd = normalize_doid(e)
                    if nd:
                        doids.append(nd)

    # Fallback: scan text fields
    text_fields = [
        record.get("indications_original", ""),
        record.get("contraindications_original", ""),
        record.get("text", ""),
    ]
    for tf in text_fields:
        if isinstance(tf, str) and tf:
            for m in DOID_REGEX.findall(tf):
                nd = normalize_doid(m)
                if nd:
                    doids.append(nd)

    # Deduplicate preserving order
    seen = set()
    out = []
    for d in doids:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def extract_doids_from_text(content: str) -> List[str]:
    """Extracts DOIDs from plain text content."""
    if not content:
        return []
    doids = []
    for m in DOID_REGEX.findall(content):
        nd = normalize_doid(m)
        if nd:
            doids.append(nd)
    # Preserve order, remove duplicates
    return list(dict.fromkeys(doids))


# ----------------------------
# INPUT PARSERS
# ----------------------------
def parse_json_file(path: str) -> Optional[dict]:
    """Attempts to parse a file as JSON."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def parse_csv_file(path: str) -> List[Tuple[str, str]]:
    """
    Parses CSV to extract (drug, doid) pairs.

    Supported layouts:
    - triples format: subject,predicate,object
    - pair format: drug,doid (or drug_id,doid_id)
    """
    pairs: List[Tuple[str, str]] = []
    try:
        df = pd.read_csv(path)
        cols_lower = {c.lower(): c for c in df.columns}

        # triples-style
        if {"subject", "predicate", "object"}.issubset(cols_lower.keys()):
            subject_col = cols_lower["subject"]
            predicate_col = cols_lower["predicate"]
            object_col = cols_lower["object"]

            for s, p, o in zip(df[subject_col], df[predicate_col], df[object_col]):
                if str(p).strip() != PREDICATE:
                    continue
                nd = normalize_doid(o)
                if nd:
                    pairs.append((normalize_text_id(s), nd))
            return pairs

        # drug/doid columns
        drug_candidates = ["drug", "drug_id", "name", "subject"]
        doid_candidates = ["doid", "doid_id", "disease", "object"]

        drug_col = next((cols_lower[c] for c in drug_candidates if c in cols_lower), None)
        doid_col = next((cols_lower[c] for c in doid_candidates if c in cols_lower), None)

        if drug_col and doid_col:
            for d, o in zip(df[drug_col], df[doid_col]):
                drug = normalize_text_id(d)
                nd = normalize_doid(o)
                if drug and nd:
                    pairs.append((drug, nd))
            return pairs

    except Exception:
        pass

    return pairs


def parse_txt_file(path: str) -> List[Tuple[str, str]]:
    """
    Parses TXT to extract (drug, doid) pairs.

    Supported layouts:
    - triplets per line: subject<TAB>predicate<TAB>object
    - triplets per line: subject,predicate,object
    - free text: extract DOIDs and infer drug from filename stem (weak fallback)
    """
    pairs: List[Tuple[str, str]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return pairs

    # Structured lines
    for ln in lines:
        parts = []
        if "\t" in ln:
            parts = ln.split("\t")
        elif "," in ln and ln.count(",") >= 2:
            parts = [p.strip() for p in ln.split(",", 2)]

        if len(parts) == 3:
            s, p, o = parts
            if p != PREDICATE:
                continue
            nd = normalize_doid(o)
            if nd:
                pairs.append((normalize_text_id(s), nd))

    if pairs:
        return pairs

    # Fallback: infer drug from filename
    content = "\n".join(lines)
    doids = extract_doids_from_text(content)
    if doids:
        drug_guess = os.path.splitext(os.path.basename(path))[0]
        drug_guess = normalize_text_id(drug_guess.upper())
        for d in doids:
            pairs.append((drug_guess, d))

    return pairs


def parse_xml_file(path: str) -> Tuple[Optional[str], List[str]]:
    """
    Attempts to parse XML and extract:
    - a drug label/id (best-effort)
    - DOIDs appearing anywhere in the document text
    """
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception:
        return None, []

    # Best-effort name extraction
    drug_name = None
    for tag in ["name", "drug", "drugName", "productName", "title"]:
        el = root.find(f".//{tag}")
        if el is not None and el.text and el.text.strip():
            drug_name = el.text.strip()
            break

    xml_text = ET.tostring(root, encoding="unicode", method="text")
    doids = extract_doids_from_text(xml_text)

    drug_id = normalize_text_id(drug_name.upper()) if drug_name else None
    return drug_id, doids


# ----------------------------
# MAIN EXTRACTION PIPELINE
# ----------------------------
def run_extraction() -> None:
    doid_map = load_doid_map()
    
    print(f"DOID map loaded: {len(doid_map)} entries")
    if doid_map:
        sample_k = next(iter(doid_map.keys()))
        print(f"DOID map sample: {sample_k} -> {doid_map[sample_k]}")

    triples_set = set()
    entity_labels: Dict[str, str] = {}

    input_files: List[str] = []
    for d in INPUT_DIRS:
        input_files.extend(glob.glob(os.path.join(d, "*.*")))

    if not input_files:
        raise FileNotFoundError(f"No input files found in: {INPUT_DIRS}")

    print(f"Input files found: {len(input_files)}")

    for fp in input_files:
        ext = os.path.splitext(fp)[1].lower()

        # Try JSON first (even if extension is not .json)
        record = parse_json_file(fp)
        if isinstance(record, dict):
            drug_id, drug_label = get_drug_id_and_label_from_record(record)
            if drug_id:
                entity_labels[drug_id] = drug_label if drug_label else drug_id

            for doid in extract_doids_from_json_record(record):
                if doid in BLACKLIST_DOIDS:
                    continue
                doid_label = doid_map.get(doid, doid)
                if str(doid_label).upper().startswith("OBSOLETE"):
                    continue
                entity_labels[doid] = doid_label
                safe_add_triple(triples_set, drug_id, PREDICATE, doid)
            continue

        # CSV
        if ext == ".csv":
            for drug, doid in parse_csv_file(fp):
                if doid in BLACKLIST_DOIDS:
                    continue
                doid_label = doid_map.get(doid, doid)
                if str(doid_label).upper().startswith("OBSOLETE"):
                    continue
                drug_id = normalize_text_id(str(drug).upper())
                entity_labels.setdefault(drug_id, str(drug))
                entity_labels[doid] = doid_label
                safe_add_triple(triples_set, drug_id, PREDICATE, doid)
            continue

        # TXT/TSV
        if ext in {".txt", ".tsv"}:
            for drug, doid in parse_txt_file(fp):
                if doid in BLACKLIST_DOIDS:
                    continue
                doid_label = doid_map.get(doid, doid)
                if str(doid_label).upper().startswith("OBSOLETE"):
                    continue
                drug_id = normalize_text_id(str(drug).upper())
                entity_labels.setdefault(drug_id, str(drug))
                entity_labels[doid] = doid_label
                safe_add_triple(triples_set, drug_id, PREDICATE, doid)
            continue

        # XML
        if ext == ".xml":
            drug_id, doids = parse_xml_file(fp)
            if drug_id:
                entity_labels.setdefault(drug_id, drug_id)
            for doid in doids:
                if doid in BLACKLIST_DOIDS:
                    continue
                doid_label = doid_map.get(doid, doid)
                if str(doid_label).upper().startswith("OBSOLETE"):
                    continue
                entity_labels[doid] = doid_label
                # Do not create edges without a valid subject (drug_id)
                safe_add_triple(triples_set, drug_id, PREDICATE, doid)
            continue

    triples = sorted(triples_set)
    df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])
    df.to_csv(TRIPLES_OUT, index=False)

    with open(ENTITY_NAMES_OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for ent_id, label in entity_labels.items():
            w.writerow([ent_id, label])

    print("Extraction completed.")
    print(f"Triples written to: {TRIPLES_OUT} (n={len(df)})")
    print(f"Entity names written to: {ENTITY_NAMES_OUT} (n={len(entity_labels)})")


if __name__ == "__main__":
    run_extraction()
