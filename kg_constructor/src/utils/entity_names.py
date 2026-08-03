from __future__ import annotations

import argparse
import csv
import os
import urllib.request
from pathlib import Path


DOID_OBO_URL = "http://purl.obolibrary.org/obo/doid.obo"


def download_if_missing(url: str, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        return

    print(f"[INFO] A descarregar DOID OBO: {url}")
    urllib.request.urlretrieve(url, out_path)
    print(f"[OK] Guardado em: {out_path}")


def parse_doid_obo(obo_path: str | Path) -> dict[str, str]:
    """Parse a DOID OBO file and return a mapping of DOID IDs to labels."""
    mapping: dict[str, str] = {}
    current_id = None
    current_name = None
    in_term = False

    with Path(obo_path).open("r", encoding="utf-8", errors="replace") as file_obj:
        for line in file_obj:
            line = line.strip()

            if line == "[Term]":
                in_term = True
                current_id = None
                current_name = None
                continue

            if line == "" and in_term:
                if current_id and current_name:
                    mapping[current_id] = current_name
                in_term = False
                current_id = None
                current_name = None
                continue

            if not in_term:
                continue

            if line.startswith("id: DOID:"):
                current_id = line.split("id:", 1)[1].strip()
            elif line.startswith("name:"):
                current_name = line.split("name:", 1)[1].strip()

    if in_term and current_id and current_name:
        mapping[current_id] = current_name

    print(f"[OK] DOID termos mapeados: {len(mapping)}")
    return mapping


def load_entity_table(entity_table_csv: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    with Path(entity_table_csv).open("r", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        rows.extend(reader)

    return rows


def build_entity_names(
    entity_table_csv: str | Path,
    out_csv: str | Path,
    obo_path: str | Path,
) -> None:
    entity_table_csv = Path(entity_table_csv)
    out_csv = Path(out_csv)
    obo_path = Path(obo_path)

    if not entity_table_csv.exists():
        raise FileNotFoundError(f"Não encontrei: {entity_table_csv}")

    download_if_missing(DOID_OBO_URL, obo_path)

    doid_map = parse_doid_obo(obo_path)
    entities = load_entity_table(entity_table_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing_doid = 0

    with out_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["id", "name"])

        for entity in entities:
            entity_id = entity["entity_id"]
            entity_type = entity["entity_type"]
            label = entity["label"]

            if entity_type == "DISEASE" and entity_id.startswith("DOID:"):
                resolved = doid_map.get(entity_id)
                if resolved:
                    writer.writerow([entity_id, resolved])
                else:
                    missing_doid += 1
                    writer.writerow([entity_id, label])
            else:
                writer.writerow([entity_id, label])

            written += 1

    print(f"[OK] Escrevi {written} labels em: {out_csv}")
    if missing_doid:
        print(f"[WARN] DOIDs sem nome na ontologia: {missing_doid} (ficaram com fallback)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build human-readable labels for KG entities."
    )
    parser.add_argument("--entity_table_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--obo_path", required=True)
    args = parser.parse_args()

    build_entity_names(
        entity_table_csv=args.entity_table_csv,
        out_csv=args.out_csv,
        obo_path=args.obo_path,
    )

if __name__ == "__main__":
    main()