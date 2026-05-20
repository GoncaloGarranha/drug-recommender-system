#!/usr/bin/env python3
"""Prepare processed KG artifacts from a raw triples CSV.

Creates a processed directory compatible with `models/rgcn` training code.

Output files (placed under `output_dir`):
- `entity2id.json` (entity -> id)
- `id2entity.json` (id -> entity)
- `rel2id.json` (relation -> id)
- `id2rel.json` (id -> relation)
- `triples_indexed.csv` (columns: src_id,rel_id,dst_id)
- `entity_types.json` (entity -> "UNKNOWN")
- `graph_metadata.json` (contains counts and distributions)

Usage:
    python kg_constructor/scripts/prepare_processed_from_csv.py \
        --input kg_constructor/data/output/triples/triples.csv \
        --out models/rgcn/data/processed/kg_constructor
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict, Counter
from pathlib import Path
from typing import Dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare processed KG artifacts from CSV")
    parser.add_argument("--input", required=True, help="Input triples CSV (src,rel,dst or src,rel,dst,split)")
    parser.add_argument("--out", required=True, help="Output processed directory")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    entities: "OrderedDict[str,int]" = OrderedDict()
    relations: "OrderedDict[str,int]" = OrderedDict()
    triples_indexed = []
    rel_counter: Counter = Counter()

    with inp.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        # Expect headers like src,rel,dst (maybe extra 'split')
        for row in reader:
            src = row.get("src") or row.get("head") or row.get("h")
            rel = row.get("rel") or row.get("relation")
            dst = row.get("dst") or row.get("tail") or row.get("t")
            if src is None or rel is None or dst is None:
                raise ValueError("Input CSV must have columns 'src','rel','dst' (or equivalent)")

            src = src.strip()
            rel = rel.strip()
            dst = dst.strip()

            if src not in entities:
                entities[src] = len(entities)
            if dst not in entities:
                entities[dst] = len(entities)
            if rel not in relations:
                relations[rel] = len(relations)

            h_id = entities[src]
            t_id = entities[dst]
            r_id = relations[rel]

            triples_indexed.append((h_id, r_id, t_id))
            rel_counter[rel] += 1

    # Write mappings
    entity2id: Dict[str, int] = dict(entities)
    id2entity: Dict[str, str] = {str(v): k for k, v in entity2id.items()}
    rel2id: Dict[str, int] = dict(relations)
    id2rel: Dict[str, str] = {str(v): k for k, v in rel2id.items()}

    (out / "entity2id.json").write_text(json.dumps(entity2id, ensure_ascii=False, indent=2))
    (out / "id2entity.json").write_text(json.dumps(id2entity, ensure_ascii=False, indent=2))
    (out / "rel2id.json").write_text(json.dumps(rel2id, ensure_ascii=False, indent=2))
    (out / "id2rel.json").write_text(json.dumps(id2rel, ensure_ascii=False, indent=2))

    # entity_types: mark all UNKNOWN
    entity_types = {k: "UNKNOWN" for k in entity2id.keys()}
    (out / "entity_types.json").write_text(json.dumps(entity_types, ensure_ascii=False, indent=2))

    # triples_indexed.csv
    triples_path = out / "triples_indexed.csv"
    with triples_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["src_id", "rel_id", "dst_id"])
        for h, r, t in triples_indexed:
            writer.writerow([h, r, t])

    # graph metadata
    metadata = {
        "num_entities": len(entity2id),
        "num_relations": len(rel2id),
        "num_triples": len(triples_indexed),
        "relation_distribution": {rel: int(cnt) for rel, cnt in rel_counter.items()},
        "entity_type_distribution": {"UNKNOWN": len(entity2id)},
    }
    (out / "graph_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    print(f"Wrote processed artifacts to: {out}")
    print(f"Entities: {len(entity2id)}, Relations: {len(rel2id)}, Triples: {len(triples_indexed)}")


if __name__ == "__main__":
    main()
