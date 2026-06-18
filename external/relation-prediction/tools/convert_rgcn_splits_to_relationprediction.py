from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


Triple = Tuple[str, str, str]


def load_split(path: Path) -> List[Triple]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    triples: List[Triple] = []
    for row in rows:
        src = str(row["src"]).strip()
        rel = str(row["rel"]).strip()
        dst = str(row["dst"]).strip()
        if src and rel and dst:
            triples.append((src, rel, dst))

    return triples


def write_dict(path: Path, values: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for idx, value in enumerate(values):
            f.write(f"{idx}\t{value}\n")


def write_triples(path: Path, triples: List[Triple]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for src, rel, dst in triples:
            f.write(f"{src}\t{rel}\t{dst}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits_dir",
        required=True,
        help="Directory containing train_triples.json, valid_triples.json and test_triples.json",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory in RelationPrediction-master/data format",
    )
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(splits_dir / "train_triples.json")
    valid = load_split(splits_dir / "valid_triples.json")
    test = load_split(splits_dir / "test_triples.json")

    all_triples = train + valid + test

    entities = sorted({src for src, _, _ in all_triples} | {dst for _, _, dst in all_triples})
    relations = sorted({rel for _, rel, _ in all_triples})

    write_dict(out_dir / "entities.dict", entities)
    write_dict(out_dir / "relations.dict", relations)

    write_triples(out_dir / "train.txt", train)
    write_triples(out_dir / "valid.txt", valid)
    write_triples(out_dir / "test.txt", test)

    metadata: Dict[str, object] = {
        "num_entities": len(entities),
        "num_relations": len(relations),
        "num_train": len(train),
        "num_valid": len(valid),
        "num_test": len(test),
        "relations": relations,
    }

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Dataset convertido para RelationPrediction-master")
    print(f"Output: {out_dir}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()