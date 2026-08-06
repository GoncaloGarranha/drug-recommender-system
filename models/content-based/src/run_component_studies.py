from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(
    "models/content-based/config/"
    "disease_to_drug_with_dgidb.yaml"
)

BASE_RUN_DIR = Path(
    "results/runs/content-based/biomedical/disease_to_drug/"
    "six_months_ingredient_gene_content_based/seed_42"
)

STUDIES_ROOT = Path(
    "results/runs/content-based/biomedical/disease_to_drug/"
    "six_months_component_studies"
)

STUDIES: dict[str, dict[str, float]] = {
    "has_ingredient": {
        "ingredient_knn": 1.0,
        "direct_gene_knn": 0.0,
        "train_indication_gene_knn": 0.0,
        "disease_gene_overlap": 0.0,
        "contraindication_knn": 0.0,
        "popularity": 0.0,
        "direct_contraindication": 0.0,
    },
    "indication_gene_profile": {
        "ingredient_knn": 0.0,
        "direct_gene_knn": 0.0,
        "train_indication_gene_knn": 1.0,
        "disease_gene_overlap": 0.0,
        "contraindication_knn": 0.0,
        "popularity": 0.0,
        "direct_contraindication": 0.0,
    },
    "combined": {
        "ingredient_knn": 0.5,
        "direct_gene_knn": 0.0,
        "train_indication_gene_knn": 0.5,
        "disease_gene_overlap": 0.0,
        "contraindication_knn": 0.0,
        "popularity": 0.0,
        "direct_contraindication": 0.0,
    },
}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_command(command: list[str]) -> None:
    print()
    print("$ " + " ".join(command))

    completed = subprocess.run(
        command,
        check=False,
    )

    if completed.returncode != 0:
        raise SystemExit(
            f"Command failed with exit code {completed.returncode}"
        )


def copy_result(source_name: str, target: Path) -> None:
    source = BASE_RUN_DIR / source_name

    if not source.is_file():
        raise SystemExit(f"Expected result not found: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    if not CONFIG_PATH.is_file():
        raise SystemExit(f"Config not found: {CONFIG_PATH}")

    STUDIES_ROOT.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, Any]] = {}

    for study_name, weights in STUDIES.items():
        print()
        print("=" * 80)
        print(f"STUDY: {study_name}")
        print("=" * 80)

        study_dir = STUDIES_ROOT / study_name
        study_dir.mkdir(parents=True, exist_ok=True)

        weights_path = study_dir / "weights.json"

        write_json(
            weights_path,
            {
                "selection_split": "valid",
                "study_name": study_name,
                "selected_weights": weights,
            },
        )

        for split in ("valid", "test"):
            run_command(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "models.content-based.src.baseline",
                    "--config",
                    str(CONFIG_PATH),
                    "--split",
                    split,
                    "--weights-file",
                    str(weights_path),
                ]
            )

            copy_result(
                f"metrics_{split}.json",
                study_dir / f"metrics_{split}.json",
            )

            diagnostics_source = (
                BASE_RUN_DIR / f"diagnostics_{split}.json"
            )

            if diagnostics_source.is_file():
                shutil.copy2(
                    diagnostics_source,
                    study_dir / f"diagnostics_{split}.json",
                )

        valid_metrics = json.loads(
            (study_dir / "metrics_valid.json").read_text(
                encoding="utf-8"
            )
        )

        test_metrics = json.loads(
            (study_dir / "metrics_test.json").read_text(
                encoding="utf-8"
            )
        )

        summary[study_name] = {
            "weights": weights,
            "valid": valid_metrics,
            "test": test_metrics,
        }

    write_json(
        STUDIES_ROOT / "component_studies_summary.json",
        summary,
    )

    print()
    print("=" * 100)
    print("COMPONENT STUDIES SUMMARY")
    print("=" * 100)

    print(
        f"{'Study':30}"
        f"{'Valid MRR':>12}"
        f"{'Test MRR':>12}"
        f"{'Test H@1':>12}"
        f"{'Test H@3':>12}"
        f"{'Test H@10':>12}"
    )

    print("-" * 90)

    for study_name, result in summary.items():
        valid = result["valid"]
        test = result["test"]

        print(
            f"{study_name:30}"
            f"{valid['mrr']:12.4f}"
            f"{test['mrr']:12.4f}"
            f"{test['hits@1']:12.4f}"
            f"{test['hits@3']:12.4f}"
            f"{test['hits@10']:12.4f}"
        )

    print()
    print(f"Results saved to: {STUDIES_ROOT}")


if __name__ == "__main__":
    main()
