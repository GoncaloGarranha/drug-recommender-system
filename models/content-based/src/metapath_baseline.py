import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.shared.benchmark_tracker import log_experiment
from models.shared.config import load_config
from models.shared.experiment_integrity import validate_experiment_integrity
from models.shared.io import read_json
from models.shared.metrics import compute_ranking_metrics
from models.shared.runs import (
    capture_environment,
    capture_git_commit,
    resolve_run_dir,
    save_metrics,
    write_run_artifacts,
)
from models.shared.splits import (
    build_known_positives,
    get_or_create_splits,
    get_split_ratios,
)


def resolve_existing_path(*relative_candidates):
    for candidate in relative_candidates:
        candidate_path = (REPO_ROOT / candidate).resolve()
        if candidate_path.exists():
            return candidate_path

    searched = "\n".join(
        f"- {(REPO_ROOT / candidate).resolve()}" for candidate in relative_candidates
    )
    raise FileNotFoundError(f"Files not found:\n{searched}")


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class MetapathContentBaseline:
    """
    Structure-aware content-based baseline.

    For each disease, candidate drugs are scored by their similarity to the
    drugs known in the training split for that disease.

    This avoids aggregating all disease features into a large diluted profile.
    """

    def __init__(
        self,
        triples_full_path: Path,
        candidate_entities: List[str],
        train_triples: pd.DataFrame,
        target_relation: str = "INDICATION",
        ingredient_relation: str = "HAS_INGREDIENT",
        contraindication_relation: str = "CONTRAINDICATION",
        gene_relation: str = "ASSOCIATED_WITH_GENE",
        use_ingredients: bool = True,
        use_contraindications: bool = True,
        use_train_indication_genes: bool = True,
        w_ingredients: float = 1.0,
        w_contraindications: float = 0.5,
        w_genes: float = 1.0,
        aggregation: str = "max",
        top_k: int = 3,
    ):
        print("--- [1/3] Initializing metapath kNN content baseline ---")

        self.df_all = pd.read_csv(triples_full_path, dtype=str)

        self.candidate_entities = list(candidate_entities)
        self.entity_to_idx = {
            entity: idx for idx, entity in enumerate(self.candidate_entities)
        }

        self.target_relation = target_relation

        self.use_ingredients = use_ingredients
        self.use_contraindications = use_contraindications
        self.use_train_indication_genes = use_train_indication_genes

        self.w_ingredients = float(w_ingredients)
        self.w_contraindications = float(w_contraindications)
        self.w_genes = float(w_genes)

        self.aggregation = aggregation
        self.top_k = int(top_k)

        if self.aggregation not in {"max", "mean", "topk_mean"}:
            raise ValueError(
                "Invalid aggregation. Use one of: max, mean, topk_mean."
            )

        candidate_set = set(self.candidate_entities)

        self.drug_to_ingredients = self._build_drug_feature_map(
            relation=ingredient_relation,
            candidate_set=candidate_set,
        )

        self.drug_to_contraindications = self._build_drug_feature_map(
            relation=contraindication_relation,
            candidate_set=candidate_set,
        )

        self.disease_to_genes = (
            self.df_all[self.df_all["rel"] == gene_relation]
            .groupby("src")["dst"]
            .apply(lambda values: set(map(str, values)))
            .to_dict()
        )

        train_indications = train_triples[
            (train_triples["rel"] == target_relation)
            & (train_triples["src"].isin(candidate_set))
        ]

        self.disease_to_train_drugs = (
            train_indications.groupby("dst")["src"]
            .apply(lambda values: list(map(str, values)))
            .to_dict()
        )

        self.drug_to_train_genes = self._build_train_drug_gene_profiles(
            train_indications=train_indications,
        )

        print(
            "-> Profile stats:",
            {
                "candidate_drugs": len(self.candidate_entities),
                "drugs_with_ingredients": len(self.drug_to_ingredients),
                "drugs_with_contraindications": len(self.drug_to_contraindications),
                "diseases_with_genes": len(self.disease_to_genes),
                "diseases_with_train_drugs": len(self.disease_to_train_drugs),
                "drugs_with_train_genes": len(self.drug_to_train_genes),
                "use_ingredients": self.use_ingredients,
                "use_contraindications": self.use_contraindications,
                "use_train_indication_genes": self.use_train_indication_genes,
                "aggregation": self.aggregation,
                "top_k": self.top_k,
                "weights": {
                    "ingredients": self.w_ingredients,
                    "contraindications": self.w_contraindications,
                    "genes": self.w_genes,
                },
            },
        )

    def _build_drug_feature_map(
        self,
        relation: str,
        candidate_set: Set[str],
    ) -> Dict[str, Set[str]]:
        rows = self.df_all[
            (self.df_all["src"].isin(candidate_set))
            & (self.df_all["rel"] == relation)
        ]

        if rows.empty:
            return {}

        return (
            rows.groupby("src")["dst"]
            .apply(lambda values: set(map(str, values)))
            .to_dict()
        )

    def _build_train_drug_gene_profiles(
        self,
        train_indications: pd.DataFrame,
    ) -> Dict[str, Set[str]]:
        """
        Build train-only drug gene profiles.

        In disease_to_drug, train INDICATION triples are:
        Drug --INDICATION--> Disease

        For each train drug, we collect genes from diseases it treats in train.
        """
        drug_to_genes: Dict[str, Set[str]] = {}

        if not self.use_train_indication_genes:
            return drug_to_genes

        for _, row in train_indications.iterrows():
            drug = str(row["src"])
            disease = str(row["dst"])
            genes = self.disease_to_genes.get(disease, set())
            if genes:
                drug_to_genes.setdefault(drug, set()).update(genes)

        return drug_to_genes

    def _drug_similarity(self, candidate: str, reference_drug: str) -> float:
        score = 0.0

        if self.use_ingredients:
            score += self.w_ingredients * jaccard(
                self.drug_to_ingredients.get(candidate, set()),
                self.drug_to_ingredients.get(reference_drug, set()),
            )

        if self.use_contraindications:
            score += self.w_contraindications * jaccard(
                self.drug_to_contraindications.get(candidate, set()),
                self.drug_to_contraindications.get(reference_drug, set()),
            )

        if self.use_train_indication_genes:
            score += self.w_genes * jaccard(
                self.drug_to_train_genes.get(candidate, set()),
                self.drug_to_train_genes.get(reference_drug, set()),
            )

        return score

    def _aggregate_similarities(self, similarities: List[float]) -> float:
        if not similarities:
            return 0.0

        values = np.asarray(similarities, dtype=np.float64)

        if self.aggregation == "max":
            return float(np.max(values))

        if self.aggregation == "mean":
            return float(np.mean(values))

        if self.aggregation == "topk_mean":
            k = min(self.top_k, len(values))
            top_values = np.sort(values)[-k:]
            return float(np.mean(top_values))

        raise ValueError(f"Unsupported aggregation: {self.aggregation}")

    def _score_candidate(self, disease: str, candidate: str) -> float:
        train_drugs = self.disease_to_train_drugs.get(disease, [])

        if not train_drugs:
            return 0.0

        similarities = [
            self._drug_similarity(candidate, reference_drug)
            for reference_drug in train_drugs
            if reference_drug != candidate
        ]

        return self._aggregate_similarities(similarities)

    def predict_ranks(
        self,
        eval_triples: pd.DataFrame,
        known_positives: Dict[str, set],
    ) -> List[float]:
        print("--- [2/3] Generating filtered rankings with metapath kNN scores ---")

        query_col = "dst"
        candidate_col = "src"

        ranks = []

        for _, row in tqdm(eval_triples.iterrows(), total=len(eval_triples)):
            disease = str(row[query_col])
            true_candidate = str(row[candidate_col])

            if true_candidate not in self.entity_to_idx:
                continue

            scores = np.zeros(len(self.candidate_entities), dtype=np.float64)

            for idx, candidate in enumerate(self.candidate_entities):
                scores[idx] = self._score_candidate(disease, candidate)

            for known_candidate in known_positives.get(disease, set()):
                if known_candidate == true_candidate:
                    continue

                known_idx = self.entity_to_idx.get(known_candidate)
                if known_idx is not None:
                    scores[known_idx] = -1e9

            true_idx = self.entity_to_idx[true_candidate]
            true_score = scores[true_idx]

            num_better = int(np.sum(scores > true_score))
            num_equal = int(np.sum(scores == true_score))

            rank = 1 + num_better + (num_equal - 1) / 2.0
            ranks.append(rank)

        return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        config_path = resolve_existing_path(args.config)
        cfg = load_config(config_path)

        triples_path = resolve_existing_path(cfg["data"]["triples_path"])
        processed_dir = resolve_existing_path(cfg["data"]["processed_dir"])

        triples_indexed_path = processed_dir / "triples_indexed.csv"
        entity2id_path = processed_dir / "entity2id.json"
        entity_types_path = processed_dir / "entity_types.json"
    except FileNotFoundError as err:
        print(err)
        sys.exit(1)

    if not triples_indexed_path.exists():
        print(f"Missing required file: {triples_indexed_path}")
        sys.exit(1)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    target_relation = cfg["task"]["target_relation"]
    task_name = cfg["task"].get("name", "disease_to_drug")

    train_ratio, valid_ratio, test_ratio = get_split_ratios(cfg["split"])

    triples_df = pd.read_csv(triples_indexed_path, dtype=str)
    splits_dir = processed_dir / "splits"

    train_rows, valid_rows, test_rows, _ = get_or_create_splits(
        output_dir=splits_dir,
        triples=triples_df.to_dict(orient="records"),
        target_relation=target_relation,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=valid_ratio,
        test_ratio=test_ratio,
        split_mode=cfg["split"].get("mode") if isinstance(cfg.get("split"), dict) else None,
        force=args.force,
    )

    split_integrity_report = validate_experiment_integrity(
        cfg,
        config_path=config_path,
        output_path=Path(splits_dir) / "split_integrity_report_metapath.json",
        force=args.force,
    )

    train_df = pd.DataFrame(train_rows)
    valid_df = pd.DataFrame(valid_rows)
    test_df = pd.DataFrame(test_rows)

    eval_df = valid_df if args.split == "valid" else test_df

    all_target_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    known_positives = build_known_positives(
        all_target_df.to_dict(orient="records"),
        query_key="dst",
        candidate_key="src",
    )

    entity2id = read_json(entity2id_path)
    entity_types = read_json(entity_types_path)

    candidate_type = cfg["task"].get("candidate_entity_type", "DRUG")

    candidate_entities = sorted(
        [
            entity
            for entity, entity_type in entity_types.items()
            if candidate_type == "ALL" or entity_type == candidate_type
        ],
        key=lambda entity: int(entity2id[entity]),
    )

    baseline_cfg = cfg.get("metapath_baseline", {})

    model = MetapathContentBaseline(
        triples_full_path=triples_path,
        candidate_entities=candidate_entities,
        train_triples=train_df,
        target_relation=target_relation,
        use_ingredients=bool(baseline_cfg.get("use_ingredients", True)),
        use_contraindications=bool(
            baseline_cfg.get("use_contraindications", True)
        ),
        use_train_indication_genes=bool(
            baseline_cfg.get("use_train_indication_genes", True)
        ),
        w_ingredients=float(baseline_cfg.get("w_ingredients", 1.0)),
        w_contraindications=float(baseline_cfg.get("w_contraindications", 0.5)),
        w_genes=float(baseline_cfg.get("w_genes", 1.0)),
        aggregation=baseline_cfg.get("aggregation", "max"),
        top_k=int(baseline_cfg.get("top_k", 3)),
    )

    run_dir = resolve_run_dir(
        cfg["runs"]["out_dir"],
        cfg,
        model_name="metapath_content_based",
        seed=seed,
    )

    split_summary = {
        "task_name": task_name,
        "target_relation": target_relation,
        "train_size": len(train_df),
        "valid_size": len(valid_df),
        "test_size": len(test_df),
        "candidate_pool_size": len(candidate_entities),
        "candidate_entity_type": candidate_type,
        "split_integrity_report": split_integrity_report["report_path"],
    }

    model_summary = {
        "model_name": "metapath_content_based",
        "seed": seed,
        "baseline": baseline_cfg,
        "candidate_count": len(candidate_entities),
    }

    write_run_artifacts(
        run_dir=run_dir,
        cfg=cfg,
        split_summary=split_summary,
        model_summary=model_summary,
        train_log=[],
        environment=capture_environment(),
        git_commit=capture_git_commit(),
    )

    ranks = model.predict_ranks(eval_df, known_positives)

    if not ranks:
        print("No evaluation ranks computed.")
        sys.exit(1)

    metrics = compute_ranking_metrics(ranks)

    print("\n--- [3/3] Final metrics (metapath kNN content baseline) ---")
    print(f"Hits@1:  {metrics['hits@1']:.4f}")
    print(f"Hits@3:  {metrics['hits@3']:.4f}")
    print(f"Hits@10: {metrics['hits@10']:.4f}")
    print(f"MRR:     {metrics['mrr']:.4f}")

    log_experiment(
        dataset=Path(config_path).stem,
        model_name="Metapath kNN Content-Based",
        hits1=metrics["hits@1"],
        hits3=metrics["hits@3"],
        hits10=metrics["hits@10"],
        mrr=metrics["mrr"],
        notes=(
            f"Structure-aware kNN content baseline | Config: {Path(config_path).name} "
            f"| Split: {args.split} | aggregation={model.aggregation}"
        ),
    )

    save_metrics(run_dir, args.split, metrics)


if __name__ == "__main__":
    main()