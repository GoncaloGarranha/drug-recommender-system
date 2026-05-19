import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Repository-relative path configuration.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.shared.benchmark_tracker import log_experiment
from models.shared.config import load_config
from models.shared.experiment_integrity import validate_experiment_integrity
from models.shared.io import read_json
from models.shared.metrics import compute_ranking_metrics
from models.shared.runs import capture_environment, capture_git_commit, resolve_run_dir, save_metrics, write_run_artifacts
from models.shared.splits import build_known_positives, get_split_ratios, get_or_create_splits


def resolve_existing_path(*relative_candidates):
    """Return the first existing repository-relative path from the candidates."""
    for candidate in relative_candidates:
        candidate_path = (REPO_ROOT / candidate).resolve()
        if candidate_path.exists():
            return candidate_path
    searched = "\n".join(f"- {(REPO_ROOT / candidate).resolve()}" for candidate in relative_candidates)
    raise FileNotFoundError(f"Files not found:\n{searched}")


class ContentBasedBaseline:
    def __init__(self, triples_full_path: Path, candidate_entities: List[str], use_ingredients: bool = True, feature_relations: list | None = None, target_relation: str | None = None, task_name: str | None = None):
        print("--- [1/3] Initializing aligned content-based baseline (TF-IDF + cosine) ---")
        self.task_name = task_name or "disease_to_drug"
        
        # Validate that the task is supported
        if self.task_name != "disease_to_drug":
            raise ValueError(
                f"Content-Based baseline is currently supported only for disease_to_drug. "
                f"Received: {self.task_name}. The ingredient-based representation is defined over drug candidates."
            )
        
        df_all = pd.read_csv(triples_full_path, dtype=str)

        if not candidate_entities:
            raise ValueError("No candidate entities found in processed graph artifacts.")

        self.entity_list = list(candidate_entities)
        self.entity_to_idx = {entity: index for index, entity in enumerate(self.entity_list)}
        self.use_ingredients = use_ingredients
        self.feature_relations = feature_relations or []
        self.target_relation = target_relation

        # Build document representation from configured feature relations
        allowed_rels = set(self.feature_relations)
        
        if self.use_ingredients:
            allowed_rels.add("HAS_INGREDIENT")
        
        # Remove target relation to avoid data leakage
        if self.target_relation is not None and self.target_relation in allowed_rels:
            allowed_rels.discard(self.target_relation)
        
        if not allowed_rels:
            raise ValueError(
                "No feature relations available for Content-Based baseline. "
                "Set config.baseline.feature_relations and/or config.baseline.use_ingredients=true."
            )
        
        docs_df = df_all[
            (df_all["src"].isin(self.entity_list)) &
            (df_all["rel"].isin(allowed_rels))
        ].copy()
        
        if docs_df.empty:
            raise ValueError(
                f"No feature triples found for candidate entities using relations: {sorted(allowed_rels)}. "
                f"Available entities: {self.entity_list[:5]}..."
            )
        
        # Create token map: src → [rel:dst, rel:dst, ...]
        doc_map = (
            docs_df.assign(token=docs_df["rel"].astype(str) + ":" + docs_df["dst"].astype(str))
            .groupby("src")["token"]
            .apply(list)
            .to_dict()
        )
        
        self.documents = [" ".join(doc_map.get(entity, [])) for entity in self.entity_list]

        if not any(doc.strip() for doc in self.documents):
            # Fallback: if no textual features are available, use the entity id as a token
            self.documents = [str(e) for e in self.entity_list]

        self.vectorizer = TfidfVectorizer(
            max_df=0.9,
            min_df=1,
            token_pattern=r"[a-zA-Z0-9:]+"
        )

        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        except ValueError as e:
            print(f"Vectorization error: {e}")
            print("Sample text passed to the vectorizer:", self.documents[0])
            sys.exit(1)

        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        print(f"-> TF-IDF matrix: {self.tfidf_matrix.shape} (entities x features)")

    def predict_ranks(
        self,
        eval_triples: pd.DataFrame,
        train_triples: pd.DataFrame,
        known_positives: Dict[str, set],
    ) -> List[float]:
        print("--- [2/3] Generating filtered rankings with cosine similarity ---")

        # For disease_to_drug: query=disease(dst), candidate=drug(src)
        query_col, candidate_col = "dst", "src"

        query_to_train_indices = (
            train_triples.groupby(query_col)[candidate_col]
            .apply(lambda x: [self.entity_to_idx[e] for e in x if e in self.entity_to_idx])
            .to_dict()
        )

        ranks = []

        for _, row in tqdm(eval_triples.iterrows(), total=len(eval_triples)):
            true_candidate = row[candidate_col]
            query = row[query_col]

            if true_candidate not in self.entity_to_idx:
                continue

            true_idx = self.entity_to_idx[true_candidate]
            known_train_indices = query_to_train_indices.get(query, [])

            if known_train_indices:
                scores = np.max(self.similarity_matrix[:, known_train_indices], axis=1).astype(np.float64)
            else:
                scores = np.zeros(len(self.entity_list), dtype=np.float64)

            for known_candidate in known_positives.get(query, set()):
                if known_candidate == true_candidate:
                    continue
                known_idx = self.entity_to_idx.get(known_candidate)
                if known_idx is not None:
                    scores[known_idx] = -1e9

            true_score = scores[true_idx]
            num_better = int(np.sum(scores > true_score))
            num_equal = int(np.sum(scores == true_score))
            rank = 1 + num_better + (num_equal - 1) / 2.0
            ranks.append(rank)

        return ranks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="models/rgcn/configs/disease_to_drug.yaml")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--force", action="store_true", help="Recreate splits even if existing metadata mismatches.")
    args = parser.parse_args()

    try:
        config_path = resolve_existing_path(args.config)
        cfg = load_config(config_path)

        # Use triples path from config to allow filtered / alternate triple files
        triples_cfg = cfg["data"].get("triples_path", "kg_constructor/data/output/triples/triples.csv")
        triples_full_path = resolve_existing_path(triples_cfg)

        processed_dir_cfg = cfg["data"]["processed_dir"]
        processed_dir = resolve_existing_path(
            processed_dir_cfg,
            f"models/{processed_dir_cfg}",
        )
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
    target_relation = cfg["task"]["target_relation"]
    task_name = cfg["task"].get("name", "disease_to_drug")
    run_dir = resolve_run_dir(cfg["runs"]["out_dir"], cfg, model_name="content_based", seed=seed)

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
        output_path=Path(splits_dir) / "split_integrity_report.json",
        force=args.force,
    )

    train_df = pd.DataFrame(train_rows)
    valid_df = pd.DataFrame(valid_rows)
    test_df = pd.DataFrame(test_rows)

    eval_df = valid_df if args.split == "valid" else test_df
    all_target_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    # Build known_positives for disease_to_drug
    known_positives = build_known_positives(
        all_target_df.to_dict(orient="records"),
        query_key="dst",
        candidate_key="src",
    )

    entity2id = read_json(entity2id_path)
    try:
        entity_types = read_json(entity_types_path)
    except Exception:
        entity_types = {}

    candidate_type_cfg = cfg["task"].get("candidate_entity_type", None)
    if candidate_type_cfg is None or candidate_type_cfg == "ALL":
        candidate_entities = sorted(list(entity2id.keys()), key=lambda entity: int(entity2id[entity]))
    else:
        candidate_entities = sorted(
            [entity for entity, entity_type in entity_types.items() if entity_type == candidate_type_cfg],
            key=lambda entity: int(entity2id[entity]),
        )

    baseline_cfg = cfg.get("baseline", {})
    use_ingredients = bool(baseline_cfg.get("use_ingredients", True))
    feature_relations = baseline_cfg.get("feature_relations", None)

    if (not use_ingredients) and (not feature_relations):
        print("Error: baseline.use_ingredients is false but baseline.feature_relations is not set in the config.")
        sys.exit(1)

    split_summary = {
        "task_name": task_name,
        "target_relation": target_relation,
        "train_size": len(train_df),
        "valid_size": len(valid_df),
        "test_size": len(test_df),
        "candidate_pool_size": len(candidate_entities),
        "candidate_entity_type": candidate_type_cfg or "ALL",
        "split_mode": cfg["split"].get("mode") if isinstance(cfg.get("split"), dict) else None,
        "split_integrity_report": split_integrity_report["report_path"],
    }

    model_summary = {
        "model_name": "content_based",
        "task_name": task_name,
        "seed": seed,
        "baseline": {
            "use_ingredients": use_ingredients,
            "feature_relations": feature_relations,
            "target_relation": target_relation,
        },
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

    model = ContentBasedBaseline(
        triples_full_path=triples_full_path,
        candidate_entities=candidate_entities,
        use_ingredients=use_ingredients,
        feature_relations=feature_relations,
        target_relation=target_relation,
        task_name=task_name,
    )
    ranks = model.predict_ranks(eval_df, train_df, known_positives)

    if not ranks:
        print("No evaluation ranks computed.")
        sys.exit(1)

    metrics = compute_ranking_metrics(ranks)

    print("\n--- [3/3] Final metrics (aligned with R-GCN protocol) ---")
    hits_at_1 = metrics["hits@1"]
    hits_at_3 = metrics["hits@3"]
    hits_at_10 = metrics["hits@10"]
    mrr = metrics["mrr"]

    print(f"Hits@1:  {hits_at_1:.4f}")
    print(f"Hits@3:  {hits_at_3:.4f}")
    print(f"Hits@10: {hits_at_10:.4f}")
    print(f"MRR:     {mrr:.4f}")

    log_experiment(
        dataset=Path(config_path).stem,
        model_name="Content-Based (TF-IDF baseline)",
        hits1=hits_at_1,
        hits3=hits_at_3,
        hits10=hits_at_10,
        mrr=mrr,
        notes=(
            f"Aligned with R-GCN protocol | Config: {Path(config_path).name} "
            f"| Split: {args.split}"
        ),
    )

    save_metrics(run_dir, args.split, metrics)