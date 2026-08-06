"""Leakage-safe hybrid content-based baseline for disease-to-drug ranking.

The model deliberately remains non-neural and interpretable. It reuses the
same graph, target splits, candidate pool, filtered-ranking protocol and
metrics as the R-GCN experiment, while keeping every score component explicit.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
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


SIMILARITY_COMPONENTS = (
    "ingredient_knn",
    "direct_gene_knn",
    "train_indication_gene_knn",
    "disease_gene_overlap",
    "contraindication_knn",
    "popularity",
)
PENALTY_COMPONENT = "direct_contraindication"


def resolve_existing_path(*relative_candidates: str | Path) -> Path:
    for candidate in relative_candidates:
        candidate_path = (REPO_ROOT / candidate).resolve()
        if candidate_path.exists():
            return candidate_path

    searched = "\n".join(
        f"- {(REPO_ROOT / candidate).resolve()}" for candidate in relative_candidates
    )
    raise FileNotFoundError(f"Files not found:\n{searched}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def relation_map(
    triples: pd.DataFrame,
    relation: str,
    allowed_sources: set[str] | None = None,
) -> Dict[str, set[str]]:
    rows = triples[triples["rel"] == relation]
    if allowed_sources is not None:
        rows = rows[rows["src"].isin(allowed_sources)]
    if rows.empty:
        return {}
    return (
        rows.groupby("src")["dst"]
        .apply(lambda values: set(map(str, values)))
        .to_dict()
    )


def build_documents(
    entities: Sequence[str],
    feature_map: Mapping[str, set[str]],
) -> List[str]:
    return [" ".join(sorted(feature_map.get(entity, set()))) for entity in entities]


def fit_tfidf(documents: Sequence[str]) -> tuple[TfidfVectorizer | None, sparse.csr_matrix]:
    if not any(document.strip() for document in documents):
        return None, sparse.csr_matrix((len(documents), 0), dtype=np.float64)

    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"[^\s]+",
        min_df=1,
        norm="l2",
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(documents).tocsr()
    return vectorizer, matrix


def normalized_centroid(
    matrix: sparse.csr_matrix,
    row_indices: Sequence[int],
) -> sparse.csr_matrix | None:
    if matrix.shape[1] == 0 or not row_indices:
        return None

    centroid = sparse.csr_matrix(matrix[row_indices].mean(axis=0))
    norm = float(sparse.linalg.norm(centroid))
    if norm == 0.0:
        return None
    return centroid / norm


def score_against_centroid(
    matrix: sparse.csr_matrix,
    row_indices: Sequence[int],
) -> np.ndarray:
    centroid = normalized_centroid(matrix, row_indices)
    if centroid is None:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    return (matrix @ centroid.T).toarray().reshape(-1).astype(
        np.float64,
        copy=False,
    )


def tie_aware_rank(scores: np.ndarray, true_index: int) -> float:
    true_score = scores[true_index]
    num_better = int(np.sum(scores > true_score))
    num_equal = int(np.sum(scores == true_score))
    return 1.0 + num_better + (num_equal - 1) / 2.0


def simplex_weights(names: Sequence[str], step: float) -> Iterable[Dict[str, float]]:
    if step <= 0.0 or step > 1.0:
        raise ValueError("weight_search.step must be in (0, 1].")

    units_float = 1.0 / step
    units = int(round(units_float))
    if not math.isclose(units * step, 1.0, abs_tol=1e-9):
        raise ValueError("weight_search.step must divide 1.0 exactly, e.g. 0.5 or 0.25.")

    def compositions(total: int, parts: int, prefix: tuple[int, ...] = ()):
        if parts == 1:
            yield prefix + (total,)
            return
        for value in range(total + 1):
            yield from compositions(total - value, parts - 1, prefix + (value,))

    for composition in compositions(units, len(names)):
        yield {
            name: round(value * step, 10)
            for name, value in zip(names, composition)
        }


class HybridContentBasedBaseline:
    """Interpretable hybrid content-based scorer with separate components."""

    def __init__(
        self,
        triples_path: Path,
        candidate_entities: Sequence[str],
        train_triples: pd.DataFrame,
        config: Mapping[str, object],
        target_relation: str = "INDICATION",
    ) -> None:
        print("--- [1/4] Building leakage-safe content profiles ---")

        self.triples = pd.read_csv(triples_path, dtype=str)
        required_columns = {"src", "rel", "dst"}
        if not required_columns.issubset(self.triples.columns):
            raise ValueError(
                f"Triples file must contain {sorted(required_columns)}; "
                f"found {list(self.triples.columns)}."
            )

        self.candidates = list(map(str, candidate_entities))
        if not self.candidates:
            raise ValueError("Candidate pool is empty.")
        self.candidate_to_index = {
            candidate: index for index, candidate in enumerate(self.candidates)
        }
        self.candidate_set = set(self.candidates)
        self.target_relation = target_relation
        self.cfg = dict(config)

        relations_cfg = dict(self.cfg.get("relations", {}))
        self.ingredient_relation = str(
            relations_cfg.get("ingredient", "HAS_INGREDIENT")
        )
        self.direct_gene_relation = str(
            relations_cfg.get("drug_gene", "INTERACTS_WITH_GENE")
        )
        self.disease_gene_relation = str(
            relations_cfg.get("disease_gene", "ASSOCIATED_WITH_GENE")
        )
        self.contraindication_relation = str(
            relations_cfg.get("contraindication", "CONTRAINDICATION")
        )

        enabled_cfg = dict(self.cfg.get("enabled_components", {}))
        self.enabled = {
            name: bool(enabled_cfg.get(name, True))
            for name in (*SIMILARITY_COMPONENTS, PENALTY_COMPONENT)
        }

        aggregation_cfg = dict(self.cfg.get("aggregation", {}))
        self.aggregation = str(aggregation_cfg.get("method", "centroid"))
        if self.aggregation != "centroid":
            raise ValueError(
                "Only aggregation.method=centroid is supported in the final baseline."
            )

        train_indications = train_triples[
            (train_triples["rel"] == target_relation)
            & (train_triples["src"].isin(self.candidate_set))
        ].copy()
        self.disease_to_train_drugs = (
            train_indications.groupby("dst")["src"]
            .apply(lambda values: sorted(set(map(str, values))))
            .to_dict()
        )
        self.train_popularity_counts = Counter(map(str, train_indications["src"]))

        self.drug_to_ingredients = relation_map(
            self.triples, self.ingredient_relation, self.candidate_set
        )
        self.drug_to_direct_genes = relation_map(
            self.triples, self.direct_gene_relation, self.candidate_set
        )
        self.drug_to_contraindications = relation_map(
            self.triples, self.contraindication_relation, self.candidate_set
        )
        self.disease_to_genes = relation_map(
            self.triples, self.disease_gene_relation
        )

        # This is the only component derived through INDICATION. The bridge uses
        # train triples exclusively, so validation and test labels never enter a profile.
        self.drug_to_train_indication_genes: Dict[str, set[str]] = defaultdict(set)
        for row in train_indications.itertuples(index=False):
            drug = str(row.src)
            disease = str(row.dst)
            self.drug_to_train_indication_genes[drug].update(
                self.disease_to_genes.get(disease, set())
            )
        self.drug_to_train_indication_genes = dict(
            self.drug_to_train_indication_genes
        )

        _, self.ingredient_matrix = fit_tfidf(
            build_documents(self.candidates, self.drug_to_ingredients)
        )
        _, self.direct_gene_matrix = fit_tfidf(
            build_documents(self.candidates, self.drug_to_direct_genes)
        )
        _, self.train_indication_gene_matrix = fit_tfidf(
            build_documents(self.candidates, self.drug_to_train_indication_genes)
        )
        _, self.contraindication_matrix = fit_tfidf(
            build_documents(self.candidates, self.drug_to_contraindications)
        )

        # Fit one shared vocabulary for direct Disease--Gene / Drug--Gene comparison.
        disease_entities = sorted(self.disease_to_genes)
        shared_gene_documents = (
            build_documents(self.candidates, self.drug_to_direct_genes)
            + build_documents(disease_entities, self.disease_to_genes)
        )
        self.shared_gene_vectorizer, shared_gene_matrix = fit_tfidf(
            shared_gene_documents
        )
        self.drug_gene_shared_matrix = shared_gene_matrix[: len(self.candidates)]
        disease_gene_matrix = shared_gene_matrix[len(self.candidates) :]
        self.disease_gene_rows = {
            disease: disease_gene_matrix[index]
            for index, disease in enumerate(disease_entities)
        }

        max_popularity = max(self.train_popularity_counts.values(), default=0)
        denominator = math.log1p(max_popularity) if max_popularity else 1.0
        self.popularity_scores = np.asarray(
            [
                math.log1p(self.train_popularity_counts.get(candidate, 0))
                / denominator
                for candidate in self.candidates
            ],
            dtype=np.float64,
        )

        self.component_cache: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
        self.coverage_report = self._build_coverage_report(train_indications)
        print(json.dumps(self.coverage_report["summary"], indent=2))

    def _feature_coverage(self, feature_map: Mapping[str, set[str]]) -> dict:
        counts = np.asarray(
            [len(feature_map.get(candidate, set())) for candidate in self.candidates],
            dtype=np.int64,
        )
        covered = int(np.sum(counts > 0))
        return {
            "covered_candidates": covered,
            "candidate_count": len(self.candidates),
            "coverage": covered / len(self.candidates),
            "mean_features": float(np.mean(counts)),
            "median_features": float(np.median(counts)),
            "max_features": int(np.max(counts, initial=0)),
        }

    def _build_coverage_report(self, train_indications: pd.DataFrame) -> dict:
        all_profile_counts = np.asarray(
            [
                len(self.drug_to_ingredients.get(candidate, set()))
                + len(self.drug_to_direct_genes.get(candidate, set()))
                + len(self.drug_to_train_indication_genes.get(candidate, set()))
                + len(self.drug_to_contraindications.get(candidate, set()))
                for candidate in self.candidates
            ],
            dtype=np.int64,
        )
        diseases_with_train_drugs = set(map(str, train_indications["dst"]))
        return {
            "components": {
                "ingredients": self._feature_coverage(self.drug_to_ingredients),
                "direct_genes": self._feature_coverage(self.drug_to_direct_genes),
                "train_indication_genes": self._feature_coverage(
                    self.drug_to_train_indication_genes
                ),
                "contraindications": self._feature_coverage(
                    self.drug_to_contraindications
                ),
            },
            "summary": {
                "candidate_count": len(self.candidates),
                "candidates_with_any_profile": int(np.sum(all_profile_counts > 0)),
                "candidates_without_profile": int(np.sum(all_profile_counts == 0)),
                "diseases_with_train_drugs": len(diseases_with_train_drugs),
                "diseases_with_gene_profile": len(self.disease_to_genes),
                "train_indication_count": int(len(train_indications)),
            },
            "leakage_policy": {
                "static_relations": [
                    self.ingredient_relation,
                    self.direct_gene_relation,
                    self.disease_gene_relation,
                    self.contraindication_relation,
                ],
                "target_relation": self.target_relation,
                "target_derived_features": "train split only",
            },
        }

    def _train_reference_indices(self, disease: str) -> List[int]:
        return [
            self.candidate_to_index[drug]
            for drug in self.disease_to_train_drugs.get(disease, [])
            if drug in self.candidate_to_index
        ]

    def _cached_knn_component(
        self,
        disease: str,
        component_name: str,
        matrix: sparse.csr_matrix,
    ) -> np.ndarray:
        cache = self.component_cache[component_name]
        if disease not in cache:
            cache[disease] = score_against_centroid(
                matrix, self._train_reference_indices(disease)
            )
        return cache[disease]

    def component_scores(self, disease: str) -> Dict[str, np.ndarray]:
        scores: Dict[str, np.ndarray] = {}

        if self.enabled["ingredient_knn"]:
            scores["ingredient_knn"] = self._cached_knn_component(
                disease, "ingredient_knn", self.ingredient_matrix
            )

        if self.enabled["direct_gene_knn"]:
            scores["direct_gene_knn"] = self._cached_knn_component(
                disease, "direct_gene_knn", self.direct_gene_matrix
            )

        if self.enabled["train_indication_gene_knn"]:
            scores["train_indication_gene_knn"] = self._cached_knn_component(
                disease,
                "train_indication_gene_knn",
                self.train_indication_gene_matrix,
            )

        if self.enabled["contraindication_knn"]:
            scores["contraindication_knn"] = self._cached_knn_component(
                disease, "contraindication_knn", self.contraindication_matrix
            )

        if self.enabled["disease_gene_overlap"]:
            if disease not in self.component_cache["disease_gene_overlap"]:
                disease_row = self.disease_gene_rows.get(disease)
                if disease_row is None or disease_row.shape[1] == 0:
                    direct_scores = np.zeros(len(self.candidates), dtype=np.float64)
                else:
                    direct_scores = (
                        self.drug_gene_shared_matrix @ disease_row.T
                    ).toarray().reshape(-1).astype(
                        np.float64,
                        copy=False,
                    )
                self.component_cache["disease_gene_overlap"][disease] = direct_scores
            scores["disease_gene_overlap"] = self.component_cache[
                "disease_gene_overlap"
            ][disease]

        if self.enabled["popularity"]:
            scores["popularity"] = self.popularity_scores

        if self.enabled[PENALTY_COMPONENT]:
            if disease not in self.component_cache[PENALTY_COMPONENT]:
                penalties = np.asarray(
                    [
                        1.0
                        if disease
                        in self.drug_to_contraindications.get(candidate, set())
                        else 0.0
                        for candidate in self.candidates
                    ],
                    dtype=np.float64,
                )
                self.component_cache[PENALTY_COMPONENT][disease] = penalties
            scores[PENALTY_COMPONENT] = self.component_cache[PENALTY_COMPONENT][
                disease
            ]

        return scores

    def combine_scores(
        self,
        disease: str,
        weights: Mapping[str, float],
    ) -> np.ndarray:
        components = self.component_scores(disease)
        combined = np.zeros(len(self.candidates), dtype=np.float64)

        for component in SIMILARITY_COMPONENTS:
            if component in components:
                combined += float(weights.get(component, 0.0)) * components[component]

        if PENALTY_COMPONENT in components:
            combined -= float(weights.get(PENALTY_COMPONENT, 0.0)) * components[
                PENALTY_COMPONENT
            ]

        return combined

    def evaluate(
        self,
        eval_triples: pd.DataFrame,
        known_positives: Mapping[str, set[str]],
        weights: Mapping[str, float],
        show_progress: bool = True,
    ) -> tuple[List[float], dict]:
        ranks: List[float] = []
        zero_signal_queries = 0
        no_train_drug_queries = 0
        no_disease_gene_queries = 0
        direct_contraindication_true_cases = 0

        iterator = eval_triples.itertuples(index=False)
        if show_progress:
            iterator = tqdm(iterator, total=len(eval_triples))

        for row in iterator:
            disease = str(row.dst)
            true_candidate = str(row.src)
            true_index = self.candidate_to_index.get(true_candidate)
            if true_index is None:
                continue

            if not self.disease_to_train_drugs.get(disease):
                no_train_drug_queries += 1
            if not self.disease_to_genes.get(disease):
                no_disease_gene_queries += 1
            if disease in self.drug_to_contraindications.get(true_candidate, set()):
                direct_contraindication_true_cases += 1

            scores = self.combine_scores(disease, weights).copy()
            if np.all(scores == scores[0]):
                zero_signal_queries += 1

            for known_candidate in known_positives.get(disease, set()):
                if known_candidate == true_candidate:
                    continue
                known_index = self.candidate_to_index.get(known_candidate)
                if known_index is not None:
                    scores[known_index] = -np.inf

            ranks.append(tie_aware_rank(scores, true_index))

        diagnostics = {
            "evaluated_triples": len(ranks),
            "zero_signal_queries": zero_signal_queries,
            "queries_without_train_drugs": no_train_drug_queries,
            "queries_without_disease_genes": no_disease_gene_queries,
            "true_candidates_directly_contraindicated": (
                direct_contraindication_true_cases
            ),
        }
        return ranks, diagnostics

    def evaluate_negative_set(
        self,
        evaluation_payload: Mapping[str, object],
        weights: Mapping[str, float],
        show_progress: bool = True,
    ) -> tuple[
        List[float],
        dict,
        List[dict],
        np.ndarray,
        np.ndarray,
    ]:
        """Evaluate fixed 1-vs-N candidate sets shared across models."""

        examples = evaluation_payload.get("examples")

        if not isinstance(examples, list):
            raise ValueError(
                "Evaluation-set payload must contain an 'examples' list."
            )

        ranks: List[float] = []
        detailed_rows: List[dict] = []
        positive_scores: List[float] = []
        negative_score_rows: List[np.ndarray] = []

        zero_signal_examples = 0
        no_train_drug_queries = 0
        no_disease_gene_queries = 0
        direct_contraindication_true_cases = 0

        iterator = examples

        if show_progress:
            iterator = tqdm(iterator, total=len(examples))

        for position, example in enumerate(iterator):
            if not isinstance(example, Mapping):
                raise ValueError(
                    f"Invalid evaluation example at position {position}."
                )

            disease = str(example["query"])
            relation = str(example["relation"])
            true_candidate = str(example["positive_candidate"])

            negative_candidates_raw = example.get(
                "negative_candidates",
                [],
            )

            if not isinstance(negative_candidates_raw, list):
                raise ValueError(
                    f"Example {position} has invalid negative candidates."
                )

            negative_candidates = [
                str(candidate)
                for candidate in negative_candidates_raw
            ]

            if relation != self.target_relation:
                raise ValueError(
                    f"Example {position} uses relation {relation!r}; "
                    f"expected {self.target_relation!r}."
                )

            candidate_names = [
                true_candidate,
                *negative_candidates,
            ]

            if len(set(candidate_names)) != len(candidate_names):
                raise ValueError(
                    f"Example {position} contains duplicate candidates."
                )

            missing_candidates = [
                candidate
                for candidate in candidate_names
                if candidate not in self.candidate_to_index
            ]

            if missing_candidates:
                raise ValueError(
                    f"Example {position} contains candidates outside "
                    f"the model candidate pool: {missing_candidates[:5]}"
                )

            if not self.disease_to_train_drugs.get(disease):
                no_train_drug_queries += 1

            if not self.disease_to_genes.get(disease):
                no_disease_gene_queries += 1

            if disease in self.drug_to_contraindications.get(
                true_candidate,
                set(),
            ):
                direct_contraindication_true_cases += 1

            all_scores = self.combine_scores(
                disease,
                weights,
            )

            candidate_indices = np.asarray(
                [
                    self.candidate_to_index[candidate]
                    for candidate in candidate_names
                ],
                dtype=np.int64,
            )

            sampled_scores = all_scores[candidate_indices]

            if np.all(sampled_scores == sampled_scores[0]):
                zero_signal_examples += 1

            rank = tie_aware_rank(
                sampled_scores,
                true_index=0,
            )

            ranks.append(rank)

            negative_scores = sampled_scores[1:]

            positive_scores.append(
                float(sampled_scores[0])
            )
            negative_score_rows.append(
                np.asarray(
                    negative_scores,
                    dtype=np.float64,
                )
            )

            detailed_rows.append(
                {
                    "example_index": int(
                        example.get("example_index", position)
                    ),
                    "query": disease,
                    "relation": relation,
                    "positive_candidate": true_candidate,
                    "positive_score": float(sampled_scores[0]),
                    "rank": float(rank),
                    "candidate_count": len(candidate_names),
                    "negative_count": len(negative_candidates),
                    "max_negative_score": (
                        float(np.max(negative_scores))
                        if len(negative_scores)
                        else None
                    ),
                    "mean_negative_score": (
                        float(np.mean(negative_scores))
                        if len(negative_scores)
                        else None
                    ),
                    "negative_candidates": "|".join(
                        negative_candidates
                    ),
                }
            )

        diagnostics = {
            "evaluated_examples": len(ranks),
            "candidate_count_per_example": (
                1
                + int(
                    evaluation_payload.get(
                        "metadata",
                        {},
                    ).get("num_negatives", 0)
                )
            ),
            "zero_signal_examples": zero_signal_examples,
            "queries_without_train_drugs": no_train_drug_queries,
            "queries_without_disease_genes": no_disease_gene_queries,
            "true_candidates_directly_contraindicated": (
                direct_contraindication_true_cases
            ),
        }

        positive_scores_array = np.asarray(
            positive_scores,
            dtype=np.float64,
        )

        negative_scores_array = np.vstack(
            negative_score_rows
        ) if negative_score_rows else np.empty(
            (0, 0),
            dtype=np.float64,
        )

        return (
            ranks,
            diagnostics,
            detailed_rows,
            positive_scores_array,
            negative_scores_array,
        )


def load_fixed_weights(config: Mapping[str, object]) -> Dict[str, float]:
    weights_cfg = dict(config.get("weights", {}))
    weights = {
        name: float(weights_cfg.get(name, 0.0))
        for name in (*SIMILARITY_COMPONENTS, PENALTY_COMPONENT)
    }
    positive_total = sum(weights[name] for name in SIMILARITY_COMPONENTS)
    if positive_total <= 0.0:
        raise ValueError("At least one positive component weight must be greater than zero.")
    return weights


def tune_weights(
    model: HybridContentBasedBaseline,
    valid_df: pd.DataFrame,
    known_positives: Mapping[str, set[str]],
    config: Mapping[str, object],
) -> tuple[Dict[str, float], dict, List[dict]]:
    search_cfg = dict(config.get("weight_search", {}))
    step = float(search_cfg.get("step", 0.5))
    penalty_values = [
        float(value)
        for value in search_cfg.get("direct_contraindication_values", [0.0, 0.5, 1.0])
    ]

    enabled_positive = [
        name for name in SIMILARITY_COMPONENTS if model.enabled.get(name, False)
    ]
    if not enabled_positive:
        raise ValueError("No positive score components are enabled.")

    print("--- [2/4] Selecting weights on validation only ---")
    results: List[dict] = []
    best_weights: Dict[str, float] | None = None
    best_metrics: dict | None = None

    candidates = list(simplex_weights(enabled_positive, step))
    for positive_weights, penalty in tqdm(
        itertools.product(candidates, penalty_values),
        total=len(candidates) * len(penalty_values),
    ):
        weights = {name: 0.0 for name in (*SIMILARITY_COMPONENTS, PENALTY_COMPONENT)}
        weights.update(positive_weights)
        weights[PENALTY_COMPONENT] = penalty if model.enabled[PENALTY_COMPONENT] else 0.0

        ranks, _ = model.evaluate(
            valid_df,
            known_positives,
            weights,
            show_progress=False,
        )
        metrics = compute_ranking_metrics(ranks)
        result = {"weights": weights, "metrics": metrics}
        results.append(result)

        if best_metrics is None or (
            metrics["mrr"], metrics["hits@10"], metrics["hits@1"]
        ) > (
            best_metrics["mrr"],
            best_metrics["hits@10"],
            best_metrics["hits@1"],
        ):
            best_weights = weights
            best_metrics = metrics

    assert best_weights is not None and best_metrics is not None
    return best_weights, best_metrics, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Select component weights using the validation split. Not allowed on test.",
    )
    parser.add_argument(
        "--weights-file",
        help="JSON file produced by a previous validation tuning run.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.tune and args.split != "valid":
        raise SystemExit("Weight tuning is allowed only with --split valid.")

    try:
        config_path = resolve_existing_path(args.config)
        cfg = load_config(config_path)
        triples_path = resolve_existing_path(cfg["data"]["triples_path"])
        processed_dir = resolve_existing_path(cfg["data"]["processed_dir"])
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error

    triples_indexed_path = processed_dir / "triples_indexed.csv"
    entity2id_path = processed_dir / "entity2id.json"
    entity_types_path = processed_dir / "entity_types.json"
    for required_path in (triples_indexed_path, entity2id_path, entity_types_path):
        if not required_path.exists():
            raise SystemExit(f"Missing required file: {required_path}")

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    task_cfg = cfg["task"]
    task_name = str(task_cfg.get("name", "disease_to_drug"))
    if task_name != "disease_to_drug":
        raise SystemExit("This baseline supports only task.name=disease_to_drug.")
    target_relation = str(task_cfg["target_relation"])

    triples_df = pd.read_csv(triples_indexed_path, dtype=str)
    train_ratio, valid_ratio, test_ratio = get_split_ratios(cfg["split"])
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

    integrity_report = validate_experiment_integrity(
        cfg,
        config_path=config_path,
        output_path=splits_dir / "split_integrity_report_content_based.json",
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
    candidate_type = str(task_cfg.get("candidate_entity_type", "DRUG"))
    candidate_entities = sorted(
        [
            entity
            for entity, entity_type in entity_types.items()
            if candidate_type == "ALL" or entity_type == candidate_type
        ],
        key=lambda entity: int(entity2id[entity]),
    )

    baseline_cfg = dict(cfg.get("content_based", cfg.get("baseline", {})))
    model = HybridContentBasedBaseline(
        triples_path=triples_path,
        candidate_entities=candidate_entities,
        train_triples=train_df,
        config=baseline_cfg,
        target_relation=target_relation,
    )

    run_dir = resolve_run_dir(
        cfg["runs"]["out_dir"],
        cfg,
        model_name="content_based",
        seed=seed,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "coverage_report.json", model.coverage_report)

    tuning_summary = None
    if args.tune:
        weights, validation_metrics, search_results = tune_weights(
            model, valid_df, known_positives, baseline_cfg
        )
        tuning_summary = {
            "selection_split": "valid",
            "selected_weights": weights,
            "validation_metrics": validation_metrics,
            "search_space_size": len(search_results),
        }
        write_json(run_dir / "selected_weights.json", tuning_summary)
        write_json(run_dir / "weight_search_results.json", search_results)
        print("Selected weights:")
        print(json.dumps(tuning_summary, indent=2))
    elif args.weights_file:
        weights_payload = json.loads(
            resolve_existing_path(args.weights_file).read_text(encoding="utf-8")
        )
        weights = {
            key: float(value)
            for key, value in weights_payload["selected_weights"].items()
        }
    else:
        weights = load_fixed_weights(baseline_cfg)

    print(f"--- [3/4] Evaluating {args.split} split with filtered ranking ---")
    ranks, diagnostics = model.evaluate(eval_df, known_positives, weights)
    if not ranks:
        raise SystemExit("No evaluation ranks were computed.")
    metrics = compute_ranking_metrics(ranks)

    split_summary = {
        "task_name": task_name,
        "target_relation": target_relation,
        "train_size": len(train_df),
        "valid_size": len(valid_df),
        "test_size": len(test_df),
        "candidate_pool_size": len(candidate_entities),
        "candidate_entity_type": candidate_type,
        "split_mode": cfg["split"].get("mode") if isinstance(cfg.get("split"), dict) else None,
        "split_integrity_report": integrity_report["report_path"],
    }
    model_summary = {
        "model_name": "hybrid_content_based",
        "task_name": task_name,
        "seed": seed,
        "selected_weights": weights,
        "tuning": tuning_summary,
        "enabled_components": model.enabled,
        "relations": {
            "ingredient": model.ingredient_relation,
            "drug_gene": model.direct_gene_relation,
            "disease_gene": model.disease_gene_relation,
            "contraindication": model.contraindication_relation,
        },
        "aggregation": model.aggregation,
        "leakage_policy": (
            "INDICATION-derived drug features use train triples only; "
            "validation and test are used only for evaluation, except validation-only tuning."
        ),
        "diagnostics": diagnostics,
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
    write_json(run_dir / f"diagnostics_{args.split}.json", diagnostics)

    print("--- [4/4] Final metrics ---")
    print(f"Hits@1:  {metrics['hits@1']:.4f}")
    print(f"Hits@3:  {metrics['hits@3']:.4f}")
    print(f"Hits@10: {metrics['hits@10']:.4f}")
    print(f"MRR:     {metrics['mrr']:.4f}")

    log_experiment(
        dataset=Path(config_path).stem,
        model_name="Hybrid Content-Based",
        hits1=metrics["hits@1"],
        hits3=metrics["hits@3"],
        hits10=metrics["hits@10"],
        mrr=metrics["mrr"],
        notes=(
            f"Same splits/candidates/filtered ranking as R-GCN | split={args.split} "
            f"| weights={json.dumps(weights, sort_keys=True)}"
        ),
    )
    save_metrics(run_dir, args.split, metrics)


if __name__ == "__main__":
    main()
