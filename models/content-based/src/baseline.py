import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Repository-relative path configuration.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.benchmark_tracker import log_experiment


def resolve_existing_path(*relative_candidates):
    """Return the first existing repository-relative path from the candidates."""
    for candidate in relative_candidates:
        candidate_path = (REPO_ROOT / candidate).resolve()
        if candidate_path.exists():
            return candidate_path
    searched = "\n".join(f"- {(REPO_ROOT / candidate).resolve()}" for candidate in relative_candidates)
    raise FileNotFoundError(f"Files not found:\n{searched}")


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def get_split_ratios(split_cfg: Dict) -> Tuple[float, float, float]:
    if "train_ratio" in split_cfg:
        train_ratio = float(split_cfg["train_ratio"])
        valid_ratio = float(split_cfg.get("val_ratio", split_cfg.get("valid_ratio", 0.1)))
        test_ratio = float(split_cfg["test_ratio"])
    else:
        train_ratio = float(split_cfg["train"])
        valid_ratio = float(split_cfg["valid"])
        test_ratio = float(split_cfg["test"])
    return train_ratio, valid_ratio, test_ratio


def split_target_triples(
    triples_df: pd.DataFrame,
    target_relation: str,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0.")

    target_df = triples_df[triples_df["rel"] == target_relation].copy()
    if target_df.empty:
        raise ValueError(f"No triples found for target relation: {target_relation}")

    indices = target_df.index.tolist()
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]

    return target_df.loc[train_idx], target_df.loc[valid_idx], target_df.loc[test_idx]


def build_known_positives(target_df: pd.DataFrame) -> Dict[str, set]:
    known = {}
    grouped = target_df.groupby("dst")["src"]
    for disease_id, drug_series in grouped:
        known[disease_id] = set(drug_series.tolist())
    return known

class ContentBasedBaseline:
    def __init__(self, triples_full_path: Path, candidate_drugs: List[str]):
        print("--- [1/3] Initializing aligned content-based baseline (TF-IDF + cosine) ---")

        df_all = pd.read_csv(triples_full_path)
        ingredients_df = df_all[df_all["rel"] == "HAS_INGREDIENT"].copy()

        ingredient_map = (
            ingredients_df.groupby("src")["dst"]
            .apply(list)
            .to_dict()
        )

        if not candidate_drugs:
            raise ValueError("No DRUG candidates found in processed graph artifacts.")

        self.drug_list = list(candidate_drugs)
        self.drug_to_idx = {drug: index for index, drug in enumerate(self.drug_list)}
        self.documents = [" ".join(ingredient_map.get(drug, [])) for drug in self.drug_list]

        if not any(doc.strip() for doc in self.documents):
            raise ValueError("No ingredient text was found for candidate drugs.")

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
        print(f"-> TF-IDF matrix: {self.tfidf_matrix.shape} (drugs x features)")

    def predict_ranks(
        self,
        eval_triples: pd.DataFrame,
        train_triples: pd.DataFrame,
        known_positives: Dict[str, set],
    ) -> List[float]:
        print("--- [2/3] Generating filtered rankings with cosine similarity ---")

        disease_to_train_indices = (
            train_triples.groupby("dst")["src"]
            .apply(lambda x: [self.drug_to_idx[d] for d in x if d in self.drug_to_idx])
            .to_dict()
        )

        ranks = []

        for _, row in tqdm(eval_triples.iterrows(), total=len(eval_triples)):
            true_drug = row["src"]
            disease = row["dst"]

            if true_drug not in self.drug_to_idx:
                continue

            true_idx = self.drug_to_idx[true_drug]
            known_train_indices = disease_to_train_indices.get(disease, [])

            if known_train_indices:
                scores = np.max(self.similarity_matrix[:, known_train_indices], axis=1).astype(np.float64)
            else:
                scores = np.zeros(len(self.drug_list), dtype=np.float64)

            for known_drug in known_positives.get(disease, set()):
                if known_drug == true_drug:
                    continue
                known_idx = self.drug_to_idx.get(known_drug)
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
    args = parser.parse_args()

    try:
        config_path = resolve_existing_path(args.config)
        with open(config_path, "r", encoding="utf-8") as file_obj:
            cfg = yaml.safe_load(file_obj)

        triples_full_path = resolve_existing_path("kg_constructor/data/output/triples/triples.csv")
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
    train_ratio, valid_ratio, test_ratio = get_split_ratios(cfg["split"])

    triples_df = pd.read_csv(triples_indexed_path)
    train_df, valid_df, test_df = split_target_triples(
        triples_df=triples_df,
        target_relation=target_relation,
        seed=seed,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )

    eval_df = valid_df if args.split == "valid" else test_df
    all_target_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    known_positives = build_known_positives(all_target_df)

    entity2id = read_json(entity2id_path)
    entity_types = read_json(entity_types_path)
    candidate_drugs = sorted(
        [entity for entity, entity_type in entity_types.items() if entity_type == "DRUG"],
        key=lambda entity: int(entity2id[entity]),
    )

    model = ContentBasedBaseline(triples_full_path=triples_full_path, candidate_drugs=candidate_drugs)
    ranks = model.predict_ranks(eval_df, train_df, known_positives)

    if not ranks:
        print("No evaluation ranks computed.")
        sys.exit(1)

    ranks_array = np.array(ranks, dtype=np.float64)

    print("\n--- [3/3] Final metrics (aligned with R-GCN protocol) ---")
    hits_at_1 = float(np.mean(ranks_array <= 1))
    hits_at_3 = float(np.mean(ranks_array <= 3))
    hits_at_10 = float(np.mean(ranks_array <= 10))
    mrr = float(np.mean(1.0 / ranks_array))

    print(f"Hits@1:  {hits_at_1:.4f}")
    print(f"Hits@3:  {hits_at_3:.4f}")
    print(f"Hits@10: {hits_at_10:.4f}")
    print(f"MRR:     {mrr:.4f}")

    log_experiment(
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