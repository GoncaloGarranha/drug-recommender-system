import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import tensorflow as tf

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common import io, model_builder, settings_reader

def resolve_path(path_text: str, must_exist: bool = False) -> Path:
    """Resolve absolute, cwd-relative, or repository-root-relative paths."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.exists() or (not must_exist and path.parent.exists()):
        return path

    for parent in Path.cwd().resolve().parents:
        candidate = parent / path
        if candidate.exists() or (not must_exist and candidate.parent.exists()):
            return candidate

    return path


def default_existing_path(default_path, fallback_path, label):
    """Use default_path when present, otherwise fall back to train.txt for legacy datasets."""
    default_path = Path(default_path)
    fallback_path = Path(fallback_path)
    if default_path.exists():
        return default_path
    if fallback_path.exists():
        print(
            "WARNING: {} not found at {}. Falling back to legacy train file {}. "
            "Rebuild the dataset with tools/build_fair_biomedical_dataset.py for the FAIR protocol.".format(
                label, default_path, fallback_path
            )
        )
        return fallback_path
    return default_path

def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as file_obj:
        return [line.strip() for line in file_obj if line.strip()]


def read_triplets(path: Path, entities_path: Path, relations_path: Path) -> np.ndarray:
    return np.array(io.read_triplets_as_list(str(path), str(entities_path), str(relations_path)), dtype=np.int32)


def build_model(settings, entities, relations, graph_triplets):
    encoder_settings = settings["Encoder"]
    decoder_settings = settings["Decoder"]
    shared_settings = settings["Shared"]
    general_settings = settings["General"]

    general_settings.put("EntityCount", len(entities))
    general_settings.put("RelationCount", len(relations))
    general_settings.put("EdgeCount", len(graph_triplets))

    encoder_settings.merge(shared_settings)
    encoder_settings.merge(general_settings)
    decoder_settings.merge(shared_settings)
    decoder_settings.merge(general_settings)

    encoder = model_builder.build_encoder(encoder_settings, graph_triplets)
    model = model_builder.build_decoder(encoder, decoder_settings)
    model.preprocess(graph_triplets)
    model.register_for_test(graph_triplets)
    model.initialize_train()
    return model


def known_subjects_by_relation_object(triples: Iterable[Sequence[int]]) -> Dict[Tuple[int, int], Set[int]]:
    known = {}
    for subject_id, relation_id, object_id in triples:
        key = (int(relation_id), int(object_id))
        known.setdefault(key, set()).add(int(subject_id))
    return known


def compute_metrics(ranks: Sequence[int], hits_at: Sequence[int]) -> Dict[str, float]:
    ranks_array = np.array(ranks, dtype=np.float64)
    metrics = {"mrr": float(np.mean(1.0 / ranks_array))}
    for k in hits_at:
        metrics["hits@{}".format(k)] = float(np.mean(ranks_array <= k))
    return metrics


def filtered_subject_rank(
    scores: np.ndarray,
    candidate_subject_ids: np.ndarray,
    candidate_subject_set: Set[int],
    gold_subject_id: int,
    known_subject_ids: Set[int],
) -> int:
    candidate_scores = scores[candidate_subject_ids]
    gold_score = scores[gold_subject_id]
    raw_rank = int(np.sum(candidate_scores >= gold_score))

    filtered_known = [
        subject_id
        for subject_id in known_subject_ids
        if subject_id != gold_subject_id and subject_id in candidate_subject_set
    ]
    filtered_penalty = int(np.sum(scores[filtered_known] >= gold_score)) if filtered_known else 0
    return raw_rank - filtered_penalty


def evaluate_subject_prediction(
    model,
    triples: np.ndarray,
    candidate_subject_ids: np.ndarray,
    known_subjects: Dict[Tuple[int, int], Set[int]],
    chunk_size: int,
    hits_at: Sequence[int],
) -> Dict[str, float]:
    ranks = []
    candidate_subject_set = set(int(x) for x in candidate_subject_ids.tolist())
    for chunk_start in range(0, len(triples), chunk_size):
        chunk = triples[chunk_start : chunk_start + chunk_size]
        all_subject_scores = model.score_all_subjects(chunk)
        for scores, triple in zip(all_subject_scores, chunk):
            subject_id = int(triple[0])
            relation_id = int(triple[1])
            object_id = int(triple[2])
            key = (relation_id, object_id)
            ranks.append(
                filtered_subject_rank(
                    scores=np.asarray(scores),
                    candidate_subject_ids=candidate_subject_ids,
                    candidate_subject_set=candidate_subject_set,
                    gold_subject_id=subject_id,
                    known_subject_ids=known_subjects.get(key, set()),
                )
            )
    metrics = compute_metrics(ranks, hits_at)
    metrics["num_queries"] = len(ranks)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RelationPrediction checkpoints with disease-to-drug restricted ranking."
    )
    parser.add_argument("--settings", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True, help="TensorFlow checkpoint prefix, e.g. models/Run-10")
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--graph_train", help="Defaults to dataset/graph_train.txt")
    parser.add_argument("--loss_train", help="Defaults to dataset/loss_train.txt")
    parser.add_argument("--candidate_subjects", help="Defaults to dataset/candidate_drugs.txt")
    parser.add_argument("--hits_at", nargs="+", type=int, default=[1, 3, 10])
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    dataset = resolve_path(args.dataset, must_exist=True)
    entities_path = dataset / "entities.dict"
    relations_path = dataset / "relations.dict"
    legacy_train_path = dataset / "train.txt"
    graph_train_path = resolve_path(args.graph_train, must_exist=True) if args.graph_train else default_existing_path(dataset / "graph_train.txt", legacy_train_path, "graph_train.txt")
    loss_train_path = resolve_path(args.loss_train, must_exist=True) if args.loss_train else default_existing_path(dataset / "loss_train.txt", legacy_train_path, "loss_train.txt")
    candidate_subjects_path = resolve_path(args.candidate_subjects, must_exist=True) if args.candidate_subjects else dataset / "candidate_drugs.txt"
    eval_path = dataset / "{}.txt".format(args.split)

    settings_path = resolve_path(args.settings, must_exist=True)
    settings = settings_reader.read(str(settings_path))
    entities = io.read_dictionary(str(entities_path))
    relations = io.read_dictionary(str(relations_path))
    entity_to_id = io.read_dictionary(str(entities_path), id_lookup=False)

    graph_triplets = read_triplets(graph_train_path, entities_path, relations_path)
    loss_triplets = read_triplets(loss_train_path, entities_path, relations_path)
    valid_triplets = read_triplets(dataset / "valid.txt", entities_path, relations_path)
    test_triplets = read_triplets(dataset / "test.txt", entities_path, relations_path)
    eval_triplets = read_triplets(eval_path, entities_path, relations_path)

    candidate_subject_ids = np.array(
        [entity_to_id[entity] for entity in read_lines(candidate_subjects_path)], dtype=np.int32
    )
    known_subjects = known_subjects_by_relation_object(
        np.concatenate((loss_triplets, valid_triplets, test_triplets), axis=0)
    )

    model = build_model(settings, entities, relations, graph_triplets)
    _ = model.predict_all_subject_scores()
    saver = tf.train.Saver(var_list=model.get_weights())
    model.session = tf.Session()
    model.session.run(tf.initialize_all_variables())
    saver.restore(model.session, args.checkpoint)

    metrics = evaluate_subject_prediction(
        model=model,
        triples=eval_triplets,
        candidate_subject_ids=candidate_subject_ids,
        known_subjects=known_subjects,
        chunk_size=args.chunk_size,
        hits_at=args.hits_at,
    )
    metrics.update(
        {
            "protocol": "disease_to_drug_subject_prediction_restricted_to_candidate_drugs",
            "split": args.split,
            "checkpoint": args.checkpoint,
            "dataset": str(dataset),
            "num_candidate_subjects": int(len(candidate_subject_ids)),
            "graph_train_triples": int(len(graph_triplets)),
            "loss_train_triples": int(len(loss_triplets)),
        }
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file_obj:
            json.dump(metrics, file_obj, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
