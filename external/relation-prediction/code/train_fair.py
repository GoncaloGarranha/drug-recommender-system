import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from common import auxilliaries, evaluation, io, model_builder, optimizer_parameter_parser, settings_reader
from optimization.optimize import build_tensorflow


def resolve_path(path_text, must_exist=False):
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
        return str(default_path)
    if fallback_path.exists():
        print(
            "WARNING: {} not found at {}. Falling back to legacy train file {}. "
            "Rebuild the dataset with tools/build_fair_biomedical_dataset.py for the FAIR protocol.".format(
                label, default_path, fallback_path
            )
        )
        return str(fallback_path)
    return str(default_path)


def optional_existing_path(default_path, label):
    """Return a path string if it exists; otherwise disable the optional input."""
    default_path = Path(default_path)
    if default_path.exists():
        return str(default_path)
    print("WARNING: {} not found at {}. Disabling this optional input.".format(label, default_path))
    return None


class SubjectTypedNegativeSampler:
    """Generate negatives by replacing only the subject with a typed candidate."""

    def __init__(self, negative_sample_rate, candidate_subject_ids, known_positive_triplets):
        self.negative_sample_rate = int(negative_sample_rate)
        self.candidate_subject_ids = np.array(sorted(set(int(x) for x in candidate_subject_ids)), dtype=np.int32)
        if self.candidate_subject_ids.size == 0:
            raise ValueError("Typed negative sampling requires at least one candidate subject id.")

        self.known_subjects = {}
        for subject_id, relation_id, object_id in known_positive_triplets:
            key = (int(relation_id), int(object_id))
            self.known_subjects.setdefault(key, set()).add(int(subject_id))

    def transform(self, triplets):
        positives = np.array(triplets, dtype=np.int32)
        negative_triplets = []

        for subject_id, relation_id, object_id in positives:
            key = (int(relation_id), int(object_id))
            forbidden = set(self.known_subjects.get(key, set()))
            forbidden.add(int(subject_id))

            collected = 0
            attempts = 0
            while collected < self.negative_sample_rate:
                picks = np.random.choice(
                    self.candidate_subject_ids,
                    size=max(8, (self.negative_sample_rate - collected) * 4),
                    replace=True,
                )
                for candidate_subject_id in picks:
                    candidate_subject_id = int(candidate_subject_id)
                    if candidate_subject_id in forbidden:
                        continue
                    negative_triplets.append([candidate_subject_id, int(relation_id), int(object_id)])
                    collected += 1
                    if collected >= self.negative_sample_rate:
                        break

                attempts += 1
                if attempts > 200 and collected < self.negative_sample_rate:
                    raise ValueError(
                        "Unable to sample typed negatives. Candidate pool may be exhausted "
                        "for relation/object pair: {}".format(key)
                    )

        labels = np.zeros((len(positives) + len(negative_triplets)), dtype=np.float32)
        labels[: len(positives)] = 1.0
        indexes = np.concatenate((positives, np.array(negative_triplets, dtype=np.int32)), axis=0)
        return indexes, labels


def build_known_subjects(triplets):
    known_subjects = {}
    for subject_id, relation_id, object_id in triplets:
        key = (int(relation_id), int(object_id))
        known_subjects.setdefault(key, set()).add(int(subject_id))
    return known_subjects


def filtered_subject_rank(scores, candidate_subject_ids, candidate_subject_set, gold_subject_id, known_subject_ids):
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


def compute_restricted_subject_metrics(model, triples, candidate_subject_ids, known_subjects, hits_at=(1, 3, 10), chunk_size=256):
    ranks = []
    candidate_subject_set = set(int(x) for x in candidate_subject_ids.tolist())

    for chunk_start in range(0, len(triples), chunk_size):
        chunk = triples[chunk_start : chunk_start + chunk_size]
        all_subject_scores = model.score_all_subjects(chunk)
        for scores, triple in zip(all_subject_scores, chunk):
            subject_id = int(triple[0])
            relation_id = int(triple[1])
            object_id = int(triple[2])
            ranks.append(
                filtered_subject_rank(
                    scores=np.asarray(scores),
                    candidate_subject_ids=candidate_subject_ids,
                    candidate_subject_set=candidate_subject_set,
                    gold_subject_id=subject_id,
                    known_subject_ids=known_subjects.get((relation_id, object_id), set()),
                )
            )

    ranks = np.array(ranks, dtype=np.float64)
    metrics = {"MRR": float(np.mean(1.0 / ranks))}
    for k in hits_at:
        metrics["H@{}".format(k)] = float(np.mean(ranks <= k))
    return metrics


def print_restricted_metrics(metrics):
    print("	Filtered")
    for key in ["MRR", "H@1", "H@3", "H@10"]:
        print("{}	{}".format(key, round(metrics[key], 3)))


def read_entity_ids(path, entity_to_id):
    ids = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            entity = line.strip()
            if entity:
                ids.append(entity_to_id[entity])
    return ids


def sample_edge_neighborhood(triplets, degrees, adj_list, sample_size):
    sample_size = min(int(sample_size), int(triplets.shape[0]))
    edges = np.zeros((sample_size), dtype=np.int32)

    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in triplets])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = weights / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]), p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    return edges


def build_adjacency(triplets, entities):
    adj_list = [[] for _ in entities]
    for i, triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return degrees, adj_list


def main():
    parser = argparse.ArgumentParser(description="Train RelationPrediction with separate graph and target-loss triples.")
    parser.add_argument("--settings", help="Filepath for settings file.", required=True)
    parser.add_argument("--dataset", help="Filepath for dataset.", required=True)
    parser.add_argument("--graph_train", help="Graph/context triples file. Defaults to dataset/graph_train.txt.")
    parser.add_argument("--loss_train", help="Target loss triples file. Defaults to dataset/loss_train.txt.")
    parser.add_argument("--candidate_subjects", help="Candidate subject file for typed negatives.")
    args = parser.parse_args()

    settings_path = resolve_path(args.settings, must_exist=True)
    dataset_path = resolve_path(args.dataset, must_exist=True)

    settings = settings_reader.read(str(settings_path))
    print(settings)

    dataset = str(dataset_path)
    relations_path = str(dataset_path / "relations.dict")
    entities_path = str(dataset_path / "entities.dict")
    legacy_train_path = dataset_path / "train.txt"
    graph_train_path = str(resolve_path(args.graph_train, must_exist=True)) if args.graph_train else default_existing_path(dataset_path / "graph_train.txt", legacy_train_path, "graph_train.txt")
    loss_train_path = str(resolve_path(args.loss_train, must_exist=True)) if args.loss_train else default_existing_path(dataset_path / "loss_train.txt", legacy_train_path, "loss_train.txt")
    valid_path = str(dataset_path / "valid.txt")
    test_path = str(dataset_path / "test.txt")
    candidate_subjects_path = str(resolve_path(args.candidate_subjects, must_exist=True)) if args.candidate_subjects else optional_existing_path(dataset_path / "candidate_drugs.txt", "candidate_drugs.txt")

    graph_triplets = np.array(io.read_triplets_as_list(graph_train_path, entities_path, relations_path))
    loss_triplets = np.array(io.read_triplets_as_list(loss_train_path, entities_path, relations_path))
    valid_triplets = np.array(io.read_triplets_as_list(valid_path, entities_path, relations_path))
    test_triplets = np.array(io.read_triplets_as_list(test_path, entities_path, relations_path))

    entities = io.read_dictionary(entities_path)
    relations = io.read_dictionary(relations_path)
    entity_to_id = io.read_dictionary(entities_path, id_lookup=False)

    candidate_subject_ids_for_eval = None
    if candidate_subjects_path is not None:
        candidate_subject_ids_for_eval = np.array(read_entity_ids(candidate_subjects_path, entity_to_id), dtype=np.int32)
        print("Using FAIR restricted validation with {} candidate subjects.".format(len(candidate_subject_ids_for_eval)))

    known_subjects_for_eval = build_known_subjects(np.concatenate((loss_triplets, valid_triplets, test_triplets), axis=0))

    encoder_settings = settings["Encoder"]
    decoder_settings = settings["Decoder"]
    shared_settings = settings["Shared"]
    general_settings = settings["General"]
    optimizer_settings = settings["Optimizer"]
    evaluation_settings = settings["Evaluation"]

    general_settings.put("EntityCount", len(entities))
    general_settings.put("RelationCount", len(relations))
    general_settings.put("EdgeCount", len(graph_triplets))

    encoder_settings.merge(shared_settings)
    encoder_settings.merge(general_settings)
    decoder_settings.merge(shared_settings)
    decoder_settings.merge(general_settings)
    optimizer_settings.merge(general_settings)
    evaluation_settings.merge(general_settings)

    encoder = model_builder.build_encoder(encoder_settings, graph_triplets)
    model = model_builder.build_decoder(encoder, decoder_settings)

    opp = optimizer_parameter_parser.Parser(optimizer_settings)
    opp.set_save_function(model.save)

    scorer = evaluation.Scorer(evaluation_settings)
    scorer.register_data(graph_triplets)
    scorer.register_data(loss_triplets)
    scorer.register_data(valid_triplets)
    scorer.register_data(test_triplets)
    scorer.register_degrees(graph_triplets)
    scorer.register_model(model)
    scorer.finalize_frequency_computation(np.concatenate((graph_triplets, valid_triplets, test_triplets), axis=0))

    def score_validation_data(validation_data):
        if candidate_subject_ids_for_eval is not None and evaluation_settings["Metric"] == "MRR":
            validation_metrics = compute_restricted_subject_metrics(
                model,
                validation_data,
                candidate_subject_ids_for_eval,
                known_subjects_for_eval,
            )
            test_metrics = compute_restricted_subject_metrics(
                model,
                test_triplets,
                candidate_subject_ids_for_eval,
                known_subjects_for_eval,
            )
            print_restricted_metrics(test_metrics)
            return validation_metrics["MRR"]

        score_summary = scorer.compute_scores(validation_data, verbose=False).get_summary()

        if evaluation_settings["Metric"] == "MRR":
            lookup_string = score_summary.mrr_string()
        elif evaluation_settings["Metric"] == "Accuracy":
            lookup_string = score_summary.accuracy_string()
        else:
            raise ValueError("Unsupported evaluation metric: {}".format(evaluation_settings["Metric"]))

        early_stopping = score_summary.results["Filtered"][lookup_string]

        score_summary = scorer.compute_scores(test_triplets, verbose=False).get_summary()
        score_summary.pretty_print()

        return early_stopping

    opp.set_early_stopping_score_function(score_validation_data)

    print("Graph triples: {}".format(len(graph_triplets)))
    print("Loss triples: {}".format(len(loss_triplets)))

    degrees, adj_list = build_adjacency(graph_triplets, entities)

    if "NegativeSampleRate" in general_settings:
        negative_sample_rate = int(general_settings["NegativeSampleRate"])
        known_positive_triplets = np.concatenate((loss_triplets, valid_triplets, test_triplets), axis=0)

        if candidate_subject_ids_for_eval is not None:
            sampler = SubjectTypedNegativeSampler(negative_sample_rate, candidate_subject_ids_for_eval, known_positive_triplets)
            print("Using subject-typed negative sampling with {} candidate subjects.".format(len(candidate_subject_ids_for_eval)))
        else:
            sampler = auxilliaries.NegativeSampler(negative_sample_rate, general_settings["EntityCount"])
            sampler.set_known_positives(known_positive_triplets)
            print("Using original untyped negative sampling.")

        def t_func(x):
            target_batch = np.array(x)
            if not encoder.needs_graph():
                return sampler.transform(target_batch)

            if "GraphBatchSize" in general_settings:
                graph_batch_size = int(general_settings["GraphBatchSize"])
                graph_batch_ids = sample_edge_neighborhood(graph_triplets, degrees, adj_list, graph_batch_size)
            else:
                graph_batch_ids = np.arange(graph_triplets.shape[0])

            graph_batch = np.array(graph_triplets)[graph_batch_ids]
            graph_percentage = float(general_settings["GraphSplitSize"])
            split_size = max(1, int(graph_percentage * graph_batch.shape[0]))
            graph_split_ids = np.random.choice(graph_batch_ids, size=split_size, replace=False)
            graph_split = np.array(graph_triplets)[graph_split_ids]

            transformed_targets = sampler.transform(target_batch)

            if "StoreEdgeData" in encoder_settings and encoder_settings["StoreEdgeData"] == "Yes":
                return (graph_split, graph_split_ids, transformed_targets[0], transformed_targets[1])
            return (graph_split, transformed_targets[0], transformed_targets[1])

        opp.set_sample_transform_function(t_func)

    model.preprocess(graph_triplets)
    model.register_for_test(graph_triplets)
    model.initialize_train()

    optimizer_weights = model.get_weights()
    optimizer_input = model.get_train_input_variables()
    loss = model.get_loss(mode="train") + model.get_regularization()
    print(optimizer_input)

    for add_op in model.get_additional_ops():
        opp.additional_ops.append(add_op)

    optimizer_parameters = opp.get_parametrization()

    model.session = tf.Session()
    optimizer = build_tensorflow(loss, optimizer_weights, optimizer_parameters, optimizer_input)
    optimizer.set_session(model.session)
    optimizer.fit(loss_triplets, validation_data=valid_triplets)


if __name__ == "__main__":
    main()
