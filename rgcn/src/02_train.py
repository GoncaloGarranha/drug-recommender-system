"""
R-GCN Training Script

Trains a Relational Graph Convolutional Network (R-GCN) encoder with a DistMult
decoder for drug-disease link prediction.

Usage:
    python -m rgcn.src.02_train --config rgcn/configs/baseline.yaml
"""

import argparse
import os
import time
import yaml
import numpy as np
import torch
import torch.nn.functional as F

from rgcn.src.utils.seed import set_seed
from rgcn.src.utils.io import load_processed_artifacts
from rgcn.src.models.rgcn_encoder import RGCNEncoder
from rgcn.src.models.decoders import DistMultDecoder


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    """Return current timestamp in YYYY-MM-DD_HH-MM-SS format."""
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def build_known_forward_triples(splits: dict, relation_id: int) -> set:
    """Build set of all known (drug, INDICATION, disease) triples."""
    known = set()
    for split_name in ["train", "valid", "test"]:
        split = splits[split_name]
        for head, _, tail in zip(split["src"], split["rel"], split["dst"]):
            known.add((int(head), int(relation_id), int(tail)))
    return known


@torch.no_grad()
def evaluate_disease_to_drug(
    encoder,
    decoder,
    graph,
    edge_types,
    eval_edges: dict,
    relation_id: int,
    known_triples: set,
    device,
    all_drugs_tensor,
    hits_k=(1, 3, 10),
):
    """
    Evaluate Disease → Drug ranking with filtered metrics.
    
    Given a disease, rank the correct drug among all candidate drugs.
    """
    encoder.eval()
    decoder.eval()

    node_embeddings = encoder(graph, edge_types)
    drug_ids = torch.tensor(eval_edges["src"], dtype=torch.long, device=device)
    disease_ids = torch.tensor(eval_edges["dst"], dtype=torch.long, device=device)

    candidate_drugs = all_drugs_tensor
    ranks = []
    hits = {k: 0 for k in hits_k}

    for i in range(len(drug_ids)):
        true_drug_id = drug_ids[i].item()
        disease_id = disease_ids[i].item()

        # Score all candidate drugs given the disease
        disease_emb = node_embeddings[disease_id].unsqueeze(0).repeat(len(candidate_drugs), 1)
        rel = decoder.rel_emb(torch.tensor([relation_id], device=device)).repeat(len(candidate_drugs), 1)
        drug_candidates = node_embeddings[candidate_drugs]
        scores = torch.sum(disease_emb * rel * drug_candidates, dim=1)

        # Filtered ranking: remove known drugs for this disease
        for idx, candidate_id in enumerate(candidate_drugs.tolist()):
            if candidate_id == true_drug_id:
                continue
            if (candidate_id, relation_id, disease_id) in known_triples:
                scores[idx] = -1e9

        true_score = scores[candidate_drugs == true_drug_id].item()
        num_better = torch.sum(scores > true_score).item()
        num_equal = torch.sum(scores == true_score).item()
        rank = 1 + num_better + (num_equal - 1) / 2.0

        ranks.append(rank)
        for k in hits_k:
            if rank <= k:
                hits[k] += 1

    ranks = np.array(ranks, dtype=np.float64)
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        **{f"hits@{k}": float(hits[k] / len(ranks)) for k in hits_k},
    }


def negative_sampling_disease(
    positive_drugs: torch.Tensor,
    relation_id: int,
    positive_diseases: torch.Tensor,
    k: int,
    device,
    all_drugs_tensor: torch.Tensor,
):
    """
    Create negative samples by corrupting the drug (head) during Disease→Drug training.
    
    For each (drug, INDICATION, disease) triple, create k negatives by replacing the drug.
    """
    batch_size = positive_drugs.shape[0]
    src_neg = all_drugs_tensor[torch.randint(0, len(all_drugs_tensor), (batch_size * k,), device=device)]
    rel_neg = torch.full((batch_size * k,), relation_id, dtype=torch.long, device=device)
    dst_neg = positive_diseases.repeat_interleave(k)
    return src_neg, rel_neg, dst_neg


def main() -> None:
    """Run the training loop and save the best checkpoint by validation MRR."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rgcn/configs/baseline.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    graph, edge_types, metadata, splits = load_processed_artifacts(cfg["data"]["processed_dir"])
    device = torch.device(cfg["training"].get("device", "cpu"))
    graph = graph.to(device)
    edge_types = edge_types.to(device)

    num_nodes = metadata["n_entities"]
    num_relations = metadata["n_relations"]
    relation_id = metadata["rel2id"].get("INDICATION")
    if relation_id is None:
        raise ValueError("Relation 'INDICATION' was not found in metadata.")

    all_drugs = sorted(
        set(splits["train"]["src"] + splits["valid"]["src"] + splits["test"]["src"])
    )
    all_drugs_tensor = torch.tensor(all_drugs, dtype=torch.long, device=device)

    encoder = RGCNEncoder(
        num_nodes,
        cfg["training"]["emb_dim"],
        num_relations,
        cfg["training"]["num_bases"],
        cfg["training"]["num_layers"],
        cfg["training"]["dropout"],
    ).to(device)
    decoder = DistMultDecoder(num_relations, cfg["training"]["emb_dim"]).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    train_drugs = torch.tensor(splits["train"]["src"], dtype=torch.long, device=device)
    train_diseases = torch.tensor(splits["train"]["dst"], dtype=torch.long, device=device)
    known_triples = build_known_forward_triples(splits, relation_id)

    run_dir = os.path.join(cfg["runs"]["out_dir"], f"{timestamp()}_disease_to_drug")
    ensure_dir(os.path.join(run_dir, "checkpoints"))

    print(f"Targeting: Disease → Drug | Device: {device}")
    best_mrr = -1.0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        encoder.train()
        decoder.train()

        node_embeddings = encoder(graph, edge_types)
        positive_scores = decoder.score(
            node_embeddings[train_drugs],
            decoder.rel_emb(torch.full_like(train_drugs, relation_id)),
            node_embeddings[train_diseases],
        )

        src_neg, rel_neg, dst_neg = negative_sampling_disease(
            train_drugs,
            relation_id,
            train_diseases,
            cfg["training"]["neg_k"],
            device,
            all_drugs_tensor,
        )
        negative_scores = decoder(node_embeddings, src_neg, rel_neg, dst_neg)

        loss = F.binary_cross_entropy_with_logits(
            positive_scores, torch.ones_like(positive_scores)
        ) + F.binary_cross_entropy_with_logits(
            negative_scores, torch.zeros_like(negative_scores)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % cfg["training"]["eval_every"] == 0 or epoch == cfg["training"]["epochs"]:
            metrics = evaluate_disease_to_drug(
                encoder,
                decoder,
                graph,
                edge_types,
                splits["valid"],
                relation_id,
                known_triples,
                device,
                all_drugs_tensor,
                hits_k=cfg["training"]["hits_k"],
            )
            print(
                f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
                f"Valid MRR (Disease→Drug): {metrics['mrr']:.4f}"
            )
            if metrics["mrr"] > best_mrr:
                best_mrr = metrics["mrr"]
                torch.save(
                    {"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "config": cfg},
                    os.path.join(run_dir, "checkpoints", "best.pt"),
                )

    print(f"Training complete. Best Valid MRR: {best_mrr:.4f}")


if __name__ == "__main__":
    main()