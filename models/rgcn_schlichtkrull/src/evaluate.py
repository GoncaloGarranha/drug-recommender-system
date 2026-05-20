from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.shared.config import load_config
from models.shared.runs import find_latest_run
from models.rgcn.src.utils.seed import set_seed

from .data import build_encoder_graph, known_positives, load_link_prediction_data
from .model import DistMult, SchlichtkrullRGCN
from .train import filtered_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Schlichtkrull-style R-GCN link prediction model")
    parser.add_argument("--runs_dir", default="models/rgcn_schlichtkrull/runs")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    args = parser.parse_args()

    run_dir = Path(args.run_dir if args.run_dir else find_latest_run(args.runs_dir))
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg["training"].get("device", "cpu"))

    data = load_link_prediction_data(cfg)
    entity2id = data["entity2id"]
    rel2id = data["rel2id"]
    train = data["train"]
    valid = data["valid"]
    test = data["test"]
    eval_triples = valid if args.split == "valid" else test

    num_nodes = len(entity2id)
    num_relations = len(rel2id)
    add_inverse = bool(cfg.get("model", {}).get("add_inverse_edges", True))
    encoder_relations = num_relations * 2 if add_inverse else num_relations
    graph, edge_types = build_encoder_graph(train, num_nodes, num_relations, device, add_inverse_edges=add_inverse)
    candidate_ids = torch.tensor(sorted(entity2id.values()), dtype=torch.long, device=device)
    known = known_positives(train + valid + test)

    training_cfg = cfg["training"]
    encoder = SchlichtkrullRGCN(
        num_nodes=num_nodes,
        h_dim=int(training_cfg["emb_dim"]),
        num_rels=encoder_relations,
        num_bases=int(training_cfg["num_bases"]),
        num_layers=int(training_cfg["num_layers"]),
        dropout=0.0,
        regularizer=cfg.get("model", {}).get("regularizer", "basis"),
    ).to(device)
    decoder = DistMult(num_relations, int(training_cfg["emb_dim"])).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    hits_k = tuple(int(k) for k in training_cfg.get("hits_k", [1, 3, 10]))
    results = filtered_eval(encoder, decoder, graph, edge_types, eval_triples, known, candidate_ids, hits_k, device)
    for key, value in results.items():
        print(f"{key}: {value:.6f}")
    output_path = run_dir / f"metrics_{args.split}.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
