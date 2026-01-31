import argparse
import json
import os
import glob
import numpy as np
import torch
from typing import Dict, List, Set, Tuple

from rgcn.src.utils.io import load_processed_artifacts
from rgcn.src.models.rgcn_encoder import RGCNEncoder
from rgcn.src.models.decoders import DistMultDecoder
from rgcn.src.utils.seed import set_seed

def find_latest_run(runs_dir: str) -> str:
    candidates = [p for p in glob.glob(os.path.join(runs_dir, "*")) if os.path.isdir(p)]
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def build_known_forward_triples(splits: dict, rel_fwd_id: int) -> Set[Tuple[int, int, int]]:
    """
    Cria o conjunto de todos os triplos conhecidos (Drug, indicated_for, Disease).
    """
    known_fwd = set()
    for split_name in ["train", "valid", "test"]:
        s = splits[split_name]
        for h_drug, r_fwd, t_dis in zip(s["src"], s["rel"], s["dst"]):
            known_fwd.add((int(h_drug), int(rel_fwd_id), int(t_dis)))
    return known_fwd

def infer_typed_partitions(splits: dict) -> Tuple[List[int], List[int]]:
    drug_ids = set()
    disease_ids = set()
    for split_name in ["train", "valid", "test"]:
        s = splits[split_name]
        drug_ids.update(int(x) for x in s["src"])
        disease_ids.update(int(x) for x in s["dst"])
    return sorted(list(drug_ids)), sorted(list(disease_ids))

@torch.no_grad()
def evaluate_indications(node_emb, decoder, edges, candidate_diseases, known_fwd, hits_k, device):
    """
    Avalia a predição de Indicações: ranqueia a doença correta face a todas as outras doenças.
    """
    drug_ids = [int(x) for x in edges["src"]]
    true_dis_ids = [int(x) for x in edges["dst"]]
    rel_fwd_id = int(edges["rel"][0]) 

    cand_tensor = torch.tensor(candidate_diseases, dtype=torch.long, device=device)
    cand_emb = node_emb[cand_tensor]

    ranks = []
    hits = {k: 0 for k in hits_k}

    for drug_h, dis_t_true in zip(drug_ids, true_dis_ids):
        h_emb = node_emb[drug_h]
        r_emb = decoder.rel_emb(torch.tensor([rel_fwd_id], device=device)).squeeze(0)

        # Scoring DistMult: <h * r, t>
        query_vec = (h_emb * r_emb).unsqueeze(0)
        scores = torch.matmul(query_vec, cand_emb.T).squeeze(0)

        # Filtered Ranking: ignorar outras doenças conhecidas para este fármaco
        for j, dis_cand in enumerate(candidate_diseases):
            if dis_cand == dis_t_true: continue
            if (drug_h, rel_fwd_id, dis_cand) in known_fwd:
                scores[j] = -1e9

        try:
            true_pos = candidate_diseases.index(dis_t_true)
            order = torch.argsort(scores, descending=True)
            rank = int((order == true_pos).nonzero(as_tuple=False)[0].item()) + 1
            ranks.append(rank)
            for k in hits_k:
                if rank <= k: hits[k] += 1
        except ValueError: continue

    ranks = np.array(ranks)
    return {"mrr": float(np.mean(1.0/ranks)), **{f"hits@{k}": float(hits[k]/len(ranks)) for k in hits_k}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="kg_constructor/data/processed")
    ap.add_argument("--runs_dir", default="rgcn/runs")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    args = ap.parse_args()

    # 1. Carregar Dados e Modelo
    g, etype, meta, splits = load_processed_artifacts(args.processed_dir)
    run_dir = find_latest_run(args.runs_dir)
    ckpt = torch.load(os.path.join(run_dir, "checkpoints", "best.pt"), map_location="cpu")
    cfg = ckpt["config"]
    device = torch.device("cpu")
    set_seed(cfg["seed"])
    encoder = RGCNEncoder(meta["n_entities"], cfg["training"]["emb_dim"], meta["n_relations"], cfg["training"]["num_bases"], cfg["training"]["num_layers"], 0.0).to(device)
    decoder = DistMultDecoder(meta["n_relations"], cfg["training"]["emb_dim"]).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    # 2. Configurar Avaliação
    node_emb = encoder(g.to(device), etype.to(device))
    drug_ids, disease_ids = infer_typed_partitions(splits)
    # IMPORTANTE: Agora usamos a relação direta
    rel_fwd = meta["rel2id"]["INDICATION"]
    known_fwd = build_known_forward_triples(splits, rel_fwd)

    # 3. Preparar Task: (Drug, indicated_for, ?)
    eval_edges = {
        "src": splits[args.split]["src"], # Drugs
        "rel": [rel_fwd] * len(splits[args.split]["rel"]),
        "dst": splits[args.split]["dst"]  # Diseases
    }

    # 4. Avaliar
    print(f"\nEvaluating Drug Repurposing (Drug -> Disease) on '{args.split}' split...")
    results = evaluate_indications(node_emb, decoder, eval_edges, disease_ids, known_fwd, cfg["training"]["hits_k"], device)

    # 5. Output
    print("DEBUG candidates:", candidate_diseases.numel())
    print("DEBUG unique dst in valid:", len(set(eval_edges["dst"])))
    print("-" * 60)
    print(f"RESULTS FOR TASK: DRUG -> DISEASE")
    print("-" * 60)
    print(f"Mean Reciprocal Rank (MRR): {results['mrr']:.4f}")
    for k in cfg["training"]["hits_k"]:
        print(f"Hits@{k}: {results[f'hits@{k}']:.4f}")
    print("-" * 60)

    with open(os.path.join(run_dir, f"final_indication_metrics_{args.split}.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()