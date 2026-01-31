import argparse
import os
import time
import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F

from rgcn.src.utils.seed import set_seed
from rgcn.src.utils.io import load_processed_artifacts
from rgcn.src.models.rgcn_encoder import RGCNEncoder
from rgcn.src.models.decoders import DistMultDecoder

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")

def build_known_forward_triples(splits: dict, rel_id: int) -> set:
    """
    Build a set of all known FORWARD triples: (drug, indicated_for, disease).
    Used for filtered evaluation in Drug -> Disease task.
    """
    known = set()
    for split_name in ["train", "valid", "test"]:
        s = splits[split_name]
        # Em splits: src=Drug, dst=Disease, rel=indicated_for
        for h, r, t in zip(s["src"], s["rel"], s["dst"]):
            known.add((int(h), int(rel_id), int(t)))
    return known

@torch.no_grad()
def evaluate_drug_to_disease(encoder, decoder, g, etype, eval_edges: dict, rel_id: int, known_triples: set, num_nodes: int, device, all_diseases_t, hits_k=(1, 3, 10)):
    """
    Evaluates the model on Drug -> Disease prediction.
    Ranks all possible diseases for a given drug.
    """
    encoder.eval()
    decoder.eval()
    node_emb = encoder(g, etype)
    drug_ids = torch.tensor(eval_edges["src"], dtype=torch.long, device=device)
    true_dis_ids = torch.tensor(eval_edges["dst"], dtype=torch.long, device=device)
    ranks = []
    hits = {k: 0 for k in hits_k}
    candidate_diseases = all_diseases_t
    for i in range(drug_ids.shape[0]):
        drug_idx = drug_ids[i].item()
        true_dis_idx = true_dis_ids[i].item()
        h_rep = node_emb[drug_idx].unsqueeze(0).repeat(len(candidate_diseases), 1)
        r_rep = decoder.rel_emb(torch.tensor([rel_id], device=device)).repeat(len(candidate_diseases), 1)
        t_rep = node_emb[candidate_diseases]
        scores = torch.sum(h_rep * r_rep * t_rep, dim=1)
        for idx, cand_idx in enumerate(candidate_diseases.tolist()):
            if cand_idx == true_dis_idx: continue
            if (drug_idx, rel_id, cand_idx) in known_triples:
                scores[idx] = -1e9
        # --- NOVA LÓGICA DE RANK (JUSTA) ---
        true_score = scores[candidate_diseases == true_dis_idx].item()
        num_better = torch.sum(scores > true_score).item()
        num_equal = torch.sum(scores == true_score).item()
        # Rank médio em caso de empate (Tie-breaking)
        rank = 1 + num_better + (num_equal - 1) / 2.0
        ranks.append(rank)
        for k in hits_k:
            if rank <= k: hits[k] += 1
    ranks = np.array(ranks, dtype=np.float64)
    return {"mrr": float(np.mean(1.0 / ranks)), **{f"hits@{k}": float(hits[k]/len(ranks)) for k in hits_k}}

def negative_sampling_disease(drug_pos: torch.Tensor, rel_id: int, dis_pos: torch.Tensor, num_nodes: int, k: int, device, all_diseases_t) -> tuple:
    """
    Corrupts the DISEASE (target) to create negative examples for Drug -> Disease.
    """
    batch_size = drug_pos.shape[0]
    src_neg = drug_pos.repeat_interleave(k)
    rel_neg = torch.full((batch_size * k,), rel_id, dtype=torch.long, device=device)
    disease_nodes = all_diseases_t
    if len(disease_nodes) < 2:
        raise ValueError("Conjunto de doenças para negative sampling é muito pequeno!")
    idxs = torch.randint(0, len(disease_nodes), (batch_size * k,), device=device)
    dst_neg = disease_nodes[idxs]
    return src_neg, rel_neg, dst_neg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rgcn/configs/baseline.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    # Loader
    g, etype, meta, splits = load_processed_artifacts(cfg["data"]["processed_dir"])
    device = torch.device(cfg["training"].get("device", "cpu"))
    g = g.to(device)
    etype = etype.to(device)
    print("[DEBUG] Grafo carregado: num_nodes=", g.num_nodes(), "num_edges=", g.num_edges())
    print("[DEBUG] Device do grafo:", g.device)
    print("[DEBUG] Device do etype:", etype.device)
    print("[INSPECT] g.edata.keys() após carregamento:", list(g.edata.keys()))
    num_nodes, num_rels = meta["n_entities"], meta["n_relations"]
    rel_id = meta["rel2id"].get("INDICATION")
    # DEBUG CRÍTICO
    print(f"DEBUG: Rel ID para INDICATION: {rel_id}")
    print(f"DEBUG: Mapeamento de relações: {meta.get('rel2id')}")
    if rel_id is None:
        raise ValueError("ERRO: A relação 'INDICATION' não foi encontrada no meta!")
    all_diseases = sorted(set(splits["train"]["dst"] + splits["valid"]["dst"] + splits["test"]["dst"]))
    all_diseases_t = torch.tensor(all_diseases, dtype=torch.long, device=device)
    # Forçar dropout a 0.0 para debug
    cfg["training"]["dropout"] = 0.0
    encoder = RGCNEncoder(num_nodes, cfg["training"]["emb_dim"], num_rels, cfg["training"]["num_bases"], cfg["training"]["num_layers"], cfg["training"]["dropout"]).to(device)
    decoder = DistMultDecoder(num_rels, cfg["training"]["emb_dim"]).to(device)
    # Debug: Devices dos modelos
    print("[DEBUG] Device dos embeddings:", encoder.emb.weight.device)
    # Debug: Output da primeira camada do encoder
    encoder.eval()
    with torch.no_grad():
        x = encoder.emb.weight
        print("[DEBUG] Embeddings antes do forward (primeiros 5):", x[:5].detach().cpu().numpy())
        x1 = encoder.layers[0](g, x, etype)
        print("[DEBUG] Output primeira camada (primeiros 5):", x1[:5].detach().cpu().numpy())
    encoder.train()
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    train_drug = torch.tensor(splits["train"]["src"], dtype=torch.long, device=device)
    train_disease = torch.tensor(splits["train"]["dst"], dtype=torch.long, device=device)
    known_forward = build_known_forward_triples(splits, rel_id)
    run_dir = os.path.join(cfg["runs"]["out_dir"], f"{timestamp()}_drug_to_disease")
    ensure_dir(os.path.join(run_dir, "checkpoints"))
    print(f"Targeting: Drug -> Disease | Device: {device}")
    best_mrr = -1.0
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        encoder.train(); decoder.train()
        node_emb = encoder(g, etype)
        # Debug: Output do encoder imediatamente antes dos scores
        if epoch == 1:
            print("[DEBUG] Output do encoder antes dos scores (primeiros 5):", node_emb[:5].detach().cpu().numpy())
        # Debug: Embeddings
        if epoch == 1:
            print("[DEBUG] Embeddings (primeiros 5):", node_emb[:5].detach().cpu().numpy())
        pos_scores = decoder.score(node_emb[train_drug], decoder.rel_emb(torch.full_like(train_drug, rel_id)), node_emb[train_disease])
        # Debug: Scores positivos
        if epoch == 1:
            print("[DEBUG] Pos scores (primeiros 10):", pos_scores[:10].detach().cpu().numpy())
        src_neg, rel_neg, dst_neg = negative_sampling_disease(train_drug, rel_id, train_disease, num_nodes, cfg["training"]["neg_k"], device, all_diseases_t)
        # Debug: Negative sampling
        if epoch == 1:
            print("[DEBUG] Neg samples src (primeiros 10):", src_neg[:10].detach().cpu().numpy())
            print("[DEBUG] Neg samples dst (primeiros 10):", dst_neg[:10].detach().cpu().numpy())
        neg_scores = decoder(node_emb, src_neg, rel_neg, dst_neg)
        # Debug: Scores negativos
        if epoch == 1:
            print("[DEBUG] Neg scores (primeiros 10):", neg_scores[:10].detach().cpu().numpy())
        loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) + F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        if epoch == 1:
            print("[DEBUG] Loss inicial:", loss.item())
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % cfg["training"]["eval_every"] == 0 or epoch == cfg["training"]["epochs"]:
            metrics = evaluate_drug_to_disease(encoder, decoder, g, etype, splits["valid"], rel_id, known_forward, num_nodes, device, all_diseases_t)
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Valid MRR (Drug->Dis): {metrics['mrr']:.4f}")
            if metrics["mrr"] > best_mrr:
                best_mrr = metrics["mrr"]
                torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "config": cfg}, os.path.join(run_dir, "checkpoints", "best.pt"))
    print(f"Training complete. Best Valid MRR: {best_mrr:.4f}")
    # Estatísticas do grafo e features
    print("[INSPECT] train_g.edata['etype'] unique:", set(g.edata['etype'].cpu().numpy()))
    print("[INSPECT] train_g.edata['etype'] min:", g.edata['etype'].min().item(), "max:", g.edata['etype'].max().item())
    print("[INSPECT] train_g.edata['norm'] stats: min:", g.edata['norm'].min().item(), "max:", g.edata['norm'].max().item(), "mean:", g.edata['norm'].mean().item())
    print("[INSPECT] train_g.number_of_nodes():", g.number_of_nodes())
    print("[INSPECT] train_g.number_of_edges():", g.number_of_edges())
    print("[INSPECT] Embeddings shape:", encoder.emb.weight.shape)
    print("[INSPECT] Embeddings min:", encoder.emb.weight.min().item(), "max:", encoder.emb.weight.max().item(), "mean:", encoder.emb.weight.mean().item())
    # Estatísticas detalhadas dos atributos do grafo
    import numpy as np
    norm = g.edata['norm'].cpu().numpy()
    etype = g.edata['etype'].cpu().numpy()
    print("[DEBUG] g.edata['norm'] shape:", norm.shape, "dtype:", norm.dtype)
    print("[DEBUG] g.edata['norm'] min:", np.min(norm), "max:", np.max(norm), "mean:", np.mean(norm), "std:", np.std(norm))
    print("[DEBUG] g.edata['norm'] valores únicos:", np.unique(norm))
    print("[DEBUG] g.edata['etype'] shape:", etype.shape, "dtype:", etype.dtype)
    print("[DEBUG] g.edata['etype'] min:", np.min(etype), "max:", np.max(etype))
    print("[DEBUG] g.edata['etype'] valores únicos:", np.unique(etype))

if __name__ == "__main__":
    main()