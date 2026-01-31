import argparse
import os
import csv
import glob
import torch
from rgcn.src.utils.io import load_processed_artifacts
from rgcn.src.models.rgcn_encoder import RGCNEncoder
from rgcn.src.models.decoders import DistMultDecoder

def find_latest_run(runs_dir: str):
    candidates = [p for p in glob.glob(os.path.join(runs_dir, "*")) if os.path.isdir(p)]
    if not candidates:
        raise FileNotFoundError("Nenhum treino encontrado em rgcn/runs")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def load_entity_labels(csv_path: str):
    mapping = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f); next(reader)
            for row in reader:
                if len(row) >= 2: mapping[row[0]] = row[1]
    return mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="Nome do Fármaco (ex: Albuterol)")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    # 1. Carregar o modelo do último treino (Foco: Drug -> Disease)
    run_dir = find_latest_run("rgcn/runs")
    ckpt = torch.load(os.path.join(run_dir, "checkpoints", "best.pt"), map_location="cpu")
    cfg = ckpt["config"]
    print(f"Modelo carregado de: {run_dir}")

    # 2. Carregar Grafo e Nomes
    g, etype, meta, splits = load_processed_artifacts(cfg["data"]["processed_dir"])
    labels = load_entity_labels("kg_constructor/data/output/entity_names.csv")
    ent_to_idx = meta["entity2id"]
    idx_to_ent = {v: k for k, v in ent_to_idx.items()}

    # IMPORTANTE: Agora usamos a relação direta (INDICATION)
    rel_id = meta["rel2id"]["INDICATION"]

    # 3. Filtrar apenas IDs que são DOENÇAS (para o ranking ser justo)
    # Pegamos nos nós que aparecem como 'dst' nos splits
    disease_nodes = list(set(splits["train"]["dst"] + splits["valid"]["dst"] + splits["test"]["dst"]))
    disease_tensor = torch.tensor(disease_nodes)

    # 4. Procurar o Fármaco (Query) - Melhor matching
    query = args.query.lower()
    # Só considerar entidades que aparecem como src nos splits (drugs reais)
    drug_ids = set(splits["train"]["src"] + splits["valid"]["src"] + splits["test"]["src"])
    # Map idx->eid
    idx_to_eid = {v: k for k, v in ent_to_idx.items()}
    drug_eids = [idx_to_eid[idx] for idx in drug_ids if idx in idx_to_eid]

    # 1. Match exato (case-insensitive)
    exact_matches = [eid for eid in drug_eids if labels.get(eid, '').lower() == query or eid.lower() == query]
    # 2. Começa por
    starts_matches = [eid for eid in drug_eids if labels.get(eid, '').lower().startswith(query) or eid.lower().startswith(query)]
    # 3. Substring
    substr_matches = [eid for eid in drug_eids if query in labels.get(eid, '').lower() or query in eid.lower()]

    # Prioridade: exato > começa por > substring
    candidates = exact_matches or starts_matches or substr_matches
    if candidates:
        # Escolher o mais curto
        best_eid = min(candidates, key=lambda eid: len(labels.get(eid, eid)))
        drug_node = ent_to_idx[best_eid]
        print(f"Fármaco detetado: {labels.get(best_eid, best_eid)} (ID: {best_eid})")
    else:
        drug_node = None

    if drug_node is None:
        print(f"Erro: O fármaco '{args.query}' não foi encontrado no Grafo.")
        return

    # 5. Inferência
    encoder = RGCNEncoder(meta["n_entities"], cfg["training"]["emb_dim"], meta["n_relations"], 
                          cfg["training"]["num_bases"], cfg["training"]["num_layers"], 0.0)
    decoder = DistMultDecoder(meta["n_relations"], cfg["training"]["emb_dim"])
    encoder.load_state_dict(ckpt["encoder"]); decoder.load_state_dict(ckpt["decoder"])
    encoder.eval(); decoder.eval()

    with torch.no_grad():
        emb = encoder(g, g.edata["etype"])
        # Score: (Drug_Query, indicated_for, TODAS_AS_DOENÇAS)
        h = emb[drug_node].unsqueeze(0).repeat(len(disease_nodes), 1)
        r = decoder.rel_emb(torch.tensor([rel_id]).repeat(len(disease_nodes)))
        t = emb[disease_tensor]
        
        # Scoring DistMult: sum(h * r * t)
        scores = torch.sum(h * r * t, dim=1)
        topk_scores, topk_indices = torch.topk(scores, k=min(args.topk, len(scores)))

    # 6. Mostrar Resultados
    print(f"\nTop-{args.topk} Indicações Prováveis (Doenças):")
    print("-" * 65)
    for i in range(len(topk_indices)):
        dis_idx = disease_nodes[topk_indices[i].item()]
        dis_eid = idx_to_ent[dis_idx]
        name = labels.get(dis_eid, dis_eid)
        print(f"{i+1:02d}. {name:45} | Score: {topk_scores[i].item():.4f}")

if __name__ == "__main__":
    main()