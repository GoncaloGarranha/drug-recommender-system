import argparse
import os
import csv
import glob
import json
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
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if len(row) >= 2:
                    mapping[row[0]] = row[1]
    return mapping


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=False, help="Nome do fármaco (ex: Albuterol)")
    ap.add_argument("--entity_id", required=False, help="ID exato do fármaco no grafo (ex: DRUG:ALBUTEROL SULFATE)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--show_ids", action="store_true", help="Mostrar IDs (recomendado para debug/auditoria)")
    ap.add_argument("--json_out", default=None, help="Guardar resultados em JSON (ex: out.json)")
    args = ap.parse_args()

    # 1) Carregar último treino
    run_dir = find_latest_run("rgcn/runs")
    ckpt = torch.load(os.path.join(run_dir, "checkpoints", "best.pt"), map_location="cpu")
    cfg = ckpt["config"]
    print(f"Modelo carregado de: {run_dir}")

    # 2) Carregar grafo e labels
    g, etype, meta, splits = load_processed_artifacts(cfg["data"]["processed_dir"])
    labels = load_entity_labels("kg_constructor/data/output/entity_names.csv")

    ent_to_idx = meta["entity2id"]
    idx_to_ent = meta["id2entity"]

    if "INDICATION" not in meta["rel2id"]:
        raise KeyError("Relação 'INDICATION' não existe no rel2id.")
    rel_id = meta["rel2id"]["INDICATION"]

    # 3) Doenças candidatas (dst de train/valid/test para ranking consistente)
    disease_nodes = list(set(splits["train"]["dst"] + splits["valid"]["dst"] + splits["test"]["dst"]))
    disease_tensor = torch.tensor(disease_nodes)

    # 4) Resolver o fármaco (por ID exato ou por query)
    drug_node = None
    drug_eid = None

    if args.entity_id:
        if args.entity_id not in ent_to_idx:
            print(f"Erro: entity_id '{args.entity_id}' não existe no grafo.")
            return
        drug_eid = args.entity_id
        drug_node = ent_to_idx[drug_eid]
        print(f"Fármaco selecionado por ID: {labels.get(drug_eid, drug_eid)} (ID: {drug_eid})")

    else:
        if not args.query:
            print("Erro: tens de fornecer --query ou --entity_id.")
            return

        query = args.query.lower()

        # Considerar só entidades que aparecem como src nos splits (drugs reais)
        drug_ids = set(splits["train"]["src"] + splits["valid"]["src"] + splits["test"]["src"])
        drug_eids = [idx_to_ent[idx] for idx in drug_ids if idx in idx_to_ent]

        def lab(eid: str) -> str:
            return labels.get(eid, eid)

        exact = [eid for eid in drug_eids if lab(eid).lower() == query or eid.lower() == query]
        starts = [eid for eid in drug_eids if lab(eid).lower().startswith(query) or eid.lower().startswith(query)]
        subs  = [eid for eid in drug_eids if query in lab(eid).lower() or query in eid.lower()]

        candidates = exact or starts or subs
        if not candidates:
            print(f"Erro: o fármaco '{args.query}' não foi encontrado no grafo.")
            return

        # Se houver vários, mostrar sugestões (user friendly)
        # Ordenar por “melhor”: match mais curto no label
        candidates = sorted(set(candidates), key=lambda eid: len(lab(eid)))

        if len(candidates) > 1:
            print("\nForam encontrados vários fármacos possíveis. Sugestões:")
            for i, eid in enumerate(candidates[:5], start=1):
                print(f"  {i}) {lab(eid)} (ID: {eid})")
            print("Dica: usa --entity_id para escolher exatamente o pretendido.\n")

        drug_eid = candidates[0]
        drug_node = ent_to_idx[drug_eid]
        print(f"Fármaco detetado: {labels.get(drug_eid, drug_eid)} (ID: {drug_eid})")

    # 5) “Known vs Novel”: indicações já presentes no grafo de treino
    known_indications = set()
    tg = splits["train_graph"]  # edges do grafo de treino (inclui também não-prediction edges)
    for s, r, t in zip(tg["src"], tg["rel"], tg["dst"]):
        if s == drug_node and r == rel_id:
            known_indications.add(t)  # t é node id (int)

    # 6) Inferência
    encoder = RGCNEncoder(
        meta["n_entities"],
        cfg["training"]["emb_dim"],
        meta["n_relations"],
        cfg["training"]["num_bases"],
        cfg["training"]["num_layers"],
        0.0,
    )
    decoder = DistMultDecoder(meta["n_relations"], cfg["training"]["emb_dim"])
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        emb = encoder(g, g.edata["etype"])

        h = emb[drug_node].unsqueeze(0).repeat(len(disease_nodes), 1)
        r = decoder.rel_emb(torch.tensor([rel_id]).repeat(len(disease_nodes)))
        t = emb[disease_tensor]

        scores = torch.sum(h * r * t, dim=1)  # DistMult raw score
        probs = sigmoid(scores)               # interpretável (não é calibrado, mas ajuda)

        topk = min(args.topk, len(scores))
        topk_scores, topk_idx = torch.topk(scores, k=topk)
        topk_probs = probs[topk_idx]

    # 7) Mostrar resultados (mais user friendly)
    print(f"\nTop-{topk} Indicações Prováveis (Drug → Disease):")
    print("-" * 92)
    print(f"{'RK':>2}  {'Doença':40}  {'DOID':14}  {'Score':>8}  {'Sigmoid':>8}  {'Estado':>7}")
    print("-" * 92)

    results = []
    for rank in range(topk):
        dis_node = disease_nodes[topk_idx[rank].item()]
        dis_eid = idx_to_ent[dis_node]
        dis_name = labels.get(dis_eid, dis_eid)

        status = "KNOWN" if dis_node in known_indications else "CANDIDATE"

        doid_str = dis_eid if dis_eid.startswith("DOID:") else ("-" if not args.show_ids else dis_eid)

        print(
            f"{rank+1:02d}  {dis_name[:40]:40}  {doid_str:14}  {topk_scores[rank].item():8.4f}  {topk_probs[rank].item():8.4f}  {status:>7}"
        )

        results.append({
            "rank": rank + 1,
            "drug": {"id": drug_eid, "name": labels.get(drug_eid, drug_eid)},
            "disease": {"id": dis_eid, "name": dis_name},
            "score": float(topk_scores[rank].item()),
            "sigmoid": float(topk_probs[rank].item()),
            "status": status,
        })

    # 8) JSON opcional
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Resultados guardados em: {args.json_out}")


if __name__ == "__main__":
    main()
