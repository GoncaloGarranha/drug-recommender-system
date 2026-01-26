import torch
import os
import json
from data_loader import KGDataLoader
from model import RGCNModel
from entity_mapper import EntityMapper

# Configurações
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../kg_constructor/data/processed")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "../checkpoints/rgcn_model.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 100

def get_drug_id(query, clinical_labels):
    # Limpa espaços e mete tudo em maiúsculas para comparar
    query = query.upper().strip()
    
    # 1. Tentar correspondência exata (ignora case)
    for model_id, name in clinical_labels.items():
        if query == name.upper().strip():
            return int(model_id), name
            
    # 2. Tentar correspondência parcial (ex: "RITO" encontra "RITONAVIR")
    for model_id, name in clinical_labels.items():
        if query in name.upper():
            return int(model_id), name
            
    return None, None

def run_inference(drug_model_id, official_name, k=20):
    loader = KGDataLoader(DATA_PATH)
    mapper = EntityMapper(DATA_PATH)
    
    # Carregar Modelo
    model = RGCNModel(loader.num_nodes, loader.num_relations * 2, EMBED_DIM).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    h_train, r_train, t_train = loader.load_triples('train.txt')
    edge_index, edge_type = loader.get_graph_structure(h_train, r_train, t_train)
    
    with torch.no_grad():
        nodes_emb = model.encode(edge_index.to(DEVICE), edge_type.to(DEVICE))
        
        # Consulta: (Fármaco, Relação [0], ?)
        head_idx = torch.tensor([drug_model_id]).to(DEVICE).expand(loader.num_nodes)
        rel_idx = torch.tensor([0]).to(DEVICE).expand(loader.num_nodes)
        all_tails = torch.arange(loader.num_nodes).to(DEVICE)
        
        scores = model.decode(nodes_emb, head_idx, rel_idx, all_tails)
        top_scores, top_indices = torch.topk(scores, k=min(20, loader.num_nodes))

    print(f"\n" + "="*85)
    print(f"🏛️  INFERÊNCIA CLÍNICA PARA: {official_name}")
    print("="*85)
    
    count = 0
    for idx_tensor, score_tensor in zip(top_indices, top_scores):
        target_id = idx_tensor.item()
        if target_id == drug_model_id: continue
        
        prob = torch.sigmoid(score_tensor).item()
        clean_name = mapper.map_id_to_name(target_id)
        tag = mapper.get_clinical_context(prob)
        
        print(f"{count+1:2d}. {clean_name:.<55} Prob: {prob:.4f} | {tag}")
        count += 1
        if count >= k: break

if __name__ == "__main__":
    mapper = EntityMapper(DATA_PATH)
    print("\n--- R-GCN Bio-Inference Engine ---")
    u_query = input("💊 Introduza o nome do fármaco (ex: ALBUTEROL): ")
    
    mid, o_name = get_drug_id(u_query, mapper.clinical_labels)
    if mid is not None:
        run_inference(mid, o_name)
    else:
        print(f"❌ '{u_query}' não encontrado.")