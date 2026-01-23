import torch
import os
from data_loader import KGDataLoader
from model import RGCNModel
from entity_mapper import EntityMapper  # <--- A nossa nova peça do puzzle

# --- CONFIGURAÇÕES DINÂMICAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../kg_constructor/data/processed")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "../checkpoints/rgcn_model.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 100

def run_inference(drug_query, rel_name="indicated_for", k=10):
    # Inicializar utilitários
    loader = KGDataLoader(DATA_PATH)
    mapper = EntityMapper(DATA_PATH)
    
    ent2id = {v: k for k, v in loader.entities_dict.items()}
    rel2id = {v: k for k, v in loader.relations_dict.items()}
    
    if drug_query not in ent2id:
        print(f"❌ Erro: '{drug_query}' não existe no Grafo.")
        return

    # Obter ID da relação de forma robusta
    rel_id = next((i for name, i in rel2id.items() if rel_name.lower() in name.lower()), 5)

    # Carregar Modelo e Pesos
    model = RGCNModel(loader.num_nodes, loader.num_relations * 2, EMBED_DIM).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # Preparar Grafo para a Passagem Forward
    h_train, r_train, t_train = loader.load_triples('train.txt')
    edge_index, edge_type = loader.get_graph_structure(h_train, r_train, t_train)
    
    with torch.no_grad():
        nodes_emb = model.encode(edge_index.to(DEVICE), edge_type.to(DEVICE))
        
        # Definir Consulta: (Fármaco, Relação, ?)
        head_idx = torch.tensor([ent2id[drug_query]]).to(DEVICE).expand(loader.num_nodes)
        rel_idx = torch.tensor([rel_id]).to(DEVICE).expand(loader.num_nodes)
        all_tails = torch.arange(loader.num_nodes).to(DEVICE)
        
        scores = model.decode(nodes_emb, head_idx, rel_idx, all_tails)
        top_scores, top_indices = torch.topk(scores, k=min(100, loader.num_nodes))

    print(f"\n" + "="*85)
    print(f"🏛️  SISTEMA DE INFERÊNCIA CLÍNICA - RESULTADOS PARA: {drug_query}")
    print("="*85)
    
    count = 0
    for idx_tensor, score_tensor in zip(top_indices, top_scores):
        raw_id = loader.entities_dict[idx_tensor.item()]
        
        if raw_id == drug_query: continue # Ignorar self-loops
        
        prob = torch.sigmoid(score_tensor).item()
        clean_name = mapper.map_id_to_name(raw_id)
        tag = mapper.get_clinical_context(prob)
        
        print(f"{count+1:2d}. {clean_name:.<55} Prob: {prob:.4f} | {tag}")
        count += 1
        if count >= k: break

if __name__ == "__main__":
    print("--- R-GCN Bio-Inference Engine ---")
    query = input("Introduza o nome do fármaco: ").strip().upper()
    run_inference(query)