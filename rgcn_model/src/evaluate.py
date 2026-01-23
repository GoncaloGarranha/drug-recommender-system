import torch
from data_loader import KGDataLoader
from model import RGCNModel
import os

# Configurações
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = "kg_constructor/data/processed"
CHECKPOINT_PATH = "../checkpoints/rgcn_model.pth"
EMBED_DIM = 100

def evaluate():
    # 1. Carregar Dados
    loader = KGDataLoader(DATA_PATH)
    h_test, r_test, t_test = loader.load_triples('test.txt')
    h_train, r_train, t_train = loader.load_triples('train.txt')
    
    # Precisamos da estrutura de treino para o Encoder gerar os embeddings
    edge_index, edge_type = loader.get_graph_structure(h_train, r_train, t_train)
    edge_index, edge_type = edge_index.to(DEVICE), edge_type.to(DEVICE)

    # 2. Carregar Modelo
    model = RGCNModel(loader.num_nodes, loader.num_relations * 2, EMBED_DIM).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    print(f"🔬 A avaliar {len(h_test)} triplos de teste...")

    mrr = 0
    hits_10 = 0
    hits_3 = 0
    hits_1 = 0

    with torch.no_grad():
        # Gerar embeddings globais
        nodes_emb = model.encode(edge_index, edge_type)
        
        # Para cada triplo no set de teste
        for i in range(len(h_test)):
            head = h_test[i].to(DEVICE)
            rel = r_test[i].to(DEVICE)
            true_tail = t_test[i].to(DEVICE)

            # O modelo tenta pontuar TODOS os nós possíveis como sendo o objeto (tail)
            all_tails = torch.arange(loader.num_nodes).to(DEVICE)
            # Expandir head e rel para o tamanho de todos os nós
            heads = head.expand(loader.num_nodes)
            rels = rel.expand(loader.num_nodes)
            
            # Calcular scores para todas as combinações (head, rel, ?)
            scores = model.decode(nodes_emb, heads, rels, all_tails)
            
            # Ranking: ver em que posição ficou a cauda verdadeira
            # Ordenar scores do maior para o menor
            _, indices = torch.sort(scores, descending=True)
            rank = (indices == true_tail).nonzero(as_tuple=True)[0].item() + 1
            
            # Acumular métricas
            mrr += 1.0 / rank
            if rank <= 10: hits_10 += 1
            if rank <= 3: hits_3 += 1
            if rank <= 1: hits_1 += 1

            if (i + 1) % 200 == 0:
                print(f" Processados: {i+1}/{len(h_test)}")

    # Médias Finais
    num_tests = len(h_test)
    print(f"\n📊 --- Resultados Finais (Link Prediction) --- 📊")
    print(f"MRR:     {mrr/num_tests:.4f}")
    print(f"Hits@10: {hits_10/num_tests:.4f} ({hits_10} acertos)")
    print(f"Hits@3:  {hits_3/num_tests:.4f}")
    print(f"Hits@1:  {hits_1/num_tests:.4f}")

if __name__ == "__main__":
    evaluate()