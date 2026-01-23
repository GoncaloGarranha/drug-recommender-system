import torch
import torch.nn.functional as F
from data_loader import KGDataLoader
from model import RGCNModel
import os

# Configurações
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = "kg_constructor/data/processed"
EPOCHS = 100
LR = 0.01
EMBED_DIM = 100

def generate_negative_samples(head, rel, tail, num_nodes):
    """Gera exemplos falsos trocando o objeto (tail) aleatoriamente."""
    negative_tail = torch.randint(0, num_nodes, tail.size(), device=DEVICE)
    return negative_tail

def train():
    # 1. Carregar Dados
    loader = KGDataLoader(DATA_PATH)
    h_train, r_train, t_train = loader.load_triples('train.txt')
    h_train, r_train, t_train = h_train.to(DEVICE), r_train.to(DEVICE), t_train.to(DEVICE)
    
    # Preparar a estrutura do grafo para as convoluções
    edge_index, edge_type = loader.get_graph_structure(h_train, r_train, t_train)
    edge_index, edge_type = edge_index.to(DEVICE), edge_type.to(DEVICE)

    # 2. Inicializar Modelo
    # Usamos num_relations * 2 por causa das inversas
    model = RGCNModel(loader.num_nodes, loader.num_relations * 2, EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"🚀 A iniciar treino em {DEVICE}...")
    
    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        
        # Gerar amostras negativas para este passo
        neg_t = generate_negative_samples(h_train, r_train, t_train, loader.num_nodes)
        
        # Codificar o grafo (Message Passing)
        nodes_emb = model.encode(edge_index, edge_type)
        
        # Pontuar triplos positivos e negativos
        pos_scores = model.decode(nodes_emb, h_train, r_train, t_train)
        neg_scores = model.decode(nodes_emb, h_train, r_train, neg_t)
        
        # Loss Function: Queremos que o score positivo seja alto e o negativo baixo
        # Usamos Margin Ranking Loss ou Binary Cross Entropy
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))]).to(DEVICE)
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    # Guardar o modelo treinado
    os.makedirs("../checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "../checkpoints/rgcn_model.pth")
    print(f"\n✅ Treino concluído! Modelo guardado em checkpoints/rgcn_model.pth")

if __name__ == "__main__":
    train()