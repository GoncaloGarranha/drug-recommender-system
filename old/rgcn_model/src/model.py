import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RGCNModel(nn.Module):
    def __init__(self, num_nodes, num_relations, embedding_dim=100):
        super(RGCNModel, self).__init__()
        
        # 1. Camada de Embeddings Inicial: Transforma IDs em vetores treináveis
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)

        # 2. Camadas Convolucionais Relacionais (Encoder)
        # Usamos 2 camadas para permitir que a informação viaje até 2 "saltos" de distância
        self.conv1 = RGCNConv(embedding_dim, embedding_dim, num_relations)
        self.conv2 = RGCNConv(embedding_dim, embedding_dim, num_relations)
        
        # 3. Matrizes de Relação para o Decoder (DistMult)
        # Cada relação tem o seu próprio vetor de peso no espaço latente
        self.rel_emb = nn.Parameter(torch.Tensor(num_relations, embedding_dim))
        nn.init.xavier_uniform_(self.rel_emb)

    def encode(self, edge_index, edge_type):
        """Gera as representações (embeddings) finais dos nós."""
        x = self.node_embedding.weight
        
        # Primeira convolução + ativação ReLU
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        
        # Segunda convolução
        x = self.conv2(x, edge_index, edge_type)
        return x

    def decode(self, nodes_emb, head_idx, rel_idx, tail_idx):
        """Calcula a pontuação (score) para triplos específicos."""
        # Extrair vetores para os sujeitos, relações e objetos
        s = nodes_emb[head_idx]
        r = self.rel_emb[rel_idx]
        o = nodes_emb[tail_idx]
        
        # Score DistMult: produto escalar triplo (s * r * o)
        # Quanto maior o score, mais provável é a relação existir
        score = torch.sum(s * r * o, dim=-1)
        return score

    def forward(self, edge_index, edge_type, head_idx, rel_idx, tail_idx):
        """Fluxo completo: Codifica o grafo e pontua os triplos."""
        nodes_emb = self.encode(edge_index, edge_type)
        return self.decode(nodes_emb, head_idx, rel_idx, tail_idx)