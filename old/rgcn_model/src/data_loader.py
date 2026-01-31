import torch
import pandas as pd
import os

class KGDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.entities_dict = self.load_dict('entities.dict')
        self.relations_dict = self.load_dict('relations.dict')
        
        self.num_nodes = len(self.entities_dict)
        self.num_relations = len(self.relations_dict)

    def load_dict(self, filename):
        path = os.path.join(self.data_path, filename)
        df = pd.read_csv(path, sep='\t', header=None, names=['id', 'name'])
        return dict(zip(df['id'], df['name']))

    def load_triples(self, filename):
        """Carrega os triplos (h, r, t) e retorna tensores."""
        path = os.path.join(self.data_path, filename)
        df = pd.read_csv(path, sep='\t', header=None, names=['h', 'r', 't'])
        
        # Converter para Tensores do PyTorch
        head = torch.tensor(df['h'].values, dtype=torch.long)
        rel = torch.tensor(df['r'].values, dtype=torch.long)
        tail = torch.tensor(df['t'].values, dtype=torch.long)
        
        return head, rel, tail

    def get_graph_structure(self, head, rel, tail):
        """
        Prepara a edge_index e edge_type para a R-GCN.
        Inclui relações diretas e inversas para melhor propagação de mensagens.
        """
        # Arestas Diretas (h -> t)
        edge_index_direct = torch.stack([head, tail], dim=0)
        edge_type_direct = rel
        
        # Arestas Inversas (t -> h) com novos IDs de relação
        edge_index_inverse = torch.stack([tail, head], dim=0)
        edge_type_inverse = rel + self.num_relations
        
        # Concatenar tudo
        edge_index = torch.cat([edge_index_direct, edge_index_inverse], dim=1)
        edge_type = torch.cat([edge_type_direct, edge_type_inverse], dim=0)
        
        return edge_index, edge_type

# Exemplo de teste rápido
if __name__ == "__main__":
    # Caminho para os dados processados no bloco anterior
    PROCESSED_DATA = "kg_constructor/data/processed"
    
    loader = KGDataLoader(PROCESSED_DATA)
    h, r, t = loader.load_triples('train.txt')
    edge_index, edge_type = loader.get_graph_structure(h, r, t)
    
    print(f"📊 Estatísticas do Data Loader:")
    print(f" - Nós: {loader.num_nodes}")
    print(f" - Relações Originais: {loader.num_relations}")
    print(f" - Triplos de Treino: {h.shape[0]}")
    print(f" - Arestas no Grafo (com inversas): {edge_index.shape[1]}")