import pandas as pd
from pyvis.network import Network
import os

# 1. Caminho robusto (ajustado para a tua estrutura de pastas)
# Se estás a correr da raiz do projeto, este é o caminho:
file_path = 'kg_constructor/data/output/triples/triples.filtered.csv'

if not os.path.exists(file_path):
    print(f"Erro: O ficheiro não foi encontrado em: {os.path.abspath(file_path)}")
    exit()

# 2. Carregar amostra
df = pd.read_csv(file_path).head(100)

# 3. Configurar a rede (directed=True para ver as setas)
net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white', directed=True)

# 4. Adicionar nós e relações
for _, row in df.iterrows():
    # Adicionar Fármaco (Azul)
    net.add_node(row['src'], label=row['src'], color='#2E86C1', size=25, title="Fármaco")
    # Adicionar Doença (Laranja)
    net.add_node(row['dst'], label=row['dst'], color='#E67E22', size=20, title="Doença")
    
    # Adicionar a relação
    # O segredo para ver as duas direções é a física do Pyvis que curva os arcos
    net.add_edge(row['src'], row['dst'], label=row['rel'], color='#5D6D7E', width=2)

# 5. O TRUQUE para ver a bidirecionalidade (Curvar as setas)
net.set_edge_smooth('curvedCW') # Isto curva as linhas para não se sobreporem

net.toggle_physics(True)

# Guardar e abrir
output_file = 'grafo_final.html'
net.write_html(output_file)
print(f"Sucesso! Abre o ficheiro {output_file} no teu browser para ver o grafo.")