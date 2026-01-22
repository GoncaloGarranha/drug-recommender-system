import pandas as pd
import os

# Caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIPLES_PATH = os.path.join(BASE_DIR, 'data', 'output', 'triples.csv')
STATS_PATH = os.path.join(BASE_DIR, 'data', 'output', 'graph_stats_summary.txt')

def run_analytics():
    if not os.path.exists(TRIPLES_PATH):
        print(f"❌ Erro: Ficheiro {TRIPLES_PATH} não encontrado.")
        return

    df = pd.read_csv(TRIPLES_PATH)
    
    print("📊 --- Relatório de Caracterização do Grafo --- 📊\n")
    
    # 1. Estatísticas Gerais
    total_triples = len(df)
    unique_heads = df['head'].nunique()
    unique_relations = df['relation'].nunique()
    all_entities = set(df['head'].astype(str)) | set(df['tail'].astype(str))
    
    print(f"Total de Tripletos (Arestas): {total_triples}")
    print(f"Total de Entidades Únicas (Nós): {len(all_entities)}")
    print(f"Tipos de Relações Distintas: {unique_relations}\n")
    
    # 2. Distribuição por Tipo de Relação
    print("📈 Distribuição por Tipo de Relação:")
    rel_counts = df['relation'].value_counts()
    for rel, count in rel_counts.items():
        percentage = (count / total_triples) * 100
        print(f" - {rel}: {count} ({percentage:.1f}%)")
    print("")

    # 3. Distribuição de Nós por Categoria (Estimada pelas relações)
    diseases = df[df['relation'].isin(['indicated_for', 'contraindicated_for'])]['tail'].nunique()
    substances = df[df['relation'] == 'has_substance']['tail'].nunique()
    orgs = df[df['relation'].isin(['has_organization', 'has_applicant'])]['tail'].nunique()
    proper_names = df[df['relation'] == 'has_proper_name']['tail'].nunique()
    drugs = df['head'].nunique()

    print("🏷️  Distribuição de Nós por Categoria:")
    print(f" - Fármacos (Produtos): {drugs}")
    print(f" - Substâncias Ativas: {substances}")
    print(f" - Patologias (Doenças): {diseases}")
    print(f" - Organizações/Fabricantes: {orgs}")
    print(f" - Nomes Próprios (Biológicos): {proper_names}\n")

    # 4. Conetividade
    avg_degree = total_triples / drugs
    print(f"🕸️  Densidade: Cada fármaco tem, em média, {avg_degree:.2f} relações.")

    # Guardar resultados
    with open(STATS_PATH, 'w', encoding='utf-8') as f:
        f.write("RESUMO ESTATÍSTICO DO GRAFO\n")
        f.write("===========================\n")
        f.write(f"Total Triplos: {total_triples}\n")
        f.write(f"Entidades: {len(all_entities)}\n")
        f.write(f"Grau Médio: {avg_degree:.2f}\n\n")
        f.write("RELAÇÕES:\n")
        f.write(rel_counts.to_string())
    
    print(f"\n✅ Estatísticas guardadas em {STATS_PATH}")

if __name__ == "__main__":
    run_analytics()