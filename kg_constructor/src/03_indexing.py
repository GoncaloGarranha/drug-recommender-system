import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split

# --- CONFIGURAÇÃO DE CAMINHOS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'output', 'triples.csv')
ENTITY_NAMES_PATH = os.path.join(BASE_DIR, 'data', 'output', 'entity_names.csv')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def run_indexing():
    # Criar pasta de destino se não existir
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # 1. Carregar as triplas
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Erro: Ficheiro {INPUT_PATH} não encontrado!")
        return
        
    df = pd.read_csv(INPUT_PATH)
    
    # 2. Carregar o mapeamento de nomes (ID -> Nome Humano) que criámos no 01_extraction
    print("📖 Loading Entity Names...")
    entity_lookup = {}
    if os.path.exists(ENTITY_NAMES_PATH):
        with open(ENTITY_NAMES_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',', 1)
                if len(parts) == 2:
                    entity_lookup[parts[0]] = parts[1]

    print("🔢 Generating technical mapping dictionaries...")
    
    # 3. Criar dicionário de entidades usando SUBJECT e OBJECT (novos nomes das colunas)
    all_entities = sorted(list(set(df['subject']) | set(df['object'])))
    entity2id = {ent: i for i, ent in enumerate(all_entities)}
    
    # 4. Criar dicionário de relações usando PREDICATE
    all_relations = sorted(list(df['predicate'].unique()))
    rel2id = {rel: i for i, rel in enumerate(all_relations)}

    # 5. Criar a "Pedra de Roseta" Clínica (clinical_labels.json)
    # Mapeia: ID_NUMÉRICO -> NOME_HUMANO
    print("🧠 Building Clinical Translation Layer...")
    clinical_labels = {}
    for ent_text, model_id in entity2id.items():
        # Busca o nome limpo (ex: PSORIASIS) no nosso lookup
        clinical_labels[model_id] = entity_lookup.get(ent_text, ent_text)

    # --- GUARDAR DICIONÁRIOS TÉCNICOS ---
    with open(os.path.join(PROCESSED_DIR, 'entities.dict'), 'w', encoding='utf-8') as f:
        for ent, idx in entity2id.items():
            f.write(f"{idx}\t{ent}\n")

    with open(os.path.join(PROCESSED_DIR, 'relations.dict'), 'w', encoding='utf-8') as f:
        for rel, idx in rel2id.items():
            f.write(f"{idx}\t{rel}\n")

    with open(os.path.join(PROCESSED_DIR, 'clinical_labels.json'), 'w', encoding='utf-8') as f:
        json.dump(clinical_labels, f, indent=4, ensure_ascii=False)

    # 6. Mapear triplas para IDs numéricos
    df['h_id'] = df['subject'].map(entity2id)
    df['r_id'] = df['predicate'].map(rel2id)
    df['t_id'] = df['object'].map(entity2id)

    print(f"✂️  Splitting {len(df)} triples (80/10/10)...")
    
    # 7. Split: 80% Train, 20% Remaining
    train_df, rem_df = train_test_split(
        df, train_size=0.8, random_state=42
    )
    
    # Split remaining: 50% Valid, 50% Test
    valid_df, test_df = train_test_split(
        rem_df, test_size=0.5, random_state=42
    )

    # 8. Guardar ficheiros .txt finais
    for name, data in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
        output_file = os.path.join(PROCESSED_DIR, f'{name}.txt')
        data[['h_id', 'r_id', 't_id']].to_csv(output_file, sep='\t', index=False, header=False)

    print(f"\n✅ Indexing completed successfully!")
    print(f" - Entities: {len(entity2id)}")
    print(f" - Relations: {len(rel2id)}")
    print(f" - Files saved in: {PROCESSED_DIR}")

if __name__ == "__main__":
    run_indexing()