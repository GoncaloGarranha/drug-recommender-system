import time
import pandas as pd
import os
import json
import glob
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# --- CONFIGURAÇÃO DE CAMINHOS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'output', 'triples.csv')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Fontes para tradução semântica
DRUG_JSON_DIR = os.path.join(BASE_DIR, 'data', 'output')
# Caminho para os XMLs (subindo um nível para chegar ao MedJsonify-main)
DISEASE_XML_DIR = os.path.join(BASE_DIR, '..', 'MedJsonify-main', 'NER', 'data', 'entities', 'xml_files')

def get_clinical_names():
    """Varrer ficheiros com lógica de 'deep search' para nomes."""
    drug_names = {}
    disease_names = {}

    print("🔍 Início da Extração Agressiva de Nomes...")

    # 1. FÁRMACOS - Procura em múltiplos campos
    json_files = glob.glob(os.path.join(DRUG_JSON_DIR, "*.json"))
    for f in json_files:
        try:
            with open(f, 'r', encoding='utf-8') as j:
                data = json.load(j)
                d_id = str(data.get('id', ''))
                
                # Hierarquia de nomes: Genérico -> Marca -> Nome do Produto
                openfda = data.get('openfda', {})
                name = openfda.get('generic_name', [None])[0]
                if not name:
                    name = openfda.get('brand_name', [None])[0]
                if not name and 'product_data_elements' in data:
                    # Tenta extrair do texto bruto se os campos estruturados falharem
                    name = data['product_data_elements'][0].split('[')[0].strip()

                if d_id and name:
                    drug_names[d_id] = name.upper()
        except: continue

    # 2. DOENÇAS - Normalização de IDs
    xml_files = glob.glob(os.path.join(DISEASE_XML_DIR, "*.json"))
    for f in xml_files:
        try:
            with open(f, 'r', encoding='utf-8') as j:
                data = json.load(j)
                doid_entities = data.get('indications', {}).get('doid_entities', [])
                for entity in doid_entities:
                    doid_url = entity.get('doid_id', '')
                    if 'DOID_' in doid_url:
                        doid = doid_url.split('DOID_')[-1]
                        text = entity.get('text', '')
                        # Limpa nomes como "Diabetes (DOID:123)" -> "Diabetes"
                        name = text.split(' (DOID:')[0].strip() if ' (DOID:' in text else text.strip()
                        if doid and name:
                            disease_names[doid] = name.capitalize()
        except: continue

    return drug_names, disease_names

def run_indexing():
    # Criar pasta de destino se não existir
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Carregar as triplas
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Erro: Ficheiro {INPUT_PATH} não encontrado!")
        return
        
    df = pd.read_csv(INPUT_PATH)
    df = df.astype(str) # Garantir que tudo é string

    print("🔢 Generating technical mapping dictionaries...")
    
    # 1. Criar dicionário de entidades (heads e tails)
    all_entities = sorted(list(set(df['head']) | set(df['tail'])))
    entity2id = {ent: i for i, ent in enumerate(all_entities)}
    
    # 2. Criar dicionário de relações
    all_relations = sorted(list(df['relation'].unique()))
    rel2id = {rel: i for i, rel in enumerate(all_relations)}

    # --- NOVO: CONSTRUÇÃO DA "PEDRA DE ROSETA" CLÍNICA ---
    print("📖 Building Clinical Translation Layer...")
    drugs_lookup, diseases_lookup = get_clinical_names()
    
    # Mapear: ID_INTERNO_MODELO -> NOME_REAL_HUMANO
    clinical_labels = {}
    for ent, model_id in entity2id.items():
        # Tenta encontrar o nome real
        clean_ent = ent.replace("DOID:", "")
        
        if ent in drugs_lookup:
            clinical_labels[model_id] = drugs_lookup[ent]
        elif clean_ent in diseases_lookup:
            clinical_labels[model_id] = diseases_lookup[clean_ent]
        else:
            # Se não encontrar, mantém o ID original (ex: 31174)
            clinical_labels[model_id] = ent

    # --- UNIVERSAL ENTITY RESOLVER ---
    unresolved = [idx for idx, name in clinical_labels.items() if name.isdigit() or len(name) > 20]
    resolved_count = 0
    unresolved_count = 0
    if unresolved:
        print(f"📡 A tentar resolver {len(unresolved)} IDs via APIs especializadas...")
        import requests

        def fetch_from_openfda(uid):
            for field in ['id', 'set_id']:
                url = f"https://api.fda.gov/drug/label.json?search={field}:\"{uid}\""
                try:
                    r = requests.get(url, timeout=3)
                    if r.status_code == 200:
                        res = r.json().get('results', [{}])[0]
                        openfda = res.get('openfda', {})
                        name = openfda.get('generic_name', [None])[0] or openfda.get('brand_name', [None])[0]
                        if name:
                            return name.upper(), 'openFDA'
                except: continue
            return None, None

        def fetch_from_ols(doid):
            url = f"https://www.ebi.ac.uk/ols/api/ontologies/doid/terms?obo_id=DOID:{doid}"
            try:
                r = requests.get(url, timeout=3)
                if r.status_code == 200:
                    res = r.json()
                    terms = res.get('_embedded', {}).get('terms', [])
                    if terms:
                        label = terms[0].get('label')
                        if label:
                            return label, 'OLS'
            except Exception as e:
                pass
            return None, None

        def fetch_from_rxnav(rxid):
            url = f"https://rxnav.nlm.nih.gov/REST/displaynames.json?id={rxid}"
            try:
                r = requests.get(url, timeout=3)
                if r.status_code == 200:
                    res = r.json()
                    terms = res.get('displayTermsList', {}).get('term', [])
                    if terms:
                        name = terms[0]
                        if name and len(name) < 50:
                            return name.upper(), 'RxNav'
            except Exception as e:
                pass
            return None, None

        def universal_resolver(tech_id):
            tech_id = tech_id.strip()
            # REGRA 1: UUID da FDA (fármaco)
            if "-" in tech_id and len(tech_id) > 30:
                print(f"💊 [FDA] A pesquisar Fármaco: {tech_id}")
                return fetch_from_openfda(tech_id)
            # REGRA 2: DOID de 7 dígitos (doença)
            if tech_id.isdigit() and len(tech_id) == 7:
                print(f"🧬 [OLS] A pesquisar Doença: DOID_{tech_id}")
                return fetch_from_ols(tech_id)
            # REGRA 3: RxNorm curto (ingrediente)
            if tech_id.isdigit() and len(tech_id) < 7:
                print(f"🧪 [RxNav] A pesquisar Ingrediente: {tech_id}")
                return fetch_from_rxnav(tech_id)
            return None, None

        for i, model_id in enumerate(unresolved):
            tech_id = clinical_labels[model_id]
            print(f"[DEBUG] ({i+1}/{len(unresolved)}) Resolving: {tech_id}")
            name, fonte = universal_resolver(tech_id)
            if name:
                print(f"[DEBUG] Resolved: {tech_id} -> {name} ({fonte})")
                clinical_labels[model_id] = f"{name} ({fonte})"
                resolved_count += 1
            else:
                print(f"[DEBUG] Could not resolve: {tech_id}")
                unresolved_count += 1
            time.sleep(0.2)  # Evita sobrecarregar as APIs

    print(f"[DEBUG] clinical_labels size: {len(clinical_labels)}")
    print(f"[RESUMO] IDs resolvidos via API: {resolved_count}")
    print(f"[RESUMO] IDs NÃO resolvidos via API: {unresolved_count}")

    # Guardar os dicionários técnicos (ID \t Nome_Original)
    entities_path = os.path.join(PROCESSED_DIR, 'entities.dict')
    relations_path = os.path.join(PROCESSED_DIR, 'relations.dict')
    clinical_labels_path = os.path.join(PROCESSED_DIR, 'clinical_labels.json')
    with open(entities_path, 'w', encoding='utf-8') as f:
        for ent, idx in entity2id.items():
            f.write(f"{idx}\t{ent}\n")
    print(f"[DEBUG] entities.dict escrito em: {entities_path}")

    with open(relations_path, 'w', encoding='utf-8') as f:
        for rel, idx in rel2id.items():
            f.write(f"{idx}\t{rel}\n")
    print(f"[DEBUG] relations.dict escrito em: {relations_path}")

    # Guardar a tradução clínica (JSON para leitura fácil no predict.py)
    with open(clinical_labels_path, 'w', encoding='utf-8') as f:
        json.dump(clinical_labels, f, indent=4, ensure_ascii=False)
    print(f"[DEBUG] clinical_labels.json escrito em: {clinical_labels_path}")

    # 3. Mapear triplas para IDs numéricos
    df['h_id'] = df['head'].map(entity2id)
    df['r_id'] = df['relation'].map(rel2id)
    df['t_id'] = df['tail'].map(entity2id)

    print("✂️  Performing data split (80/10/10) with stratification...")
    
    # 4. Split: 80% Train, 20% Remaining
    train_df, rem_df = train_test_split(
        df, train_size=0.8, random_state=42, stratify=df['r_id']
    )
    
    # Split remaining: 50% Valid, 50% Test
    valid_df, test_df = train_test_split(
        rem_df, test_size=0.5, random_state=42, stratify=rem_df['r_id']
    )

    # 5. Guardar ficheiros finais (.txt) para o modelo R-GCN
    for name, data in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
        output_file = os.path.join(PROCESSED_DIR, f'{name}.txt')
        data[['h_id', 'r_id', 't_id']].to_csv(output_file, sep='\t', index=False, header=False)

    print(f"\n✅ Processing completed successfully!")
    print(f" - Technical dictionaries and .txt files saved in: {PROCESSED_DIR}")
    print(f" - Clinical Rosetta Stone saved as: clinical_labels.json")

if __name__ == "__main__":
    run_indexing()