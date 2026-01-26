import os
import json
import csv

# --- CONFIGURAÇÃO DE CAMINHOS ---
INPUT_DIR = "kg_constructor/data/input/xml_files"
OUTPUT_DIR = "kg_constructor/data/output"
MAP_PATH = "kg_constructor/data/processed/disease_map.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Carregar a "Pedra de Roseta" (Disease Map)
with open(MAP_PATH, 'r', encoding='utf-8') as f:
    disease_map = json.load(f)

def clean_doid(uri):
    """Transforma 'http://.../DOID_8893' em 'DOID:8893'"""
    if not uri or not isinstance(uri, str): return None
    part = uri.split('/')[-1].replace('_', ':')
    return part if 'DOID' in part else None

def run_extraction():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    triples = []
    entity_names = set() # Para o entity_names.csv

    print(f"🚀 A processar {len(files)} ficheiros...")

    for fname in files:
        with open(os.path.join(INPUT_DIR, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Sujeito: Nome do Fármaco
            drug_name = str(data.get('name', '')).strip().upper()
            if not drug_name or drug_name == 'NONE': continue

            # Guardar par (ID, Nome) para o fármaco
            entity_names.add((drug_name, drug_name))

            # Objeto: Indicações Estruturadas
            indications = data.get('indications', {})
            if isinstance(indications, dict):
                entities = indications.get('doid_entities', [])
                for ent in entities:
                    raw_id = ent.get('doid_id')
                    clean_id = clean_doid(raw_id)
                    
                    if clean_id:
                        # Tentar tradução oficial, senão usa o ID como nome temporário
                        clinical_name = disease_map.get(clean_id, clean_id)
                        
                        # Criar a Tripla
                        triples.append([drug_name, "indicated_for", clean_id])
                        
                        # Guardar par (ID, Nome Clínico)
                        entity_names.add((clean_id, clinical_name))

    # --- GUARDAR RESULTADOS ---
    # 1. triples.csv
    with open(os.path.join(OUTPUT_DIR, "triples.csv"), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "predicate", "object"])
        writer.writerows(triples)

    # 2. entity_names.csv (Mapeamento Humano)
    with open(os.path.join(OUTPUT_DIR, "entity_names.csv"), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for eid, ename in entity_names:
            writer.writerow([eid, ename])

    print(f"\n✅ Extração concluída!")
    print(f"📊 Triplas geradas: {len(triples)}")
    print(f"🏷️  Entidades mapeadas: {len(entity_names)}")

if __name__ == "__main__":
    run_extraction()