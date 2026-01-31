import os
import json
import random
from collections import Counter

# --- CONFIGURAÇÃO ---
INPUT_DIR = "kg_constructor/data/input/xml_files"  # Vamos focar-nos na maior fonte primeiro
SAMPLE_SIZE = 50  # Analisar 50 ficheiros aleatórios para ter uma amostra real

def inspect_data():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    if not files:
        print(f"❌ Erro: Não foram encontrados ficheiros em {INPUT_DIR}")
        return

    sample_files = random.sample(files, min(SAMPLE_SIZE, len(files)))
    
    report = {
        "drug_name_keys": Counter(),
        "has_indications": 0,
        "has_doid_in_indications": 0,
        "id_examples": [],
        "sample_mappings": []
    }

    print(f"🔍 A analisar {len(sample_files)} ficheiros aleatórios...")

    for fname in sample_files:
        with open(os.path.join(INPUT_DIR, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 1. Verificar chaves do nome do fármaco
            for k in ['name', 'brand_name', 'generic_name', 'proprietary_name']:
                if data.get(k): report["drug_name_keys"][k] += 1
            
            # 2. Inspecionar Indicações
            indications = data.get('indications', [])
            if indications:
                report["has_indications"] += 1
                
                # Se o MedJsonify estruturou isto, costuma estar em doid_entities
                doid_entities = indications.get('doid_entities', []) if isinstance(indications, dict) else []
                
                if doid_entities:
                    report["has_doid_in_indications"] += 1
                    for ent in doid_entities[:2]: # Ver os primeiros 2 exemplos
                        eid = ent.get('doid_id') or ent.get('id')
                        ename = ent.get('name') or ent.get('label')
                        if eid:
                            report["id_examples"].append(eid)
                            report["sample_mappings"].append(f"{ename} -> {eid}")

    # --- APRESENTAÇÃO DOS RESULTADOS ---
    print("\n" + "="*50)
    print("📊 RELATÓRIO DE AUDITORIA DE DADOS")
    print("="*50)
    
    print(f"\n✅ Chaves encontradas para NOME DO FÁRMACO:")
    for k, count in report["drug_name_keys"].items():
        print(f"   - {k}: {count} ficheiros")

    print(f"\n🧬 Cobertura de Indicações:")
    print(f"   - Ficheiros com secção 'indications': {report['has_indications']}/{SAMPLE_SIZE}")
    print(f"   - Ficheiros com IDs estruturados (DOID): {report['has_doid_in_indications']}/{SAMPLE_SIZE}")

    print(f"\n🧪 Amostra de Identificadores encontrados:")
    print(f"   - IDs: {list(set(report['id_examples']))[:10]}")
    
    print(f"\n📖 Exemplo de Mapeamento Real (O que o modelo vai ver):")
    for mapping in list(set(report["sample_mappings"]))[:5]:
        print(f"   - {mapping}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    inspect_data()