import os
import json
from collections import Counter

INPUT_DIR = "kg_constructor/data/input/xml_files"

def full_audit():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    total = len(files)
    
    stats = {
        "missing_name": [],
        "missing_indications": 0,
        "valid_doid_uris": 0,
        "found_a_label": 0, # Ficheiros onde o nome da doença NÃO é None
        "uri_formats": Counter()
    }

    print(f"🕵️  A iniciar auditoria total a {total} ficheiros...")

    for fname in files:
        with open(os.path.join(INPUT_DIR, fname), 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                
                # 1. Verificar Sujeito
                if not data.get('name'):
                    stats["missing_name"].append(fname)
                
                # 2. Verificar Objeto
                indications = data.get('indications', {})
                if not indications:
                    stats["missing_indications"] += 1
                    continue
                
                entities = indications.get('doid_entities', [])
                for ent in entities:
                    doid = ent.get('doid_id')
                    label = ent.get('name') or ent.get('label')
                    
                    if doid:
                        stats["valid_doid_uris"] += 1
                        # Guardar o formato da URI para termos certeza
                        example_format = doid.split('/')[2] if '/' in doid else "format_unknown"
                        stats["uri_formats"][example_format] += 1
                    
                    if label and label != "None":
                        stats["found_a_label"] += 1

            except Exception as e:
                print(f"⚠️ Erro no ficheiro {fname}: {e}")

    print("\n" + "="*50)
    print("🏆 RELATÓRIO DE CERTEZA MÁXIMA")
    print("="*50)
    print(f"✅ Ficheiros analisados: {total}")
    print(f"❌ Sem nome de fármaco: {len(stats['missing_name'])}")
    print(f"⚠️ Sem secção de indicações: {stats['missing_indications']}")
    print(f"🧬 IDs de doenças encontrados: {stats['valid_doid_uris']}")
    print(f"🌟 Milagre! Nomes de doenças encontrados: {stats['found_a_label']}")
    print(f"\n🌍 Formatos de URI detectados: {dict(stats['uri_formats'])}")
    print("="*50)

if __name__ == "__main__":
    full_audit()