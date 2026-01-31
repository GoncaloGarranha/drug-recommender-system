import json
import requests
import os

def download_ontology():
    url = "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.json"
    print("🌐 A descarregar a Disease Ontology (isto pode demorar um pouco)...")
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("❌ Erro ao descarregar a ontologia.")
        return None

def build_map():
    data = download_ontology()
    if not data: return

    disease_map = {}
    # A estrutura do doid.json tem uma lista de 'graphs'
    nodes = data.get('graphs', [{}])[0].get('nodes', [])

    print(f"⚙️  A processar {len(nodes)} nós da ontologia...")

    for node in nodes:
        uri = node.get('id', '')
        # Extrair o ID final (ex: http://.../DOID_5575 -> DOID:5575)
        if 'DOID_' in uri:
            clean_id = uri.split('/')[-1].replace('_', ':')
            label = node.get('lbl', 'Unknown')
            disease_map[clean_id] = label.upper()

    output_path = "kg_constructor/data/processed/disease_map.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(disease_map, f, indent=4)
    
    print(f"✅ Sucesso! {len(disease_map)} doenças mapeadas em {output_path}")

if __name__ == "__main__":
    build_map()