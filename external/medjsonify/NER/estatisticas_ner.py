import os
import json
from collections import defaultdict
from tabulate import tabulate

# Caminho para a pasta NER/data/entities
BASE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'entities')


# Estatísticas
unique_products = set()  # Produto Medicinal
unique_substances = set()  # Substância Ativa
unique_diseases = set()  # Doença
unique_indications = set()  # Indicação Terapêutica
unique_contras = set()  # Contraindicação
normalized_substances = 0
normalized_diseases = 0
total_substances = 0
total_diseases = 0
relation_count = 0
example_schema = None

# Campos de ontologia possíveis
ONTOLOGY_KEYS = {'chebi_id', 'doid_id', 'ordo_id', 'mesh_id', 'omim_id', 'snomed_id', 'atc_code', 'rxnorm_id'}

def process_json_file(filepath):
    global example_schema, normalized_entities, all_entities, relation_count
    global total_substances, normalized_substances, total_diseases, normalized_diseases
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if example_schema is None:
            example_schema = list(data.keys())
        # 1. Produto Medicinal (Produto comercial)
        product_id = data.get('drug_id') or data.get('id')
        if product_id:
            unique_products.add(product_id)

        # 2. Substância Ativa (ingredients, normalizada via chebi_id ou drugbank_id)
        ingredients = data.get('ingredients', [])
        for ing in ingredients:
            name = ing.get('name')
            if name:
                unique_substances.add(name)
                total_substances += 1
                if ing.get('chebi_id') or ing.get('drugbank_id'):
                    normalized_substances += 1

        # 3. Doença / Patologia (doid_entities em indications/contraindications)
        for section in ['indications', 'contraindications']:
            sec = data.get(section, {})
            if isinstance(sec, dict):
                doid_ents = sec.get('doid_entities', [])
                for ent in doid_ents:
                    text = ent.get('text')
                    doid_id = ent.get('doid_id')
                    if text:
                        unique_diseases.add(text)
                        total_diseases += 1
                        if doid_id:
                            normalized_diseases += 1

        # 4. Indicação Terapêutica (indications)
        ind_sec = data.get('indications', {})
        if isinstance(ind_sec, dict):
            ind_text = ind_sec.get('text')
            if ind_text:
                unique_indications.add(ind_text)

        # 5. Contraindicação (contraindications)
        contra_sec = data.get('contraindications', {})
        if isinstance(contra_sec, dict):
            contra_text = contra_sec.get('text')
            if contra_text:
                unique_contras.add(contra_text)

        # Relações (mantém original)
        relations = data.get('relations', [])
        relation_count += len(relations)

def walk_and_process(base_dir):
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                process_json_file(os.path.join(root, file))

if __name__ == '__main__':
    walk_and_process(BASE_DIR)
    # Estatísticas
    normalization_rate_substances = (normalized_substances / total_substances * 100) if total_substances else 0
    normalization_rate_diseases = (normalized_diseases / total_diseases * 100) if total_diseases else 0
    table = [
        ['Produto Medicinal Único', len(unique_products)],
        ['Substância Ativa Única', len(unique_substances)],
        ['Doença/Patologia Única', len(unique_diseases)],
        ['Indicações Terapêuticas Únicas', len(unique_indications)],
        ['Contraindicações Únicas', len(unique_contras)],
    ]
    print('# Estatísticas dos Dados Extraídos (Núcleo Essencial)')
    print(tabulate(table, headers=['Entidade/Métrica', 'Valor'], tablefmt='github'))
    if example_schema:
        print('\n## Exemplo de Schema (keys de um JSON)')
        print('```json')
        print(json.dumps(example_schema, indent=2, ensure_ascii=False))
        print('```')
    else:
        print('Nenhum ficheiro JSON encontrado para mostrar o schema.')
