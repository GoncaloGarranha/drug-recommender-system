import os
import json
import pandas as pd
from typing import Any, Dict, List, Optional

 # --- Config paths ---
# Adjusted for your structure: kg_constructor/src/01_extraction.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_PATH = os.path.join(BASE_DIR, 'config', 'mapping_rules.json')
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'input')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'output', 'triples.csv')

def load_rules(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_id(val: str, filters: List[str]) -> str:
    """Cleans URIs and transforms into canonical IDs (e.g., DOID:4989)"""
    if not val: return ""
    for prefix in filters:
        if val.startswith(prefix):
            # Remove prefix and replace _ with :
            return val.replace(prefix, '').replace('_', ':')
    # If it's a DrugBank link
    if "drugbank.com" in val:
        return "DB:" + val.split('/')[-1]
    return val

def resolve_path(obj: Any, path: str) -> Any:
    """Recursively navigates dictionaries and lists."""
    parts = path.split('.')
    for part in parts:
        if isinstance(obj, dict):
            obj = obj.get(part)
        elif isinstance(obj, list):
            # If it's a list, try to extract the remaining part from each item
            remaining_path = ".".join(parts[parts.index(part):])
            return [resolve_path(item, remaining_path) for item in obj]
        else:
            return None
    return obj

def extract_triples(data: Dict, mapper: Dict, filters: List[str]) -> List[List[str]]:
    triples = []
    subject = str(resolve_path(data, mapper['subject']))
    
    if not subject or subject == "None":
        return []

    for rule in mapper['predicate_map']:
        path = rule['path']
        relation = rule['relation']
        attr = rule.get('attr')

        # Get the value (can be a single value or a list)
        values = resolve_path(data, path)

        # Normalize to list to process everything the same way
        if not isinstance(values, list):
            values = [values]
            
        for v in values:
            if isinstance(v, dict) and attr:
                # If the value is a dict and we have an 'attr' defined
                v = v.get(attr)
            # Otherwise, try default keys if it's a dict
            elif isinstance(v, dict):
                v = v.get('chebi_id') or v.get('drugbank_id') or v.get('doid_id')

            if v:
                cleaned_v = clean_id(str(v), filters)
                if cleaned_v:
                    triples.append([subject, relation, cleaned_v])
    return triples

def main():
    print("Starting triple extraction...")
    rules = load_rules(RULES_PATH)
    all_triples = []

    # Iterate through xml, csv, txt folders inside data/input
    for root, _, files in os.walk(INPUT_DIR):
        for fname in files:
            if not fname.endswith('.json'):
                continue

            fpath = os.path.join(root, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Detect which mapper to use based on JSON keys
            selected_mapper = None
            for mapper in rules['mappers']:
                if mapper['detector'] in data:
                    selected_mapper = mapper
                    break
            
            if selected_mapper:
                file_triples = extract_triples(data, selected_mapper, rules['ontology_prefix_filter'])
                all_triples.extend(file_triples)
            else:
                print(f"Warning: No rule for {fname}")

    # Save results
    df = pd.DataFrame(all_triples, columns=['head', 'relation', 'tail'])
    df = df.drop_duplicates()  # Remove duplicate facts

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Extraction complete! {len(df)} unique triples saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()