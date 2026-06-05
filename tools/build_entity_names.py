import csv
import os
import re
import urllib.request

DOID_OBO_URL = "http://purl.obolibrary.org/obo/doid.obo"

def download_if_missing(url: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return
    print(f"[INFO] A descarregar DOID OBO: {url}")
    urllib.request.urlretrieve(url, out_path)
    print(f"[OK] Guardado em: {out_path}")

def parse_doid_obo(obo_path: str) -> dict:
    """
    Parser simples e robusto para OBO:
    captura blocos [Term] com:
      id: DOID:xxxx
      name: ...
    """
    mapping = {}
    current_id = None
    current_name = None
    in_term = False

    with open(obo_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()

            if line == "[Term]":
                in_term = True
                current_id = None
                current_name = None
                continue

            if line == "" and in_term:
                # fim de bloco
                if current_id and current_name:
                    mapping[current_id] = current_name
                in_term = False
                current_id = None
                current_name = None
                continue

            if not in_term:
                continue

            if line.startswith("id: DOID:"):
                current_id = line.split("id:", 1)[1].strip()
            elif line.startswith("name:"):
                current_name = line.split("name:", 1)[1].strip()

    # caso o ficheiro não termine com linha vazia
    if in_term and current_id and current_name:
        mapping[current_id] = current_name

    print(f"[OK] DOID termos mapeados: {len(mapping)}")
    return mapping

def load_entity_table(entity_table_csv: str):
    rows = []
    with open(entity_table_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def main():
    # 1) caminhos (ajusta se tiveres estrutura diferente)
    entity_table_csv = os.path.join("kg_constructor", "data", "output", "mappings", "entity_table.csv")
    out_csv = os.path.join("kg_constructor", "data", "output", "entity_names.csv")

    if not os.path.exists(entity_table_csv):
        raise FileNotFoundError(f"Não encontrei: {entity_table_csv}")

    # 2) DOID obo local
    obo_path = os.path.join("kg_constructor", "data", "input", "ontologies", "doid.obo")
    download_if_missing(DOID_OBO_URL, obo_path)

    # 3) parse DOID
    doid_map = parse_doid_obo(obo_path)

    # 4) carregar entidades e construir labels finais
    entities = load_entity_table(entity_table_csv)

    # entity_names.csv tem de ter: id,name (é isso que o infer.py lê)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    written = 0
    missing_doid = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name"])

        for e in entities:
            eid = e["entity_id"]
            etype = e["entity_type"]
            label = e["label"]

            if etype == "DISEASE" and eid.startswith("DOID:"):
                # se ainda estiver "cego", substituir por nome oficial
                resolved = doid_map.get(eid)
                if resolved:
                    w.writerow([eid, resolved])
                else:
                    missing_doid += 1
                    w.writerow([eid, label])  # fallback
            else:
                # drugs/ingredients/etc já têm label resolvido
                w.writerow([eid, label])

            written += 1

    print(f"[OK] Escrevi {written} labels em: {out_csv}")
    if missing_doid:
        print(f"[WARN] DOIDs sem nome na ontologia: {missing_doid} (ficaram com fallback)")

if __name__ == "__main__":
    main()
