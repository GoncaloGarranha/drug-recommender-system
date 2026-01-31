import pandas as pd
import os

# No Mac, os caminhos devem ser precisos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIPLES_PATH = os.path.join(BASE_DIR, "data", "output", "triples.csv")

# --- LISTA NEGRA DE IDS (LIXO ESTRUTURAL) ---
BLACK_LIST = [
    "DOID:4",        # "DISEASE" (Muito genérico)
    "DOID:0050156",  # VIH (Probável erro de extração)
    "DOID:2377",     # Esclerose Múltipla (Probável erro de extração)
    "DOID:14566",    # Tuberculose (Probável erro de extração)
    "DOID:630",      # Genetic Disease (Genérico)
    "DOID:150"       # Disease of Mental Health (Genérico)
]

def purify():
    if not os.path.exists(TRIPLES_PATH):
        print(f"❌ Ficheiro {TRIPLES_PATH} não encontrado.")
        return

    df = pd.read_csv(TRIPLES_PATH)
    initial_count = len(df)

    # 1. Remover triplas com IDs da Blacklist
    df_clean = df[~df['object'].isin(BLACK_LIST)]
    
    # 2. Remover triplas onde o nome da doença é apenas "DISEASE" 
    # (Caso tenhas IDs diferentes que mapeiam para o mesmo nome)
    # Por agora, o filtro por ID é o mais seguro.

    print(f"🧹 Limpeza concluída!")
    print(f"📉 Triplas originais: {initial_count}")
    print(f"✨ Triplas limpas: {len(df_clean)}")
    
    # Gravar por cima do original
    df_clean.to_csv(TRIPLES_PATH, index=False)
    print(f"✅ Ficheiro triples.csv purificado com sucesso.")

if __name__ == "__main__":
    purify()