import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

# --- PATH CONFIGURATION ---
TRIPLES_PATH = "kg_constructor/data/output/triples.csv"
ENTITY_NAMES_PATH = "kg_constructor/data/output/entity_names.csv"
PLOT_OUTPUT_DIR = "kg_constructor/data/output/plots"

os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

REQUIRED_COLUMNS = {"subject", "predicate", "object"}


def load_entity_names(path: str) -> dict:
    """
    Loads ID-to-label mapping from entity_names.csv.

    Expected header: id,label
    This implementation safely ignores malformed rows.
    """
    mapping = {}
    if not os.path.exists(path):
        return mapping

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # DictReader uses the first row as header automatically
        for row in reader:
            ent_id = (row.get("id") or "").strip()
            label = (row.get("label") or "").strip()
            if ent_id:
                mapping[ent_id] = label if label else ent_id

    return mapping


def run_analytics() -> None:
    """
    Basic EDA over triples.csv and plot generation.
    """
    print("Starting Knowledge Graph Analytics...")

    if not os.path.exists(TRIPLES_PATH):
        print(f"Error: {TRIPLES_PATH} not found. Run extraction first.")
        return

    df = pd.read_csv(TRIPLES_PATH)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"Error: triples.csv is missing columns: {sorted(missing)}")
        return

    name_map = load_entity_names(ENTITY_NAMES_PATH)

    n_triples = len(df)
    n_drugs = df["subject"].nunique()
    n_diseases = df["object"].nunique()

    print("----------------------------------------")
    print("Dataset Summary Statistics")
    print(f"Total Triples: {n_triples}")
    print(f"Unique Drugs (Subjects): {n_drugs}")
    print(f"Unique Diseases (Objects): {n_diseases}")
    print("----------------------------------------")

    # Top drugs
    top_drugs = df["subject"].value_counts().head(15)
    top_drugs_names = [name_map.get(str(x), str(x)) for x in top_drugs.index]

    plt.figure(figsize=(12, 7))
    plt.bar(top_drugs_names, top_drugs.values)
    plt.title("Top 15 Drugs by Number of Indications", fontsize=14)
    plt.ylabel("Number of Indications (Edges)", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "top_drugs_distribution.png"))
    print("Saved: top_drugs_distribution.png")

    # Top diseases
    top_diseases = df["object"].value_counts().head(15)
    top_diseases_names = [name_map.get(str(x), str(x)) for x in top_diseases.index]

    plt.figure(figsize=(12, 7))
    plt.bar(top_diseases_names, top_diseases.values)
    plt.title("Top 15 Diseases by Drug Prevalence", fontsize=14)
    plt.ylabel("Number of Associated Drugs", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "top_diseases_distribution.png"))
    print("Saved: top_diseases_distribution.png")

    # Degree distribution (drug side)
    drug_degree = df["subject"].value_counts()

    plt.figure(figsize=(10, 6))
    plt.hist(drug_degree, bins=50, edgecolor="black", alpha=0.8)
    plt.title("Drug Degree Distribution (Connectivity)", fontsize=14)
    plt.xlabel("Number of Indications per Drug", fontsize=12)
    plt.ylabel("Frequency (Log Scale)", fontsize=12)
    plt.yscale("log")
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "degree_distribution_log.png"))
    print("Saved: degree_distribution_log.png")

    print(f"Analytics completed. Plots saved in: {PLOT_OUTPUT_DIR}")


if __name__ == "__main__":
    run_analytics()
