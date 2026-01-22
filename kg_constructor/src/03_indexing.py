import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'output', 'triples.csv')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def run_indexing():
    # Create destination folder if it does not exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Load the triples
    df = pd.read_csv(INPUT_PATH)
    df = df.astype(str) # Ensure everything is treated as string for indexing

    print("🔢 Generating mapping dictionaries...")
    
    # 1. Create entity dictionary (all heads and tails)
    all_entities = sorted(list(set(df['head']) | set(df['tail'])))
    entity2id = {ent: i for i, ent in enumerate(all_entities)}
    
    # 2. Create relation dictionary
    all_relations = sorted(list(df['relation'].unique()))
    rel2id = {rel: i for i, rel in enumerate(all_relations)}

    # Save dictionaries (format: ID \t Name)
    with open(os.path.join(PROCESSED_DIR, 'entities.dict'), 'w', encoding='utf-8') as f:
        for ent, idx in entity2id.items():
            f.write(f"{idx}\t{ent}\n")
            
    with open(os.path.join(PROCESSED_DIR, 'relations.dict'), 'w', encoding='utf-8') as f:
        for rel, idx in rel2id.items():
            f.write(f"{idx}\t{rel}\n")

    # 3. Map triples to numeric IDs
    df['h_id'] = df['head'].map(entity2id)
    df['r_id'] = df['relation'].map(rel2id)
    df['t_id'] = df['tail'].map(entity2id)

    print("✂️  Performing data split (80/10/10) with stratification...")
    
    # 4. Split: 80% Train, 20% Remaining (with stratification by relation to maintain balance)
    train_df, rem_df = train_test_split(
        df, train_size=0.8, random_state=42, stratify=df['r_id']
    )
    
    # Split the remaining: 50% Validation, 50% Test (which gives 10% of the total for each)
    valid_df, test_df = train_test_split(
        rem_df, test_size=0.5, random_state=42, stratify=rem_df['r_id']
    )

    # 5. Save final files in (h, r, t) format for R-GCN
    # This format is standard for Deep Learning Graph frameworks
    for name, data in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
        output_file = os.path.join(PROCESSED_DIR, f'{name}.txt')
        data[['h_id', 'r_id', 't_id']].to_csv(output_file, sep='\t', index=False, header=False)

    print(f"\n✅ Processing completed successfully!")
    print(f" - Dictionaries and files saved in: {PROCESSED_DIR}")
    print(f" - Train: {len(train_df)} triples")
    print(f" - Validation: {len(valid_df)} triples")
    print(f" - Test: {len(test_df)} triples")

if __name__ == "__main__":
    run_indexing()