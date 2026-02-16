# Drug Recommender System

A knowledge graph-based system for drug-disease relationship prediction using R-GCN (Relational Graph Convolutional Networks). This project extracts biomedical data, constructs a knowledge graph, and trains a neural network to predict drug indications and contraindications.

## Project Structure

```
.
├── kg_constructor/        # Knowledge graph construction pipeline
│   ├── configs/          # Configuration files
│   ├── data/             # Input, processed, and output data
│   └── src/              # Source code for the 9-step pipeline
├── rgcn/                 # R-GCN model for link prediction
│   ├── configs/          # Model configurations
│   ├── runs/             # Training outputs and checkpoints
│   └── src/              # Training, evaluation, and inference scripts
├── MedJsonify-main/      # Data extraction and NER pipeline
├── tools/                # Utility scripts
├── docs/                 # Documentation
└── requirements.txt      # Python dependencies
```

## Quick Start

### 1. Environment Setup

#### Create Virtual Environment

**macOS/Linux:**
source venv/bin/activate

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the Knowledge Graph

Run the complete knowledge graph construction pipeline:

```bash
python -m kg_constructor.src.run_kg_pipeline --config kg_constructor/configs/config.yaml
```

This will:
- Import data from `MedJsonify-main/NER/data/entities/`
- Parse and normalize drug and disease information
- Resolve entities to canonical IDs
- Build triples (drug-disease relationships)
- Filter out unwanted edges (HAS_INGREDIENT, blacklisted diseases)
- Split edges into train/valid/test sets (80/10/10)
- Create DGL graph for model training
- Generate analytics and visualizations

**Output**: Processed data in `kg_constructor/data/processed/`

See detailed documentation in `kg_constructor/src/steps/README.md`

### 3. Train the R-GCN Model

Train the neural network to predict drug-disease relationships:

```bash
python -m rgcn.src.02_train --config rgcn/configs/baseline.yaml
```

This will:
- Load the knowledge graph from `kg_constructor/data/processed/`
- Train an R-GCN encoder with DistMult decoder
- Evaluate on validation set every 10 epochs
- Save checkpoints to `rgcn/runs/<timestamp>_<name>/`

**Output**: 
- Best model: `rgcn/runs/.../checkpoints/best.pt`
- Training logs: `rgcn/runs/.../train_log.json`

See detailed documentation in `rgcn/README.md`

### 4. Make Predictions

Generate drug-disease predictions for a specific drug:

```bash
# Search by drug name
python -m rgcn.src.04_infer --query "Albuterol" --topk 10

# Search by exact entity ID
python -m rgcn.src.04_infer --entity_id "DRUG:ALBUTEROL SULFATE" --topk 10

# Save results to JSON
python -m rgcn.src.04_infer --query "Metformin" --topk 10 --json_out predictions.json
```

## System Overview

### Knowledge Graph Construction

The KG constructor (`kg_constructor/`) processes raw biomedical data through a 9-step pipeline:

1. **Import Data** - Copy files from MedJsonify to working directory
2. **Statistics** - Generate audit reports on raw data
3. **Parse Sources** - Extract and normalize drugs, ingredients, diseases
4. **Resolve Entities** - Create canonical entity IDs and alias mappings
5. **Build Triples** - Convert raw edges to canonical triples
6. **Filter Graph** - Remove self-loops, blacklisted diseases, HAS_INGREDIENT
7. **Split Edges** - Create train/valid/test splits (80/10/10)
8. **Index DGL** - Build DGL graph for model training
9. **Analytics** - Generate visualizations and statistics

**Key Features**:
- Entity normalization: 3,359 drug names → 2,470 canonical entities
- Relation filtering: Removes HAS_INGREDIENT (not useful for prediction)
- Blacklist filtering: Removes generic diseases (e.g., "disease", "syndrome")
- Final output: 8,951 triples, 2 relations (INDICATION, CONTRAINDICATION)

### R-GCN Model

The R-GCN module (`rgcn/`) implements a neural network for link prediction:

**Architecture**:
- **Encoder**: R-GCN (learns node embeddings from multi-relational graph)
- **Decoder**: DistMult (calculates relationship scores)
- **Training**: Negative sampling with binary cross-entropy loss
- **Evaluation**: Filtered ranking with MRR and Hits@K metrics

**Current Performance** (8,951 triples, 2 relations):
- Test MRR: **0.60** (correct answer at position ~1.67 on average)
- Test Hits@1: **0.49** (48.5% top-1 accuracy)
- Test Hits@10: **0.81** (80.9% top-10 coverage)

## Data Flow

```
MedJsonify NER Output
        ↓
kg_constructor (9 steps)
        ↓
Knowledge Graph (8,951 triples)
        ↓
R-GCN Training
        ↓
Trained Model
        ↓
Drug-Disease Predictions
```

## Configuration

### Knowledge Graph Constructor

Edit `kg_constructor/configs/config.yaml`:

```yaml
filters:
  remove_self_loops: true          # Remove drug-drug or disease-disease edges
  remove_has_ingredient: true      # Remove HAS_INGREDIENT relations
  blacklist_diseases:              # Generic diseases to filter
    - "DOID:4"    # disease
    - "DOID:225"  # syndrome
    - "DOID:162"  # cancer
    - "DOID:848"  # arthritis

edge_split:
  train_ratio: 0.8
  valid_ratio: 0.1
  test_ratio: 0.1
  seed: 42
```

### R-GCN Model

Edit `rgcn/configs/baseline.yaml`:

```yaml
training:
  epochs: 200              # Training epochs
  lr: 0.001                # Learning rate
  emb_dim: 128             # Embedding dimension
  num_layers: 2            # R-GCN layers
  neg_k: 50                # Negative samples per positive
  dropout: 0.2             # Dropout rate
```

## Common Tasks

### Run Full Pipeline from Scratch

```bash
# Activate virtual environment (see section 1 above)

# Build knowledge graph
python -m kg_constructor.src.run_kg_pipeline --config kg_constructor/configs/config.yaml

# Train model
python -m rgcn.src.02_train --config rgcn/configs/baseline.yaml

# Evaluate model
python -m rgcn.src.03_evaluate

# Make predictions
python -m rgcn.src.04_infer --query "Aspirin" --topk 10
```

### Run from Specific KG Constructor Step

```bash
# Run from step 5 onwards, forcing re-execution
python -m kg_constructor.src.run_kg_pipeline --from_step 05_filter_graph --force
```

### Load and Validate Graph Data

```bash
python -m rgcn.src.01_load_data --processed_dir kg_constructor/data/processed
```

## Troubleshooting

### Virtual Environment Issues

**macOS/Linux**: If `source venv/bin/activate` fails, try:
```bash
chmod +x venv/bin/activate
source venv/bin/activate
```

**Windows**: If PowerShell script execution is disabled:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Module Not Found Errors

Make sure to set `PYTHONPATH` when running scripts:

```bash
# macOS/Linux
PYTHONPATH=/path/to/project python -m kg_constructor.src.run_kg_pipeline

# Windows (PowerShell)
$env:PYTHONPATH="/path/to/project"; python -m kg_constructor.src.run_kg_pipeline

# Windows (Command Prompt)
set PYTHONPATH=C:\path\to\project
python -m kg_constructor.src.run_kg_pipeline
```

### Low Model Performance

If R-GCN metrics are too low:
1. Increase `emb_dim` (e.g., 256)
2. Decrease `lr` (e.g., 0.0005)
3. Increase `epochs` (e.g., 300)
4. Increase `neg_k` (e.g., 100)

See `rgcn/README.md` for more troubleshooting tips.

## Requirements

- Python 3.8+
- PyTorch 1.12+
- DGL 0.9+
- NumPy, Pandas, scikit-learn
- See `requirements.txt` for full list

## Project Statistics

**Knowledge Graph**:
- Entities: 2,161 (drugs + diseases)
- Relations: 2 (INDICATION, CONTRAINDICATION)
- Triples: 8,951
- Train/Valid/Test: 7,160 / 895 / 896

**Entity Resolution**:
- Drug names: 3,359 → 2,470 canonical entities
- DrugBank mappings: 882 drugs
- Normalized name mappings: 1,892 drugs

## Documentation

- **Knowledge Graph Constructor**: `kg_constructor/src/steps/README.md`
- **R-GCN Model**: `rgcn/README.md`
- **Important Commands**: `docs/important-commands.txt`

## License

This project is for academic research purposes.

## Citation

If you use this implementation, please cite:

```
@inproceedings{schlichtkrull2018modeling,
  title={Modeling relational data with graph convolutional networks},
  author={Schlichtkrull, Michael and Kipf, Thomas N and Bloem, Peter and Berg, Rianne van den and Titov, Ivan and Welling, Max},
  booktitle={European semantic web conference},
  pages={593--607},
  year={2018}
}
```
