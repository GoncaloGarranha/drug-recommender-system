# R-GCN for Drug-Disease Link Prediction

This module trains a neural network (R-GCN) to predict which drugs are indicated or contraindicated for specific diseases. The model learns from the knowledge graph and can suggest drug-disease relationships.

## Overview

The R-GCN pipeline has 4 steps:

1. **Load Data** (`01_load_data.py`) - Load and validate the graph data
2. **Train Model** (`02_train.py`) - Train the neural network
3. **Evaluate Model** (`03_evaluate.py`) - Measure prediction quality
4. **Make Predictions** (`04_infer.py`) - Generate recommendations for specific drugs

## How It Works

The model has these components:
- **R-GCN Encoder**: Learns vector representations of drugs and diseases by analyzing their connections in the graph
- **DistMult Decoder**: Calculates scores to predict if a drug-disease relationship exists
- **Negative Sampling**: During training, creates fake negative examples to teach the model what relationships don't exist
- **Filtered Ranking**: During evaluation, removes known relationships so the model isn't penalized for correct predictions

## Configuration

Configuration files are in `configs/`:
- `baseline.yaml` - Standard settings for drug-disease prediction
- `ablations.yaml` - Alternative settings for experiments

Main parameters:
```yaml
training:
  epochs: 200            # How many times to go through the data
  lr: 0.001              # Learning rate (how fast the model learns)
  weight_decay: 0.0001   # Prevents overfitting
  dropout: 0.2           # Randomly turns off neurons to improve generalization
  emb_dim: 128           # Size of the learned vectors
  num_bases: 3           # Reduces model parameters (simplifies the model)
  num_layers: 2          # Number of R-GCN layers
  neg_k: 50              # Negative examples per positive example
  eval_every: 10         # Test the model every 10 epochs
  hits_k: [1, 3, 10]     # Metrics to calculate
```

## How to Run

### 1. Load and Validate Data

Check if all required files exist and are correct:

python -m rgcn.src.01_load_data --processed_dir kg_constructor/data/processed

**Output**: Graph statistics and split counts

### 2. Train the Model

Train the R-GCN network on the graph data:

python -m rgcn.src.02_train --config rgcn/configs/baseline.yaml

**Options**:
- `--config`: Path to YAML config file (required)
- `--seed`: Random seed for reproducibility (default: from config)

**Output**: 
- Model checkpoints saved to `rgcn/runs/<timestamp>_<run_name>/checkpoints/`
- Training logs in `train_log.json`
- Best model saved as `best.pt`

**Metrics saved**:
- MRR (Mean Reciprocal Rank - average position of correct answer)
- Hits@1, Hits@3, Hits@10 (% of times correct answer is in top-1, top-3, top-10)
- Loss per epoch

### 3. Evaluate the Model

Test the trained model on held-out test data:

python -m rgcn.src.03_evaluate --runs_dir rgcn/runs

**Options**:
- `--runs_dir`: Directory with training runs (default: `rgcn/runs`)
- `--run_dir`: Specific run to evaluate (default: most recent)

**Output**: 
- Test set evaluation results
- Metrics for each split (train/valid/test)
- Saved to `evaluation_results.json`

**Metrics**:
- MRR (average position of correct answer)
- Hits@1, Hits@3, Hits@10 (success rate in top-K)
- Mean Rank, Median Rank (average and median position)

### 4. Generate Predictions

Get top-K disease predictions for a specific drug:

# Search by drug name (fuzzy match)
python -m rgcn.src.04_infer --query "Albuterol" --topk 10

# Save results to JSON file
python -m rgcn.src.04_infer --query "Metformin" --topk 10 --json_out predictions.json

# Show internal entity IDs (for debugging)
python -m rgcn.src.04_infer --query "Aspirin" --topk 5 --show_ids

**Options**:
- `--query`: Drug name (fuzzy matching)
- `--entity_id`: Exact entity ID from the graph
- `--topk`: Number of predictions to return (default: 10)
- `--show_ids`: Show internal entity IDs
- `--json_out`: Save results to JSON file

**Output format**:
```
Top 10 Indicações para 'Albuterol':
  1. Asthma                    (score: 0.8523, DOID:2841)
  2. Chronic Obstructive...    (score: 0.7891, DOID:3083)
  3. Bronchitis                (score: 0.7234, DOID:6132)
  ...
```

## Expected Performance

Based on the current graph (8,951 triples, 2 relations) with baseline configuration:

| Metric    | Test Set |
|-----------|----------|
| MRR       | **0.60** |
| Hits@1    | **0.49** |
| Hits@3    | **0.67** |
| Hits@10   | **0.81** |

**Interpretation**:
- **MRR 0.60**: The correct disease appears on average at position 1.67 in the ranked list
- **Hits@1 48.5%**: The model's top prediction is correct almost half the time
- **Hits@3 66.9%**: The correct disease is in the top-3 predictions 2 out of 3 times
- **Hits@10 80.9%**: The correct disease is in the top-10 predictions 4 out of 5 times

**Note**: Performance may vary with different hyperparameters, seeds, or data splits.

## Directory Structure

```
rgcn/
├── configs/
│   ├── baseline.yaml          # Standard configuration
│   └── ablations.yaml         # Experimental configurations
├── runs/
│   └── <timestamp>_<name>/    # Training outputs
│       ├── checkpoints/
│       │   ├── best.pt        # Best model
│       │   └── last.pt        # Last epoch
│       ├── train_log.json     # Training metrics log
│       └── config.yaml        # Config used for this run
├── src/
│   ├── 01_load_data.py        # Data validation
│   ├── 02_train.py            # Training script
│   ├── 03_evaluate.py         # Evaluation script
│   ├── 04_infer.py            # Prediction script
│   ├── models/
│   │   ├── rgcn_encoder.py   # R-GCN encoder
│   │   └── decoders.py       # DistMult decoder
│   └── utils/
│       ├── io.py             # Data loading functions
│       └── seed.py           # Reproducibility functions
└── README.md                  # This file
```

## Architecture Details

### R-GCN Encoder

The encoder learns node representations by combining information from neighbors, weighted by relationship type:

```
h_i^(l+1) = σ(Σ_r W_r^(l) Σ_{j∈N_i^r} h_j^(l) / |N_i^r| + W_0^(l) h_i^(l))
```

- `h_i^(l)`: Vector representation of node i at layer l
- `N_i^r`: Neighbors of node i through relation r
- `W_r^(l)`: Weight matrix for relation r
- Basis decomposition reduces parameters: `W_r = Σ_b a_{rb} V_b`

### DistMult Decoder

Calculates triple scores using a simple bilinear function:

```
score(h, r, t) = h^T R_r t
```

- `h, t`: Vector representations of source and target nodes
- `R_r`: Diagonal matrix for relation r
- Higher scores mean more likely relationships

### Loss Function

Binary cross-entropy with negative sampling:

```
L = - Σ_{(h,r,t)∈D+} log σ(score(h,r,t)) 
    - Σ_{(h,r,t')∈D-} log(1 - σ(score(h,r,t')))
```

- `D+`: Real edges from the dataset
- `D-`: Fake negative examples (randomly generated)

## Graph Statistics

Current knowledge graph (after removing HAS_INGREDIENT):

- **Entities**: 2,161 (drugs + diseases)
- **Relations**: 2 (INDICATION, CONTRAINDICATION)
- **Triples**: 8,951 total
  - INDICATION: 6,568 (73.4%)
  - CONTRAINDICATION: 2,383 (26.6%)
- **Splits**: 80/10/10 train/valid/test
  - Train: 7,160 edges
  - Valid: 895 edges
  - Test: 896 edges

## Implementation Details

### Filtered Ranking

During evaluation, known triples are removed from candidates to avoid penalizing correct predictions:

```python
# Build set of all known (drug, INDICATION, disease) triples
known_triples = {(h, r, t) for h, r, t in train + valid + test}

# Remove known triples when ranking candidates
scores[known_mask] = -float('inf')
```

### Drug → Disease Focus

The model predicts **drug → disease** relationships:
- Source entities: Drugs
- Target entities: Diseases
- Candidate set: All diseases in the graph

Contraindications are treated the same way during inference (drug → contraindicated disease).

### Entity Resolution

Entity IDs from the KG constructor are preserved:
- Drugs: `DRUG:<NORMALIZED_NAME>` (e.g., `DRUG:ALBUTEROL SULFATE`)
- Diseases: `DOID:<ID>` (e.g., `DOID:2841`)

The inference script supports both:
1. Fuzzy name matching (`--query "Albuterol"`)
2. Exact ID matching (`--entity_id "DRUG:ALBUTEROL SULFATE"`)

## Troubleshooting

### Low Performance

If MRR < 0.30 or Hits@10 < 0.50:
1. Increase `emb_dim` (e.g., 256)
2. Decrease `lr` (e.g., 0.0005)
3. Increase `epochs` (e.g., 300)
4. Adjust `dropout` (try 0.1 or 0.3)
5. Increase `neg_k` (e.g., 100)

### Out of Memory

If training crashes with memory error:
1. Reduce `emb_dim`
2. Reduce `neg_k`
3. Use `device: "cpu"` instead of GPU
4. Reduce batch size (if batching is implemented)

### No Improvement During Training

If validation metrics don't improve:
1. Check for data leakage (overlap between train/valid/test)
2. Check if splits are balanced
3. Reduce `lr` to prevent divergence
4. Check for NaN losses (increase weight decay)

### Unexpected Predictions

If predictions seem wrong:
1. Verify entity ID exists: use `--show_ids` flag
2. Check model checkpoint is loading correctly
3. Make sure latest KG artifacts are being used
4. Compare scores with known ground truth

## Dependencies

- Python 3.8+
- PyTorch 1.12+
- DGL 0.9+
- NumPy
- PyYAML

Install via:
```bash
pip install -r requirements.txt
```

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
