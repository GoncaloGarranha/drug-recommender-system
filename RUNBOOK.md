# Reproducibility Runbook

## Final six-month disease-to-drug experiment

The main thesis task queries a disease and ranks candidate drugs for the
`INDICATION` relation.

### 1. Build the biomedical Knowledge Graph

```bash
python -m kg_constructor.src.run_kg_pipeline \
  --config kg_constructor/configs/biomedical/periods/2025-10_to_2026-03.yaml
```

Add `--force` only when the existing generated artefacts should be rebuilt.

### 2. Prepare the R-GCN graph artefacts and fixed splits

```bash
python -m models.rgcn.src.prepare_graph \
  --config models/rgcn/configs/biomedical/disease_to_drug/2025-10_to_2026-03/main/all_supported_relations.yaml \
  --force
```

### 3. Validate the prepared artefacts

```bash
python -m models.rgcn.src.load_data \
  --processed_dir models/rgcn/data/processed/disease_to_drug/six_months_random_split
```

### 4. Train the R-GCN

```bash
python -m models.rgcn.src.train \
  --config models/rgcn/configs/biomedical/disease_to_drug/2025-10_to_2026-03/main/all_supported_relations.yaml
```

### 5. Evaluate the trained run

```bash
python -m models.rgcn.src.evaluate \
  --runs_dir results/runs/rgcn \
  --split test
```

### 6. Run the content-based baseline on the same split

```bash
python -m models.content-based.src.baseline \
  --config models/rgcn/configs/biomedical/disease_to_drug/2025-10_to_2026-03/main/all_supported_relations.yaml \
  --split test
```

The historical three-month experiment uses the corresponding configurations
under `2025-10_to_2025-12`.
