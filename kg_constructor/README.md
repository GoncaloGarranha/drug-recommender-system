# KG Constructor Steps

This directory contains the steps for the Knowledge Graph construction. Each script is responsible for a specific stage in the data processing workflow.

## Running the Complete Pipeline

To run the entire knowledge graph construction pipeline:

# Run all steps in sequence
python -m kg_constructor.src.run_kg_pipeline --config kg_constructor/configs/(choosen_config).yaml

# Individual Steps

## 00_import_data.py

Imports files (xml, csv, txt) from `MedJsonify-main/NER/data/entities/` to `kg_constructor/data/input/`.

### Run command
```bash
python -m kg_constructor.src.steps.00_import_data
```

## 01_stats.py

Generates statistics and audit report from the input files in `kg_constructor/data/input/`.

### Run command
```bash
python -m kg_constructor.src.steps.01_stats
```

## 02_parse_sources.py

Parses input JSON files and generates clean CSV tables with normalized data (drugs, ingredients, diseases, edges).

### Run command
```bash
python -m kg_constructor.src.steps.02_parse_sources
```

## 03_resolve_entities.py

Builds canonical entity IDs and creates alias mappings from the parsed CSV tables.

### Run command
```bash
python -m kg_constructor.src.steps.03_resolve_entities
```

## 04_build_triples.py

Converts raw edges into canonical triples (src, rel, dst) using entity mappings.

### Run command
```bash
python -m kg_constructor.src.steps.04_build_triples
```

## 05_filter_graph.py

Applies cleaning filters to canonical triples (removes self-loops, blacklisted diseases).

### Run command
```bash
python -m kg_constructor.src.steps.05_filter_graph
```

---

**Note**: Train/validation/test splitting and DGL artifact generation now happen inside the R-GCN pipeline (`models/rgcn/src/` and `models/shared/`). The KG constructor produces both:

- `triples.csv`: full biomedical graph;
- `triples.filtered.csv`: graph without selected relations/noise, used for ablation or focused therapeutic experiments.

Use `triples.csv` for the main full-graph experiments, including `HAS_INGREDIENT` when the study design requires it. Use `triples.filtered.csv` for filtered ablations or other focused experiments where `HAS_INGREDIENT` is excluded. These two outputs represent different experimental settings and should not be compared as if they were the same dataset.