# KG Constructor Steps

This directory contains the steps for the Knowledge Graph construction. Each script is responsible for a specific stage in the data processing workflow.

## Running the Complete Pipeline

To run the entire knowledge graph construction pipeline:

# Run all steps in sequence
python -m kg_constructor.src.run_kg_pipeline --config kg_constructor/configs/config.yaml

# Individual Steps

## 00_import_data.py

Imports files (xml, csv, txt) from `MedJsonify-main/NER/data/entities/` to `kg_constructor/data/input/`.

### Run command
python -m kg_constructor.src.steps.00_import_data

## 01_stats.py

Generates statistics and audit report from the input files in `kg_constructor/data/input/`.

### Run command
python -m kg_constructor.src.steps.01_stats

## 02_parse_sources.py

Parses input JSON files and generates clean CSV tables with normalized data (drugs, ingredients, diseases, edges).

### Run command
python -m kg_constructor.src.steps.02_parse_sources

## 03_resolve_entities.py

Builds canonical entity IDs and creates alias mappings from the parsed CSV tables.

### Run command
python -m kg_constructor.src.steps.03_resolve_entities

## 04_build_triples.py

Converts raw edges into canonical triples (src, rel, dst) using entity mappings.

### Run command
python -m kg_constructor.src.steps.04_build_triples

## 05_filter_graph.py

Applies cleaning filters to canonical triples (removes self-loops, blacklisted diseases).

### Run command
python -m kg_constructor.src.steps.05_filter_graph

---

**Note**: Train/validation/test splitting and DGL artifact generation now happen inside the R-GCN pipeline (`rgcn/src/utils/io.py`). The KG constructor produces `triples.filtered.csv` as its final output.