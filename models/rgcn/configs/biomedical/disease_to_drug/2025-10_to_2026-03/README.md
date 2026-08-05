# Disease-to-drug configurations: 2025-10 to 2026-03

These configurations use the consolidated biomedical knowledge graph:

`kg_constructor/data/builds/2025-10_to_2026-03/triples/triples_with_dgidb.csv`

## Main experiment

- `main/all_supported_relations.yaml`

Uses all supported relations:

- `INDICATION`
- `CONTRAINDICATION`
- `HAS_INGREDIENT`
- `ASSOCIATED_WITH_GENE`
- `INTERACTS_WITH_GENE`

## Relation studies

- `relation_studies/without_has_ingredient.yaml`
- `relation_studies/without_contraindication.yaml`
- `relation_studies/without_associated_with_gene.yaml`
- `relation_studies/without_interacts_with_gene.yaml`

Each relation study removes exactly one contextual relation while preserving the
dataset, split strategy, seed and training hyperparameters.
