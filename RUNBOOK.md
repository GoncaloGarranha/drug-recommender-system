## DISEASE_TO_DRUG_FULL — Ingredients included

```text
1 - python -m kg_constructor.src.run_kg_pipeline --config kg_constructor/configs/config.yaml --force
2 - python -m models.rgcn.src.prepare_graph --config models/rgcn/configs/biomedical/disease_to_drug.yaml
3 - python -m models.rgcn.src.load_data --processed_dir models/rgcn/data/processed/disease_to_drug_full
4 - python -m models.rgcn.src.train --config models/rgcn/configs/biomedical/disease_to_drug.yaml
5 - python -m models.rgcn.src.evaluate --runs_dir results/runs/rgcn --split test
```

## DISEASE_TO_DRUG_WITH_GENES — Main thesis task

This is the main biomedical task used in the thesis.

Target relation:

## DISEASE_TO_DRUG_WITH_GENES — Main thesis task

This is the main biomedical task used in the thesis.

Target relation:

```text
Disease → INDICATION → Drug

1 - python -m kg_constructor.src.run_kg_pipeline --config kg_constructor/configs/config.yaml --force
2 - python -m models.rgcn.src.prepare_graph --config models/rgcn/configs/biomedical/disease_to_drug_with_genes.yaml
3 - python -m models.rgcn.src.load_data --processed_dir models/rgcn/data/processed/disease_to_drug_with_genes
4 - python -m models.rgcn.src.train --config models/rgcn/configs/biomedical/disease_to_drug_with_genes.yaml
5 - python -m models.rgcn.src.evaluate --runs_dir results/runs/rgcn --split test
```