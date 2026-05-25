DISEASE_TO_DRUG_FULL (INGREDIENTS INCLUDED)
1 - python -m kg_constructor.src.run_kg_pipeline --config kg_constructor/configs/config.yaml --force
2 - python -m models.rgcn.src.00_prepare_graph --config .\models\rgcn\configs\disease_to_drug.yaml
3 - python -m models.rgcn.src.01_load_data --processed_dir models/rgcn/data/processed/disease_to_drug_full
4 - python -m models.rgcn.src.02_train --config .\models\rgcn\configs\disease_to_drug.yaml
5 - python -m models.rgcn.src.03_evaluate --runs_dir .\models\rgcn\runs

Notes
- Run the steps in order.
- If `00_prepare_graph` or `02_train` complains about split metadata, remove the existing `splits` folder for the dataset and rerun the prepare step.
- `03_evaluate` can also be pointed to a specific `--run_dir` if you want to evaluate a single trained run.
