# Benchmark datasets

This directory contains static benchmark graphs used by the R-GCN experiments.

The processed `triples.csv` files are versioned. Raw source files under `raw/` are kept locally and ignored by Git.

| Dataset | Triples | Entities | Relations | Train | Valid | Test | Config |
|---|---:|---:|---:|---:|---:|---:|---|
| `fb15k` | 592213 | 14951 | 1345 | 483142 | 50000 | 59071 | fb15k.yaml |
| `fb15k237` | 310116 | 14541 | 237 | 272115 | 17535 | 20466 | fb15k237.yaml |
| `fb15k237_sampled` | 36251 | 12250 | 237 | 35103 | 573 | 575 | fb15k237_sampled.yaml |
| `hetionet_biomedical` | 14117 | 1462 | 3 | 11293 | 1411 | 1413 | none |
| `hetionet_enriched` | 31825 | 7758 | 4 | 25460 | 3182 | 3183 | none |
| `wn18` | 151442 | 40943 | 18 | 141442 | 5000 | 5000 | wn18.yaml |
| `wn18rr` | 93003 | 40943 | 11 | 86835 | 3034 | 3134 | wn18rr.yaml |
| `wn18rr_sampled` | 36251 | 29656 | 11 | 33847 | 1183 | 1221 | wn18rr_sampled.yaml |

Hashes and machine-readable metadata are recorded in `manifest.json`.

The Hetionet datasets are retained as static benchmark assets but currently have no active YAML configuration in `models/rgcn/configs/benchmarks/`.
