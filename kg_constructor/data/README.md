# Knowledge Graph Data

This directory contains the datasets used to construct and evaluate the
biomedical knowledge graph.

The data are organised into four main areas:

- `input/`: MedJsonify entity files;
- `external/`: external biomedical sources;
- `builds/`: canonical biomedical graph builds;
- `benchmarks/`: standard and biomedical benchmark datasets.

## Directory structure

```text
data/
├── README.md
├── input/
│   └── medjsonify/
│       ├── 2025-10_to_2025-12/
│       │   ├── csv_files/
│       │   ├── txt_files/
│       │   ├── xml_files/
│       │   └── manifest.json
│       └── 2025-10_to_2026-03/
│           ├── csv_files/
│           ├── txt_files/
│           ├── xml_files/
│           ├── manifest.json
│           └── merge_manifest.csv
├── external/
│   ├── dgidb/
│   ├── open_targets/
│   └── diseases_for_genes.csv
├── builds/
│   ├── 2025-10_to_2025-12/
│   │   ├── interim/
│   │   ├── mappings/
│   │   ├── processed/
│   │   ├── reports/
│   │   ├── triples/
│   │   └── manifest.json
│   └── 2025-10_to_2026-03/
│       ├── interim/
│       ├── mappings/
│       ├── processed/
│       ├── reports/
│       ├── triples/
│       └── manifest.json
└── benchmarks/
```

## MedJsonify inputs

The MedJsonify inputs are grouped according to the format of the original
source documents:

- `csv_files/`: entities extracted from CSV documents;
- `txt_files/`: entities extracted from plain-text documents;
- `xml_files/`: entities extracted from XML documents.

Although the directories retain the original document type in their names,
their contents are JSON entity files produced by MedJsonify.

| Period | CSV-derived | TXT-derived | XML-derived | Total |
|---|---:|---:|---:|---:|
| 2025-10 to 2025-12 | 631 | 114 | 2650 | 3395 |
| 2025-10 to 2026-03 | 1278 | 139 | 4248 | 5665 |

Each period contains a `manifest.json` recording its temporal coverage, file
counts, sizes and provenance.

The combined `2025-10_to_2026-03` dataset also contains
`merge_manifest.csv`, which records how the source files were incorporated
into the final input dataset.

## External biomedical sources

The `external/` directory contains the sources used to enrich the biomedical
graph:

- `open_targets/`: disease–gene associations;
- `dgidb/`: drug–gene interactions;
- `diseases_for_genes.csv`: disease identifiers used during the integration
  of gene associations.

These datasets are stored locally because of their size and are not tracked
by Git.

## Canonical graph builds

The `builds/` directory contains the canonical biomedical graphs used in the
project.

### 2025-10 to 2025-12

Historical graph used as the reference build.

- Graph:
  `builds/2025-10_to_2025-12/triples/triples_with_dgidb.csv`
- Number of triples: `40746`
- SHA-256:
  `77603322dc1777c4fa231338af58a82734844fec86cfaa70d650ae386fb7b71d`

Relation distribution:

| Relation | Triples |
|---|---:|
| `INDICATION` | 5739 |
| `CONTRAINDICATION` | 2220 |
| `HAS_INGREDIENT` | 18349 |
| `ASSOCIATED_WITH_GENE` | 9943 |
| `INTERACTS_WITH_GENE` | 4495 |

### 2025-10 to 2026-03

Main graph used for the final biomedical experiments.

- Graph:
  `builds/2025-10_to_2026-03/triples/triples_with_dgidb.csv`
- Number of triples: `49903`
- SHA-256:
  `713df059da00efa45479aa4179fea533211c2ed0571abf128470caaee10126bb`

Relation distribution:

| Relation | Triples |
|---|---:|
| `INDICATION` | 7216 |
| `CONTRAINDICATION` | 2752 |
| `HAS_INGREDIENT` | 24978 |
| `ASSOCIATED_WITH_GENE` | 10423 |
| `INTERACTS_WITH_GENE` | 4534 |

The configuration files associated with these builds are located in:

```text
kg_constructor/configs/biomedical/periods/
├── 2025-10_to_2025-12.yaml
└── 2025-10_to_2026-03.yaml
```

## Build directories

A build can contain the following directories:

- `triples/`: generated graph files;
- `reports/`: pipeline execution and validation reports;
- `interim/`: intermediate files generated during graph construction;
- `mappings/`: entity-resolution tables and mappings;
- `processed/`: reusable artefacts derived from the graph.

Some of these directories may be empty. This occurs when only the canonical
graph, its manifest and its validation reports were preserved from the
original execution.

## Benchmarks

The `benchmarks/` directory contains datasets used to evaluate the
link-prediction models independently from the biomedical graph.

It includes standard datasets such as:

- FB15k;
- FB15k-237;
- WN18;
- WN18RR;

as well as sampled and biomedical benchmark variants.

Converted benchmark files use the following format:

```csv
src,rel,dst,split
```

The `split` column preserves the original training, validation and test
partitions when available.

## Storage policy

Large or generated data files are excluded from Git, including:

- MedJsonify entity files;
- raw external datasets;
- raw benchmark files;
- generated graph triples;
- intermediate and processed artefacts;
- pipeline reports.

The following lightweight files remain tracked:

- `README.md`;
- dataset and build manifests;
- configuration files;
- provenance metadata.

This keeps the repository small while preserving the information required to
identify, validate and reproduce each dataset.