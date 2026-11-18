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

| Dataset identifier | Source months | CSV-derived | TXT-derived | XML-derived | Total |
|---|---|---:|---:|---:|---:|
| `2025-10_to_2025-12` | July, August and November 2025 | 631 | 114 | 2,650 | 3,395 |
| `2025-10_to_2026-03` | July, August, November and December 2025; January and February 2026 | 1,278 | 139 | 4,248 | 5,665 |

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

These datasets are intended to be stored locally because of their size. Ignore
rules prevent new untracked copies from being added accidentally. Exact source
versions, expected paths and checksums are documented in
[`external/README.md`](external/README.md). Do not rely on historical Git
objects as a distribution mechanism for these snapshots.

## Canonical graph builds

The `builds/` directory contains the canonical biomedical graphs used in the
project.

The final graph is distributed separately through
[Zenodo](https://doi.org/10.5281/zenodo.22687285). In a new clone, extract the
published package and copy `data/triples.csv` to
`builds/2025-10_to_2026-03/triples/triples_with_dgidb.csv` before preparing the
model artefacts. The graph checksum can then be verified against the build
manifest documented below.

### Initial three-month build (`2025-10_to_2025-12`)

Historical graph used as the reference build.

- Graph:
  `builds/2025-10_to_2025-12/triples/triples_with_dgidb.csv`
- Source months: July, August and November 2025
- Number of triples: `44088`
- SHA-256:
  `f3ee51e22307d7e9c6106e47db6445322ec38fbfe9fa3c3c1fb5c1fb80d83321`

Relation distribution:

| Relation | Triples |
|---|---:|
| `HAS_INGREDIENT` | 18,349 |
| `ASSOCIATED_WITH_GENE` | 9,951 |
| `INTERACTS_WITH_GENE` | 8,983 |
| `INDICATION` | 4,983 |
| `CONTRAINDICATION` | 1,822 |

### Final six-month build (`2025-10_to_2026-03`)

Main graph used for the final biomedical experiments.

- Graph:
  `builds/2025-10_to_2026-03/triples/triples_with_dgidb.csv`
- Source months: July, August, November and December 2025; January and
  February 2026
- Number of triples: `51759`
- SHA-256:
  `3d5c7df7d021a077b3d5c7fdd986ee6ac1343fec57d585b66d2f622eb387ad72`

Relation distribution:

| Relation | Triples |
|---|---:|
| `HAS_INGREDIENT` | 24,030 |
| `ASSOCIATED_WITH_GENE` | 9,951 |
| `INTERACTS_WITH_GENE` | 9,414 |
| `INDICATION` | 6,143 |
| `CONTRAINDICATION` | 2,221 |

The directory names `2025-10_to_2025-12` and `2025-10_to_2026-03` are legacy
build identifiers retained to preserve the existing experiment configurations.
They do not represent the exact temporal coverage of the regulatory data.

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
