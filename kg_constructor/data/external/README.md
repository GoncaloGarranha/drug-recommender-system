# External Biomedical Data

The Knowledge Graph construction pipeline uses external biomedical datasets
that are too large to distribute as part of the source-code repository. Keep
the downloaded snapshots under this directory using the layout documented
below.

The canonical graph and its provenance are published separately on Zenodo. The
versions in this document describe the source snapshots used to construct the
final dissertation dataset.

## Open Targets Platform

- Release: `26.03`
- Retrieved: `2026-06-12`
- Official release directory:
  <https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.03/output/>
- Purpose: disease-gene associations and entity metadata

Obtain the Parquet datasets named `disease`, `association_overall_direct` and
`target` from that release and arrange them as follows:

```text
kg_constructor/data/external/open_targets/
├── disease/
│   └── disease.parquet
├── association_overall_direct/
│   └── *.parquet
└── target/
    └── *.parquet
```

For the snapshot used in the final experiments, the combined SHA-256 digest of
the sorted Parquet-file checksum list is:

```text
8b518ffcfde9f23eb6df467723df3fc383333e395d0cf78500b77946d2a990c5
```

It can be recalculated from the `open_targets` directory with:

```bash
find . -type f -name '*.parquet' -print0 \
  | sort -z \
  | xargs -0 shasum -a 256 \
  | shasum -a 256
```

## DGIdb

- Application version: `v.5.0.7`
- Data version: `Dec-2024`
- Retrieved: `2026-07-06`
- Official project and release history: <https://github.com/dgidb/dgidb-v5>
- Purpose: drug-gene interactions

Download the December 2024 TSV release and arrange it as follows:

```text
kg_constructor/data/external/dgidb/
├── interactions.tsv
├── drugs.tsv
└── genes.tsv
```

Expected SHA-256 checksums:

```text
ee255dce75924ef5075d8204fc7d0276a463984c555e41b9d55f6a46473db70b  interactions.tsv
0f665056cd7db45cdc132c96445699c7196e0815f935d9fca8ecd26e6382efe8  drugs.tsv
538b827679cf1f0cd837aa9c9b17cada62359adee459b8688aa95f9919427196  genes.tsv
```

Only `interactions.tsv` is required by the current graph-construction
configuration. The other files are retained as supporting source metadata.

## Validation

Before building the graph, verify that the paths above exist. The final build
manifest records the same source versions and the checksum of the DGIdb
interaction file:

```text
kg_constructor/data/builds/2025-10_to_2026-03/manifest.json
```

These datasets remain subject to the terms and citation requirements of their
respective upstream providers.
