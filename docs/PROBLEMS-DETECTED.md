# Problems Detected and Review Status

This document records technical and methodological findings discovered during
the development and review of the project. It is a living review register, not
a list of defects that are all still active.

## Status definitions

| Status | Meaning |
|---|---|
| Resolved | The current implementation contains a verified correction or validation. |
| Mitigated | Controls exist, but part of the original limitation remains. |
| Accepted limitation | The behaviour is known and intentionally retained; it must be disclosed when reporting results. |
| Open | Additional implementation or documentation work is still required. |

## Current summary

| ID | Title | Severity | Status |
|---|---|---:|---|
| KG-001 | Duplicate triples in generated biomedical graph | Critical | Resolved |
| KG-002 | Duplicate triples overlap across train/validation/test splits | Critical | Resolved |
| KG-003 | DrugBank identifiers are not normalized | High | Resolved |
| KG-004 | Loss of provenance when building final triples | High | Mitigated |
| KG-005 | Assumption that each input file has one primary drug | Medium/High | Accepted limitation |
| KG-006 | Drugs may be resolved by normalized name when an external identifier is unavailable | Medium | Accepted limitation |
| DOC-001 | README references missing `kg_constructor/configs/config.yaml` | High | Open |
| DOC-002 | Inconsistent processed-directory names | Medium/High | Mitigated |
| METH-001 | Biomedical results affected by target-edge split leakage | Critical | Resolved |
| METH-002 | Generic benchmark evaluation protocol | High | Resolved |
| METH-003 | `INDICATION` and `CONTRAINDICATION` coexist for the same drug-disease pairs | High | Accepted limitation |
| EVAL-001 | Need explicit split-integrity validation | Critical | Resolved |
| REPRO-001 | Three-month graph does not match its tracked manifest | High | Open |
| REPRO-002 | Open Targets and DGIdb source versions are not recorded | High | Open |
| REPRO-003 | Final configurations reference a missing six-month `publication` build | High | Open |
| REPRO-004 | Six-month quality report records an inconsistent input path | Medium | Open |
| CODE-001 | CSV read/write helpers are duplicated across KG steps | Low/Medium | Open |
| TEST-001 | Automated test coverage and test-runner configuration are insufficient | Medium | Open |

## Resolved

### KG-001 — Duplicate triples in the biomedical graph

Canonical triples are deduplicated during graph construction and again where
external gene relations are added. The final quality gate fails when duplicate
`(src, rel, dst)` triples are present. The final six-month quality report records
zero duplicate triples.

### KG-002 — Duplicate triples across experimental splits

The shared split-integrity validator checks duplicates inside each split and
exact-triple overlap between the training graph, validation set, and test set.
The persisted splits are copied into each run directory so that evaluation can
reuse the exact assignments used during training.

### KG-003 — DrugBank identifier normalization

Entity resolution normalizes identifier prefixes and validates DrugBank entity
formats. The final quality report records zero bad DrugBank entities. Name-based
fallback resolution remains possible and is tracked separately under KG-006.

### METH-001 — Target-edge split leakage

Only training `INDICATION` edges are included in the message-passing graph.
Validation and test indications are excluded from that graph and are used only
for model selection/evaluation and for the complete known-positive set required
by filtered ranking. This resolves direct leakage of held-out target edges.

This status does not cover the cross-relation ambiguity described in METH-003.

### EVAL-001 — Explicit split-integrity validation

`models/shared/experiment_integrity.py` now validates duplicate rows, exact
split disjointness, known-positive completeness, candidate-pool coverage, and
preservation of official benchmark splits. It emits a machine-readable report
and stops the experiment when a required check fails.

## Mitigated

### KG-004 — Provenance in final triples

The constructor produces `triples_with_provenance.csv` and build reports during
the canonical construction stage. Dataset manifests also record graph hashes,
counts, and selected construction metadata. However, provenance is not carried
as edge-level metadata through every enrichment and final model artifact.

Remaining work: document the provenance boundary explicitly and, if required,
propagate source identifiers through the Open Targets and DGIdb enrichment
steps.

### DOC-002 — Processed-directory naming

`models/shared/graph_artifacts.py` contains compatibility resolution for older
and newer processed-directory layouts. This reduces runtime failures, but the
repository still contains historical names such as `publication`, `legacy`,
`corrected`, and `main`. Documentation and final configurations should converge
on one canonical layout.

## Accepted limitations

### KG-005 — One primary drug per input file

Source parsing currently treats the first detected drug as the primary drug for
parts of the extraction process. This is retained as an assumption about the
MedJsonify input structure. It may misattribute relations in documents that
describe multiple primary drugs and should be stated in the data-construction
limitations.

### KG-006 — Name-based entity resolution fallback

When a stable external identifier is unavailable, entity resolution can use a
normalized name to select a canonical DrugBank or ChEBI identifier. Conservative
rules and ambiguity rejection reduce the risk, but name-based identity is not
equivalent to identifier-based identity. Resolution reports should be retained
with each canonical build.

### METH-002 — Generic benchmark evaluation protocol

Generic link-prediction benchmarks are now evaluated using filtered ranking in
both directions, head and tail, with the complete entity set as the candidate
pool. Online validation during training uses the same bidirectional protocol
used by the external evaluator.

A separate evaluation bug was also corrected: when `target_relation=None`,
`evaluate.py` previously reconstructed the encoder graph with validation and
test triples included as context, causing held-out-edge leakage and inflated
metrics. Generic evaluation now builds the encoder graph from training triples
only. Historical benchmark rows affected by this bug are explicitly marked
`INVALID_EVALUATION_LEAKAGE` in `results/benchmarks.csv` and must not be used
for reporting.

### METH-003 — Coexisting indication and contraindication relations

The six-month graph contains 1,427 drug-disease pairs represented by both
`INDICATION` and `CONTRAINDICATION`. They are intentionally retained because the
graph preserves the extracted evidence and because indication and
contraindication can depend on clinical context, dose, population, formulation,
or source.

This choice may favour link prediction: 149 of the 615 test indications also
have a direct `CONTRAINDICATION` edge in the message-passing graph. The model can
therefore observe that the held-out drug and disease are directly connected by
another relation. This is not direct target-edge leakage, but it is a potential
cross-relation structural shortcut.

Reporting requirement: disclose the retained conflicts, their frequency, and
their possible effect on performance. Results should be described as applying
to the graph as constructed, rather than as evidence that the ambiguity has no
effect.

## Open reproducibility issues

### REPRO-001 — Three-month manifest mismatch

The current `2025-10_to_2025-12` canonical CSV contains 44,088 triples, while
its tracked manifest declares 40,746 triples and a different SHA-256 hash. The
quality report and temporal result table use 44,088 triples. Either the manifest
must be regenerated for the current graph or the graph matching the manifest
must be restored and clearly versioned.

### REPRO-002 — External-source versions are missing

The canonical manifests identify Open Targets and DGIdb as sources but record
their versions as `not recorded`. Record the release or download date, source
URL, file hashes, and any extraction filters used.

### REPRO-003 — Missing six-month publication build

Final PyKEEN and content-based configurations reference
`kg_constructor/data/builds/2025-10_to_2026-03_publication`, but that canonical
build directory is not present in the current workspace. Processed artifacts
and results exist, but those executions cannot be reconstructed solely from the
saved configuration unless the source build is restored or the paths are
updated to a hash-equivalent canonical graph.

### REPRO-004 — Inconsistent quality-report path

The quality report stored under `2025-10_to_2026-03/reports` declares its input
as the missing `2025-10_to_2026-03_publication` path. Its counts match the
current six-month graph, but the recorded provenance path should be corrected or
explained.

## Other open issues

### DOC-001 — Missing default KG configuration

The root and KG constructor READMEs still show commands using
`kg_constructor/configs/config.yaml`, which is not present. Replace these
examples with one of the period-specific configurations or add a maintained
default configuration.

### CODE-001 — Duplicated CSV utilities

Several KG steps contain overlapping CSV manipulation logic. Consolidating
these operations in the shared I/O and validation modules would reduce drift,
but this does not currently affect the validity of the final results.

### TEST-001 — Insufficient automated tests

The Python modules compile and the two encoder diagnostic scripts execute, but
the project does not currently provide a configured automated test suite. The
diagnostic scripts print tensors without asserting expected behaviour, and
`pytest` is not installed in the project environment.

Priority tests should cover split disjointness, filtered ranking, tie handling,
negative sampling, inverse-edge construction, candidate typing, manifest/hash
consistency, and an end-to-end miniature experiment.
