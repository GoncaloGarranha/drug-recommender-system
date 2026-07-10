# Future extension: external indication sources

## 1. Purpose

This document describes how external drug indication sources could be integrated into the project in future work.

The current project does not implement this extension. The goal is to leave the repository prepared and well documented by specifying:

- where external indications would enter the pipeline;
- the expected input format;
- the new relation types that would be created;
- how data leakage would be avoided;
- how the impact of the extension would be evaluated.

This extension is intentionally left as future work to preserve the current experimental setup and avoid changing the definition of the main prediction target at this stage.

## 2. Current project scope

The current biomedical recommendation task is disease-to-drug recommendation.

The target relation is:

```text
INDICATION
```

Internally, the knowledge graph stores indication triples as:

```text
Drug --INDICATION--> Disease
```

The R-GCN model is then evaluated by ranking candidate drugs for a given disease.

The current main KG contains relation types such as:

```text
INDICATION
CONTRAINDICATION
HAS_INGREDIENT
INTERACTS_WITH_GENE
ASSOCIATED_WITH_GENE
```

The `INDICATION` relation is treated as the main supervised target for training, validation, and test splits.

## 3. Why external indications are not added now

External indication sources could increase the number of known drug-disease associations. However, adding them directly at this stage would change the experimental setting.

The main risks are:

1. target leakage;
2. changing the definition of the prediction target;
3. mixing evidence types with different reliability levels;
4. making results incomparable with previous experiments;
5. inflating performance if validation or test indications are reintroduced as graph context.

For these reasons, external indications should not be merged directly into the existing `INDICATION` relation without redesigning the split and evaluation protocol.

## 4. Where this extension would enter the project

The extension would enter after the current KG construction pipeline has already resolved entities and built the base drug-disease graph.

A possible future pipeline position would be after entity resolution and graph filtering, but before the final quality gate.

Conceptually:

```text
raw sources
→ parse sources
→ resolve entities
→ build base triples
→ filter graph
→ add current gene/drug-gene relations
→ add external indications        [future extension]
→ quality gate
→ R-GCN preprocessing
→ training and evaluation
```

A possible future step file would be:

```text
kg_constructor/src/steps/13_add_external_indications.py
```

The step should remain disabled by default.

## 5. Proposed input location

External indication files should be stored separately from the current KG sources.

Suggested location:

```text
kg_constructor/data/external/indications/
```

Example files:

```text
kg_constructor/data/external/indications/external_indications.csv
kg_constructor/data/external/indications/README.md
```

The external source should not overwrite or directly modify the current manually validated `INDICATION` triples.

## 6. Proposed input format

A future external indications file should follow a strict tabular schema.

Recommended CSV columns:

| Column | Description |
|---|---|
| `source` | Name of the external source |
| `source_version` | Version or release date of the source |
| `source_drug_id` | Drug identifier in the external source |
| `source_drug_name` | Drug name in the external source |
| `mapped_drug_id` | Canonical project drug identifier |
| `source_disease_id` | Disease identifier in the external source |
| `source_disease_name` | Disease name in the external source |
| `mapped_disease_id` | Canonical project disease identifier |
| `evidence_type` | Type of evidence supporting the indication |
| `confidence` | Confidence level assigned to the association |
| `relation` | Relation to be created in the KG |
| `provenance_url` | Optional URL or reference to the evidence |
| `notes` | Optional comments |

Example:

```csv
source,source_version,source_drug_id,source_drug_name,mapped_drug_id,source_disease_id,source_disease_name,mapped_disease_id,evidence_type,confidence,relation,provenance_url,notes
ExternalSource,2026-01,DB00000,ExampleDrug,DRUGBANK:DB00000,DOID:0000,Example disease,DOID:0000,curated,high,INDICATION_EXTERNAL,,example row
```

## 7. Required entity mapping

External indication sources should not introduce raw drug or disease names directly into the final KG.

Every external row should be mapped to the project canonical entity identifiers.

Expected canonical identifiers:

```text
DRUGBANK:<id>
DRUG:<name>
CHEBI:<id>
DOID:<id>
```

Minimum mapping requirements:

```text
mapped_drug_id must not be empty
mapped_disease_id must not be empty
mapped_disease_id should use DOID whenever possible
mapped_drug_id should use the project canonical drug namespace
```

Rows that cannot be mapped should be exported to an unmapped report instead of being silently discarded.

Suggested reports:

```text
kg_constructor/data/output/reports/external_indications_report.json
kg_constructor/data/output/reports/external_indications_unmapped_drugs.csv
kg_constructor/data/output/reports/external_indications_unmapped_diseases.csv
```

## 8. Proposed new relation types

External indications should be represented with separate relation types instead of being merged directly into `INDICATION`.

Recommended relation types:

```text
INDICATION_EXTERNAL
INDICATION_LABEL
INDICATION_CURATED
INDICATION_INFERRED
```

Possible interpretation:

| Relation | Meaning |
|---|---|
| `INDICATION` | Current primary target relation |
| `INDICATION_EXTERNAL` | External indication evidence, generic |
| `INDICATION_LABEL` | Indication extracted from drug labels |
| `INDICATION_CURATED` | Curated external indication |
| `INDICATION_INFERRED` | Computationally inferred or lower-confidence indication |

The safest default is:

```text
ExternalSourceDrug --INDICATION_EXTERNAL--> Disease
```

This keeps external indication evidence separate from the supervised target.

## 9. Configuration design

The extension should be controlled by configuration and disabled by default.

Suggested config block:

```yaml
external_indications:
  enabled: false
  input_path: "kg_constructor/data/external/indications/external_indications.csv"
  output_triples_path: "kg_constructor/data/output/triples/triples_with_external_indications.csv"
  report_path: "kg_constructor/data/output/reports/external_indications_report.json"

  relation_name: "INDICATION_EXTERNAL"
  min_confidence: "high"

  merge_into_target_indication: false
  require_mapped_drug: true
  require_mapped_disease: true
  keep_unmapped_report: true
```

Important default:

```yaml
merge_into_target_indication: false
```

This prevents external evidence from being accidentally treated as supervised target data.

## 10. Leakage prevention

External indications can easily cause leakage if they overlap with validation or test target triples.

The following rules should be enforced.

### Rule 1: Do not merge external indications directly into `INDICATION`

Unsafe:

```text
Drug --INDICATION_EXTERNAL--> Disease
converted into
Drug --INDICATION--> Disease
```

Safe:

```text
Drug --INDICATION_EXTERNAL--> Disease
kept as context only
```

### Rule 2: Remove validation/test target pairs from external context

If a pair appears in validation or test as:

```text
Drug --INDICATION--> Disease
```

then the same pair must not appear in the training graph as:

```text
Drug --INDICATION_EXTERNAL--> Disease
Drug --INDICATION_LABEL--> Disease
Drug --INDICATION_CURATED--> Disease
```

Otherwise, the model may see the answer during training.

### Rule 3: Build splits before adding external context, or explicitly filter external context

Preferred protocol:

```text
1. Build canonical INDICATION triples
2. Create train/validation/test split on INDICATION
3. Add external indication context only to the training graph
4. Remove any external triples overlapping validation/test target pairs
5. Train and evaluate
```

### Rule 4: Keep provenance

Every external indication should retain source and confidence information.

This allows experiments such as:

```text
high-confidence only
curated only
label-derived only
all external indications
```

### Rule 5: Report overlap statistics

The pipeline should report:

```text
number of external indication rows
number of mapped rows
number of unmapped drugs
number of unmapped diseases
number of duplicate external triples
number of external triples overlapping train target
number of external triples overlapping validation target
number of external triples overlapping test target
number of external triples kept after leakage filtering
```

## 11. Evaluation design

External indications should be evaluated through controlled ablation experiments.

### Baseline

Current KG:

```text
INDICATION
CONTRAINDICATION
HAS_INGREDIENT
INTERACTS_WITH_GENE
ASSOCIATED_WITH_GENE
```

### Experiment A: external indications as context

Add external indications as context only:

```text
INDICATION_EXTERNAL
```

The target relation remains:

```text
INDICATION
```

This tests whether external evidence improves recommendation without changing the supervised target.

### Experiment B: confidence-filtered external indications

Compare:

```text
high-confidence external indications only
vs
all external indications
```

This tests whether noisy external evidence harms performance.

### Experiment C: source-specific ablation

If multiple external sources are used, evaluate each source separately:

```text
source A only
source B only
source A + source B
```

### Experiment D: target expansion

This should only be done as a separate experimental setting.

In this case, external indications may be merged into the target relation, but the train/validation/test split must be rebuilt from scratch.

This experiment should not be directly compared with the original setup without clearly stating that the target definition changed.

## 12. Recommended metrics

The same evaluation metrics used in the current R-GCN experiments should be reused:

```text
MRR
Hits@1
Hits@3
Hits@10
AUC-ROC with sampled negatives
Average Precision with sampled negatives
```

The most important comparison should be full-ranking performance, because this matches the disease-to-drug recommendation setting.

Recommended reporting:

| Setup | Full MRR | Hits@1 | Hits@3 | Hits@10 | Sampled MRR | AUC-ROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current KG | | | | | | | |
| + External indications as context | | | | | | | |
| + High-confidence external only | | | | | | | |
| + External indications merged into target | | | | | | | |

## 13. Quality gate requirements

The quality gate should be extended to validate external indication relations.

Checks should include:

```text
no empty source entity
no empty target entity
valid drug prefix on source
valid disease prefix on target
no duplicate triples
no overlap with validation/test target pairs when used as context
valid relation name
valid confidence value
valid provenance source
```

Allowed source prefixes:

```text
DRUGBANK:
DRUG:
CHEBI:
```

Allowed target prefix:

```text
DOID:
```

Allowed external indication relations:

```text
INDICATION_EXTERNAL
INDICATION_LABEL
INDICATION_CURATED
INDICATION_INFERRED
```

## 14. Future implementation outline

A future implementation could follow this structure:

```text
kg_constructor/src/steps/13_add_external_indications.py
```

Expected responsibilities:

```text
1. Read external indication CSV
2. Validate required columns
3. Filter by confidence
4. Validate mapped drug and disease identifiers
5. Create external indication triples
6. Remove duplicates
7. Remove leakage overlaps with validation/test target pairs if split information is available
8. Write triples_with_external_indications.csv
9. Write mapping and leakage reports
10. Run quality gate
```

Pseudo-code:

```python
def run(cfg, paths=None, force=False):
    external_cfg = cfg.get("external_indications", {})

    if not external_cfg.get("enabled", False):
        print("External indications step disabled by config.")
        return

    # 1. Load current triples
    # 2. Load external indication rows
    # 3. Validate schema
    # 4. Filter confidence
    # 5. Convert mapped rows to triples
    # 6. Remove duplicates
    # 7. Remove leakage overlaps
    # 8. Save output triples and report
```

## 15. Suggested future config variants

Possible future R-GCN configurations:

```text
biomedical/main/disease_to_drug_with_external_indications.yaml
biomedical/relation_studies/external_indications_only.yaml
biomedical/relation_studies/no_external_indications.yaml
biomedical/relation_studies/external_indications_high_confidence.yaml
```

Suggested experiment names:

```text
with_external_indications_context
with_external_indications_high_confidence
with_external_indications_target_expansion
```

## 16. Summary

External indications are a useful future extension, but they should be added carefully.

The safest design is to treat them as additional context relations, not as direct replacements or additions to the current `INDICATION` target.

The recommended approach is:

```text
keep current INDICATION as supervised target
add external indications as separate relations
filter overlaps with validation/test targets
keep provenance and confidence
evaluate using controlled ablations
only merge into INDICATION in a separate target-expansion experiment
```

This preserves experimental validity while making the KG extensible for richer drug-disease evidence in future work.
