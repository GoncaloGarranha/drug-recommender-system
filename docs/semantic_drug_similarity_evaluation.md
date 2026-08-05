# Semantic Drug Similarity Evaluation

## Motivation

The current evaluation protocol follows the standard filtered link-prediction setting: for each query, the model receives a pool of candidate entities, scores each candidate, removes other known positives from the ranking, and computes the rank of the exact expected entity. This is appropriate for benchmarking, but it is strict in biomedical recommendation tasks because a clinically useful recommendation may be a different drug with equivalent therapeutic role, active substance, pharmacological class, or mechanism of action.

This document proposes an additional evaluation phase that preserves the traditional metrics and adds semantic metrics based on groups of similar drugs. The goal is not to replace exact MRR, but to report whether the model ranks a semantically acceptable alternative near the top when it misses the exact drug.

## Recommended approach

I recommend implementing semantic evaluation as a post-processing layer over the existing ranking pipeline, rather than changing model training or changing the construction of the training graph. This keeps the experiment comparable with previous runs and avoids leaking validation or test labels into the model.

The proposed evaluation should produce two families of metrics:

1. **Exact metrics**: the current filtered MRR and Hits@K, where only the ground-truth drug is considered correct.
2. **Semantic metrics**: a complementary version of MRR and Hits@K where a prediction is considered acceptable if the first ranked candidate belongs to the same similarity group as the ground-truth drug.

The main reported metric should be named separately, for example `semantic_mrr`, `semantic_hits@1`, `semantic_hits@3`, and `semantic_hits@10`. The exact `mrr` should remain unchanged.

## How to build similarity groups

### Phase 1: graph-derived groups

The first implementation should use evidence already present in the project Knowledge Graph, especially the `HAS_INGREDIENT` relation. Two drugs should be grouped when they share at least one active ingredient or substance. This is the safest initial criterion because it is transparent, reproducible, and grounded in the graph data already used by the system.

A practical similarity rule is:

- collect all `Drug -> HAS_INGREDIENT -> Substance` links;
- represent each drug by the set of its substances;
- compute pairwise overlap using Jaccard similarity;
- assign drugs to the same semantic group when their Jaccard similarity is above a configurable threshold, for example `0.5`;
- always include the drug itself in its own group.

This rule should create a mapping such as:

```json
{
  "DRUGBANK:DB00945": ["DRUGBANK:DB00945", "DRUGBANK:DB01023"],
  "DRUGBANK:DB01023": ["DRUGBANK:DB00945", "DRUGBANK:DB01023"]
}
```

The groups should be saved as a versioned artifact, for example `data/processed/drug_similarity_groups.json`, together with metadata describing the source triples, threshold, date, and number of grouped drugs.

### Phase 2: therapeutic-neighbour groups

After the ingredient-based version is stable, a second group builder can add therapeutic evidence:

- drugs linked to the same diseases through `INDICATION`;
- drugs sharing contraindication profiles;
- drugs close in the graph through drug-substance-disease paths.

This second layer should be configurable and reported separately because it is less strict than active-ingredient equivalence. For example, `semantic_mrr_ingredient` and `semantic_mrr_therapeutic` should not be mixed without explanation.

### Phase 3: external ontology groups

If external data are allowed, the strongest clinical version would integrate ATC classes, DrugBank categories, MeSH pharmacological actions, or mechanisms of action. This should be considered a later phase because it introduces new dependencies, licensing considerations, and an additional mapping problem between external identifiers and the project entities.

## Semantic rank definition

For each evaluation query, let `G(true_drug)` be the group containing the expected drug. Instead of calculating the rank of only the exact drug, calculate the rank of the first candidate in the ranked list that belongs to `G(true_drug)`.

Example:

```text
Ground truth: Drug A
Similarity group: {Drug A, Drug B, Drug C}
Ranked candidates: [Drug X, Drug B, Drug Y, Drug A]
Exact rank: 4
Semantic rank: 2
```

The semantic reciprocal rank for this query is therefore `1 / 2`, while the exact reciprocal rank remains `1 / 4`.

This definition is preferable to artificially moving the exact drug rank because it reflects what the model actually recommended first among clinically related alternatives.

## Filtering rule

The semantic evaluator should reuse the existing filtered ranking protocol. Known positives for the same query should still be filtered out, except for the exact test target being evaluated. Semantic candidates should not be filtered merely because they are in the same group; they should remain rankable unless they are already known positives for that query.

This distinction is important because the objective is to measure whether the model recommends a related drug, not to hide related drugs from the candidate pool.

## Suggested implementation plan

1. Add a reusable similarity-group builder under `models/shared/` or `kg_constructor/src/steps/`, depending on whether the groups are considered model artifacts or KG artifacts.
2. Generate `drug_similarity_groups.json` from the canonical triples file and entity mappings.
3. Add a function such as `semantic_filtered_rank(candidate_ids, scores, acceptable_candidate_ids, known_positive_ids)` to `models/shared/evaluation.py`.
4. Extend `models/shared/ranking.py` so `evaluate_ranking` can optionally receive a similarity-group mapping and return exact metrics plus semantic metrics.
5. Extend the content-based baseline with the same semantic metric logic, so R-GCN and baseline remain comparable.
6. Store the semantic-group configuration and group artifact path inside each run directory for reproducibility.
7. Add a report table comparing exact MRR against semantic MRR, and include a short qualitative inspection of cases where semantic MRR improves but exact MRR does not.

## Validation checks

Before trusting the metric, the following checks should be produced:

- number of drug entities in the candidate pool;
- number and percentage of drugs with at least one similar drug;
- distribution of group sizes;
- top largest groups for manual inspection;
- examples of semantic hits where the exact drug was missed;
- comparison between exact MRR and semantic MRR.

A very large average group size is a warning sign because the metric may become too permissive. For the first version, small ingredient-based groups are preferable to broad therapeutic groups.

## Opinion

This is a strong and defensible extension for the dissertation, as long as it is presented as a complementary semantic evaluation and not as a replacement for standard link-prediction metrics. The exact metric remains necessary for comparison with other models, while semantic MRR gives a more clinically meaningful view of recommendation usefulness.

The best first step is to implement ingredient-based groups because they are transparent, reproducible, and already supported by the graph schema. Therapeutic-class and mechanism-of-action groups can then be added as richer variants once the evaluation code is stable.

## File-by-file change plan

The implementation should be done in small steps so that each layer can be tested independently.

### 1. `models/shared/evaluation.py`

Add a new ranking helper for semantic evaluation, for example:

```python
def semantic_filtered_rank(
    candidate_ids,
    scores,
    acceptable_candidate_ids,
    known_positive_ids,
):
    ...
```

Expected behavior:

- apply the same filtered-ranking masking already used by `filtered_rank`;
- do not mask the acceptable candidates solely because they are semantically similar;
- compute the rank of the highest-scoring candidate that belongs to `acceptable_candidate_ids`;
- raise a clear error or return `None` when no acceptable candidate exists in the candidate pool;
- preserve tie-aware ranking, using the same logic as the current exact rank calculation.

This file is the right place because it already contains the generic filtered-ranking logic shared by model evaluators.

### 2. `models/shared/metrics.py`

No existing metric should be changed. Keep `compute_ranking_metrics` as the exact metric aggregator.

Optionally add a thin wrapper such as:

```python
def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    ...
```

This would allow semantic ranks to reuse the same MRR/Hits@K calculation while returning names such as `semantic_mrr` and `semantic_hits@10`.

### 3. `models/shared/ranking.py`

Extend `evaluate_ranking` with optional semantic inputs, for example:

```python
def evaluate_ranking(..., similarity_groups: dict[int, set[int]] | None = None, ...):
    ...
```

Required changes:

- keep the existing exact rank calculation unchanged;
- when `similarity_groups` is provided, collect semantic ranks in parallel with exact ranks;
- for each true candidate, define `acceptable_candidate_ids = similarity_groups.get(true_candidate_id, {true_candidate_id})`;
- return both exact and semantic metrics in the same result dictionary;
- ensure the `direction="both"` case averages semantic metrics consistently with exact metrics.

This file is the central integration point for the R-GCN evaluation because it maps triples into query/candidate ranking tasks and calls `filtered_rank`.

### 4. `models/shared/__init__.py`

Export the new semantic helper added to `models/shared/evaluation.py`, for example:

```python
from .evaluation import filtered_rank, filtered_ranks_from_batches, semantic_filtered_rank
```

This keeps the shared API consistent for both R-GCN and baseline code.

### 5. New file: `models/shared/drug_similarity.py`

Add reusable code for loading and validating similarity groups.

Suggested functions:

```python
def load_similarity_groups(path: Path) -> dict[int, set[int]]:
    ...

def validate_similarity_groups(groups: dict[int, set[int]], candidate_ids: set[int]) -> dict[str, object]:
    ...
```

Responsibilities:

- read `drug_similarity_groups.json`;
- convert entity identifiers to integer IDs when necessary;
- ensure each drug includes itself in its group;
- optionally remove group members outside the candidate pool;
- return validation statistics such as number of groups, covered drugs, and group-size distribution.

### 6. New file: `kg_constructor/src/steps/07_build_drug_similarity_groups.py`

Add a KG construction step that builds groups from triples.

Responsibilities:

- read the canonical triples file, such as `kg_constructor/data/output/triples/triples.csv`;
- filter `HAS_INGREDIENT` edges;
- build `drug -> set(substances)` profiles;
- compute Jaccard overlap between drugs;
- group drugs whose overlap is above a configurable threshold;
- write `drug_similarity_groups.json` and a metadata/report file.

This should be the first version of the group builder because it relies on evidence already present in the KG and does not require new external data.

### 7. `kg_constructor/src/pipeline.py`

Register the new similarity-group step after triples are built and filtered.

Suggested behavior:

- make the step optional through configuration;
- run it after the KG contains stable `HAS_INGREDIENT` triples;
- save the output path in the pipeline report.

### 8. `kg_constructor/configs/config.yaml`

Add configuration for the group-building step, for example:

```yaml
drug_similarity:
  enabled: true
  source_relation: HAS_INGREDIENT
  jaccard_threshold: 0.5
  min_shared_features: 1
  output_path: data/processed/drug_similarity_groups.json
```

These values should be explicit so that semantic evaluation results are reproducible.

### 9. `models/rgcn/configs/biomedical/*.yaml`

For each biomedical evaluation config where semantic metrics should be enabled, add an optional evaluation section such as:

```yaml
evaluation:
  similarity_groups_path: kg_constructor/data/builds/2025-10_to_2026-03/processed/drug_similarity_groups.json
  semantic_metrics: true
```

Do not enable this for benchmark datasets such as FB15k or WN18 unless equivalent semantic groups exist.

### 10. `models/rgcn/src/evaluate.py`

Update the standalone R-GCN evaluation command.

Expected changes:

- read `evaluation.similarity_groups_path` from the config;
- load the group mapping with `models/shared/drug_similarity.py`;
- pass the mapping into `evaluate_ranking`;
- persist semantic metrics in the same `metrics_valid.json` and `metrics_test.json` files, using distinct keys.

### 11. `models/rgcn/src/train.py`

Update the standard R-GCN training loop so validation-time evaluation can report semantic metrics when configured.

Expected changes:

- load similarity groups once during experiment setup, not inside every epoch;
- pass the group mapping into the validation call to `evaluate_ranking`;
- keep early stopping based on the existing exact monitor unless the config explicitly asks for a semantic monitor;
- save semantic validation metrics alongside exact validation metrics.

### 12. `models/rgcn/src/train_hardneg_corrected.py`

Mirror the same integration done in `models/rgcn/src/train.py` for the hard-negative training variant.

Expected changes:

- keep the hard-negative sampling/training logic unchanged;
- reuse the same similarity-group loader;
- pass semantic groups into the validation ranking call;
- save exact and semantic metrics with distinct names.

### 13. `models/content-based/src/baseline.py`

Apply the same semantic evaluation to the content-based baseline.

Required changes:

- add an optional CLI/config argument for `similarity_groups_path`;
- while computing exact ranks, also compute semantic ranks using the same candidate scores;
- aggregate semantic ranks through the shared metrics function;
- print and save both exact and semantic metrics.

This is important because the dissertation compares R-GCN against the content-based baseline. If only R-GCN has semantic metrics, the comparison becomes incomplete.

### 14. Tests to add

Add unit tests for the semantic rank logic. Depending on the project test layout, these could be placed under a new `tests/` directory.

Minimum test cases:

- exact target at rank 4 but semantically similar target at rank 2;
- known positives are filtered but semantic candidates are not incorrectly removed;
- groups that contain only the true drug produce the same result as exact ranking;
- ties are handled consistently with the existing filtered rank logic;
- missing group entries fall back to `{true_candidate_id}`.

### 15. Documentation updates

Update the relevant README files after the runtime implementation exists:

- explain what semantic MRR measures;
- clarify that exact MRR remains the primary benchmark metric;
- document how to generate `drug_similarity_groups.json`;
- include an example command for running evaluation with semantic metrics;
- add a short warning that large or overly broad groups can inflate semantic metrics.

## Similarity strategy decision

Before implementing code, the similarity criterion should be defined as an evaluation protocol decision. The recommended strategy is to use a **tiered definition of drug similarity**, starting with strict and explainable evidence and only later adding broader clinical evidence.

### Recommended first criterion: shared active ingredient

Two drugs should initially be considered similar when they share one or more active ingredients or substances through `HAS_INGREDIENT` relations.

This should be the first criterion because it is:

- **transparent**: the explanation is easy to inspect manually;
- **reproducible**: it uses information already present in the KG;
- **clinically conservative**: drugs with the same active substance are usually much closer than drugs that merely share an indication;
- **safe for evaluation**: it reduces the risk of artificially inflating semantic MRR with broad or vague groups.

A practical rule is:

```text
Drug A is similar to Drug B if:
Jaccard(ingredients(A), ingredients(B)) >= threshold
and shared_ingredients(A, B) >= min_shared_ingredients
```

For the first experiments, use:

```yaml
source_relation: HAS_INGREDIENT
jaccard_threshold: 0.5
min_shared_ingredients: 1
```

If the dataset contains many single-ingredient drugs, an additional stricter variant should also be tested:

```yaml
jaccard_threshold: 1.0
```

This strict variant corresponds to drugs having exactly the same ingredient set and is useful as a high-confidence semantic evaluation.

### Second criterion: therapeutic profile similarity

After the ingredient-based groups are validated, a second strategy can group drugs by therapeutic profile. This means comparing drugs according to the diseases they treat, using `INDICATION` relations.

This is clinically relevant, but less strict. Two drugs can treat the same disease while having different mechanisms, risk profiles, or lines of therapy. Therefore, this criterion should not be mixed with ingredient-based similarity in the first result table.

A possible rule is:

```text
Drug A is therapeutically similar to Drug B if:
Jaccard(indications(A), indications(B)) >= threshold
and shared_indications(A, B) >= min_shared_indications
```

This should be reported separately as `semantic_mrr_therapeutic`, not as the default semantic MRR.

### Third criterion: external pharmacological classes

A stronger future version can use external pharmacological classifications, such as ATC classes, DrugBank categories, MeSH pharmacological actions, or mechanism-of-action annotations.

This can produce clinically richer groups, but should be a later phase because it introduces:

- dependency on external sources;
- additional identifier mapping;
- possible licensing constraints;
- more subjective decisions about how broad a class can be.

### Final recommendation

For this project, the best strategy is:

1. **Primary semantic evaluation**: shared active ingredient groups using `HAS_INGREDIENT`.
2. **Sensitivity analysis**: compare `jaccard_threshold=0.5` against `jaccard_threshold=1.0`.
3. **Secondary semantic evaluation**: therapeutic profile groups based on shared `INDICATION`, reported separately.
4. **Future extension**: external ATC/DrugBank/MeSH/MoA classes, only after the internal KG-based protocol is stable.

This allows the dissertation to present a conservative and defensible metric first, then discuss broader clinical similarity as an additional analysis rather than as the main result.

## Final selected strategy for the current deadline

Given the current project context and the limited time available, the strategy to implement should be **one final protocol**, not a set of experimental alternatives.

The selected strategy is:

```text
Two drugs are semantically similar if they share at least one active ingredient/substance through the KG relation HAS_INGREDIENT.
```

In operational terms:

```text
similar(Drug A, Drug B) = true
if ingredients(A) ∩ ingredients(B) is not empty
```

This corresponds to the following fixed configuration:

```yaml
source_relation: HAS_INGREDIENT
min_shared_ingredients: 1
jaccard_threshold: 0.0
```

The Jaccard value can still be computed and stored for transparency, but it should **not** be used as a strict threshold in the final version. The reason is practical: many biomedical products can have small ingredient-set differences caused by formulation, salt forms, combinations, or data extraction differences. Requiring `jaccard_threshold=0.5` or `1.0` may create groups that are too strict for the available data and reduce the usefulness of the semantic evaluation.

### Why this is the best fit now

This strategy is the best fit for the current dissertation phase because it is:

- **directly supported by the current KG**, since the graph already models `HAS_INGREDIENT` links;
- **clinically interpretable**, because sharing an active substance is an understandable explanation for similarity;
- **simple to implement**, avoiding delays from external ontologies or complex therapeutic-class mapping;
- **less subjective** than grouping drugs only because they treat the same disease;
- **safe for reporting**, because it is conservative enough to justify semantic equivalence without overclaiming therapeutic interchangeability.

### What should not be included in the final implementation

For the final version, do **not** include therapeutic-profile groups based on shared `INDICATION` as part of the main semantic metric. This can be mentioned as future work, but it is too broad for the main result because drugs may share an indication while having different mechanisms, toxicity, clinical line of use, or contraindications.

Also do **not** include external ATC, MeSH, DrugBank categories, or mechanism-of-action grouping in the implementation for this deadline. These sources are valuable, but they introduce extra mapping and validation work that may distract from finishing a robust and explainable evaluation.

### Final metric interpretation

The final semantic MRR should therefore mean:

```text
The model receives semantic credit when the first acceptable ranked drug is either the exact expected drug or another drug that shares at least one active ingredient/substance with it.
```

The exact MRR remains the main benchmark metric, while semantic MRR becomes the complementary clinical plausibility metric.

## Reliability claim

This strategy can be described as the **most reliable option available within the current KG-based context**, but it should not be described as the universally most reliable definition of clinical drug similarity.

The recommended wording is:

```text
Among the similarity criteria available in the current Knowledge Graph, shared active ingredient is the most reliable and conservative criterion for semantic evaluation, because it is directly supported by explicit graph evidence and requires minimal subjective interpretation.
```

This wording is intentionally precise. It claims reliability **within the available data and project scope**, not absolute clinical equivalence.

The reason is that sharing an active ingredient provides strong evidence of pharmacological relatedness, but it does not automatically guarantee that two products are fully interchangeable in clinical practice. Dosage, formulation, route of administration, contraindications, salt form, combination products, and regulatory context can still differ.

Therefore, the dissertation should frame this as:

- **reliable for semantic evaluation**;
- **conservative compared with shared indication groups**;
- **more reproducible than external or manually curated groupings at this stage**;
- **not a claim of full therapeutic interchangeability**.

A safe final sentence for the report is:

```text
For this work, active-ingredient overlap was adopted as the primary semantic similarity criterion, as it represents the most reliable and reproducible signal available in the constructed Knowledge Graph, while avoiding broader and more subjective therapeutic-class assumptions.
```

## Response to metric-choice feedback

The feedback suggesting Tanimoto/Jaccard or cosine similarity is technically correct, but for the current project the final choice should remain **Tanimoto/Jaccard over active-ingredient sets**, not cosine similarity over heterogeneous profiles.

For binary ingredient sets, the Tanimoto coefficient is equivalent to the Jaccard index:

```text
Tanimoto(A, B) = Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

Therefore, the final protocol can be described as an ingredient-set Tanimoto/Jaccard strategy. However, to keep the implementation robust and deadline-friendly, the decision rule should remain based on `|A ∩ B| >= 1`, while storing the Tanimoto/Jaccard score as metadata for interpretability.

Cosine similarity is useful when drugs are represented as weighted vectors, such as TF-IDF profiles combining ingredients, contraindications, indications, and gene-derived tokens. This is already aligned with the content-based baseline, but it is less suitable as the primary semantic-equivalence rule because its explanation is less direct and depends on feature weighting.

### Gene association decision

In the current graph context, genes should be treated as directly associated with diseases, not directly with drugs, unless an explicit drug-target or drug-gene source is added.

A drug can inherit gene context indirectly through its training indications, but this should be considered a derived feature for modelling or analysis, not a direct KG edge of the form `Drug -> Gene`.

The safe modelling choice is therefore:

```text
Disease -> ASSOCIATED_WITH_GENE -> Gene
Drug -> INDICATION -> Disease
```

and not:

```text
Drug -> ASSOCIATED_WITH_GENE -> Gene
```

unless a curated source for drug-target interactions is introduced.
