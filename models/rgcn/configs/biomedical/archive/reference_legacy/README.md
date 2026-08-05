# Biomedical reference configurations

This directory contains the two explicit reference definitions used for the
disease-to-drug recommendation task.

## identifier_level_reference.yaml

Historical identifier-level benchmark.

- Preserves the historical drug entity representation.
- Uses the historical split definition.
- Treats distinct identifiers as distinct graph entities.
- Reproduces the historical R-GCN result of approximately 0.741 MRR.
- Uses the frozen snapshot associated with commit 72df15d.

## canonical_concept_reference.yaml

Canonical concept-level benchmark.

- Uses the current canonical biomedical graph.
- Merges drug aliases according to the current entity-resolution pipeline.
- Uses deterministic negative sampling.
- Provides a stricter complementary evaluation of concept-level
  generalization.

The two configurations represent different evaluation granularities. Neither
configuration should silently replace the other.
