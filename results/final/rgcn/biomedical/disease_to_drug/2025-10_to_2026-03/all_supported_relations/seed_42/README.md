# Final R-GCN run: six-month graph

Selected final experiment for the consolidated six-month graph.

## Experimental setup

- Task: `disease_to_drug`
- Seed: `42`
- Split: random 80/10/10
- Training triples: `5772`
- Validation triples: `721`
- Test triples: `723`
- Candidate entity type: `DRUG`
- Candidate pool: `2472`
- Explicit historical candidate list: no

Validation and test use the same base set of 2472 drug
candidates. Known positives are masked only during filtered
evaluation.

## Best checkpoint

- Epoch: `7690`
- Validation MRR: `0.7283990220`
- Validation Hits@1: `0.6976421637`
- Validation Hits@3: `0.7420249653`
- Validation Hits@10: `0.7961165049`

## Final evaluation

| Split | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---:|---:|---:|---:|
| Validation | 0.7284 | 0.6976 | 0.7420 | 0.7961 |
| Test | 0.7596 | 0.7331 | 0.7690 | 0.8133 |

The original external archive and model checkpoints are not currently
available locally. Their recorded names and hashes are preserved in
`artifact_reference.json` for provenance.
