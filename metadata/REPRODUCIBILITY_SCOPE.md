# Reproducibility scope

- The frozen 4-fold assignment and all local CSV/JSON metadata are the source of truth.
- `FULL` mode rebuilds conservative images, materializes the folds, trains four B2-640 models, evaluates each held-out fold, aggregates OOF predictions, and exports the prediction artifact.
- `QUICK` mode does not train. It verifies the supplied final weights and repeats inference, evaluation, OOF aggregation, and export.
- The supplied OOF prediction is development evidence: every image was predicted by a model that did not train on that fold. It is not an external held-out test result.
- The 500-image development set is entirely GT-positive. Specificity, true-negative performance, automatic PASS safety, and normal-product false-alarm workload are not estimated.
- Exact metric reproduction is expected in the documented KAMP software/GPU environment. Cross-hardware bitwise identity is not guaranteed.

