# W&B and Submission Isolation Check

## Design

- W&B imports are lazy and confined to `src/wandb_logging.py` and the explicit post-evaluation helper `src/log_stage2a_eval_to_wandb.py`.
- `src/run_stage2a_training.py` defaults to `--use-wandb` off.
- Installing W&B does not silently enable the Ultralytics built-in callback; that callback is disabled for the training process and the project's explicit opt-in logger is used instead.
- W&B initialization, epoch logging, finishing, and artifact-upload errors are fail-open and do not abort training.
- Artifact upload defaults off and requires `--wandb-log-artifacts`.

## Submission/inference isolation

- `src/evaluate_stage2a_run.py` has no W&B import and its evaluation output is locally complete without W&B.
- No prediction/inference script imports `wandb_logging` or `wandb`.
- A saved `best.pt` is a normal Ultralytics checkpoint and can be loaded without W&B credentials, internet, or the W&B package.
- The optional evaluation logger consumes completed JSON/CSV evidence after evaluation; evaluation never depends on it.

## Dependency status

W&B is not installed in the current local Stage 2A environment. It was not installed automatically and is not yet pinned in `requirements.txt`. On KAMP, the user must first confirm installation and version. Suggested command:

```bash
pip install --user wandb
```

After installation, record the confirmed W&B version in `requirements.txt` according to the project dependency policy.

## Static verification result

- `py_compile`: passed for the modified runner and all new helpers.
- Runner and post-evaluation helper `--help`: passed without importing W&B.
- Runner import with W&B absent and disabled: passed (`DISABLED`).
- Explicit W&B request with W&B absent: failed open as designed (`FAILED_OPEN`) without raising into the training caller.
- Frozen-fold hash: matched `963116078893679b859c2d5150a1ba90700ad207c54ae385a1b6acac255b98a8`.
- Fold1 oversampling preflight: 376 train images, 63 duplicated eligible train images, 439 effective entries, 124 validation images, and zero validation duplicates.
- Repository search: existing evaluation and inference scripts contain no W&B dependency.

No training or inference was run.
