# Reproducibility Dependency Audit

Audit date: 2026-10-02

## Import coverage

- All Python files under `src/` were inspected after the Stage 2A recovery scripts were added.
- Stage 0–1D, dataset preparation, OOF aggregation, and error analysis use only the Python standard library and local modules.
- `src/run_stage2a_training.py` directly imports PyTorch and Ultralytics at runtime.
- `src/evaluate_stage2a_run.py` directly imports Ultralytics at runtime.
- No project source directly imports NumPy, Pandas, SciPy, Matplotlib, OpenCV, Pillow, torchvision, scikit-learn, PyYAML, or Polars.
- The curated `requirements.txt` pins the two direct model-runtime dependencies and one confirmed CPU-compatibility runtime requirement; it is not a `pip freeze` dump.

## Python and platform

- Stage 2A interpreter: Python 3.12.10 (64-bit); pip 26.2.1.
- Earlier standard-library Stage 0–1D scripts also ran with Python 3.14.3; the model workflow should use Python 3.12.10.
- OS: Windows 10 10.0.19045 SP0, AMD64.
- CPU: Intel Pentium 4405U, 2 cores / 4 threads.
- CUDA available: no.
- PyTorch CUDA version: none (`torch.version.cuda is None`).
- Stage 2A device: CPU only.

## Verified `.venv_stage2a` versions

| Package | Version | Requirement role |
|---|---:|---|
| torch | 2.14.1 | Direct import |
| ultralytics | 8.4.158 | Direct import |
| polars | 1.44.2 | Ultralytics transitive runtime |
| polars-runtime-compat | 1.44.2 | Required compatibility runtime for this non-AVX CPU |
| torchvision | 0.29.1 | Transitive |
| numpy | 2.5.3 | Transitive |
| opencv-python | 5.0.0.93 | Transitive |
| Pillow | 12.3.0 | Transitive |
| PyYAML | 6.0.3 | Transitive |

## Packages introduced during competition work

- A separate `.venv_stage2a` environment was previously created for the interrupted pilot.
- Ultralytics 8.4.158 was installed explicitly; its model-runtime dependency set includes PyTorch, torchvision, NumPy, OpenCV, Pillow, PyYAML, and Polars.
- The Polars runtime-compat package was introduced after the default Polars runtime failed on this CPU's missing AVX/AVX2 support.
- Official Ultralytics/COCO `yolov8n.pt` is a pinned model asset, not a package. Its SHA-256 is checked by the training script.

## Recovery snapshot

`outputs/eda/pip_freeze_snapshot.txt` contains the complete environment snapshot for debugging only. It does not replace curated dependencies.

## Portability limitations and unresolved items

- CPU and CUDA PyTorch wheels can require different installation indexes. The current pins describe the verified CPU environment; final README installation guidance still needs to distinguish CPU and GPU installations.
- Heavy experiments now target the KAMP-NOTE PyTorch GPU-CUDA12 environment. Its exact preinstalled PyTorch/torchvision versions have not yet been observed; the scripts record them in each run manifest.
- `polars[rtcompat]` is included because of the current Pentium CPU. Its behavior should be smoke-tested on the eventual submission/reviewer machine.
- Two Python versions exist historically. The final submission should state Python 3.12.10 as the verified Stage 2A runtime.
- Training time on the current CPU is substantial. The scripts deliberately execute only one requested experiment and never chain folds.

## Future dependency rule

For a new third-party dependency: state its name and purpose, provide the install command, wait for user installation, verify the version, and only then update `requirements.txt`. No automatic installation is permitted.
