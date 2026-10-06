# Stage 1D — Artifact-Control Ablation Summary

## 1. Input integrity

- **VERIFIED** — 500 eligible samples, 1,147 objects, complete frozen 4-fold assignment, unique stems, and no missing BMP/TXT files.
- Frozen fold SHA-256 remained `963116078893679b859c2d5150a1ba90700ad207c54ae385a1b6acac255b98a8`; no fold or original image was modified.

## 2. Why the artifact is a leakage risk

- **VERIFIED** — strong chromatic marks occur in 500/500 images and Stage 1B established near-GT alignment in 1124/1147 objects with component-count agreement in 493/500 images.
- RAW_RGB is diagnostic only and is never a training candidate.

## 3. Grayscale-only residual leakage

- **VERIFIED** — direct grayscale retains detectable local intensity/edge structure for 36.4% of object rows; mean original-artifact edge recall=0.0227.
- **STRONGLY SUPPORTED** — channel collapse alone is insufficient for a fair baseline.

## 4. Mask robustness

- **VERIFIED** — 95.2% of images pass the fixed thin-component/proximity robustness screen; median total masked fraction=0.1658%.
- The mask is not dilated and never removes rectangle interiors or full GT boxes.

## 5. Candidate repair methods

- Conservative: median of nonmasked 5×5 neighbors, expanding to 9×9 only if needed.
- Local interpolation: arithmetic mean of nearest valid source-luma values in four cardinal directions within 5 pixels.
- Both modify only the strong-chromatic line mask and save lossless uint8 PNGs.

## 6. Signal preservation

- **VERIFIED** — maximum outside-mask pixel change is 0 across evaluated dense representations.
- Mean modified fractions: grayscale=0.0000%, conservative=0.2219%, local=0.2213%.

## 7. Residual shortcut diagnostic

- Mean artifact-edge recall: grayscale=0.0227, conservative=0.0017, local=0.0000.
- **EXPLORATORY** — this fixed nonlearned diagnostic is a shortcut screen, not foreign-object detection performance or an exhaustive adversarial test.

## 8. Repair-induced edge/texture artifacts

- Mean adjacent gradient shift: conservative=8.44, local=7.811.
- Only pixels adjacent to repaired lines can show gradient changes; nonartifact pixel intensities are preserved exactly.

## 9. GT-local signal preservation

- Median cross-feature deviation from Stage 1B reference: grayscale=4.441, conservative=1, local=0.9763.
- These are image-signal comparisons and do not prove perfect recovery of the physical foreign object.

## 10. Small-object preservation

- **VERIFIED** — smallest-quartile diagnostic flags: grayscale=True, conservative=False, local=False.
- The bottom-decile and quartile cutoffs both equal 8 px; retaining all cutoff ties makes both sensitivity groups the same 295 objects (25.7%), which is reported transparently rather than broken by arbitrary tie order.

## 11. Candidate comparison

- RAW_RGB: rejected, direct color shortcut.
- GRAYSCALE_DIRECT: rejected, residual rectangle structure.
- ARTIFACT_MASK_EXCLUSION: valid feature reference but not a dense detector input.
- Conservative/local repairs: evaluated by shortcut reduction, exact outside-mask preservation, feature stability, small-object safety, and repair-edge risk without a weighted score.

## 12. Approved baseline representation

- **STRONGLY SUPPORTED** — ARTIFACT_INPAINT_CONSERVATIVE.

## 13. Optional secondary representation

- **EXPLORATORY** — ARTIFACT_LOCAL_INTERPOLATION.

## 14. Remaining uncertainty

- True grayscale signal under overwritten colored pixels is not observable; repair estimates it locally.
- The residual diagnostic is simple and cannot prove absence of every exploitable shortcut.
- Deployment-time artifact presence and acquisition pipeline are undocumented.
- Repair-induced local gradients may interact with a future detector and require frozen-fold ablation.

## 15. Recommended next stage

- Use the frozen Stage 1C folds for a separately authorized baseline comparison of the approved representation(s). Keep RAW_RGB excluded and report artifact-control ablation separately; do not access external held-out data.
