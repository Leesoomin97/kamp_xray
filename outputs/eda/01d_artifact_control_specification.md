# Stage 1D — Artifact-Control Specification

## Scope and frozen validation

- Approved samples: 500 canonical BMP files with official TXT labels.
- Frozen folds are read from `01c_final_validation_folds.csv`; SHA-256: `963116078893679b859c2d5150a1ba90700ad207c54ae385a1b6acac255b98a8`.
- No fold assignment is modified by this pipeline.

## Shared input and mask

- Read uncompressed 8-bit indexed BMP and decode its 256-entry RGB palette.
- Convert palette RGB to uint8 BT.601 luma: `round(0.299R + 0.587G + 0.114B)`, clipped to [0,255].
- Mark a pixel only when `max(R,G,B)-min(R,G,B) >= 200`.
- Do not dilate the mask; do not mask rectangle interiors or entire GT boxes.
- Processing is deterministic and uses no random seed.

## Preferred representation: ARTIFACT_INPAINT_CONSERVATIVE

- For each masked pixel, replace it with the median of nonmasked source-luma pixels in a 5×5 neighborhood; expand to 9×9 only if no valid neighbor exists.
- Only masked pixels are written; all nonmasked pixels remain bit-identical to direct grayscale.
- Save losslessly as 8-bit grayscale PNG, original width/height, uint8 range [0,255].
- No normalization, CLAHE, sharpening, denoising, residual transform, or resizing occurs here. Model normalization belongs to a later stage.

## Optional secondary representation: ARTIFACT_LOCAL_INTERPOLATION

- Four-direction local interpolation as specified above.

## Diagnostic-only representations

- `RAW_RGB`: original palette color, never approved for training.
- `GRAYSCALE_DIRECT`: direct luma conversion; rejected if residual rectangle leakage remains.
- `ARTIFACT_MASK_EXCLUSION`: Stage 1B feature reference only; not a dense detector input.

## Version

- Processing version: `stage1d_v1`
- Implementation: `src/run_stage1d_artifact_control.py`
