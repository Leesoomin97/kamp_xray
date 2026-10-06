# Stage 2A OOF Error Analysis Summary

## Scope and observed facts

- Frozen OOF scope: 500 labeled development images and 1147 GT objects.
- Reporting threshold: confidence 0.25; this is not an operational threshold.
- Object outcomes: TP=1121, FN=26; unmatched predictions FP=51.
- Localization failures (confidence retained but IoU < 0.50): 11.
- All 500 images contain labels; there is no true-negative image corpus for image-level specificity/FPR estimation.

## Derived diagnostics

- Size, area ratio, image-edge distance, inferred product-edge distance, signed/absolute contrast, robust CNR-like proxy, object IQR, background gradient/IQR, and bbox aspect ratio are reported with counts by exploratory quartile.
- Prediction-confidence and localization-IoU relationships are reported both by exploratory quartile and Spearman association.
- Machine, date, and resolution summaries are descriptive and may be mutually confounded.
- Pre-specified interaction tables always retain object counts; sparse cells must not be overinterpreted.
- Product-edge distance is inferred from an unsupervised boundary proxy rather than GT segmentation.

## Pixel-size detectability

- Native and 640-letterbox projected bbox minimum-side sizes are reported per object and in distribution-derived bins.
- The sharpest adjacent empirical-bin recall change is recorded in `02b_pixel_size_detectability_summary.json`.
- This is candidate evidence for Stage 3 empirical-limit analysis only; no detection or safety threshold is selected.

## Hypotheses and limitations

- Associations identify candidate failure conditions for controlled follow-up; they do not establish causality.
- Intensity/contrast/CNR-like variables are image-derived proxies. They do not identify physical density, material, or thickness.
- FP background quartiles describe the FP set only and are not false-positive rates because no true-negative image denominator exists.
- No PASS/DETECT/REINSPECT or Stage 3 operational threshold is selected here.
