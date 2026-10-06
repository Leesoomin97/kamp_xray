# Stage 3 Image-Level Safety Analysis

## Dataset scope

- OOF images: 500
- GT objects: 1147
- Single-object images: 176

## Confidence 0.25

- At least one GT detected: 495/500 (0.990000)
- All GT objects detected: 475/500 (0.950000)
- Images with zero GT detected: 5

## Confidence 0.40

- At least one GT detected: 495/500 (0.990000)
- All GT objects detected: 475/500 (0.950000)
- Partial-detection images: 20
- Images with zero GT detected: 5
- Total missed GT objects: 26

## Single-object images

- confidence 0.25: 171/176 detected (0.971591)
- confidence 0.40: 171/176 detected (0.971591)

## Interpretation

- 'At least one GT detected' approximates product-level rejection when any foreign object is sufficient to reject the product.
- 'All GT detected' is stricter and reveals residual object-level misses hidden by multi-object images.
- Single-object images remove the masking effect where one detected object can hide another missed object within the same product image.

## Limitation

- All 500 images are GT-positive. There is no normal / true-negative corpus.
- Therefore production PASS specificity and real reinspection workload cannot be estimated.