# Stage 3 EDA Audit

- PASS: 18
- FAIL: 0

| Check | Status | Detail |
|---|---|---|
| B2 OOF @0.25 reproduction | PASS | TP=1121, FP=51, FN=26 |
| 0.25 -> 0.40 recall preserved | PASS | 0.25: TP=1121, FN=26; 0.40: TP=1121, FN=26 |
| 0.40 reduces FP vs 0.25 | PASS | FP 51 -> 30 |
| 0.001 minimum-FN point | PASS | FN=13, Recall=0.988666, FP=3688 |
| 0.40 object FN equals image missed-object total | PASS | object FN=26, image missed GT=26 |
| 0.40 image partition sums to 500 | PASS | all=475, partial=20, zero=5 |
| 0.25 and 0.40 image safety identical | PASS | 0.25 any/all/missed=495/475/26; 0.40=495/475/26 |
| GT-count image total = 500 | PASS | sum=500 |
| GT-count object total = 1147 | PASS | objects=1147 |
| Single-object image count | PASS | n=176 |
| Single-object results match GT-count table | PASS | detected=171, missed=5 |
| Incomplete-image table count | PASS | rows=25 |
| Zero-correct-detection case count | PASS | rows=5 |
| High-confidence wrong prediction case count | PASS | rows=5 |
| Zero-case stems match | PASS | GT stems=5, prediction stems=5 |
| All high-confidence zero-case predictions are IoU < 0.5 | PASS | max nearest IoU=0.495970 |
| All zero-case predictions have confidence >= 0.40 | PASS | min confidence=0.448172 |
| 0.001 trade-off matches threshold sweep | PASS | TP=1134, FP=3688, FN=13 |

## Key verified values

- Object-level @0.25: TP=1121, FP=51, FN=26, Recall=0.977332
- Object-level @0.40: TP=1121, FP=30, FN=26, Recall=0.977332
- Image-level @0.40: any detected=495/500, all detected=475/500, zero correct=5/500
- Single-object @0.40: 171/176
- Low-confidence extreme @0.001: TP=1134, FP=3688, FN=13
