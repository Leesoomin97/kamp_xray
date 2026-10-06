# Stage 6 candidate-guided local re-detection — Fold1 analysis

## Scope

Inference-only exploratory analysis. Candidate generation, crop construction, local matching, and correction used no GT. TXT GT was introduced only for final evaluation. No threshold or crop scale is adopted operationally.

## Candidate integrity and burden

B0 contributed 1,077 candidates over 124 images: mean 8.69, median 9, P90 15, max 24. One-scale all-candidate re-detection therefore costs 8.69 local crops/image in addition to B0, versus 7.21 P192 fixed-grid patches/image.

## Fixed-scale policy sweep

Best observed F1 configuration (diagnostic only): scale=128, policy=CONFIRM_CONFIDENCE, base-local IoU=0.3, TP/FP/FN=281/15/9, F1=0.9590.
Best observed Recall configuration (diagnostic only): scale=128, policy=CONFIRM_CONFIDENCE, IoU=0.3, Recall=0.9690, F1=0.9590.
These observations are a predeclared scale-response sweep, not operational threshold selection.

## B0 FN rescue trajectories

- `001_20200623_082414(1)#2` [LOCALIZATION_FAILURE], policy=CONFIRM_CONFIDENCE, match=0.1: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `001_20200623_082414(1)#2` [LOCALIZATION_FAILURE], policy=REFINE_BOX, match=0.1: first rescue=192, rescue scales=192;224;256, stable adjacent=192;224;256, P192 rescue=True.
- `001_20200623_082414(1)#2` [LOCALIZATION_FAILURE], policy=CONFIRM_AND_REFINE, match=0.1: first rescue=192, rescue scales=192;224;256, stable adjacent=192;224;256, P192 rescue=True.
- `001_20200623_082414(1)#2` [LOCALIZATION_FAILURE], policy=CONFIRM_CONFIDENCE, match=0.3: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `001_20200623_082414(1)#2` [LOCALIZATION_FAILURE], policy=REFINE_BOX, match=0.3: first rescue=224, rescue scales=224;256, stable adjacent=224;256, P192 rescue=True.
- `001_20200623_082414(1)#2` [LOCALIZATION_FAILURE], policy=CONFIRM_AND_REFINE, match=0.3: first rescue=224, rescue scales=224;256, stable adjacent=224;256, P192 rescue=True.
- `001_20200623_082414(1)#2` [LOCALIZATION_FAILURE], policy=CONFIRM_CONFIDENCE, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `001_20200623_082414(1)#2` [LOCALIZATION_FAILURE], policy=REFINE_BOX, match=0.5: first rescue=256, rescue scales=256, stable adjacent=, P192 rescue=True.
- `001_20200623_082414(1)#2` [LOCALIZATION_FAILURE], policy=CONFIRM_AND_REFINE, match=0.5: first rescue=256, rescue scales=256, stable adjacent=, P192 rescue=True.
- `001_20200624_083202(6)#0` [LOW_CONFIDENCE], policy=CONFIRM_CONFIDENCE, match=0.1: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `001_20200624_083202(6)#0` [LOW_CONFIDENCE], policy=REFINE_BOX, match=0.1: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `001_20200624_083202(6)#0` [LOW_CONFIDENCE], policy=CONFIRM_AND_REFINE, match=0.1: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `001_20200624_083202(6)#0` [LOW_CONFIDENCE], policy=CONFIRM_CONFIDENCE, match=0.3: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `001_20200624_083202(6)#0` [LOW_CONFIDENCE], policy=REFINE_BOX, match=0.3: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `001_20200624_083202(6)#0` [LOW_CONFIDENCE], policy=CONFIRM_AND_REFINE, match=0.3: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `001_20200624_083202(6)#0` [LOW_CONFIDENCE], policy=CONFIRM_CONFIDENCE, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `001_20200624_083202(6)#0` [LOW_CONFIDENCE], policy=REFINE_BOX, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `001_20200624_083202(6)#0` [LOW_CONFIDENCE], policy=CONFIRM_AND_REFINE, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `001_20200727_204958(2)#0` [LOCALIZATION_FAILURE], policy=CONFIRM_CONFIDENCE, match=0.1: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `001_20200727_204958(2)#0` [LOCALIZATION_FAILURE], policy=REFINE_BOX, match=0.1: first rescue=256, rescue scales=256, stable adjacent=, P192 rescue=True.
- `001_20200727_204958(2)#0` [LOCALIZATION_FAILURE], policy=CONFIRM_AND_REFINE, match=0.1: first rescue=256, rescue scales=256, stable adjacent=, P192 rescue=True.
- `001_20200727_204958(2)#0` [LOCALIZATION_FAILURE], policy=CONFIRM_CONFIDENCE, match=0.3: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `001_20200727_204958(2)#0` [LOCALIZATION_FAILURE], policy=REFINE_BOX, match=0.3: first rescue=256, rescue scales=256, stable adjacent=, P192 rescue=True.
- `001_20200727_204958(2)#0` [LOCALIZATION_FAILURE], policy=CONFIRM_AND_REFINE, match=0.3: first rescue=256, rescue scales=256, stable adjacent=, P192 rescue=True.
- `001_20200727_204958(2)#0` [LOCALIZATION_FAILURE], policy=CONFIRM_CONFIDENCE, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `001_20200727_204958(2)#0` [LOCALIZATION_FAILURE], policy=REFINE_BOX, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `001_20200727_204958(2)#0` [LOCALIZATION_FAILURE], policy=CONFIRM_AND_REFINE, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `001_20200819_124414(0)#0` [LOCALIZATION_FAILURE], policy=CONFIRM_CONFIDENCE, match=0.1: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `001_20200819_124414(0)#0` [LOCALIZATION_FAILURE], policy=REFINE_BOX, match=0.1: first rescue=224, rescue scales=224;256, stable adjacent=224;256, P192 rescue=False.
- `001_20200819_124414(0)#0` [LOCALIZATION_FAILURE], policy=CONFIRM_AND_REFINE, match=0.1: first rescue=224, rescue scales=224;256, stable adjacent=224;256, P192 rescue=False.
- `001_20200819_124414(0)#0` [LOCALIZATION_FAILURE], policy=CONFIRM_CONFIDENCE, match=0.3: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `001_20200819_124414(0)#0` [LOCALIZATION_FAILURE], policy=REFINE_BOX, match=0.3: first rescue=256, rescue scales=256, stable adjacent=, P192 rescue=False.
- `001_20200819_124414(0)#0` [LOCALIZATION_FAILURE], policy=CONFIRM_AND_REFINE, match=0.3: first rescue=256, rescue scales=256, stable adjacent=, P192 rescue=False.
- `001_20200819_124414(0)#0` [LOCALIZATION_FAILURE], policy=CONFIRM_CONFIDENCE, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `001_20200819_124414(0)#0` [LOCALIZATION_FAILURE], policy=REFINE_BOX, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `001_20200819_124414(0)#0` [LOCALIZATION_FAILURE], policy=CONFIRM_AND_REFINE, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `001_20200826_123714(5)#0` [LOW_CONFIDENCE], policy=CONFIRM_CONFIDENCE, match=0.1: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `001_20200826_123714(5)#0` [LOW_CONFIDENCE], policy=REFINE_BOX, match=0.1: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `001_20200826_123714(5)#0` [LOW_CONFIDENCE], policy=CONFIRM_AND_REFINE, match=0.1: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `001_20200826_123714(5)#0` [LOW_CONFIDENCE], policy=CONFIRM_CONFIDENCE, match=0.3: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `001_20200826_123714(5)#0` [LOW_CONFIDENCE], policy=REFINE_BOX, match=0.3: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `001_20200826_123714(5)#0` [LOW_CONFIDENCE], policy=CONFIRM_AND_REFINE, match=0.3: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `001_20200826_123714(5)#0` [LOW_CONFIDENCE], policy=CONFIRM_CONFIDENCE, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `001_20200826_123714(5)#0` [LOW_CONFIDENCE], policy=REFINE_BOX, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `001_20200826_123714(5)#0` [LOW_CONFIDENCE], policy=CONFIRM_AND_REFINE, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `002_20200623_082311(8)#0` [LOW_CONFIDENCE], policy=CONFIRM_CONFIDENCE, match=0.1: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `002_20200623_082311(8)#0` [LOW_CONFIDENCE], policy=REFINE_BOX, match=0.1: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `002_20200623_082311(8)#0` [LOW_CONFIDENCE], policy=CONFIRM_AND_REFINE, match=0.1: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `002_20200623_082311(8)#0` [LOW_CONFIDENCE], policy=CONFIRM_CONFIDENCE, match=0.3: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `002_20200623_082311(8)#0` [LOW_CONFIDENCE], policy=REFINE_BOX, match=0.3: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `002_20200623_082311(8)#0` [LOW_CONFIDENCE], policy=CONFIRM_AND_REFINE, match=0.3: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `002_20200623_082311(8)#0` [LOW_CONFIDENCE], policy=CONFIRM_CONFIDENCE, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `002_20200623_082311(8)#0` [LOW_CONFIDENCE], policy=REFINE_BOX, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `002_20200623_082311(8)#0` [LOW_CONFIDENCE], policy=CONFIRM_AND_REFINE, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `002_20200623_082318(5)#2` [LOCALIZATION_FAILURE], policy=CONFIRM_CONFIDENCE, match=0.1: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `002_20200623_082318(5)#2` [LOCALIZATION_FAILURE], policy=REFINE_BOX, match=0.1: first rescue=160, rescue scales=160;192;224, stable adjacent=160;192;224, P192 rescue=True.
- `002_20200623_082318(5)#2` [LOCALIZATION_FAILURE], policy=CONFIRM_AND_REFINE, match=0.1: first rescue=160, rescue scales=160;192;224, stable adjacent=160;192;224, P192 rescue=True.
- `002_20200623_082318(5)#2` [LOCALIZATION_FAILURE], policy=CONFIRM_CONFIDENCE, match=0.3: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `002_20200623_082318(5)#2` [LOCALIZATION_FAILURE], policy=REFINE_BOX, match=0.3: first rescue=160, rescue scales=160;192;224, stable adjacent=160;192;224, P192 rescue=True.
- `002_20200623_082318(5)#2` [LOCALIZATION_FAILURE], policy=CONFIRM_AND_REFINE, match=0.3: first rescue=160, rescue scales=160;192;224, stable adjacent=160;192;224, P192 rescue=True.
- `002_20200623_082318(5)#2` [LOCALIZATION_FAILURE], policy=CONFIRM_CONFIDENCE, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `002_20200623_082318(5)#2` [LOCALIZATION_FAILURE], policy=REFINE_BOX, match=0.5: first rescue=192, rescue scales=192;224, stable adjacent=192;224, P192 rescue=True.
- `002_20200623_082318(5)#2` [LOCALIZATION_FAILURE], policy=CONFIRM_AND_REFINE, match=0.5: first rescue=192, rescue scales=192;224, stable adjacent=192;224, P192 rescue=True.
- `002_20200624_162540(3)#0` [LOW_CONFIDENCE], policy=CONFIRM_CONFIDENCE, match=0.1: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `002_20200624_162540(3)#0` [LOW_CONFIDENCE], policy=REFINE_BOX, match=0.1: first rescue=224, rescue scales=224, stable adjacent=, P192 rescue=True.
- `002_20200624_162540(3)#0` [LOW_CONFIDENCE], policy=CONFIRM_AND_REFINE, match=0.1: first rescue=224, rescue scales=224, stable adjacent=, P192 rescue=True.
- `002_20200624_162540(3)#0` [LOW_CONFIDENCE], policy=CONFIRM_CONFIDENCE, match=0.3: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `002_20200624_162540(3)#0` [LOW_CONFIDENCE], policy=REFINE_BOX, match=0.3: first rescue=224, rescue scales=224, stable adjacent=, P192 rescue=True.
- `002_20200624_162540(3)#0` [LOW_CONFIDENCE], policy=CONFIRM_AND_REFINE, match=0.3: first rescue=224, rescue scales=224, stable adjacent=, P192 rescue=True.
- `002_20200624_162540(3)#0` [LOW_CONFIDENCE], policy=CONFIRM_CONFIDENCE, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=True.
- `002_20200624_162540(3)#0` [LOW_CONFIDENCE], policy=REFINE_BOX, match=0.5: first rescue=224, rescue scales=224, stable adjacent=, P192 rescue=True.
- `002_20200624_162540(3)#0` [LOW_CONFIDENCE], policy=CONFIRM_AND_REFINE, match=0.5: first rescue=224, rescue scales=224, stable adjacent=, P192 rescue=True.
- `002_20200714_083057(5)#2` [LOW_CONFIDENCE], policy=CONFIRM_CONFIDENCE, match=0.1: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `002_20200714_083057(5)#2` [LOW_CONFIDENCE], policy=REFINE_BOX, match=0.1: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `002_20200714_083057(5)#2` [LOW_CONFIDENCE], policy=CONFIRM_AND_REFINE, match=0.1: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `002_20200714_083057(5)#2` [LOW_CONFIDENCE], policy=CONFIRM_CONFIDENCE, match=0.3: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `002_20200714_083057(5)#2` [LOW_CONFIDENCE], policy=REFINE_BOX, match=0.3: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `002_20200714_083057(5)#2` [LOW_CONFIDENCE], policy=CONFIRM_AND_REFINE, match=0.3: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `002_20200714_083057(5)#2` [LOW_CONFIDENCE], policy=CONFIRM_CONFIDENCE, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `002_20200714_083057(5)#2` [LOW_CONFIDENCE], policy=REFINE_BOX, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.
- `002_20200714_083057(5)#2` [LOW_CONFIDENCE], policy=CONFIRM_AND_REFINE, match=0.5: first rescue=, rescue scales=, stable adjacent=, P192 rescue=False.

## Multi-scale consensus

- MS1_160_192, match=0.1: TP/FP/FN=163/145/127, F1=0.5452, rescue=1, loss=119.
- MS2_160_192_224, match=0.1: TP/FP/FN=168/156/122, F1=0.5472, rescue=2, loss=115.
- MS1_160_192, match=0.3: TP/FP/FN=210/90/80, F1=0.7119, rescue=1, loss=72.
- MS2_160_192_224, match=0.3: TP/FP/FN=227/106/63, F1=0.7287, rescue=1, loss=55.
- MS1_160_192, match=0.5: TP/FP/FN=270/26/20, F1=0.9215, rescue=0, loss=11.
- MS2_160_192_224, match=0.5: TP/FP/FN=247/59/43, F1=0.8289, rescue=1, loss=35.

Per-object best-scale rows are oracle upper bounds only and are not deployable.

## Candidate subset burden sweep

- low_base_confidence at 10%: candidates=108, images=64, calls/image=0.87, rescue=0/9, localization rescue=0, low-confidence rescue=0, F1=0.9558.
- low_base_confidence at 20%: candidates=216, images=77, calls/image=1.74, rescue=0/9, localization rescue=0, low-confidence rescue=0, F1=0.9509.
- low_base_confidence at 30%: candidates=324, images=93, calls/image=2.61, rescue=0/9, localization rescue=0, low-confidence rescue=0, F1=0.9445.
- low_base_confidence at 40%: candidates=431, images=105, calls/image=3.48, rescue=0/9, localization rescue=0, low-confidence rescue=0, F1=0.9414.
- low_base_confidence at 50%: candidates=539, images=112, calls/image=4.35, rescue=0/9, localization rescue=0, low-confidence rescue=0, F1=0.9320.
- small_predicted_box at 10%: candidates=108, images=18, calls/image=0.87, rescue=0/9, localization rescue=0, low-confidence rescue=0, F1=0.9590.
- small_predicted_box at 20%: candidates=216, images=32, calls/image=1.74, rescue=1/9, localization rescue=1, low-confidence rescue=0, F1=0.9317.
- small_predicted_box at 30%: candidates=324, images=51, calls/image=2.61, rescue=1/9, localization rescue=1, low-confidence rescue=0, F1=0.8450.
- small_predicted_box at 40%: candidates=431, images=80, calls/image=3.48, rescue=1/9, localization rescue=1, low-confidence rescue=0, F1=0.7445.
- small_predicted_box at 50%: candidates=539, images=89, calls/image=4.35, rescue=1/9, localization rescue=1, low-confidence rescue=0, F1=0.7270.
- low_predicted_local_visibility at 10%: candidates=108, images=51, calls/image=0.87, rescue=0/9, localization rescue=0, low-confidence rescue=0, F1=0.9336.
- low_predicted_local_visibility at 20%: candidates=216, images=91, calls/image=1.74, rescue=1/9, localization rescue=1, low-confidence rescue=0, F1=0.8961.
- low_predicted_local_visibility at 30%: candidates=324, images=109, calls/image=2.61, rescue=1/9, localization rescue=1, low-confidence rescue=0, F1=0.8475.
- low_predicted_local_visibility at 40%: candidates=431, images=115, calls/image=3.48, rescue=1/9, localization rescue=1, low-confidence rescue=0, F1=0.7879.
- low_predicted_local_visibility at 50%: candidates=539, images=122, calls/image=4.35, rescue=1/9, localization rescue=1, low-confidence rescue=0, F1=0.7488.
- high_instability at 10%: candidates=108, images=20, calls/image=0.87, rescue=0/9, localization rescue=0, low-confidence rescue=0, F1=0.9590.
- high_instability at 20%: candidates=216, images=41, calls/image=1.74, rescue=0/9, localization rescue=0, low-confidence rescue=0, F1=0.9590.
- high_instability at 30%: candidates=324, images=75, calls/image=2.61, rescue=0/9, localization rescue=0, low-confidence rescue=0, F1=0.9556.
- high_instability at 40%: candidates=431, images=103, calls/image=3.48, rescue=0/9, localization rescue=0, low-confidence rescue=0, F1=0.9239.
- high_instability at 50%: candidates=539, images=113, calls/image=4.35, rescue=0/9, localization rescue=0, low-confidence rescue=0, F1=0.8744.

All routing signals are prediction-time observable, but their Fold1 ranking performance is diagnostic and not a deployable router. GT is used only to score coverage after routing.

## P192 comparison

The existing P192 fixed-grid model rescues 5/9 B0 FN. Candidate-local common/unique rescue relationships are reported in the rescue matrix; fixed-grid and candidate-local costs are not directly interchangeable because one covers the image and the other repeats inference around every candidate.

## Decision framework

A candidate-guided policy is promising only if it reduces FN, preserves or improves F1, avoids material FP growth and B0-TP loss, improves localization/low-confidence errors, and offers a burden advantage over 7.21 fixed-grid patches/image. Otherwise B0 remains primary and the method is rejected.

## Required decision questions

- Scale 128: best diagnostic fixed-scale row has TP/FP/FN=281/15/9, F1=0.9590, rescue/loss=0/0.
- Scale 160: best diagnostic fixed-scale row has TP/FP/FN=281/15/9, F1=0.9590, rescue/loss=0/0.
- Scale 192: best diagnostic fixed-scale row has TP/FP/FN=281/16/9, F1=0.9574, rescue/loss=0/0.
- Scale 224: best diagnostic fixed-scale row has TP/FP/FN=264/32/26, F1=0.9010, rescue/loss=2/19.
- Scale 256: best diagnostic fixed-scale row has TP/FP/FN=266/30/24, F1=0.9078, rescue/loss=1/16.
- FN reduction: best observed fixed row has FN=9 versus B0 FN=9; this is exploratory, not a selected setting.
- Scale response and possible context loss must be judged from the five predeclared rows above; monotonic improvement is not assumed.
- Confidence confirmation: the row with the fewest low-confidence FN leaves 5 such errors.
- Box refinement: the row with the fewest localization failures leaves 4 such errors versus B0=4; M6=2 is context only, not part of this inference.
- Confirmation-only preservation: the best-F1 row records 0 B0 TP losses and 0 new FP; candidates were never deleted, but bbox refinement can still change GT matching.
- P192 rescue overlap: common=4, candidate-local-only=1, P192-only=1 among the nine B0 FN.
- Multi-scale: best fixed consensus is MS1_160_192 at match=0.5, F1=0.9215; it is not automatically preferred to a single scale.
- Burden: all-candidate one/two/three-scale calls are 8.69/17.37/26.06 per image, so even one scale exceeds the 7.21 fixed-grid P192 reference before routing.
- Practicality: use the burden table to determine whether an observable candidate subset reaches the same rescue coverage with fewer calls. If not, candidate-guided local reinspection is not operationally justified by this Fold1 experiment.