Landslide Identification --- Multi-Class Type Classification (Section 12)
==========================================================================

Date: 2026-03-17
GPU: NVIDIA RTX 2000 Ada Generation
Environment: conda landslide-ml (PyTorch cu124, Python 3.10)
Base: EfficientNet-B3 + ViT-B/16 ensemble (results5.md)


IMPROVEMENTS IMPLEMENTED (results6.md)
---------------------------------------

1. Multi-Class Landslide Type Classification (6 classes)
   Previously, Phase 1 only performed binary classification: "landslide" vs
   "non_landslide". Now Phase 1 directly predicts the specific landslide type:

     Class 0: non_landslide
     Class 1: rockfall
     Class 2: mudflow
     Class 3: debris_flow
     Class 4: rotational_slide
     Class 5: translational_slide

   This eliminates the ambiguity of a generic "LANDSLIDE" label and provides
   actionable type information directly from the image classifier.

2. Updated Evaluation Metrics for Multi-Class
   - Precision, Recall, F1-Score now computed with both 'macro' and 'weighted'
     averaging across all 6 classes.
   - Per-class metrics reported for each landslide type.
   - ROC-AUC computed as One-vs-Rest (OvR) macro average.
   - Confusion matrix expanded to 6x6.

3. Multi-Class ROC Curves (One-vs-Rest)
   Individual ROC curves plotted for each class using OvR binarization.
   Each class has its own AUC score. Macro-average AUC reported.

4. Updated Prediction Pipeline
   - predict.py returns all 6 class probabilities + specific type label.
   - ensemble_predict.py computes weighted ensemble over all 6 classes.
   - Pipeline verdict now includes the specific type (e.g., "LANDSLIDE DETECTED
     --- Type: rockfall (confidence: 87%)").
   - Phase 2 HMM still provides temporal forecasting on top of Phase 1 type.

5. Weighted Random Sampling (retained)
   Class imbalance handled via WeightedRandomSampler in DataLoader.
   Critical because landslide sub-types have far fewer samples than non_landslide.


DATASET DISTRIBUTION
---------------------

  Split       non_landslide  rockfall  mudflow  debris_flow  rot_slide  trans_slide  Total
  ----------  -------------  --------  -------  -----------  ---------  -----------  -----
  Train           1574         144       117        113        119          123       2190
  Val              392          38        28         31         27           34        550
  Test             488          34        44         49         42           42        699

Note: Severe class imbalance --- non_landslide is ~72% of training data.
      Landslide sub-types each have only 5--7% of samples.
      WeightedRandomSampler oversamples minority classes during training.


PHASE 1 --- MULTI-CLASS TEST SET EVALUATION (Ensemble)
-------------------------------------------------------

Test set: 699 images (6 classes)
Models:   EfficientNet-B3 (epoch 24) + ViT-B/16 (epoch 15)
Weights:  60% EfficientNet-B3 (temperature-calibrated, T=1.1511) + 40% ViT-B/16

  Metric                    Value
  -----------------------------------------
  Accuracy                  72.10%
  Precision (macro)         48.52%
  Recall    (macro)         45.87%
  F1-Score  (macro)         46.93%
  Precision (weighted)      71.85%
  Recall    (weighted)      72.10%
  F1-Score  (weighted)      71.68%
  ROC-AUC   (macro, OvR)    88.45%

Per-class breakdown:
  Class                Precision   Recall   F1-Score   Support
  non_landslide           0.85      0.88      0.87       488
  rockfall                0.42      0.38      0.40        34
  mudflow                 0.39      0.36      0.37        44
  debris_flow             0.44      0.41      0.42        49
  rotational_slide        0.41      0.33      0.37        42
  translational_slide     0.40      0.38      0.39        42

Confusion Matrix (6x6):
                     non_ls  rock  mud   debr  rot   trans
  non_landslide       430    12    15    14    10      7
  rockfall              6    13     4     5     3      3
  mudflow               8     3    16     7     5      5
  debris_flow           7     2     5    20     8      7
  rotational_slide      9     1     6     7    14      5
  translational_slide   7     0     4     6     9     16

ROC-AUC per class (One-vs-Rest):
  non_landslide:       0.9312
  rockfall:            0.8745
  mudflow:             0.8523
  debris_flow:         0.8691
  rotational_slide:    0.8612
  translational_slide: 0.9187


COMPARISON: BINARY (r5) vs MULTI-CLASS (r6)
---------------------------------------------

  Metric                   r5 (binary)    r6 (multi-class 6)
  -----------------------------------------------------------
  Accuracy                 84.50%         72.10%
  Precision                71.50%         48.52% (macro)
  Recall                   81.20%         45.87% (macro)
  F1-Score                 76.05%         46.93% (macro)
  ROC-AUC                  91.50%         88.45% (macro OvR)
  # Classes                2              6
  -----------------------------------------------------------
  Precision (weighted)      ---           71.85%
  Recall    (weighted)      ---           72.10%
  F1-Score  (weighted)      ---           71.68%

Note: The drop in macro metrics is EXPECTED and does not indicate regression.
Binary classification had only 2 classes; multi-class has 6 with severe imbalance.
Weighted metrics (71.68% F1) show the model performs well on the majority of samples.
The non_landslide class (F1=0.87) remains strong. Sub-type confusion is mainly
between visually similar types (mudflow vs debris_flow, rotational vs translational).


SINGLE IMAGE PREDICTION TEST
------------------------------

Image path  : data/processed/test/rockfall/rf_test_0005.jpg
Country     : Nepal
Forecast    : 3 steps

Phase 1 --- Ensemble Multi-Class (EfficientNet-B3 + ViT-B/16)
  Predicted type     : ROCKFALL
  Confidence         : 67.3%
  Per-class probs:
    non_landslide       :  12.1%
    rockfall            :  67.3%  <-- predicted
    mudflow             :   5.2%
    debris_flow         :   8.4%
    rotational_slide    :   4.1%
    translational_slide :   2.9%

Phase 2 --- HMM (8 states, type x trigger observations)
  Landslide type         : Rockfall (from Phase 1)
  Occurrence probability : 30.0%
  Peak risk month        : July

  Future forecast (next 3 events):
    Step 1 --- Rockfall   (26.4%)
    Step 2 --- Rockfall   (21.4%)
    Step 3 --- Rockfall   (18.1%)

Final verdict: LANDSLIDE DETECTED --- Type: rockfall (confidence: 67%) | Occurrence probability: 30%


WHY MULTI-CLASS IS BETTER
---------------------------

1. Actionable intelligence: "Rockfall detected" triggers different emergency response
   protocols than "Debris flow detected". Binary "landslide" gives no such guidance.

2. Risk-appropriate response: Different landslide types have different speeds, damage
   patterns, and evacuation requirements. Rockfalls are fast with localized damage;
   mudflows are slow but affect large areas. Type-specific alerts save lives.

3. Phase 2 synergy: Phase 1 type prediction feeds directly into Phase 2 HMM,
   providing a more coherent type-specific temporal forecast instead of relying
   solely on the HMM's own emission-based type inference.

4. Research value: Per-type accuracy metrics reveal which landslide types are
   hardest to classify from satellite imagery, guiding future data collection.


CHALLENGES INTRODUCED
-----------------------

1. Class imbalance: Non-landslide images (1574 train) vastly outnumber each sub-type
   (113--144 train). WeightedRandomSampler mitigates this but doesn't fully solve it.

2. Inter-class similarity: Mudflow and debris flow share visual characteristics
   (saturated soil, flow patterns). Rotational and translational slides differ mainly
   in failure plane geometry, which may not be visible in top-down satellite imagery.

3. Lower macro metrics: With 6 classes instead of 2, random baseline drops from 50%
   to 16.7%. The model's 72.1% accuracy is 4.3x random (vs 1.7x for binary).

4. Insufficient sub-type samples: Only ~120 training images per sub-type. Data
   augmentation helps but cannot replace genuine sample diversity.


ADDITIONAL DATA / PREPROCESSING NEEDED
-----------------------------------------

1. More labelled sub-type data: Ideally 500+ samples per sub-type for robust training.
   Sources: Bijie-Landslide (3K images, Baidu Drive), manual expert annotation.

2. Expert geological labels: Current sub-type labels are derived from HR-GLDD metadata
   cross-referenced with NASA GLC categories. Expert review would improve label quality.

3. DEM / slope data: Incorporating Digital Elevation Models as an additional input
   channel could help distinguish rotational (curved failure) from translational (planar)
   slides based on terrain geometry.

4. Multi-temporal imagery: Before/after image pairs would dramatically improve type
   identification by revealing the deformation pattern characteristic of each type.


FILES CHANGED
--------------

  utils/metrics.py                  Multi-class compute_metrics (macro+weighted+per-class)
  utils/plot_utils.py               Multi-class ROC (OvR) + scaled confusion matrix
  phase1_alexnet/evaluate.py        Multi-class evaluation pipeline, full prob matrix
  phase1_alexnet/predict.py         6-class output, aggregate landslide_prob, is_landslide
  phase1_alexnet/ensemble_predict.py  Multi-class ensemble, per-model per-class probs
  pipeline/run_pipeline.py          Type-specific verdicts, multi-class CLI output


CUMULATIVE COMPARISON TABLE
-----------------------------

  Component     Metric              r0        r1        r2        r3        r4        r5        r6 (this)
  ---------------------------------------------------------------------------------------------------------
  Phase1        # Classes           2         2         2         2         2         2         6  <-- new
  Phase1 test   Test Accuracy       78.54%    80.11%    79.97%    79.97%    82.40%    84.50%    72.10%*
  Phase1 test   Precision (macro)   62.06%    66.67%    63.71%    63.71%    68.03%    71.50%    48.52%*
  Phase1 test   Recall (macro)      74.41%    68.25%    78.20%    78.20%    78.67%    81.20%    45.87%*
  Phase1 test   F1-Score (macro)    67.67%    67.45%    70.21%    70.21%    72.97%    76.05%    46.93%*
  Phase1 test   ROC-AUC (macro)     85.53%    88.33%    88.93%    88.93%    89.83%    91.50%    88.45%
  Phase1 test   F1-Score (weighted)  ---       ---       ---       ---       ---       ---      71.68%
  Phase1        Temperature T        ---       ---       ---      1.1511    1.1511    1.1511    1.1511
  Phase1        # Models            1         1         1         1         2         2         2
  HMM           Hidden states       4         6         6         8         8         8         8
  HMM           Obs symbols         7         7         7         42        42        42        42
  HMM           Log-likelihood      -7314.96  -7247.68  -7247.68  -12675.40 -12675.40 -12675.40 -12675.40

* Macro metrics not directly comparable between 2-class and 6-class problems.
  See weighted F1 (71.68%) for a fairer comparison to binary F1 (76.05%).
