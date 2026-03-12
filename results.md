Landslide Identification — Test Results
========================================

Date: 2026-03-12
GPU: NVIDIA RTX 2000 Ada Generation
Environment: conda landslide-ml (PyTorch cu124, Python 3.10)


SINGLE IMAGE PREDICTION TEST
------------------------------

Image path : data/processed/test/landslide/ls_test_0008.jpg
Country    : Nepal
Forecast   : 3 steps

Phase 1 — AlexNet Classification
  Result     : LANDSLIDE DETECTED
  Confidence : 97.3%
  Checkpoint : checkpoints/alexnet_best.pth (epoch 13, val_acc=78.55%)

Phase 2 — HMM Type Prediction
  Landslide type        : Landslide
  Occurrence probability: 39.6%
  Peak risk month       : July

  Future forecast (next 3 events):
    Step 1 — Landslide  (48.0%)
    Step 2 — Landslide  (64.2%)
    Step 3 — Landslide  (69.4%)

Final verdict: LANDSLIDE DETECTED — Type: Landslide (occurrence probability: 39%)


PHASE 1 — FULL TEST SET EVALUATION
-------------------------------------

Test set: 699 images (211 landslide + 488 non_landslide)
Device  : CUDA (NVIDIA RTX 2000 Ada Generation)

  Metric              Value
  ──────────────────────────
  Accuracy            78.54%
  Precision           62.06%
  Recall              74.41%
  F1-Score            67.67%
  ROC-AUC             85.53%

Confusion Matrix:
                  Predicted: non_landslide   Predicted: landslide
  Actual: non_landslide           392                 96
  Actual: landslide                54                157

Per-class breakdown:
  Class            Precision   Recall   F1-Score   Support
  non_landslide       0.88      0.80      0.84       488
  landslide           0.62      0.74      0.68       211
  macro avg           0.75      0.77      0.76       699
  weighted avg        0.80      0.79      0.79       699

Plots saved:
  plots/confusion_matrix.png
  plots/roc_curve.png


PHASE 2 — HMM MODEL SUMMARY
------------------------------

Dataset    : NASA Global Landslide Catalog (nasa_glc.csv)
Records    : 9,471 events | 1988–2016 | 141 countries
Sequences  : 112 country-level sequences | 9,442 total observations

Hidden states (4):
  0 — Shallow Landslide
  1 — Deep Landslide
  2 — Debris Flow
  3 — Rockfall

Observation symbols (7):
  Complex Landslide (232), Debris Flow (173), Earth Flow (3),
  Landslide (6756), Mudslide (1818), Rockfall (483), Translational Slide (6)

Training log-likelihood : -7314.96
Checkpoint saved        : checkpoints/hmm_model.pkl
Encoder saved           : checkpoints/hmm_encoder.pkl


TRAINING SUMMARY
------------------

AlexNet
  Epochs         : 30
  Batch size     : 64 (GPU)
  Learning rate  : 0.0001 (StepLR: step=10, gamma=0.1)
  Best epoch     : 13
  Best val acc   : 78.55%
  Training plot  : plots/training_history.png

HMM
  Hidden states  : 4
  Iterations     : 100 (Baum-Welch)
  Log-likelihood : -7314.96
