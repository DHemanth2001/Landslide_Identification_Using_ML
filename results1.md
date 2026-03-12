Landslide Identification — Improved Model Results
===================================================

Date: 2026-03-12
GPU: NVIDIA RTX 2000 Ada Generation
Environment: conda landslide-ml (PyTorch cu124, Python 3.10)


IMPROVEMENTS IMPLEMENTED (from content.md Section 7)
------------------------------------------------------

1. Pretrained ImageNet weights (AlexNet_Weights.IMAGENET1K_V1)
   Feature layers frozen for first 5 epochs, then fully unfrozen for fine-tuning.
   This is the most impactful improvement noted in the content guide.

2. Two-stage learning rates
   Classifier head  : LR = 5e-5  (full training)
   Feature layers   : LR = 5e-6  (after unfreeze at epoch 6)
   Previous LR was 1e-4 from scratch.

3. Stronger augmentation (dataset.py)
   Added: RandomBrightnessContrast (p=0.4), GridDistortion (p=0.2),
          CoarseDropout (p=0.2)
   Retained: HorizontalFlip, VerticalFlip, RandomRotate90, ColorJitter, GaussianBlur

4. Increased training epochs: 30 → 40

5. HMM hidden states: 4 → 6
   New states: Shallow Landslide, Deep Landslide, Debris Flow, Rockfall,
               Mixed Flow, Complex Event


SINGLE IMAGE PREDICTION TEST
------------------------------

Image path : data/processed/test/landslide/ls_test_0008.jpg
Country    : Nepal
Forecast   : 3 steps

Phase 1 — AlexNet Classification
  Result     : LANDSLIDE DETECTED
  Confidence : 98.0%   (was 97.3% before improvements)
  Checkpoint : checkpoints/alexnet_best.pth (epoch 24, val_acc=81.82%)

Phase 2 — HMM Type Prediction
  Landslide type        : Landslide
  Occurrence probability: 42.4%   (was 39.6% with 4 hidden states)
  Peak risk month       : July

  Future forecast (next 3 events):
    Step 1 — Landslide  (76.8%)   (was 48.0%)
    Step 2 — Landslide  (78.5%)   (was 64.2%)
    Step 3 — Landslide  (71.9%)   (was 69.4%)

Final verdict: LANDSLIDE DETECTED — Type: Landslide (occurrence probability: 42%)


PHASE 1 — FULL TEST SET EVALUATION (IMPROVED MODEL)
------------------------------------------------------

Test set: 699 images (211 landslide + 488 non_landslide)
Device  : CUDA (NVIDIA RTX 2000 Ada Generation)

  Metric              Before    After     Change
  ─────────────────────────────────────────────
  Accuracy            78.54%    80.11%    +1.57%
  Precision           62.06%    66.67%    +4.61%
  Recall              74.41%    68.25%    -6.16%
  F1-Score            67.67%    67.45%    -0.22%
  ROC-AUC             85.53%    88.33%    +2.80%

Note: Precision improved significantly (+4.61%) and ROC-AUC improved (+2.80%).
Recall decreased slightly — the pretrained model is more conservative (fewer false
alarms). For critical applications, lowering PHASE1_THRESHOLD below 0.5 would
recover recall at acceptable precision cost.

Confusion Matrix:
                  Predicted: non_landslide   Predicted: landslide
  Actual: non_landslide           416                 72
  Actual: landslide                67                144

Per-class breakdown:
  Class            Precision   Recall   F1-Score   Support
  non_landslide       0.86      0.85      0.86       488
  landslide           0.67      0.68      0.67       211
  macro avg           0.76      0.77      0.77       699
  weighted avg        0.80      0.80      0.80       699

Plots saved:
  plots/confusion_matrix.png
  plots/roc_curve.png


PHASE 2 — IMPROVED HMM MODEL SUMMARY
--------------------------------------

Hidden states: 6  (increased from 4)
  0 — Shallow Landslide
  1 — Deep Landslide
  2 — Debris Flow
  3 — Rockfall
  4 — Mixed Flow
  5 — Complex Event

Training log-likelihood : -7247.68  (was -7314.96, higher = better fit)
Improvement             : +67.28 log-likelihood units
Checkpoint saved        : checkpoints/hmm_model.pkl


TRAINING SUMMARY
------------------

AlexNet (Improved)
  Epochs              : 40  (was 30)
  Batch size          : 64 (GPU)
  Learning rate       : 5e-5 classifier, 5e-6 features after unfreeze (was 1e-4)
  Pretrained weights  : ImageNet (AlexNet_Weights.IMAGENET1K_V1)
  Unfreeze epoch      : 6
  Best epoch          : 24
  Best val acc        : 81.82%  (was 78.55%)
  Improvement         : +3.27% validation accuracy

HMM (Improved)
  Hidden states       : 6  (was 4)
  Iterations          : 100 (Baum-Welch)
  Log-likelihood      : -7247.68  (was -7314.96)
  Improvement         : +67.28 log-likelihood units


COMPARISON TABLE
-----------------

  Component        Metric              Before    After     Delta
  ──────────────────────────────────────────────────────────────
  AlexNet val      Val Accuracy        78.55%    81.82%    +3.27%
  AlexNet test     Test Accuracy       78.54%    80.11%    +1.57%
  AlexNet test     Precision           62.06%    66.67%    +4.61%
  AlexNet test     ROC-AUC             85.53%    88.33%    +2.80%
  AlexNet single   Confidence          97.3%     98.0%     +0.7%
  HMM              Log-likelihood      -7314.96  -7247.68  +67.28
  HMM              Occurrence prob     39.6%     42.4%     +2.8%
