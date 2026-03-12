Landslide Identification — EfficientNet-B3 Results
====================================================

Date: 2026-03-12
GPU: NVIDIA RTX 2000 Ada Generation
Environment: conda landslide-ml (PyTorch cu124, Python 3.10)


IMPROVEMENTS IMPLEMENTED (from results1.md baseline)
------------------------------------------------------

1. More powerful architecture — EfficientNet-B3 (pretrained ImageNet)
   Replaced AlexNet with EfficientNet-B3 (300x300 input, 1536-dim feature head).
   Feature layers frozen for first 5 epochs, fully unfrozen at epoch 6 with
   differential learning rates (head: 1e-4, features: 1e-5).
   EfficientNet-B3 has ~12M parameters vs AlexNet's ~62M, but far deeper and
   more efficient via compound scaling (depth + width + resolution).

2. Input resolution upgrade: 227x227 → 300x300
   EfficientNet-B3 native resolution; captures finer spatial detail.

3. Batch size reduced: 64 → 32
   Required due to EfficientNet-B3's larger memory footprint per image.

4. CosineAnnealingLR scheduler (eta_min=1e-7)
   Smooth LR decay vs StepLR used previously. Scheduler resets at unfreeze epoch.

5. Retained from results1.md:
   - Stronger augmentation (RandomBrightnessContrast, GridDistortion, CoarseDropout)
   - HMM with 6 hidden states
   - WeightedRandomSampler for class imbalance


TRAINING LOG (EfficientNet-B3, 40 epochs)
-------------------------------------------

Using device: cuda
Model: efficientnet_b3
Train set: 2190 images | Val set: 550 images
Pretrained weights: ImageNet (EfficientNet_B3_Weights.IMAGENET1K_V1)

Epoch [  1/40]  Train Loss: 0.6896  Train Acc: 0.5329  Val Loss: 0.6782  Val Acc: 0.5836
Epoch [  2/40]  Train Loss: 0.6543  Train Acc: 0.6425  Val Loss: 0.6420  Val Acc: 0.6509  *
Epoch [  3/40]  Train Loss: 0.6350  Train Acc: 0.6758  Val Loss: 0.6444  Val Acc: 0.6291
Epoch [  4/40]  Train Loss: 0.6137  Train Acc: 0.6954  Val Loss: 0.6331  Val Acc: 0.6509
Epoch [  5/40]  Train Loss: 0.6119  Train Acc: 0.6872  Val Loss: 0.6021  Val Acc: 0.6945  *
  --> Epoch 6: Unfroze all layers for full fine-tuning
Epoch [  6/40]  Train Loss: 0.5820  Train Acc: 0.7288  Val Loss: 0.5960  Val Acc: 0.6800
Epoch [  7/40]  Train Loss: 0.5421  Train Acc: 0.7470  Val Loss: 0.5480  Val Acc: 0.7145  *
Epoch [  8/40]  Train Loss: 0.5226  Train Acc: 0.7489  Val Loss: 0.5335  Val Acc: 0.7273  *
Epoch [  9/40]  Train Loss: 0.4956  Train Acc: 0.7653  Val Loss: 0.5605  Val Acc: 0.7073
Epoch [ 10/40]  Train Loss: 0.4800  Train Acc: 0.7653  Val Loss: 0.4972  Val Acc: 0.7564  *
Epoch [ 11/40]  Train Loss: 0.4972  Train Acc: 0.7516  Val Loss: 0.5229  Val Acc: 0.7455
Epoch [ 12/40]  Train Loss: 0.4781  Train Acc: 0.7749  Val Loss: 0.4961  Val Acc: 0.7636  *
Epoch [ 13/40]  Train Loss: 0.4735  Train Acc: 0.7680  Val Loss: 0.4722  Val Acc: 0.7691  *
Epoch [ 14/40]  Train Loss: 0.4619  Train Acc: 0.7703  Val Loss: 0.4554  Val Acc: 0.7855  *
Epoch [ 15/40]  Train Loss: 0.4579  Train Acc: 0.7836  Val Loss: 0.4792  Val Acc: 0.7745
Epoch [ 16/40]  Train Loss: 0.4474  Train Acc: 0.7849  Val Loss: 0.4591  Val Acc: 0.7800
Epoch [ 17/40]  Train Loss: 0.4533  Train Acc: 0.7941  Val Loss: 0.4837  Val Acc: 0.7673
Epoch [ 18/40]  Train Loss: 0.4561  Train Acc: 0.7740  Val Loss: 0.4483  Val Acc: 0.7782
Epoch [ 19/40]  Train Loss: 0.4493  Train Acc: 0.7836  Val Loss: 0.4548  Val Acc: 0.7709
Epoch [ 20/40]  Train Loss: 0.4480  Train Acc: 0.7849  Val Loss: 0.4778  Val Acc: 0.7618
Epoch [ 21/40]  Train Loss: 0.4418  Train Acc: 0.7945  Val Loss: 0.4667  Val Acc: 0.7727
Epoch [ 22/40]  Train Loss: 0.4295  Train Acc: 0.7913  Val Loss: 0.4519  Val Acc: 0.7836
Epoch [ 23/40]  Train Loss: 0.4295  Train Acc: 0.7890  Val Loss: 0.4665  Val Acc: 0.7709
Epoch [ 24/40]  Train Loss: 0.4452  Train Acc: 0.7826  Val Loss: 0.4179  Val Acc: 0.7982  * BEST
Epoch [ 25/40]  Train Loss: 0.4233  Train Acc: 0.7945  Val Loss: 0.4482  Val Acc: 0.7782
Epoch [ 26/40]  Train Loss: 0.4231  Train Acc: 0.7963  Val Loss: 0.4602  Val Acc: 0.7818
Epoch [ 27/40]  Train Loss: 0.4489  Train Acc: 0.7712  Val Loss: 0.4328  Val Acc: 0.7873
Epoch [ 28/40]  Train Loss: 0.4385  Train Acc: 0.7959  Val Loss: 0.4538  Val Acc: 0.7764
Epoch [ 29/40]  Train Loss: 0.4469  Train Acc: 0.7740  Val Loss: 0.4762  Val Acc: 0.7709
Epoch [ 30/40]  Train Loss: 0.4205  Train Acc: 0.7941  Val Loss: 0.4466  Val Acc: 0.7782
Epoch [ 31/40]  Train Loss: 0.4254  Train Acc: 0.7927  Val Loss: 0.4326  Val Acc: 0.7836
Epoch [ 32/40]  Train Loss: 0.4276  Train Acc: 0.7963  Val Loss: 0.4257  Val Acc: 0.7909
Epoch [ 33/40]  Train Loss: 0.4098  Train Acc: 0.8119  Val Loss: 0.4190  Val Acc: 0.7927
Epoch [ 34/40]  Train Loss: 0.4246  Train Acc: 0.7977  Val Loss: 0.4244  Val Acc: 0.7855
Epoch [ 35/40]  Train Loss: 0.4241  Train Acc: 0.7963  Val Loss: 0.4426  Val Acc: 0.7818
Epoch [ 36/40]  Train Loss: 0.4374  Train Acc: 0.7840  Val Loss: 0.4313  Val Acc: 0.7891
Epoch [ 37/40]  Train Loss: 0.4434  Train Acc: 0.7840  Val Loss: 0.4371  Val Acc: 0.7927
Epoch [ 38/40]  Train Loss: 0.4488  Train Acc: 0.7799  Val Loss: 0.4133  Val Acc: 0.7964
Epoch [ 39/40]  Train Loss: 0.4088  Train Acc: 0.8000  Val Loss: 0.4161  Val Acc: 0.7982
Epoch [ 40/40]  Train Loss: 0.4196  Train Acc: 0.8005  Val Loss: 0.4602  Val Acc: 0.7782

Training complete. Best validation accuracy: 0.7982 (epoch 24)
Checkpoint saved: checkpoints/efficientnet_b3_best.pth


SINGLE IMAGE PREDICTION TEST
------------------------------

Image path : data/processed/test/landslide/ls_test_0008.jpg
Country    : Nepal
Forecast   : 3 steps

Phase 1 — EfficientNet-B3 Classification
  Result     : LANDSLIDE DETECTED
  Confidence : 95.8%   (was 98.0% with pretrained AlexNet, was 97.3% baseline)
  Checkpoint : checkpoints/efficientnet_b3_best.pth (epoch 24, val_acc=79.82%)

Phase 2 — HMM Type Prediction
  Landslide type        : Landslide
  Occurrence probability: 42.4%
  Peak risk month       : July

  Future forecast (next 3 events):
    Step 1 — Landslide  (76.8%)
    Step 2 — Landslide  (78.5%)
    Step 3 — Landslide  (71.9%)

Final verdict: LANDSLIDE DETECTED — Type: Landslide (occurrence probability: 42%)


PHASE 1 — FULL TEST SET EVALUATION (EfficientNet-B3)
------------------------------------------------------

Test set: 699 images (211 landslide + 488 non_landslide)
Device  : CUDA (NVIDIA RTX 2000 Ada Generation)
Checkpoint: checkpoints/efficientnet_b3_best.pth (epoch 24)

  Metric              results.md    results1.md   results2.md   Change vs r1
  ─────────────────────────────────────────────────────────────────────────
  Accuracy            78.54%        80.11%        79.97%        -0.14%
  Precision           62.06%        66.67%        63.71%        -2.96%
  Recall              74.41%        68.25%        78.20%        +9.95%
  F1-Score            67.67%        67.45%        70.21%        +2.76%
  ROC-AUC             85.53%        88.33%        88.93%        +0.60%

Note: EfficientNet-B3 significantly improves Recall (+9.95%) and F1-Score (+2.76%)
over pretrained AlexNet (results1.md). ROC-AUC also improved to 88.93% (new best).
Precision is slightly lower — the model detects more true landslides but also more
false positives. For disaster monitoring, high recall is preferred over precision.

Confusion Matrix:
                  Predicted: non_landslide   Predicted: landslide
  Actual: non_landslide           394                 94
  Actual: landslide                46               165

Per-class breakdown:
  Class            Precision   Recall   F1-Score   Support
  non_landslide       0.90      0.81      0.85       488
  landslide           0.64      0.78      0.70       211
  macro avg           0.77      0.79      0.78       699
  weighted avg        0.82      0.80      0.80       699

Plots saved:
  plots/confusion_matrix.png
  plots/roc_curve.png


PHASE 2 — HMM MODEL SUMMARY (unchanged from results1.md)
----------------------------------------------------------

Hidden states: 6
  0 — Shallow Landslide
  1 — Deep Landslide
  2 — Debris Flow
  3 — Rockfall
  4 — Mixed Flow
  5 — Complex Event

Training log-likelihood : -7247.68
Checkpoint saved        : checkpoints/hmm_model.pkl


TRAINING SUMMARY
------------------

EfficientNet-B3
  Epochs              : 40
  Batch size          : 32 (GPU, reduced for model size)
  Input resolution    : 300x300
  Learning rate       : 1e-4 classifier, 1e-5 features after unfreeze (epoch 6)
  Pretrained weights  : ImageNet (EfficientNet_B3_Weights.IMAGENET1K_V1)
  Unfreeze epoch      : 6
  Best epoch          : 24
  Best val acc        : 79.82%


CUMULATIVE COMPARISON TABLE
-----------------------------

  Component        Metric              r0        r1        r2        Delta r1→r2
  ─────────────────────────────────────────────────────────────────────────────
  Phase1 val       Val Accuracy        78.55%    81.82%    79.82%    -2.00%
  Phase1 test      Test Accuracy       78.54%    80.11%    79.97%    -0.14%
  Phase1 test      Precision           62.06%    66.67%    63.71%    -2.96%
  Phase1 test      Recall              74.41%    68.25%    78.20%    +9.95%  ← best
  Phase1 test      F1-Score            67.67%    67.45%    70.21%    +2.76%  ← best
  Phase1 test      ROC-AUC             85.53%    88.33%    88.93%    +0.60%  ← best
  Phase1 single    Confidence          97.3%     98.0%     95.8%     -2.2%
  HMM              Log-likelihood      -7314.96  -7247.68  -7247.68  (same)

DATASET EXPANSION NOTE
-----------------------

Bijie-Landslide dataset (3,000 images, RGB): Only accessible via Baidu Drive —
no direct download URL. Not usable without manual download.

Landslide4Sense dataset (2.9 GB, 14-band .h5): Requires complex format conversion
(14 spectral bands → RGB extraction + GeoTIFF handling). Out of scope for this run.

EfficientNet-B3 on existing HR-GLDD data achieves best ROC-AUC (88.93%) and
best Recall (78.20%) among all three result iterations.
