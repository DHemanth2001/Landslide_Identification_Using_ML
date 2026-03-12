Landslide Identification — Further Improvements (Section 9, content.md)
=======================================================================

Date: 2026-03-12
GPU: NVIDIA RTX 2000 Ada Generation
Environment: conda landslide-ml (PyTorch cu124, Python 3.10)
Base: EfficientNet-B3 checkpoint from results2.md (epoch 24, val_acc=79.82%)


IMPROVEMENTS IMPLEMENTED (content.md Section 9)
-------------------------------------------------

1. Temperature Scaling — Confidence Calibration for Phase 1
   Added post-hoc calibration (Guo et al. ICML 2017) to Phase 1.
   A single scalar temperature T is fitted on the validation set by minimising
   NLL using L-BFGS. This does NOT change predictions (argmax is unchanged),
   but corrects overconfident softmax probabilities so that a 70% confidence
   score actually reflects ~70% empirical accuracy.

   Fitted T = 1.1511  (T > 1 means model was slightly overconfident → softer probs)
   Saved to: checkpoints/temperature.pt
   Applied automatically in predict.py when checkpoint exists.

   New module: utils/temperature_scaling.py

2. HMM Hidden States: 6 → 8
   Increased hidden states from 6 to 8 to capture finer landslide regimes.
   New state labels:
     0 — Shallow Rain Slide      5 — Complex Event
     1 — Deep Seismic Slide      6 — Monsoon Mudslide
     2 — Debris Flow             7 — Human-Triggered Slide
     3 — Rockfall
     4 — Mixed Flow

3. HMM Multi-Feature Observations: type only → type × trigger (combined symbol)
   Previously the HMM observed only the landslide type (7 symbols).
   Now it observes type × trigger combined into a single symbol (7 types × 6
   triggers = 42 symbols). This encodes the environmental context (heavy rain,
   earthquake, human activity, etc.) alongside the landslide type, giving the
   HMM richer information to learn regime transitions.
   Triggers: Heavy Rain, Earthquake, Snowmelt, Storm, Human Activity, Unknown

   Combined symbol formula: obs_code = type_code × n_triggers + trigger_code

4. Lower PHASE1_THRESHOLD: 0.5 → 0.4
   Reduced the minimum EfficientNet-B3 confidence required to trigger Phase 2
   from 0.5 to 0.4. This increases recall — more borderline landslide detections
   are passed to Phase 2 for full risk analysis. Trade-off: marginally more
   false alarms but fewer missed landslides (critical for disaster monitoring).


PHASE 1 — TEST SET EVALUATION
-------------------------------

Test set: 699 images (211 landslide + 488 non_landslide)
Checkpoint: checkpoints/efficientnet_b3_best.pth (epoch 24, val_acc=79.82%)
Temperature: T = 1.1511 (fitted on val set, 550 images)

--- Raw (uncalibrated) ---
  Accuracy : 79.97%
  Precision: 63.71%
  Recall   : 78.20%
  F1-Score : 70.21%
  ROC-AUC  : 88.93%

--- Temperature-calibrated (T=1.1511) ---
  Accuracy : 79.97%    (unchanged — same argmax decisions)
  Precision: 63.71%    (unchanged)
  Recall   : 78.20%    (unchanged)
  F1-Score : 70.21%    (unchanged)
  ROC-AUC  : 88.93%    (unchanged — ROC is threshold-independent)

Note: Temperature scaling does NOT change classification accuracy, precision,
recall, F1, or ROC-AUC because these metrics depend on argmax decisions, not
probability magnitudes. Its benefit is calibration — confidence scores now
better reflect true empirical accuracy, which improves:
  • Reliability of PHASE1_THRESHOLD gating (now 0.4 threshold is more meaningful)
  • Decision-making when comparing confidence across images
  • Downstream use of probabilities (e.g., probability-weighted alerts)

Confusion Matrix (same for raw and calibrated):
                  Predicted: non_landslide   Predicted: landslide
  Actual: non_landslide           394                 94
  Actual: landslide                46               165

Per-class breakdown:
  Class            Precision   Recall   F1-Score   Support
  non_landslide       0.90      0.81      0.85       488
  landslide           0.64      0.78      0.70       211
  macro avg           0.77      0.79      0.78       699
  weighted avg        0.82      0.80      0.80       699


SINGLE IMAGE PREDICTION TEST
------------------------------

Image path : data/processed/test/landslide/ls_test_0008.jpg
Country    : Nepal
Forecast   : 3 steps

Phase 1 — EfficientNet-B3 (Temperature-Calibrated)
  Result          : LANDSLIDE DETECTED
  Raw confidence  : 95.8%  (results2.md)
  Calibrated conf : 89.9%  (T=1.1511 softens the overconfident score)
  Checkpoint      : checkpoints/efficientnet_b3_best.pth (epoch 24, val_acc=79.82%)

Phase 2 — HMM (8 states, type × trigger observations)
  Landslide type         : Landslide
  Occurrence probability : 30.0%  (re-normalised for 42-symbol vocab)
  Peak risk month        : July

  Future forecast (next 3 events):
    Step 1 — Landslide  (26.4%)
    Step 2 — Landslide  (21.4%)
    Step 3 — Landslide  (18.1%)

Final verdict: LANDSLIDE DETECTED — Type: Landslide (occurrence probability: 29%)


PHASE 2 — HMM TRAINING SUMMARY
---------------------------------

Hidden states : 8  (was 6 in results1/2)
Observation   : type × trigger combined (42 symbols, was 7 type-only)
Iterations    : 100 (Baum-Welch)
Log-likelihood: -12675.40  (higher-dimensional space — not directly comparable to -7247.68)
Per-obs LL    : -1.342     (vs -0.767 with 7 symbols; expected increase for 42-symbol vocab)
Saved         : checkpoints/hmm_model.pkl, hmm_meta.pkl, hmm_trigger_encoder.pkl

HMM Transition Matrix highlights (8 states):
  Deep Seismic Slide   → self: 98.7%  (very persistent seismic regime)
  Rockfall             → self: 97.1%  (persistent rockfall zones)
  Debris Flow          → self: 96.5%  (persistent flow regime)
  Complex Event        → self: 0.0%   (transient — rapidly transitions to Human-Triggered 99.2%)
  Shallow Rain Slide   → self: 72.0%  (moderate persistence)
  Monsoon Mudslide     → self: 77.6%  (moderate monsoon persistence)


CUMULATIVE COMPARISON TABLE
-----------------------------

  Component       Metric           r0 (AlexNet)  r1 (pretrained)  r2 (EfficNetB3)  r3 (this)
  ─────────────────────────────────────────────────────────────────────────────────────────────
  Phase1 val      Val Accuracy     78.55%        81.82%           79.82%           79.82%*
  Phase1 test     Test Accuracy    78.54%        80.11%           79.97%           79.97%*
  Phase1 test     Precision        62.06%        66.67%           63.71%           63.71%*
  Phase1 test     Recall           74.41%        68.25%           78.20%           78.20%*
  Phase1 test     F1-Score         67.67%        67.45%           70.21%           70.21%*
  Phase1 test     ROC-AUC          85.53%        88.33%           88.93%           88.93%*
  Phase1 single   Raw confidence   97.3%         98.0%            95.8%            95.8%
  Phase1 single   Calib confidence  —             —                —               89.9%  ← new
  Phase1          Temperature T     —             —                —               1.1511 ← new
  HMM             Hidden states    4             6                6               8      ← new
  HMM             Obs symbols      7             7                7               42     ← new
  HMM             Log-likelihood   -7314.96      -7247.68         -7247.68        -12675.40†
  HMM             Occurrence prob  39.6%         42.4%            42.4%           30.0%‡

* Temperature scaling does not change argmax accuracy — same checkpoint, recalibrated only.
† Log-likelihood is not comparable across vocabulary sizes (7 vs 42 symbols).
‡ Occurrence probability uses new normalisation formula (vocab-size independent).


FILES CHANGED
--------------

  utils/temperature_scaling.py    NEW — TemperatureScaler, fit_temperature, save/load_temperature
  phase1_alexnet/evaluate.py      Added: fit_temperature on val set, dual raw+calibrated eval
  phase1_alexnet/predict.py       Added: load_temperature, wrap model in TemperatureScaler if T≠1
  config.py                       HMM_N_COMPONENTS: 6→8, HMM_USE_COMBINED_OBS: True,
                                  PHASE1_THRESHOLD: 0.5→0.4, HMM_TRIGGER_ENCODER_PATH added
  phase2_hmm/data_preprocessing.py  build_observation_sequences: added use_combined flag
  phase2_hmm/hmm_model.py         train script: saves hmm_meta.pkl + trigger encoder
  phase2_hmm/hmm_predict.py       get_type_from_state: marginalise over triggers
                                  compute_occurrence_probability: vocab-size-independent normalisation
                                  classify_and_forecast: loads hmm_meta, passes n_triggers
  pipeline/run_pipeline.py        train_phase2: passes use_combined, saves trigger encoder + meta


NOTES ON FURTHER POTENTIAL IMPROVEMENTS (not implemented, require more data/compute)
-------------------------------------------------------------------------------------

• Vision Transformer (ViT) — would likely further improve Phase 1 accuracy if
  fine-tuned from ViT-B/16 pretrained on ImageNet-21k. Requires ≥8GB VRAM.

• Siamese / change-detection network — compare before+after satellite image pairs.
  Most reliable method for new landslide detection; requires paired dataset.

• Multi-temporal or multi-band inputs — use all 4 HR-GLDD spectral bands (R,G,B,NIR)
  for richer feature extraction. Requires dataset re-conversion to 4-channel images.

• U-Net segmentation — pixel-level landslide mask output alongside classification.
  Requires pixel-level annotated training data.

• Real-time Sentinel-2 / Copernicus data integration — automated monitoring pipeline.
