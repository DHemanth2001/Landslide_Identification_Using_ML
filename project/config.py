import os

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "binary_combined_expanded")
EXCEL_DIR = os.path.join(DATA_DIR, "excel")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

EXCEL_FILE = os.path.join(EXCEL_DIR, "data set (1).xlsx")

# ─── Phase 1: Multi-Scale Attention Fusion Network ──────────────────────────
# Architecture: ConvNeXt-Base + CBAM + FPN (primary) + SwinV2-Small (ensemble)
# Base paper: ResM-FusionNet (2025), improved with ConvNeXt backbone + CBAM attention
#
# Reference papers:
#   1. ResM-FusionNet (2025) — multi-scale residual fusion for landslide detection
#   2. "A ConvNet for the 2020s" — ConvNeXt (Liu et al., CVPR 2022)
#   3. "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
#   4. "Swin Transformer V2" (Liu et al., CVPR 2022)
#   5. "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017)

# Model configuration
CONVNEXT_IMG_SIZE = (224, 224)     # ConvNeXt-Base native pretrained size
SWINV2_IMG_SIZE = (256, 256)       # SwinV2-Small native pretrained size
EFFNETV2_IMG_SIZE = (384, 384)     # EfficientNetV2-S optimal resolution
IMG_SIZE = CONVNEXT_IMG_SIZE       # Primary model image size

BATCH_SIZE = 16                    # Smaller batch for larger models + mixup
NUM_EPOCHS = 50                    # More epochs for thorough convergence
LEARNING_RATE = 5e-5               # Lower LR for pretrained backbones
UNFREEZE_EPOCH = 8                 # Unfreeze backbone after 8 warm-up epochs
DROPOUT_RATE = 0.4                 # Slightly less dropout with label smoothing
NUM_CLASSES = 2                    # Binary: non_landslide vs landslide

# Advanced training hyperparameters
LABEL_SMOOTHING = 0.1              # Prevents overconfident predictions
MIXUP_ALPHA = 0.3                  # Mixup interpolation strength
CUTMIX_ALPHA = 1.0                 # CutMix interpolation strength
MIXUP_PROB = 0.5                   # Probability of applying mixup vs cutmix
GRADIENT_CLIP_NORM = 1.0           # Max gradient norm for stability
EMA_DECAY = 0.9998                 # EMA smoothing factor (higher = smoother)
WEIGHT_DECAY = 0.05                # AdamW weight decay (ConvNeXt paper setting)
WARMUP_EPOCHS = 5                  # Linear LR warmup before cosine decay

# Active model selection
ACTIVE_MODEL = "convnext_cbam_fpn"  # "convnext_cbam_fpn" or "swinv2_s"

# Checkpoint paths — new models
CONVNEXT_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "convnext_cbam_fpn_best.pth")
SWINV2_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "swinv2_s_best.pth")
EMA_CONVNEXT_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "convnext_cbam_fpn_ema_best.pth")
EMA_SWINV2_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "swinv2_s_ema_best.pth")
EFFNETV2_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "efficientnetv2_cbam_best.pth")
EMA_EFFNETV2_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "efficientnetv2_cbam_ema_best.pth")

# Legacy checkpoint paths (kept for backward compatibility)
ALEXNET_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "alexnet_best.pth")
EFFICIENTNET_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "efficientnet_b3_best.pth")
VIT_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "vit_b_16_best.pth")

# Dataset: HR-GLDD (from Zenodo 7189381), converted from numpy to JPEG
# 6-class multi-class classification
TRAIN_LANDSLIDE = 8008
TRAIN_NON_LANDSLIDE = 7870
VAL_LANDSLIDE = 158
VAL_NON_LANDSLIDE = 392
TEST_LANDSLIDE = 211
TEST_NON_LANDSLIDE = 488
DATASET_SPLITS = ["train", "val", "test"]

# ImageNet normalization stats
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Binary classification configuration
CLASS_NAMES = ["non_landslide", "landslide"]
LABEL_MAP = {
    "non_landslide": 0,
    "landslide": 1,
}

# Ensemble weights — ConvNeXt-CBAM-FPN + SwinV2-Small
ENSEMBLE_WEIGHT_CONVNEXT = 0.55    # ConvNeXt gets slightly more weight (stronger backbone)
ENSEMBLE_WEIGHT_SWINV2 = 0.45     # SwinV2 complements with attention-based features

# Pipeline threshold: minimum confidence to trigger Phase 2
PHASE1_THRESHOLD = 0.50            # Will be re-calibrated after training

# ─── Phase 2: Bi-LSTM + Multi-Head Attention ─────────────────────────────────
# Replaces classical HMM with deep learning temporal model.
# Reference: "Attention-Based RNNs for Landslide Temporal Prediction" (2024)
LSTM_HIDDEN_DIM = 128              # LSTM hidden state dimension (256 bidirectional)
LSTM_NUM_LAYERS = 2                # Stacked Bi-LSTM layers
LSTM_NUM_HEADS = 4                 # Multi-head attention heads
LSTM_DROPOUT = 0.3                 # Dropout in LSTM + attention
LSTM_NUM_EPOCHS = 100              # Max training epochs (early stopping applies)
LSTM_LEARNING_RATE = 1e-3          # AdamW learning rate
LSTM_FORECAST_STEPS = 3            # Number of future steps to forecast
LSTM_EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement for 15 epochs

# Checkpoint paths — LSTM model
LSTM_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "bilstm_attention_best.pth")
LSTM_TYPE_ENCODER_PATH = os.path.join(CHECKPOINTS_DIR, "lstm_type_encoder.pkl")
LSTM_TRIGGER_ENCODER_PATH = os.path.join(CHECKPOINTS_DIR, "lstm_trigger_encoder.pkl")

# Active Phase 2 model: "lstm" (new) or "hmm" (legacy)
ACTIVE_PHASE2 = "lstm"

# ─── Phase 2 Legacy: HMM (kept for comparison) ───────────────────────────────
HMM_N_COMPONENTS = 8
HMM_N_ITER = 100
HMM_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "hmm_model.pkl")
HMM_ENCODER_PATH = os.path.join(CHECKPOINTS_DIR, "hmm_encoder.pkl")
HMM_TRIGGER_ENCODER_PATH = os.path.join(CHECKPOINTS_DIR, "hmm_trigger_encoder.pkl")
HMM_USE_COMBINED_OBS = True

HMM_STATE_LABELS = {
    0: "Shallow Rain Slide",
    1: "Deep Seismic Slide",
    2: "Debris Flow",
    3: "Rockfall",
    4: "Mixed Flow",
    5: "Complex Event",
    6: "Monsoon Mudslide",
    7: "Human-Triggered Slide",
}

# ─── General ──────────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# Create output directories at import time
for _dir in [CHECKPOINTS_DIR, PLOTS_DIR]:
    os.makedirs(_dir, exist_ok=True)
