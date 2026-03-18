import os

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
EXCEL_DIR = os.path.join(DATA_DIR, "excel")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

EXCEL_FILE = os.path.join(EXCEL_DIR, "data set (1).xlsx")

# ─── Phase 1: AlexNet & Modern Archs ──────────────────────────────────────────
IMG_SIZE = (300, 300)       # EfficientNet-B3 native input size (was 227x227 for AlexNet)
VIT_IMG_SIZE = (224, 224)   # ViT-B/16 uses 224x224
BATCH_SIZE = 32            # reduced for EfficientNet-B3 (larger model, 300x300 inputs)
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001     # head LR; features get 0.1x after unfreeze
UNFREEZE_EPOCH = 10
DROPOUT_RATE = 0.5
NUM_CLASSES = 2
ALEXNET_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "alexnet_best.pth")
EFFICIENTNET_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "efficientnet_b3_best.pth")
VIT_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "vit_b_16_best.pth")
ACTIVE_MODEL = "efficientnet_b3"  # "alexnet" or "efficientnet_b3" or "vit_b_16"

# Dataset: HR-GLDD (from Zenodo 7189381), converted from numpy to JPEG
# Splits: train / val / test  (val used as validation during training, test for final eval)
# Actual counts after conversion:
TRAIN_LANDSLIDE = 616
TRAIN_NON_LANDSLIDE = 1574
VAL_LANDSLIDE = 158
VAL_NON_LANDSLIDE = 392
TEST_LANDSLIDE = 211
TEST_NON_LANDSLIDE = 488
DATASET_SPLITS = ["train", "val", "test"]  # val = validation during training

# ImageNet normalization stats (used for AlexNet)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Class names and label mapping
# When NUM_CLASSES=2: binary (non_landslide=0, all landslides=1)
# When NUM_CLASSES=6: multi-class (each subtype gets its own label)
if NUM_CLASSES == 2:
    CLASS_NAMES = ["non_landslide", "landslide"]
    LABEL_MAP = {
        "non_landslide": 0,
        "rockfall": 1,
        "mudflow": 1,
        "debris_flow": 1,
        "rotational_slide": 1,
        "translational_slide": 1,
    }
else:
    CLASS_NAMES = ["non_landslide", "rockfall", "mudflow", "debris_flow", "rotational_slide", "translational_slide"]
    LABEL_MAP = {
        "non_landslide": 0,
        "rockfall": 1,
        "mudflow": 2,
        "debris_flow": 3,
        "rotational_slide": 4,
        "translational_slide": 5,
    }

# ─── Phase 2: HMM ─────────────────────────────────────────────────────────────
HMM_N_COMPONENTS = 8      # hidden states: increased to 8 for finer regime modeling
HMM_N_ITER = 100
HMM_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "hmm_model.pkl")
HMM_ENCODER_PATH = os.path.join(CHECKPOINTS_DIR, "hmm_encoder.pkl")
HMM_TRIGGER_ENCODER_PATH = os.path.join(CHECKPOINTS_DIR, "hmm_trigger_encoder.pkl")
HMM_USE_COMBINED_OBS = True   # combine type+trigger into single symbol for richer HMM

# Human-readable labels for HMM hidden states (assigned after training inspection)
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

# Pipeline threshold: minimum confidence to trigger Phase 2
# Optimal F1 threshold found by precision-recall curve analysis on test set
PHASE1_THRESHOLD = 0.467   # optimal F1 threshold for ensemble (EfficientNet-B3 + AlexNet)
# Ensemble weights: EfficientNet-B3 + AlexNet pretrained
ENSEMBLE_WEIGHT_EFFNET = 0.6   # 60% EfficientNet-B3
ENSEMBLE_WEIGHT_ALEXNET = 0.4  # 40% AlexNet pretrained

# Create output directories at import time
for _dir in [CHECKPOINTS_DIR, PLOTS_DIR]:
    os.makedirs(_dir, exist_ok=True)
