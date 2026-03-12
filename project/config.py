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

# ─── Phase 1: AlexNet ─────────────────────────────────────────────────────────
IMG_SIZE = (227, 227)
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.5
NUM_CLASSES = 2
ALEXNET_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "alexnet_best.pth")

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
CLASS_NAMES = ["non_landslide", "landslide"]  # index 0 and 1
LABEL_MAP = {"non_landslide": 0, "landslide": 1}

# ─── Phase 2: HMM ─────────────────────────────────────────────────────────────
HMM_N_COMPONENTS = 4      # hidden states: shallow, deep, debris, rockfall
HMM_N_ITER = 100
HMM_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "hmm_model.pkl")
HMM_ENCODER_PATH = os.path.join(CHECKPOINTS_DIR, "hmm_encoder.pkl")

# Human-readable labels for HMM hidden states (assigned after training inspection)
HMM_STATE_LABELS = {
    0: "Shallow Landslide",
    1: "Deep Landslide",
    2: "Debris Flow",
    3: "Rockfall",
}

# ─── General ──────────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# Pipeline threshold: minimum AlexNet confidence to trigger Phase 2
PHASE1_THRESHOLD = 0.5

# Create output directories at import time
for _dir in [CHECKPOINTS_DIR, PLOTS_DIR]:
    os.makedirs(_dir, exist_ok=True)
