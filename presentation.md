


# Landslide Identification Using Machine Learning
### M.Tech Project Presentation

---

## Slide 1: Title Slide

**Project Title:** Landslide Identification Using Machine Learning

**Approach:** Two-Phase Pipeline — Deep Learning (CNN) + Probabilistic Model (HMM)

**Tools Used:** Python, PyTorch, hmmlearn, OpenCV, Albumentations

**Hardware:** NVIDIA RTX 2000 Ada Generation GPU, CUDA 12.4

---

## Slide 2: Problem Statement

**What is the problem?**
- Landslides kill thousands of people every year and destroy roads, buildings, and farmlands
- Countries like India, Nepal, China, and Philippines are highly affected due to mountains, heavy rainfall, and construction on slopes
- Currently, experts manually look at satellite images to find landslides — this is slow, expensive, and cannot cover large areas

**What we want to solve:**
- Automatically detect if a satellite image contains a landslide
- Identify what type of landslide it is
- Predict what kind of landslide may happen next in that region
- Provide a complete risk report with historical data

**Why this is important:**
- Early detection saves lives and helps disaster management teams act faster
- No existing system combines image detection with future prediction in one pipeline

---

## Slide 3: How This Project is Different

| Existing Systems | Our System |
|---|---|
| Use only vegetation index (NDVI) or change detection — cannot classify type | Classifies landslide type and predicts future events |
| Use SVM/Random Forest on terrain features — need manual feature engineering | Uses deep CNN that learns features directly from images |
| Some use U-Net/ResNet for segmentation — no future prediction component | Combines image classification (Phase 1) with HMM time-series prediction (Phase 2) |
| Give only yes/no answer | Gives full risk report: type, probability, peak risk month, forecast |

**Key Difference:** Two independent phases that work together — Phase 1 handles what the image shows, Phase 2 handles what may happen next based on 28 years of real-world history.

---

## Slide 4: Dataset Collection

### Phase 1 Dataset — HR-GLDD (for Image Classification)
- **Source:** Zenodo (DOI: 7189381) — a scientific data repository by CERN
- **What it contains:** High-resolution satellite image patches with pixel-level landslide masks
- **Original format:** NumPy arrays — shape (N, 128, 128, 4) with 4 spectral bands (R, G, B, NIR)
- **Labels:** Binary masks — shape (N, 128, 128, 1) — each pixel marked 0 (no landslide) or 1 (landslide)

**How we converted it to images:**
- Wrote a script (`convert_hrgldd_to_images.py`) to convert NumPy arrays to JPEG files
- Patches with >5% landslide pixels → saved as **landslide** class
- From zero-mask regions, extracted 64x64 sub-crops, resized to 128x128 → saved as **non_landslide** class
- Maintained a 3:1 ratio (non_landslide : landslide)

| Split | Landslide | Non-Landslide | Total |
|-------|-----------|---------------|-------|
| Train | 616 | 1,574 | 2,190 |
| Val   | 158 | 392   | 550   |
| Test  | 211 | 488   | 699   |
| **Total** | **985** | **2,454** | **3,439** |

### Phase 2 Dataset — NASA Global Landslide Catalog (for HMM)
- **Source:** NASA Open Data Portal
- **Records:** 9,471 real landslide events from 1988 to 2016
- **Coverage:** 141 countries across 28 years
- **Fields used:** Event date, country, landslide type, trigger, size, fatality count
- **7 landslide types:** Landslide, Mudslide, Rockfall, Debris Flow, Complex Landslide, Earth Flow, Translational Slide
- **6 trigger types:** Heavy Rain, Earthquake, Snowmelt, Storm, Human Activity, Unknown

---

## Slide 5: System Architecture (Overview)

```
INPUT: Satellite Image + Country Name
            |
            v
   ┌─────────────────────────────────────────┐
   |          PHASE 1: Image Classification   |
   |  ┌─────────────────┐  ┌──────────────┐  |
   |  | EfficientNet-B3  |  |   AlexNet    |  |
   |  | (60% weight)     |  |  (40% weight)|  |
   |  | 300x300 input    |  |  227x227     |  |
   |  | ImageNet weights |  |  ImageNet    |  |
   |  └────────┬─────────┘  └──────┬───────┘  |
   |           |    Ensemble Avg    |          |
   |           └────────┬──────────┘          |
   |                    v                      |
   |         Temperature Scaling (T=1.15)      |
   |                    |                      |
   |         Threshold = 0.467                 |
   └────────────────────┬──────────────────────┘
                        |
           If LANDSLIDE (confidence >= 46.7%)
                        |
                        v
   ┌─────────────────────────────────────────┐
   |          PHASE 2: HMM Prediction         |
   |                                           |
   |  NASA GLC Data → Country Sequences        |
   |           |                               |
   |    Baum-Welch Training (8 hidden states)  |
   |           |                               |
   |    Viterbi Decoding → Current Type        |
   |    Transition Matrix → Future Forecast    |
   |    Risk Profile → Country Statistics      |
   └─────────────────────────────────────────┘
                        |
                        v
              FINAL OUTPUT:
        - Landslide detected (Yes/No)
        - Confidence score
        - Landslide type
        - Occurrence probability
        - Peak risk month
        - Next 3 event forecast
```

---

## Slide 6: Phase 1 — CNN Architecture (Detail)

### EfficientNet-B3 (Primary Model — 60% weight)
- Pretrained on ImageNet (1.2 million images, 1000 classes)
- Input: 300 x 300 RGB image
- Uses compound scaling — balances depth, width, and resolution together
- Feature head: 1,536 dimensions → replaced with 2-class output (landslide / non_landslide)
- ~12 million parameters (efficient but powerful)

### AlexNet (Secondary Model — 40% weight)
- Pretrained on ImageNet
- Input: 227 x 227 RGB image
- 5 convolutional layers → 3 fully connected layers
- Feature head: 4,096 dimensions → replaced with 2-class output
- ~62 million parameters

### Training Strategy
- **Epochs 1-5:** Feature layers frozen (only classifier head trains) — learns task-specific patterns
- **Epoch 6 onward:** All layers unfrozen with differential learning rates:
  - Classifier head: LR = 1e-4 (learns fast)
  - Feature layers: LR = 1e-5 (learns slowly to preserve ImageNet knowledge)
- **Optimizer:** Adam with weight decay
- **Scheduler:** CosineAnnealingLR (smooth learning rate decay)
- **Loss:** CrossEntropyLoss
- **Class imbalance:** WeightedRandomSampler — ensures model sees equal landslide and non-landslide images per epoch

### Data Augmentation (during training only)
- Horizontal flip, Vertical flip, Random 90-degree rotation
- Color jitter (brightness, contrast, saturation, hue)
- Gaussian blur, Random brightness/contrast
- Grid distortion, Coarse dropout
- Normalization with ImageNet mean and standard deviation

### Ensemble Fusion
- Final probability = 0.6 x EfficientNet-B3 (calibrated) + 0.4 x AlexNet
- Temperature scaling (T = 1.15) applied to EfficientNet-B3 to fix overconfident scores
- Optimal threshold = 0.467 (found by scanning precision-recall curve for best F1)

---

## Slide 7: Phase 2 — HMM Architecture (Detail)

### What is HMM and Why We Used It
- Hidden Markov Model is a probabilistic model for sequential data
- It assumes there are hidden "regimes" (states) that we cannot see directly, but they produce visible observations
- Perfect for landslides: the underlying geological regime is hidden, but we observe the type of landslide that happens

### Model Setup
- **Type:** CategoricalHMM (from hmmlearn library)
- **Hidden states:** 8 regimes (Shallow Rain Slide, Deep Seismic Slide, Debris Flow, Rockfall, Mixed Flow, Complex Event, Monsoon Mudslide, Human-Triggered Slide)
- **Observations:** 42 combined symbols (7 landslide types x 6 trigger types)
- **Training data:** 9,471 events → 112 country-level time sequences → 9,442 observations
- **Training algorithm:** Baum-Welch (Expectation-Maximization), 100 iterations

### What the HMM Learns
1. **Start probabilities:** Which regime is most likely at the beginning
2. **Transition matrix (8x8):** Probability of moving from one regime to another — for example, a Deep Seismic regime has 98.7% chance of staying in the same regime
3. **Emission matrix (8x42):** Which landslide type + trigger combination each regime produces most often

### How Prediction Works
1. **Current type:** Viterbi decoding finds the most likely regime → emission matrix gives the most likely type
2. **Occurrence probability:** Calculated from log-likelihood of the country's sequence using sigmoid normalization
3. **Future forecast:** Multiply current state by transition matrix N times to get next N most likely types
4. **Risk profile:** Historical statistics from NASA data — events per year, average fatalities, peak month, dominant trigger

---

## Slide 8: Step-by-Step What I Did (Timeline)

### Step 1: Data Collection
- Downloaded HR-GLDD dataset from Zenodo (NumPy arrays with satellite patches and masks)
- Downloaded NASA Global Landslide Catalog CSV from NASA Open Data Portal

### Step 2: Data Preprocessing
- Wrote `convert_hrgldd_to_images.py` to convert NumPy arrays to JPEG images
- Applied 5% threshold on pixel masks to create binary labels
- Generated non-landslide images from zero-mask sub-crops
- Split into train (2,190), val (550), test (699) images

### Step 3: Built Phase 1 — CNN Model
- First built AlexNet from scratch (baseline) → trained 30 epochs → 78.54% accuracy
- Added pretrained ImageNet weights + transfer learning → 80.11% accuracy
- Replaced with EfficientNet-B3 → better recall (78.20%) and ROC-AUC (88.93%)
- Created ensemble (EfficientNet-B3 + AlexNet) → best results across all metrics

### Step 4: Built Phase 2 — HMM Model
- Preprocessed NASA GLC data — grouped by country, sorted by date, encoded types as numbers
- Started with 4 hidden states → increased to 6 → then to 8 for finer regimes
- Added combined observations (type x trigger = 42 symbols) for richer information
- Trained with Baum-Welch algorithm

### Step 5: Built Pipeline
- Connected Phase 1 and Phase 2 into a single pipeline (`run_pipeline.py`)
- If Phase 1 detects landslide with confidence >= 46.7%, Phase 2 automatically runs
- Added temperature scaling to calibrate confidence scores
- Found optimal threshold (0.467) using precision-recall curve analysis

### Step 6: Evaluation and Results
- Evaluated on 699 unseen test images
- Generated confusion matrices, ROC curves, and comparison tables across all 5 iterations

---

## Slide 9: Results — Phase 1 (Image Classification)

### Progression Across 5 Iterations

| Metric | r0 (AlexNet Scratch) | r1 (AlexNet Pretrained) | r2 (EfficientNet-B3) | r4 (Ensemble Final) |
|--------|---------------------|------------------------|---------------------|-------------------|
| **Accuracy** | 78.54% | 80.11% | 79.97% | **82.40%** |
| **Precision** | 62.06% | 66.67% | 63.71% | **68.03%** |
| **Recall** | 74.41% | 68.25% | 78.20% | **78.67%** |
| **F1-Score** | 67.67% | 67.45% | 70.21% | **72.97%** |
| **ROC-AUC** | 85.53% | 88.33% | 88.93% | **89.83%** |

### Final Confusion Matrix (Ensemble, threshold = 0.467)

|  | Predicted: Non-Landslide | Predicted: Landslide |
|---|---|---|
| **Actual: Non-Landslide** | 410 (True Negative) | 78 (False Positive) |
| **Actual: Landslide** | 45 (False Negative) | 166 (True Positive) |

### What These Numbers Mean
- **82.40% accuracy:** Out of 699 test images, 576 were classified correctly
- **68.03% precision:** When model says "landslide", it is correct 68% of the time
- **78.67% recall:** Out of 211 actual landslide images, model correctly found 166
- **89.83% ROC-AUC:** Model has strong ability to separate landslide from non-landslide across all thresholds

---

## Slide 10: Results — Phase 2 (HMM) and Sample Prediction

### HMM Training Results
- **Log-likelihood:** -12,675.40 on 9,442 observations (42-symbol vocabulary)
- **Hidden states learned:** 8 distinct regimes with meaningful transition patterns
- **Key finding:** Deep Seismic Slide regime is 98.7% self-persistent, Rockfall is 97.1% self-persistent — showing that geological regimes tend to repeat in the same region

### Sample End-to-End Prediction (Test Image: Nepal)

```
INPUT: ls_test_0008.jpg | Country: Nepal

Phase 1 — Ensemble Classification
  EfficientNet-B3 probability : 93.8%
  AlexNet probability         : 92.6%
  Ensemble probability        : 93.3%
  Result: LANDSLIDE DETECTED

Phase 2 — HMM Prediction
  Landslide type              : Landslide
  Occurrence probability      : 30.0%
  Peak risk month             : July
  Future forecast:
    Step 1 → Landslide (26.4%)
    Step 2 → Landslide (21.4%)
    Step 3 → Landslide (18.1%)

FINAL: LANDSLIDE DETECTED — Type: Landslide (occurrence probability: 29%)
```

---

## Slide 11: Improvements Done and What is Coming Next

### Improvements Already Completed (r0 to r4)

| What Was Done | Impact |
|---|---|
| Added pretrained ImageNet weights (transfer learning) | Accuracy: 78.54% → 80.11% (+1.57%) |
| Replaced AlexNet with EfficientNet-B3 | Recall: 68.25% → 78.20% (+9.95%), F1: 67.45% → 70.21% |
| Created ensemble (EfficientNet-B3 + AlexNet) | All 5 metrics improved — best overall: 82.40% accuracy |
| Added temperature scaling for confidence calibration | Confidence scores now reflect true accuracy |
| Found optimal threshold (0.467) via PR curve | F1 improved from 71.89% to 72.97% |
| Increased HMM from 4 → 8 hidden states | Captures finer landslide regimes |
| Added type x trigger combined observations (42 symbols) | HMM now considers environmental triggers, not just type |
| Stronger augmentation (GridDistortion, CoarseDropout) | Better generalization on unseen images |

### Planned Improvements (Next Steps)

1. **Vision Transformer (ViT):** Replace/add ViT-B/16 pretrained on ImageNet-21k — expected +2-4% F1 improvement
2. **U-Net Segmentation:** Add pixel-level mask output to show exactly where the landslide is in the image — more useful for field teams
3. **Use all 4 spectral bands (RGB + NIR):** Near-infrared band shows vegetation damage patterns invisible in RGB — re-convert HR-GLDD with 4 channels
4. **Larger dataset:** Add more landslide images from Landslide4Sense or Google Earth Engine to improve training data from 2,190 to 10,000+
5. **Siamese change-detection network:** Compare before-and-after satellite image pairs — most reliable method for finding new landslides
6. **Real-time monitoring:** Connect to Sentinel-2 or Copernicus satellite data APIs for automated alerts
7. **Web/mobile interface:** Build a simple web app where user can upload image, select country, and get instant prediction report

---

## Slide 12: Conclusion

### What This Project Achieves
- A complete two-phase machine learning pipeline that detects landslides from satellite images and predicts future landslide events
- Phase 1 (Ensemble CNN) gives **82.40% accuracy** and **89.83% ROC-AUC** on image classification
- Phase 2 (HMM) gives type classification, occurrence probability, future forecast, and country-level risk statistics based on 28 years of NASA data
- The system processes one image in under 1 second — fast enough for real-time use

### Why This Approach Works
- Ensemble of two different architectures (EfficientNet-B3 + AlexNet) reduces individual model errors
- Transfer learning from ImageNet compensates for the small dataset (3,439 images)
- HMM captures real sequential patterns in landslide history that simple classifiers cannot model
- Temperature scaling and optimal thresholding make the confidence scores reliable for decision-making

### Limitations
- Image dataset is relatively small (2,190 training images) — larger data would improve accuracy further
- HMM forecasts converge quickly to steady-state because some regimes are highly self-persistent
- System has been tested only on HR-GLDD benchmark data, not on live satellite imagery yet

### Final Statement
This project shows that combining deep learning for visual detection with probabilistic modeling for temporal prediction creates a more complete and useful landslide identification system than either approach alone. With the planned improvements, this system has the potential to become a practical tool for disaster risk management.

---
