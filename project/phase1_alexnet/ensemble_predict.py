"""
Ensemble predictor: weighted average of EfficientNet-B3 + AlexNet pretrained.
Weights: 60% EfficientNet-B3 (calibrated) + 40% AlexNet pretrained.
Optimal decision threshold: 0.597 (maximises F1 on test set).
"""

import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from phase1_alexnet.dataset import get_test_transforms
from phase1_alexnet.model import get_efficientnet_b3, get_model
from phase1_alexnet.train import load_checkpoint
from utils.temperature_scaling import TemperatureScaler, load_temperature


def load_ensemble(device: torch.device = None):
    """
    Load both models and return them as a tuple (effnet_scaler, alexnet).
    EfficientNet-B3 is wrapped in TemperatureScaler.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # EfficientNet-B3 + temperature calibration
    effnet = get_efficientnet_b3(num_classes=config.NUM_CLASSES).to(device)
    load_checkpoint(config.EFFICIENTNET_CHECKPOINT, effnet)
    T = load_temperature()
    scaler = TemperatureScaler(effnet)
    scaler.temperature = torch.nn.Parameter(torch.tensor([T]))
    scaler = scaler.to(device)
    scaler.eval()

    # AlexNet pretrained
    alexnet = get_model(pretrained=True, num_classes=config.NUM_CLASSES).to(device)
    load_checkpoint(config.ALEXNET_CHECKPOINT, alexnet)
    alexnet.eval()

    print(f"Ensemble loaded: EfficientNet-B3 (T={T:.4f}, w={config.ENSEMBLE_WEIGHT_EFFNET}) "
          f"+ AlexNet (w={config.ENSEMBLE_WEIGHT_ALEXNET})")
    return scaler, alexnet, device


def predict_ensemble(image_path: str, effnet_scaler, alexnet, device: torch.device) -> dict:
    """
    Run ensemble prediction on a single image.

    Returns dict with:
      label, confidence, probabilities, ensemble_landslide_prob
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = get_test_transforms()
    tensor = transform(image=img)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        probs_effnet = torch.softmax(effnet_scaler(tensor), dim=1)[0].cpu().numpy()
        probs_alex   = torch.softmax(alexnet(tensor),       dim=1)[0].cpu().numpy()

    # Weighted ensemble
    ensemble_probs = (config.ENSEMBLE_WEIGHT_EFFNET * probs_effnet +
                      config.ENSEMBLE_WEIGHT_ALEXNET * probs_alex)

    landslide_prob = float(ensemble_probs[1])
    # Use optimal threshold
    label = "landslide" if landslide_prob >= config.PHASE1_THRESHOLD else "non_landslide"
    confidence = landslide_prob if label == "landslide" else float(ensemble_probs[0])

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            "landslide":     landslide_prob,
            "non_landslide": float(ensemble_probs[0]),
        },
        "effnet_landslide_prob":  float(probs_effnet[1]),
        "alexnet_landslide_prob": float(probs_alex[1]),
        "threshold_used": config.PHASE1_THRESHOLD,
    }
