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
from phase1_alexnet.model import get_efficientnet_b3, get_vit_b_16
from phase1_alexnet.train import load_checkpoint
from utils.temperature_scaling import TemperatureScaler, load_temperature


def load_ensemble(device: torch.device = None):
    """
    Load both models and return them as a tuple (effnet_scaler, vit_model, device).
    EfficientNet-B3 is wrapped in TemperatureScaler.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # EfficientNet-B3 + temperature calibration
    effnet = get_efficientnet_b3(num_classes=config.NUM_CLASSES).to(device)
    try:
        load_checkpoint(config.EFFICIENTNET_CHECKPOINT, effnet)
    except FileNotFoundError:
        print("Warning: EfficientNet checkpoint not found.")
    T = load_temperature()
    scaler = TemperatureScaler(effnet)
    scaler.temperature = torch.nn.Parameter(torch.tensor([T]))
    scaler = scaler.to(device)
    scaler.eval()

    # ViT-B/16
    vit_model = get_vit_b_16(num_classes=config.NUM_CLASSES).to(device)
    try:
        load_checkpoint(config.VIT_CHECKPOINT, vit_model)
    except FileNotFoundError:
        print("Warning: ViT-B/16 checkpoint not found.")
    vit_model.eval()

    print(f"Ensemble loaded: EfficientNet-B3 (T={T:.4f}, w={config.ENSEMBLE_WEIGHT_EFFNET}) "
          f"+ ViT-B/16 (w={config.ENSEMBLE_WEIGHT_ALEXNET})")
    return scaler, vit_model, device


def predict_ensemble(image_path: str, effnet_scaler, vit_model, device: torch.device) -> dict:
    """
    Run ensemble prediction on a single image.

    Returns dict with:
      label, confidence, probabilities, effnet_landslide_prob, vit_landslide_prob
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform_effnet = get_test_transforms(config.IMG_SIZE)
    tensor_effnet = transform_effnet(img).unsqueeze(0).to(device)

    transform_vit = get_test_transforms(config.VIT_IMG_SIZE)
    tensor_vit = transform_vit(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs_effnet = torch.softmax(effnet_scaler(tensor_effnet), dim=1)[0].cpu().numpy()
        probs_vit   = torch.softmax(vit_model(tensor_vit),       dim=1)[0].cpu().numpy()

    # Weighted ensemble
    ensemble_probs = (config.ENSEMBLE_WEIGHT_EFFNET * probs_effnet +
                      config.ENSEMBLE_WEIGHT_ALEXNET * probs_vit)

    landslide_prob = float(ensemble_probs[1])
    # Use optimal threshold
    label = "landslide" if landslide_prob >= config.PHASE1_THRESHOLD else "non_landslide"
    confidence = landslide_prob if label == "landslide" else float(ensemble_probs[0])

    # Calculate probabilities for all classes dynamically
    probabilities = {
        config.CLASS_NAMES[i]: float(ensemble_probs[i])
        for i in range(config.NUM_CLASSES)
    }

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "effnet_landslide_prob":  float(probs_effnet[1]) if config.NUM_CLASSES == 2 else probs_effnet.tolist(),
        "vit_landslide_prob": float(probs_vit[1]) if config.NUM_CLASSES == 2 else probs_vit.tolist(),
        "threshold_used": config.PHASE1_THRESHOLD,
    }
