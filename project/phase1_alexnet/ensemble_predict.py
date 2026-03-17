"""
Ensemble predictor: weighted average of EfficientNet-B3 + ViT-B/16 (multi-class).
Weights: 60% EfficientNet-B3 (calibrated) + 40% ViT-B/16.
6-class output: non_landslide, rockfall, mudflow, debris_flow,
                rotational_slide, translational_slide.
Supports Test-Time Augmentation (TTA) for more robust predictions.
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
from utils.tta import predict_ensemble_with_tta


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


def predict_ensemble(image_path: str, effnet_scaler, vit_model, device: torch.device,
                     use_tta: bool = True, n_augments: int = 5) -> dict:
    """
    Run multi-class ensemble prediction on a single image.
    Optionally uses Test-Time Augmentation (TTA) for more robust predictions.

    Returns dict with:
      label (specific type), confidence, probabilities (all 6 classes),
      is_landslide, landslide_prob, per-model probabilities.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if use_tta:
        # TTA: average predictions over multiple augmented views
        ensemble_probs, probs_effnet, probs_vit = predict_ensemble_with_tta(
            effnet_scaler, vit_model, img, device,
            effnet_img_size=config.IMG_SIZE, vit_img_size=config.VIT_IMG_SIZE,
            effnet_weight=config.ENSEMBLE_WEIGHT_EFFNET,
            vit_weight=config.ENSEMBLE_WEIGHT_ALEXNET,
            n_augments=n_augments,
        )
    else:
        # Standard single-pass prediction
        transform_effnet = get_test_transforms(config.IMG_SIZE)
        tensor_effnet = transform_effnet(img).unsqueeze(0).to(device)

        transform_vit = get_test_transforms(config.VIT_IMG_SIZE)
        tensor_vit = transform_vit(img).unsqueeze(0).to(device)

        with torch.no_grad():
            probs_effnet = torch.softmax(effnet_scaler(tensor_effnet), dim=1)[0].cpu().numpy()
            probs_vit    = torch.softmax(vit_model(tensor_vit),        dim=1)[0].cpu().numpy()

        # Weighted ensemble over all classes
        ensemble_probs = (config.ENSEMBLE_WEIGHT_EFFNET * probs_effnet +
                          config.ENSEMBLE_WEIGHT_ALEXNET * probs_vit)

    # Multi-class: pick the highest-probability class
    label_idx = int(np.argmax(ensemble_probs))
    label = config.CLASS_NAMES[label_idx]
    confidence = float(ensemble_probs[label_idx])
    is_landslide = label != "non_landslide"

    # Aggregate landslide probability (sum of all landslide sub-types)
    landslide_prob = float(sum(ensemble_probs[i] for i in range(1, config.NUM_CLASSES)))

    # Per-class probabilities
    probabilities = {
        config.CLASS_NAMES[i]: float(ensemble_probs[i])
        for i in range(config.NUM_CLASSES)
    }

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "is_landslide": is_landslide,
        "landslide_prob": landslide_prob,
        "landslide_type": label if is_landslide else None,
        "effnet_probs": {config.CLASS_NAMES[i]: float(probs_effnet[i]) for i in range(config.NUM_CLASSES)},
        "vit_probs": {config.CLASS_NAMES[i]: float(probs_vit[i]) for i in range(config.NUM_CLASSES)},
        "threshold_used": config.PHASE1_THRESHOLD,
        "tta_enabled": use_tta,
        "tta_augments": n_augments if use_tta else 0,
    }
