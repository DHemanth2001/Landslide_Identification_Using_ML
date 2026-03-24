"""
Ensemble predictor: weighted average of ConvNeXt-CBAM-FPN + SwinV2-Small (6-class).
Weights: 55% ConvNeXt-CBAM-FPN (calibrated) + 45% SwinV2-Small.

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
from phase1_alexnet.model import get_convnext_cbam_fpn, get_swinv2_s
from phase1_alexnet.train import load_checkpoint
from utils.temperature_scaling import TemperatureScaler, load_temperature
from utils.tta import predict_ensemble_with_tta


def load_ensemble(device: torch.device = None):
    """
    Load both models and return them as a tuple.
    ConvNeXt-CBAM-FPN is wrapped in TemperatureScaler.

    Returns:
        (convnext_scaler, swinv2_model, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ConvNeXt-CBAM-FPN + temperature calibration (prefer EMA checkpoint)
    convnext = get_convnext_cbam_fpn(num_classes=config.NUM_CLASSES, freeze=False).to(device)
    ckpt_path = config.EMA_CONVNEXT_CHECKPOINT
    if not os.path.exists(ckpt_path):
        ckpt_path = config.CONVNEXT_CHECKPOINT
    try:
        load_checkpoint(ckpt_path, convnext)
    except FileNotFoundError:
        print("Warning: ConvNeXt-CBAM-FPN checkpoint not found.")

    T = load_temperature()
    scaler = TemperatureScaler(convnext)
    scaler.temperature = torch.nn.Parameter(torch.tensor([T]))
    scaler = scaler.to(device)
    scaler.eval()

    # SwinV2-Small (prefer EMA checkpoint)
    swinv2 = get_swinv2_s(num_classes=config.NUM_CLASSES, freeze=False).to(device)
    ckpt_path = config.EMA_SWINV2_CHECKPOINT
    if not os.path.exists(ckpt_path):
        ckpt_path = config.SWINV2_CHECKPOINT
    try:
        load_checkpoint(ckpt_path, swinv2)
    except FileNotFoundError:
        print("Warning: SwinV2-Small checkpoint not found.")
    swinv2.eval()

    print(f"Ensemble loaded: ConvNeXt-CBAM-FPN (T={T:.4f}, w={config.ENSEMBLE_WEIGHT_CONVNEXT}) "
          f"+ SwinV2-Small (w={config.ENSEMBLE_WEIGHT_SWINV2})")
    return scaler, swinv2, device


def predict_ensemble(image_path: str, convnext_scaler, swinv2_model,
                     device: torch.device,
                     use_tta: bool = True, n_augments: int = 5) -> dict:
    """
    Run 6-class ensemble prediction on a single image.
    Optionally uses Test-Time Augmentation (TTA).

    Returns dict with:
      label (specific type), confidence, probabilities (all 6 classes),
      is_landslide, landslide_prob, per-model probabilities.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if use_tta:
        ensemble_probs, probs_convnext, probs_swinv2 = predict_ensemble_with_tta(
            convnext_scaler, swinv2_model, img, device,
            effnet_img_size=config.CONVNEXT_IMG_SIZE,
            vit_img_size=config.SWINV2_IMG_SIZE,
            effnet_weight=config.ENSEMBLE_WEIGHT_CONVNEXT,
            vit_weight=config.ENSEMBLE_WEIGHT_SWINV2,
            n_augments=n_augments,
        )
    else:
        transform_convnext = get_test_transforms(config.CONVNEXT_IMG_SIZE)
        tensor_convnext = transform_convnext(img).unsqueeze(0).to(device)

        transform_swinv2 = get_test_transforms(config.SWINV2_IMG_SIZE)
        tensor_swinv2 = transform_swinv2(img).unsqueeze(0).to(device)

        with torch.no_grad():
            probs_convnext = torch.softmax(convnext_scaler(tensor_convnext), dim=1)[0].cpu().numpy()
            probs_swinv2 = torch.softmax(swinv2_model(tensor_swinv2), dim=1)[0].cpu().numpy()

        ensemble_probs = (config.ENSEMBLE_WEIGHT_CONVNEXT * probs_convnext +
                          config.ENSEMBLE_WEIGHT_SWINV2 * probs_swinv2)

    label_idx = int(np.argmax(ensemble_probs))
    label = config.CLASS_NAMES[label_idx]
    confidence = float(ensemble_probs[label_idx])
    is_landslide = label != "non_landslide"

    landslide_prob = float(sum(ensemble_probs[i] for i in range(1, config.NUM_CLASSES)))

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
        "convnext_probs": {config.CLASS_NAMES[i]: float(probs_convnext[i])
                           for i in range(config.NUM_CLASSES)},
        "swinv2_probs": {config.CLASS_NAMES[i]: float(probs_swinv2[i])
                         for i in range(config.NUM_CLASSES)},
        "threshold_used": config.PHASE1_THRESHOLD,
        "tta_enabled": use_tta,
        "tta_augments": n_augments if use_tta else 0,
    }
