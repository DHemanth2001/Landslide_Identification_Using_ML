"""
Single-image inference for the trained landslide classifier (6-class).
Predicts one of 6 classes: non_landslide, rockfall, mudflow, debris_flow,
                           rotational_slide, translational_slide.

Supports MSAFusionNet (ConvNeXt-CBAM-FPN) and SwinV2-Small models,
with EMA checkpoints and temperature calibration.
"""

import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from phase1_alexnet.dataset import get_test_transforms
from phase1_alexnet.model import (
    get_convnext_cbam_fpn,
    get_swinv2_s,
    get_model,
    get_efficientnet_b3,
    get_vit_b_16,
)
from phase1_alexnet.train import load_checkpoint
from utils.temperature_scaling import TemperatureScaler, load_temperature


def _create_model(model_name: str, num_classes: int):
    """Create model by name without loading checkpoint."""
    if model_name == "convnext_cbam_fpn":
        return get_convnext_cbam_fpn(num_classes=num_classes, freeze=False)
    elif model_name == "swinv2_s":
        return get_swinv2_s(num_classes=num_classes, freeze=False)
    elif model_name == "efficientnet_b3":
        return get_efficientnet_b3(num_classes=num_classes)
    elif model_name == "vit_b_16":
        return get_vit_b_16(num_classes=num_classes)
    else:
        return get_model(pretrained=False, num_classes=num_classes)


def _get_checkpoint_path(model_name: str):
    """Get best checkpoint path, preferring EMA."""
    if model_name == "convnext_cbam_fpn":
        ema = config.EMA_CONVNEXT_CHECKPOINT
        raw = config.CONVNEXT_CHECKPOINT
    elif model_name == "swinv2_s":
        ema = config.EMA_SWINV2_CHECKPOINT
        raw = config.SWINV2_CHECKPOINT
    elif model_name == "efficientnet_b3":
        return config.EFFICIENTNET_CHECKPOINT
    elif model_name == "vit_b_16":
        return config.VIT_CHECKPOINT
    else:
        return config.ALEXNET_CHECKPOINT
    return ema if os.path.exists(ema) else raw


def _get_img_size(model_name: str):
    if model_name == "swinv2_s":
        return config.SWINV2_IMG_SIZE
    return config.CONVNEXT_IMG_SIZE


def load_model(model_name: str = None, checkpoint_path: str = None,
               device: torch.device = None):
    """
    Load a trained model from checkpoint with temperature calibration.

    Returns:
        (model_or_scaler, device)
    """
    if model_name is None:
        model_name = config.ACTIVE_MODEL
    if checkpoint_path is None:
        checkpoint_path = _get_checkpoint_path(model_name)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _create_model(model_name, num_classes=config.NUM_CLASSES)
    model = model.to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    # Apply temperature scaling if available
    T = load_temperature()
    if T != 1.0:
        scaler = TemperatureScaler(model)
        scaler.temperature = torch.nn.Parameter(torch.tensor([T]))
        scaler = scaler.to(device)
        scaler.eval()
        return scaler, device

    return model, device


# Legacy alias
load_alexnet_model = load_model


def predict_single_image(image_path: str, model, device: torch.device,
                         model_name: str = None) -> dict:
    """
    Predict the class of a single image (6-class).

    Args:
        image_path: Path to the image file.
        model:      Loaded model in eval mode.
        device:     torch.device.
        model_name: Model name for selecting image size.

    Returns:
        dict with label, confidence, probabilities, is_landslide, landslide_prob.
    """
    if model_name is None:
        model_name = config.ACTIVE_MODEL

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_size = _get_img_size(model_name)
    transform = get_test_transforms(img_size)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    label_idx = int(np.argmax(probs))
    label = config.CLASS_NAMES[label_idx]
    confidence = float(probs[label_idx])

    probabilities = {
        config.CLASS_NAMES[i]: float(probs[i])
        for i in range(config.NUM_CLASSES)
    }

    landslide_prob = float(sum(probs[i] for i in range(1, config.NUM_CLASSES)))

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "is_landslide": label != "non_landslide",
        "landslide_prob": landslide_prob,
    }


def predict_batch(image_paths: list, model, device: torch.device,
                  model_name: str = None) -> list:
    """Run prediction on a list of image paths."""
    results = []
    for path in image_paths:
        try:
            result = predict_single_image(path, model, device, model_name)
            result["image_path"] = path
        except Exception as e:
            result = {"image_path": path, "error": str(e)}
        results.append(result)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--model", default=None,
                       help="Model name: convnext_cbam_fpn, swinv2_s")
    parser.add_argument("--checkpoint", default=None, help="Path to .pth checkpoint")
    args = parser.parse_args()

    model_name = args.model or config.ACTIVE_MODEL
    mdl, dev = load_model(model_name=model_name, checkpoint_path=args.checkpoint)
    result = predict_single_image(args.image, mdl, dev, model_name)

    print(f"\nImage      : {args.image}")
    print(f"Model      : {model_name}")
    print(f"Label      : {result['label']}")
    print(f"Confidence : {result['confidence']:.4f}")
    print(f"Is landslide: {result['is_landslide']}")
    print(f"Landslide probability (aggregate): {result['landslide_prob']:.4f}")
    print("Per-class probabilities:")
    for cls, prob in result["probabilities"].items():
        print(f"  {cls:>20s}: {prob:.4f}")
