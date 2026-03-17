"""
Single-image inference for the trained landslide classifier (multi-class).
Predicts one of 6 classes: non_landslide, rockfall, mudflow, debris_flow,
                           rotational_slide, translational_slide.
"""

import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from phase1_alexnet.dataset import get_test_transforms
from phase1_alexnet.model import get_model, get_efficientnet_b3, get_vit_b_16
from phase1_alexnet.train import load_checkpoint
from utils.temperature_scaling import TemperatureScaler, load_temperature


def load_alexnet_model(checkpoint_path: str = None, device: torch.device = None):
    """Load model from checkpoint, return (model, device)."""
    if checkpoint_path is None:
        if config.ACTIVE_MODEL == "efficientnet_b3":
            checkpoint_path = config.EFFICIENTNET_CHECKPOINT
        elif config.ACTIVE_MODEL == "vit_b_16":
            checkpoint_path = config.VIT_CHECKPOINT
        else:
            checkpoint_path = config.ALEXNET_CHECKPOINT
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try each architecture until one matches the checkpoint
    for model_fn in [
        lambda: get_vit_b_16(num_classes=config.NUM_CLASSES),
        lambda: get_efficientnet_b3(num_classes=config.NUM_CLASSES),
        lambda: get_model(pretrained=True, num_classes=config.NUM_CLASSES),
        lambda: get_model(pretrained=False, num_classes=config.NUM_CLASSES),
    ]:
        try:
            model = model_fn()
            model = model.to(device)
            load_checkpoint(checkpoint_path, model)
            break
        except RuntimeError:
            continue
    model.eval()
    # Apply temperature scaling if a calibration file exists
    T = load_temperature()
    if T != 1.0:
        scaler = TemperatureScaler(model)
        scaler.temperature = torch.nn.Parameter(torch.tensor([T]))
        scaler = scaler.to(device)
        scaler.eval()
        return scaler, device
    return model, device


def predict_single_image(image_path: str, model, device: torch.device) -> dict:
    """
    Predict the class of a single image (multi-class).

    Args:
        image_path: Path to the image file.
        model:      Loaded model in eval mode.
        device:     torch.device.

    Returns:
        dict with keys:
          - label: one of CLASS_NAMES (e.g. 'rockfall', 'mudflow', ...)
          - confidence: float (probability of the predicted class)
          - probabilities: dict mapping each class name to its probability
          - is_landslide: bool (True if predicted class is not 'non_landslide')
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = get_test_transforms()
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

    # Aggregate landslide probability (sum of all landslide sub-types)
    landslide_prob = float(sum(probs[i] for i in range(1, config.NUM_CLASSES)))

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "is_landslide": label != "non_landslide",
        "landslide_prob": landslide_prob,
    }


def predict_batch(image_paths: list, model, device: torch.device) -> list:
    """
    Run prediction on a list of image paths.

    Returns:
        List of result dicts (same format as predict_single_image).
    """
    results = []
    for path in image_paths:
        try:
            result = predict_single_image(path, model, device)
            result["image_path"] = path
        except Exception as e:
            result = {"image_path": path, "error": str(e)}
        results.append(result)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--checkpoint", default=None, help="Path to .pth checkpoint")
    args = parser.parse_args()

    mdl, dev = load_alexnet_model(args.checkpoint)
    result = predict_single_image(args.image, mdl, dev)
    print(f"\nImage      : {args.image}")
    print(f"Label      : {result['label']}")
    print(f"Confidence : {result['confidence']:.4f}")
    print(f"Is landslide: {result['is_landslide']}")
    print(f"Landslide probability (aggregate): {result['landslide_prob']:.4f}")
    print("Per-class probabilities:")
    for cls, prob in result["probabilities"].items():
        print(f"  {cls:>20s}: {prob:.4f}")
