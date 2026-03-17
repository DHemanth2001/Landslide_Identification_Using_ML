"""
Test-Time Augmentation (TTA) for landslide classification.

At inference time, the same image is transformed multiple ways (flips, crops),
each variant is passed through the model, and probabilities are averaged.
This produces more robust predictions at the cost of N× inference time.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms


def get_tta_transforms(img_size):
    """
    Return a list of deterministic augmentation pipelines for TTA.

    Each transform produces a different view of the same image:
      0. Original (no augmentation)
      1. Horizontal flip
      2. Vertical flip
      3. Horizontal + Vertical flip
      4. 90-degree rotation

    Args:
        img_size: Tuple (H, W) for resizing.

    Returns:
        List of torchvision.transforms.Compose objects.
    """
    from torchvision.transforms.functional import hflip, vflip, rotate
    import numpy as np

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    base = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    hflip_t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        normalize,
    ])

    vflip_t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        normalize,
    ])

    hvflip_t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        normalize,
    ])

    rot90_t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.RandomRotation(degrees=(90, 90)),
        transforms.ToTensor(),
        normalize,
    ])

    return [base, hflip_t, vflip_t, hvflip_t, rot90_t]


def predict_with_tta(model, image_rgb, img_size, device, n_augments=5):
    """
    Run TTA on a single RGB image (numpy HWC uint8) and return averaged probabilities.

    Args:
        model:      PyTorch model in eval mode.
        image_rgb:  Numpy array (H, W, 3) in RGB, uint8.
        img_size:   Tuple (H, W) for the model's expected input size.
        device:     torch.device.
        n_augments: Number of TTA views to use (1-5). Default 5.

    Returns:
        Averaged probability array of shape (num_classes,).
    """
    tta_transforms = get_tta_transforms(img_size)[:n_augments]

    all_probs = []
    with torch.no_grad():
        for t in tta_transforms:
            tensor = t(image_rgb).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            all_probs.append(probs)

    import numpy as np
    return np.mean(all_probs, axis=0)


def predict_ensemble_with_tta(
    effnet_scaler, vit_model, image_rgb, device,
    effnet_img_size=(300, 300), vit_img_size=(224, 224),
    effnet_weight=0.6, vit_weight=0.4, n_augments=5,
):
    """
    Run TTA on both ensemble models and return weighted-averaged probabilities.

    Args:
        effnet_scaler: Temperature-calibrated EfficientNet model.
        vit_model:     ViT-B/16 model.
        image_rgb:     Numpy array (H, W, 3) in RGB, uint8.
        device:        torch.device.
        effnet_img_size: Input size for EfficientNet.
        vit_img_size:  Input size for ViT.
        effnet_weight: Ensemble weight for EfficientNet.
        vit_weight:    Ensemble weight for ViT.
        n_augments:    Number of TTA views (1-5).

    Returns:
        Tuple of (ensemble_probs, effnet_probs, vit_probs) — each numpy (C,).
    """
    effnet_probs = predict_with_tta(effnet_scaler, image_rgb, effnet_img_size, device, n_augments)
    vit_probs = predict_with_tta(vit_model, image_rgb, vit_img_size, device, n_augments)

    ensemble_probs = effnet_weight * effnet_probs + vit_weight * vit_probs
    return ensemble_probs, effnet_probs, vit_probs
