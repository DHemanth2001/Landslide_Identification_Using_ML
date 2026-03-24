"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for landslide models.

Generates heatmaps showing which regions of the input image the model
focuses on when making its classification decision.

Supports ConvNeXt-CBAM-FPN, SwinV2-Small, and legacy models.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
           via Gradient-based Localization", ICCV 2017.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM: extracts gradients from a target convolutional layer to produce
    a class-discriminative heatmap.

    Usage:
        cam = GradCAM(model, target_layer)
        heatmap = cam.generate(input_tensor, class_idx=None)
        overlay = cam.overlay_on_image(heatmap, original_image)
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture activations/gradients.")

        grads = self.gradients
        acts = self.activations

        if acts.dim() == 3:
            # Transformer: (1, num_patches+1, hidden_dim)
            num_tokens = acts.shape[1] - 1
            grid_size = int(num_tokens ** 0.5)
            acts = acts[:, 1:, :]
            grads = grads[:, 1:, :]
            weights = grads.mean(dim=1, keepdim=True)
            cam = (weights * acts).sum(dim=-1)
            cam = cam.reshape(1, grid_size, grid_size)
        else:
            # CNN: standard spatial Grad-CAM
            weights = grads.mean(dim=[2, 3], keepdim=True)
            cam = (weights * acts).sum(dim=1, keepdim=True)
            cam = cam.squeeze(1)

        cam = F.relu(cam)
        cam = cam.squeeze(0).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def overlay_on_image(self, heatmap, original_image, alpha=0.5):
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(alpha * heatmap_colored + (1 - alpha) * original_image)
        return overlay

    def remove_hooks(self):
        self._forward_hook.remove()
        self._backward_hook.remove()


def get_target_layer(model, model_name: str):
    """
    Return the appropriate target layer for Grad-CAM.

    Args:
        model:      PyTorch model.
        model_name: Architecture name.

    Returns:
        The target nn.Module layer.
    """
    if model_name == "convnext_cbam_fpn":
        # Last CBAM module (applied to stage4 features)
        return model.cbam4
    elif model_name == "swinv2_s":
        # Last layer norm in SwinV2
        return model.model.norm
    elif model_name == "efficientnet_b3":
        return model.features[-1]
    elif model_name == "vit_b_16":
        return model.encoder.layers[-1].ln_1
    elif model_name == "alexnet":
        return model.features[-2]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def generate_gradcam_for_image(
    model, image_path, model_name, img_size, device,
    class_idx=None, save_path=None,
):
    """
    Generate and optionally save a Grad-CAM visualization for a single image.

    Returns:
        Tuple of (heatmap, overlay, predicted_class_idx).
    """
    from phase1_alexnet.dataset import get_test_transforms

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = get_test_transforms(img_size)
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    actual_model = model.model if hasattr(model, 'model') else model
    target_layer = get_target_layer(actual_model, model_name)
    cam = GradCAM(actual_model, target_layer)

    heatmap = cam.generate(input_tensor, class_idx=class_idx)
    predicted_idx = actual_model(input_tensor).argmax(dim=1).item() if class_idx is None else class_idx

    img_resized = cv2.resize(img_rgb, (img_size[1], img_size[0]))
    overlay = cam.overlay_on_image(heatmap, img_resized)

    cam.remove_hooks()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, overlay_bgr)
        print(f"Grad-CAM saved to {save_path}")

    return heatmap, overlay, predicted_idx
