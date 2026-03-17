"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for landslide models.

Generates heatmaps showing which regions of the input image the model
focuses on when making its classification decision. This provides
interpretability for disaster monitoring systems.

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
        """
        Args:
            model:        PyTorch model (must be in eval mode).
            target_layer: The nn.Module layer to extract activations from.
                          For EfficientNet-B3: model.features[-1]
                          For ViT-B/16: model.encoder.layers[-1].ln_1
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed image tensor, shape (1, C, H, W).
            class_idx:    Target class index. If None, uses the predicted class.

        Returns:
            Numpy heatmap of shape (H_activation, W_activation), values in [0, 1].
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for the target class
        target = output[0, class_idx]
        target.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture activations/gradients. "
                               "Check that target_layer is correct.")

        # For CNN layers: activations shape (1, C, H, W)
        # For ViT layers: activations shape (1, tokens, dim) — needs reshaping
        grads = self.gradients
        acts = self.activations

        if acts.dim() == 3:
            # ViT: (1, num_patches+1, hidden_dim) → treat as spatial
            # Remove CLS token, reshape to 2D grid
            num_tokens = acts.shape[1] - 1  # exclude CLS
            grid_size = int(num_tokens ** 0.5)
            acts = acts[:, 1:, :]  # remove CLS token
            grads = grads[:, 1:, :]

            # Global average pooling over spatial dimension
            weights = grads.mean(dim=1, keepdim=True)  # (1, 1, dim)
            cam = (weights * acts).sum(dim=-1)  # (1, num_patches)
            cam = cam.reshape(1, grid_size, grid_size)
        else:
            # CNN: standard spatial Grad-CAM
            weights = grads.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
            cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
            cam = cam.squeeze(1)  # (1, H, W)

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze(0).cpu().numpy()  # (H, W)

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def overlay_on_image(self, heatmap, original_image, alpha=0.5):
        """
        Overlay heatmap on original image.

        Args:
            heatmap:        Numpy array (H_cam, W_cam) in [0, 1].
            original_image: Numpy array (H, W, 3) in RGB, uint8.
            alpha:          Blending factor.

        Returns:
            Numpy array (H, W, 3) in RGB, uint8 — the overlaid image.
        """
        h, w = original_image.shape[:2]

        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Convert to colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = np.uint8(alpha * heatmap_colored + (1 - alpha) * original_image)
        return overlay

    def remove_hooks(self):
        """Remove registered hooks to free memory."""
        self._forward_hook.remove()
        self._backward_hook.remove()


def get_target_layer(model, model_name: str):
    """
    Return the appropriate target layer for Grad-CAM based on model architecture.

    Args:
        model:      PyTorch model.
        model_name: 'efficientnet_b3', 'vit_b_16', or 'alexnet'.

    Returns:
        The target nn.Module layer.
    """
    if model_name == "efficientnet_b3":
        return model.features[-1]
    elif model_name == "vit_b_16":
        return model.encoder.layers[-1].ln_1
    elif model_name == "alexnet":
        # Last conv layer in AlexNet features
        return model.features[-2]  # Conv5 before final MaxPool
    else:
        raise ValueError(f"Unknown model: {model_name}")


def generate_gradcam_for_image(
    model, image_path, model_name, img_size, device,
    class_idx=None, save_path=None,
):
    """
    Generate and optionally save a Grad-CAM visualization for a single image.

    Args:
        model:      PyTorch model (eval mode).
        image_path: Path to input image.
        model_name: Architecture name for selecting target layer.
        img_size:   (H, W) tuple for input transforms.
        device:     torch.device.
        class_idx:  Target class. None = use predicted class.
        save_path:  If provided, save the overlay image.

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

    # Handle TemperatureScaler wrapper
    actual_model = model.model if hasattr(model, 'model') else model
    target_layer = get_target_layer(actual_model, model_name)
    cam = GradCAM(actual_model, target_layer)

    heatmap = cam.generate(input_tensor, class_idx=class_idx)
    predicted_idx = actual_model(input_tensor).argmax(dim=1).item() if class_idx is None else class_idx

    # Resize original image for overlay
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
