"""
Multi-Scale Attention Fusion Network for landslide 6-class classification.

Architecture based on ResM-FusionNet (2025), improved with:
  - ConvNeXt-Base backbone (Liu et al., CVPR 2022) replacing ResNet-50
  - CBAM attention modules (Woo et al., ECCV 2018) at each fusion scale
  - Feature Pyramid Network (FPN) for multi-scale feature fusion
  - SwinV2-Small (Liu et al., CVPR 2022) as ensemble partner

6-class output: non_landslide, rockfall, mudflow, debris_flow,
                rotational_slide, translational_slide
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── CBAM: Convolutional Block Attention Module ──────────────────────────────
# Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018

class ChannelAttention(nn.Module):
    """Channel attention: learns which feature channels are important."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Average pool + max pool → shared MLP → sigmoid
        avg_out = self.shared_mlp(x.mean(dim=[2, 3]))          # (B, C)
        max_out = self.shared_mlp(x.amax(dim=[2, 3]))          # (B, C)
        attn = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    """Spatial attention: learns which spatial locations are important."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        max_out = x.amax(dim=1, keepdim=True)   # (B, 1, H, W)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """CBAM = Channel Attention → Spatial Attention (sequential)."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ─── Multi-Scale Attention Fusion Network (ConvNeXt + CBAM + FPN) ────────────

class MSAFusionNet(nn.Module):
    """
    Multi-Scale Attention Fusion Network for landslide classification.

    Based on ResM-FusionNet (2025), with key improvements:
      1. ConvNeXt-Base backbone (stronger than ResNet-50)
      2. CBAM attention at each pyramid level
      3. FPN-style top-down fusion for multi-scale features
      4. Dual-head: global avg pool + multi-scale pool → concat → classifier

    ConvNeXt-Base feature channels: [128, 256, 512, 1024] at 4 stages.
    """

    def __init__(self, num_classes: int = 6, dropout: float = 0.4):
        super().__init__()
        import torchvision.models as models

        # Load ConvNeXt-Base pretrained on ImageNet
        backbone = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)

        # ConvNeXt features: indices 0-1 (stage1), 2-3 (stage2), 4-5 (stage3), 6-7 (stage4)
        self.stage1 = nn.Sequential(backbone.features[0], backbone.features[1])  # 128 ch
        self.stage2 = nn.Sequential(backbone.features[2], backbone.features[3])  # 256 ch
        self.stage3 = nn.Sequential(backbone.features[4], backbone.features[5])  # 512 ch
        self.stage4 = nn.Sequential(backbone.features[6], backbone.features[7])  # 1024 ch

        stage_channels = [128, 256, 512, 1024]
        fpn_dim = 256  # Unified FPN channel dimension

        # CBAM attention at each stage
        self.cbam1 = CBAM(stage_channels[0])
        self.cbam2 = CBAM(stage_channels[1])
        self.cbam3 = CBAM(stage_channels[2])
        self.cbam4 = CBAM(stage_channels[3])

        # Lateral connections (1x1 conv to unify channel dimensions for FPN)
        self.lateral4 = nn.Conv2d(stage_channels[3], fpn_dim, 1)
        self.lateral3 = nn.Conv2d(stage_channels[2], fpn_dim, 1)
        self.lateral2 = nn.Conv2d(stage_channels[1], fpn_dim, 1)
        self.lateral1 = nn.Conv2d(stage_channels[0], fpn_dim, 1)

        # FPN smooth layers (3x3 conv after top-down addition)
        self.smooth4 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.smooth1 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)

        # Classification head: concat global avg pools from all 4 FPN levels + stage4
        # Total features: fpn_dim * 4 (FPN levels) + stage_channels[3] (direct) = 1024 + 1024
        head_dim = fpn_dim * 4 + stage_channels[3]

        self.classifier = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Dropout(p=dropout),
            nn.Linear(head_dim, 512),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes),
        )

        self._init_added_layers()

    def _init_added_layers(self):
        """Initialize non-pretrained layers with proper initialization."""
        for module in [self.lateral1, self.lateral2, self.lateral3, self.lateral4,
                       self.smooth1, self.smooth2, self.smooth3, self.smooth4,
                       self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bottom-up: extract multi-scale features
        c1 = self.stage1(x)    # (B, 128, H/4,  W/4)
        c2 = self.stage2(c1)   # (B, 256, H/8,  W/8)
        c3 = self.stage3(c2)   # (B, 512, H/16, W/16)
        c4 = self.stage4(c3)   # (B, 1024, H/32, W/32)

        # Apply CBAM attention at each scale
        c1 = self.cbam1(c1)
        c2 = self.cbam2(c2)
        c3 = self.cbam3(c3)
        c4 = self.cbam4(c4)

        # Top-down FPN with lateral connections
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[2:], mode="nearest")
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[2:], mode="nearest")

        # Smooth FPN outputs
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)

        # Global average pool each level → concat
        g1 = F.adaptive_avg_pool2d(p1, 1).flatten(1)  # (B, fpn_dim)
        g2 = F.adaptive_avg_pool2d(p2, 1).flatten(1)  # (B, fpn_dim)
        g3 = F.adaptive_avg_pool2d(p3, 1).flatten(1)  # (B, fpn_dim)
        g4 = F.adaptive_avg_pool2d(p4, 1).flatten(1)  # (B, fpn_dim)

        # Also use direct stage4 features for rich high-level representation
        g_direct = F.adaptive_avg_pool2d(c4, 1).flatten(1)  # (B, 1024)

        # Concatenate all features
        features = torch.cat([g1, g2, g3, g4, g_direct], dim=1)  # (B, 2048)

        return self.classifier(features)


# ─── SwinV2-Small Classifier ─────────────────────────────────────────────────

class SwinV2Classifier(nn.Module):
    """
    Swin Transformer V2 Small for landslide classification.
    Reference: Liu et al., "Swin Transformer V2", CVPR 2022.

    Pretrained on ImageNet, with classification head replaced.
    Input size: 256x256.
    """

    def __init__(self, num_classes: int = 6, dropout: float = 0.4):
        super().__init__()
        import torchvision.models as models

        self.model = models.swin_v2_s(weights=models.Swin_V2_S_Weights.IMAGENET1K_V1)

        # Replace classification head
        in_features = self.model.head.in_features  # 768
        self.model.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 384),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(384, num_classes),
        )

        # Initialize new head
        for m in self.model.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─── EMA (Exponential Moving Average) Model ──────────────────────────────────

class EMAModel:
    """
    Maintains an exponential moving average of model parameters.
    The EMA model typically generalizes better than the raw trained model.

    Usage:
        ema = EMAModel(model, decay=0.9998)
        # After each training step:
        ema.update(model)
        # For evaluation:
        ema.apply(model)  # copy EMA weights into model
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters after a training step."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        """Copy EMA weights into the model (for evaluation)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]


# ─── Factory Functions ────────────────────────────────────────────────────────

def get_convnext_cbam_fpn(num_classes: int = 6, dropout: float = 0.4, freeze: bool = True) -> nn.Module:
    """
    Create MSAFusionNet (ConvNeXt-Base + CBAM + FPN).
    Backbone frozen by default for warm-up phase.
    """
    model = MSAFusionNet(num_classes=num_classes, dropout=dropout)

    if freeze:
        # Freeze backbone stages (CBAM, FPN, classifier remain trainable)
        for stage in [model.stage1, model.stage2, model.stage3, model.stage4]:
            for param in stage.parameters():
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"MSAFusionNet: {total:,} total params, {trainable:,} trainable (backbone frozen)")
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"MSAFusionNet: {total:,} total params (all trainable)")

    return model


def get_swinv2_s(num_classes: int = 6, dropout: float = 0.4, freeze: bool = True) -> nn.Module:
    """
    Create SwinV2-Small classifier.
    Feature layers frozen by default for warm-up phase.
    """
    model = SwinV2Classifier(num_classes=num_classes, dropout=dropout)

    if freeze:
        # Freeze everything except the classification head
        for name, param in model.model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"SwinV2-Small: {total:,} total params, {trainable:,} trainable (backbone frozen)")
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"SwinV2-Small: {total:,} total params (all trainable)")

    return model


# ─── Legacy factory functions (backward compatibility) ────────────────────────

class AlexNet(nn.Module):
    """Legacy AlexNet (kept for checkpoint loading compatibility)."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_model(pretrained=False, num_classes=2, dropout=0.5):
    """Legacy: create AlexNet model."""
    if pretrained:
        import torchvision.models as models
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        return model
    return AlexNet(num_classes=num_classes, dropout=dropout)


def get_efficientnet_b3(num_classes=2, freeze_epochs=5):
    """Legacy: create EfficientNet-B3."""
    import torchvision.models as models
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_vit_b_16(num_classes=2):
    """Legacy: create ViT-B/16."""
    import torchvision.models as models
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    for param in model.heads.head.parameters():
        param.requires_grad = True
    return model
