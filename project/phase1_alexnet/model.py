"""
CNN architectures for landslide binary image classification.
Supports AlexNet (227x227) and EfficientNet-B3 (300x300).
Output: logits for [non_landslide, landslide]
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    AlexNet CNN with 5 convolutional layers and 3 fully-connected layers.
    Matches the original AlexNet architecture adapted for 227x227 input.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: 227x227x3 -> 55x55x96
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),   # -> 27x27x96
            # Conv2: 27x27x96 -> 27x27x256
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),   # -> 13x13x256
            # Conv3: 13x13x256 -> 13x13x384
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv4: 13x13x384 -> 13x13x384
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv5: 13x13x384 -> 13x13x256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),   # -> 6x6x256
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

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x  # raw logits; use softmax only at inference

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)


def get_model(pretrained: bool = False, num_classes: int = 2, dropout: float = 0.5) -> AlexNet:
    """
    Factory function to create an AlexNet model.

    Args:
        pretrained: If True, loads ImageNet-pretrained weights from torchvision
                    and replaces the final classifier layer. Recommended for
                    faster convergence with small datasets.
        num_classes: Number of output classes (default 2).
        dropout:     Dropout probability in FC layers.

    Returns:
        AlexNet model instance.
    """
    if pretrained:
        import torchvision.models as models

        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        # Freeze feature layers for transfer learning (optional — unfreeze after a few epochs)
        for param in model.features.parameters():
            param.requires_grad = False
        # Replace final classifier layer
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        print("Loaded pretrained AlexNet with replaced final layer.")
        return model

    model = AlexNet(num_classes=num_classes, dropout=dropout)
    print(f"Initialized AlexNet from scratch (num_classes={num_classes}).")
    return model


def get_efficientnet_b3(num_classes: int = 2, freeze_epochs: int = 5) -> nn.Module:
    """
    EfficientNet-B3 pretrained on ImageNet, with classifier head replaced.
    Feature layers frozen initially — call unfreeze_features() after warm-up.

    Args:
        num_classes: Number of output classes.
        freeze_epochs: Informational only; caller controls unfreezing.

    Returns:
        EfficientNet-B3 model instance.
    """
    import torchvision.models as models

    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    # Freeze all feature layers for warm-up phase
    for param in model.features.parameters():
        param.requires_grad = False
    # Replace classifier head: EfficientNet-B3 has 1536 features in
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    print(f"Loaded pretrained EfficientNet-B3 with replaced classifier head (in={in_features}, out={num_classes}).")
    return model


def get_vit_b_16(num_classes: int = 2) -> nn.Module:
    """
    Vision Transformer (ViT-B/16) pretrained on ImageNet.
    In torchvision, the classification head is accessed via model.heads.head.
    """
    import torchvision.models as models

    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    # Freeze the base parameters for warm-up
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the classification head
    # ViT-B/16's head operates on 768 features.
    in_features = model.heads.head.in_features
    # Unfreeze the head
    model.heads.head = nn.Linear(in_features, num_classes)
    for param in model.heads.head.parameters():
        param.requires_grad = True

    print(f"Loaded pretrained ViT-B/16 with replaced classifier head (in={in_features}, out={num_classes}).")
    return model
