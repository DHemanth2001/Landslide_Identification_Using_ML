"""
AlexNet architecture for landslide binary image classification.
Input: 227x227x3 RGB images
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
