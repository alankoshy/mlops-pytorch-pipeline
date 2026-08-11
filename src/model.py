import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(
    num_classes=10, pretrained=True, in_channels=3
) -> nn.Module:
    """
    Fine-tunes a ResNet-18 model for small-image classification tasks.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # Adjust first convolution layer if working with single-channel image for FashionMNIST
    if in_channels == 1:
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    num_ftrs = model.fc.in_features
    print("Input Features: ", num_ftrs, " Input Classes: ", num_classes)
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
