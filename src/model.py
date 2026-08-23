import torch.nn as nn
from torchvision import models

def get_model(num_classes=10, pretrained=True) -> nn.Module:
    weights = 'DEFAULT' if pretrained else None
    model = models.resnet18(weights=weights)

    num_ftrs = model.fc.in_features
    print("Input Features: ", num_ftrs, " Input Classes: ", num_classes)
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
