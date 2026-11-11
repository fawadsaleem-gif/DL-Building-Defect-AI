from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

def load_checkpoint_model(checkpoint_path, device=None):
    """Recreate the project's ResNet18 architecture and load its checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, device, checkpoint["class_names"]

def imagenet_transform():
    """Return the preprocessing transform associated with ResNet18 ImageNet weights."""
    return models.ResNet18_Weights.DEFAULT.transforms()
