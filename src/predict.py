import torch
from PIL import Image
from .model import imagenet_transform

def predict_image(model, image, class_names, device):
    """Return predicted label, confidence and sorted class probabilities."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")
    tensor = imagenet_transform()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    order = torch.argsort(probabilities, descending=True)
    ranked = [(class_names[i], float(probabilities[i])) for i in order]
    return {"label": ranked[0][0], "confidence": ranked[0][1], "ranked": ranked}
