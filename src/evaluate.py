import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, loader, class_names, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "report": classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }
