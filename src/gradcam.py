import torch
import torch.nn.functional as F
import numpy as np

def generate_gradcam(model, input_tensor, target_layer, class_index=None):
    """Create a normalised Grad-CAM heatmap for one input tensor [1,3,H,W]."""
    activations, gradients = {}, {}
    def hook(module, inputs, output):
        activations["value"] = output
        output.register_hook(lambda grad: gradients.__setitem__("value", grad))
    handle = target_layer.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        if class_index is None:
            class_index = int(logits.argmax(1).item())
        logits[0, class_index].backward()
        weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations["value"]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam, class_index
    finally:
        handle.remove()
