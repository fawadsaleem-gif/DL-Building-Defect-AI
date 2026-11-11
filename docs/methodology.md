# Methodology

## Task
Seven-class image classification of building-surface appearance categories.

## Input pipeline
Images are loaded with `torchvision.datasets.ImageFolder`. ResNet18 ImageNet weight transforms resize/crop and normalise images to the pretrained model's expected input distribution. A typical batch has shape `[B, 3, 224, 224]`; labels have shape `[B]`.

## Transfer learning
ResNet18 provides convolutional features learned from ImageNet. The final fully connected layer is replaced with a seven-output classifier. The initial experiment freezes the feature extractor and trains the classifier. Later experiments unfreeze selected deeper layers to adapt features to the target dataset.

## Optimisation and objective
Multi-class cross entropy compares logits with integer class indices. Adam updates only parameters whose `requires_grad` is true. Fine-tuning deeper layers gives more adaptation capacity but can overfit, especially on a relatively small dataset.

## Early stopping
Experiment 3 monitored validation loss and retained the checkpoint with the best validation loss. Early stopping limited unnecessary training after validation loss failed to improve.

## Explainability
Grad-CAM uses gradients flowing into a convolutional feature layer to weight its feature maps. The resulting heatmap highlights regions influential to a target class score. It does not establish the true physical location or severity of a defect.
