# Experiment Results

## Model comparison

| Experiment | Training strategy | Test accuracy |
|---|---|---:|
| Experiment 1 | Frozen ResNet18 feature extractor; train classifier | 83.14% |
| Experiment 2 | Fine-tune Layer3, Layer4 and classifier | 79.75% |
| Experiment 3 | Fine-tune Layer4 and classifier with early stopping | **87.92%** |

## Experiment 3 — Selected model

Experiment 3 achieved a test accuracy of **87.92% across 795 test images**.

The model used selective fine-tuning of ResNet18's Layer4 and final classification layer. Early stopping was applied using validation loss to limit unnecessary training.

The saved checkpoint was:

`models/building_defect_resnet18_experiment3.pth`

## Comparison and interpretation

Experiment 3 outperformed the first two experiments on the recorded test evaluation.

Experiment 2 showed a larger training-validation performance gap, suggesting overfitting. Experiment 3 used a more constrained fine-tuning strategy and achieved the strongest recorded test accuracy.

The confusion matrix also indicates that some defect categories remain difficult to distinguish, particularly the crack categories.

## Conclusion

Experiment 3 is selected as the project's final candidate model based on its measured test accuracy and training strategy.

Further work should investigate class-specific errors, dataset diversity and possible near-duplicate images across dataset splits.