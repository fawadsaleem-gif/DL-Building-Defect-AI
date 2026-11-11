# Building Defect AI

An educational computer-vision project for classifying visible building-surface conditions from images using **ResNet18 transfer learning** and **Grad-CAM**.

> **Important:** This is a research/learning prototype. It is not a structural-safety assessment, professional inspection, or diagnosis tool. Predictions can be wrong and must be reviewed by a qualified professional.

## Problem and scope
The model classifies an input image into one of seven dataset labels:

`algae`, `major_crack`, `minor_crack`, `peeling`, `plain`, `spalling`, `stain`.

The labels are preserved exactly as supplied by the dataset. In particular, `algae` and `stain` are not renamed to other building-condition categories.

## Method
1. Explore the image dataset and inspect class balance.
2. Use ImageNet-pretrained ResNet18 preprocessing.
3. Train a classification head, then compare fine-tuning strategies.
4. Evaluate on a held-out test split using accuracy, per-class precision/recall/F1 and a confusion matrix.
5. Use Grad-CAM to visualise image regions that influenced a selected prediction.
6. Serve inference and visual explanation through a Streamlit interface.

## Dataset
The project work used the BD3 dataset (3,965 images) and a 60/20/20 train/validation/test split with seed 42. The split counts recorded during development were:
- Train: 2,378
- Validation: 792
- Test: 795

The `BFD-ML-4K` folder was present in the working project but its role in the experiments has not been established. Keep it separate unless a documented experiment explicitly uses it.

**Do not upload the dataset automatically.** Confirm the dataset licence and redistribution terms first. The original dataset and split folders are intentionally not included in this package.

## Experiments recorded during development
| Experiment | Strategy | Test accuracy |
|---|---|---:|
| 1 | Frozen ResNet18 feature extractor; train classification head | 83.14% |
| 2 | Fine-tune Layer3, Layer4 and classifier | 79.75% |
| 3 | Fine-tune Layer4 and classifier; early stopping | Verify from saved test evaluation before publication |

Experiment 2 showed a substantial training/validation gap and weak recall for `major_crack`, suggesting overfitting and class-specific confusion. Experiment 3 was selected as the candidate model because it used a more constrained fine-tuning strategy and early stopping. **Do not state a final Experiment 3 test score until `04_model_evaluation.ipynb` has been run against the saved checkpoint and test set.**

## Repository layout
```text
Building_Defect_AI/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── app/app.py
├── notebooks/
├── src/building_defect_ai/
├── configs/
├── dataset/README.md
├── models/README.md
├── outputs/figures/
├── outputs/reports/
├── docs/
└── original_scripts/
```

`original_scripts/` preserves the scripts supplied for the audit. The app is copied to `app/app.py`. The package does not contain your local dataset or model checkpoint; copy those from your working project into the matching folders.

## Setup
```bash
conda env create -f environment.yml
conda activate building-defect-ai
```
Or use an existing Python environment:
```bash
pip install -r requirements.txt
```

Place the prepared dataset at `dataset/BD3_split/{train,val,test}/<class_name>/...` and the selected checkpoint at:
`models/building_defect_resnet18_experiment3.pth`

Run notebooks from the repository root. Launch the app:
```bash
streamlit run app/app.py
```

## Reproducibility notes
- The split script in the original archive used a hard-coded local Windows path. Use the relative-path version in `src/building_defect_ai/dataset.py` or edit the source path before running it.
- Confirm that the class order in the checkpoint matches the `ImageFolder` class order.
- Record the checkpoint, test metrics, random seed, library versions and dataset source with each final result.
- Check for near-duplicate images across splits before claiming independent generalisation.
- A Grad-CAM heatmap is an explanation aid, not a defect boundary, severity estimate or proof of causal reasoning.

## Learning outcomes
The notebooks explain the purpose of each method, show tensor shapes and visualisations, compare training strategies, evaluate class-wise performance, and interpret Grad-CAM outputs.

## Credits
Dataset: BD3 dataset, attributed in the original project to Praveenkottari. Verify the dataset's current licence and citation requirements before redistribution.
