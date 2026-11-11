# Model checkpoints

Copy the selected checkpoint from the working project:
`building_defect_resnet18_experiment3.pth`

The Streamlit app expects this exact path and checkpoint fields:
- `model_state_dict`
- `num_classes`
- `class_names`

Large model files can be hosted with Git LFS or the Hugging Face Hub. Do not commit all experimental checkpoints unless there is a clear reason.
