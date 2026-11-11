# Streamlit application

Run from the repository root after placing the checkpoint at `models/building_defect_resnet18_experiment3.pth`:
```bash
streamlit run app/app.py
```
The app uses the Experiment 3 checkpoint and produces a class prediction, confidence, top-three probabilities and Grad-CAM visualisation. The app file is the audited copy supplied in the source archive.
