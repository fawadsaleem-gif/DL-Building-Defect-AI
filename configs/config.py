from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "BD3_split"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "building_defect_resnet18_experiment3.pth"
CLASS_NAMES = ["algae", "major_crack", "minor_crack", "peeling", "plain", "spalling", "stain"]
BATCH_SIZE = 16
RANDOM_SEED = 42
