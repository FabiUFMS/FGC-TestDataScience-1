### Imports ###
from pathlib import Path



# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
print(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_MODEL_DIR = DATA_DIR / "model"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MACHINE_LEARNING_DIR = PROJ_ROOT / "reports" / "machine_learning"




