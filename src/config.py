from pathlib import Path

# Project root: acne-classification/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR=PROJECT_ROOT/"Data/AcneDataset"
CHECKPOINT_DIR=PROJECT_ROOT/"models"/"checkpoint"
IMG_SIZE=(224,224)

BATCH_SIZE=32