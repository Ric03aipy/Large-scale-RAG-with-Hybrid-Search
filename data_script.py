import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FOLDER = ROOT / "data"
PEPS_FOLDER = ROOT.parent / "peps/peps" 

# Reproducibility
random.seed(42)

FILES = sorted(list(PEPS_FOLDER.glob("pep-*.rst")))
N_SAMPLES = 100

if not DATA_FOLDER.exists(): Path.mkdir(DATA_FOLDER, exist_ok=True)
for file in random.sample(FILES, N_SAMPLES): shutil.copy2(file, DATA_FOLDER)
