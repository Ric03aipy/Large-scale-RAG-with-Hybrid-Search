import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = ROOT / "data"
PEPS_FOLDER = ROOT.parent / "peps/peps" 
EVAL_FOLDER = ROOT / "eval"

# Reproducibility: seed + manifest
random.seed(42)

FILES = sorted(PEPS_FOLDER.glob("pep-*.rst"))
N_SAMPLES = 100



if __name__ == "__main__":

    Path.mkdir(DATA_FOLDER, exist_ok=True) # exist_ok avoids conditional statement
    with Path.open(EVAL_FOLDER / "manifest.txt", "w") as f: 
        for file in random.sample(FILES, N_SAMPLES): 
            shutil.copy2(file, DATA_FOLDER)
            filename = file.name
            f.write(filename + "\n")
