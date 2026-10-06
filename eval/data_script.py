import random
import shutil
from pathlib import Path

from eval.eval_config import ROOT, DATA_FOLDER, PEPS_FOLDER, EVAL_FOLDER, SEED

# Reproducibility: seed + manifest
random.seed(SEED)

FILES = sorted(PEPS_FOLDER.glob("pep-*.rst"))
N_SAMPLES = 100


if __name__ == "__main__":
    Path.mkdir(DATA_FOLDER, exist_ok=True)  # exist_ok avoids conditional statement
    with Path.open(EVAL_FOLDER / "manifest.txt", "w") as f:
        for file in random.sample(FILES, N_SAMPLES):
            shutil.copy2(file, DATA_FOLDER)
            filename = file.name
            f.write(filename + "\n")
