import random
import shutil
from pathlib import Path

from eval.eval_config import DATA_FOLDER, PEPS_FOLDER, EVAL_FOLDER, SEED, N_SAMPLES_DOCS

if __name__ == "__main__":
    # Reproducibility: seed + manifest
    rng = random.Random(SEED)
    FILES = sorted(PEPS_FOLDER.glob("pep-*.rst"))

    Path.mkdir(DATA_FOLDER, exist_ok=True)  # exist_ok avoids conditional statement
    with Path.open(EVAL_FOLDER / "manifest.txt", "w") as f:
        for file in random.sample(FILES, N_SAMPLES_DOCS):
            shutil.copy2(file, DATA_FOLDER)
            filename = file.name
            f.write(filename + "\n")
