import random
import shutil
from pathlib import Path

from eval.eval_config import DATA_FOLDER, EVAL_FOLDER, N_SAMPLES_DOCS, PEPS_FOLDER, SEED

if __name__ == "__main__":
    # Reproducibility: seed + manifest
    rng = random.Random(SEED)
    FILES = sorted(PEPS_FOLDER.glob("pep-*.rst"))
    # If files is empty we don't want to erase the current collection 
    if not FILES: 
        raise FileNotFoundError(f"{DATA_FOLDER} doesn't contain any '*.rst' file. Please make sure {PEPS_FOLDER} exist before.")
       

    Path.mkdir(DATA_FOLDER, exist_ok=True)  # exist_ok avoids conditional statement
    with Path.open(EVAL_FOLDER / "manifest.txt", "w") as f:
        for file in rng.sample(FILES, N_SAMPLES_DOCS):
            shutil.copy2(file, DATA_FOLDER)
            filename = file.name
            f.write(filename + "\n")
