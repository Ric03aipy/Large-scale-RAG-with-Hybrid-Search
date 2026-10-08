from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = ROOT / "data"
PEPS_FOLDER = ROOT.parent / "peps/peps"
EVAL_FOLDER = ROOT / "eval"

SEED = 42
MAX_CHUNKS_PER_FILE = 3
N_CHUNKS_TO_EXPORT= 80

N_SAMPLES_DOCS = 100

TEST_CHUNK_SIZE = 1000
TEST_CHUNK_OVERLAP = 50
