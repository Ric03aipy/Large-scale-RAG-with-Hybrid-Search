from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = ROOT / "data"
PEPS_FOLDER = ROOT.parent / "peps/peps"
EVAL_FOLDER = ROOT / "eval"
METRIC_FILE = EVAL_FOLDER / "raw_metrics.csv"
METRIC_RESULT_FILE = EVAL_FOLDER / "eval_results.csv"


SEED = 42
MAX_CHUNKS_PER_FILE = 3
N_CHUNKS_TO_EXPORT= 80

N_SAMPLES_DOCS = 100

TEST_CHUNK_SIZE = 1000
TEST_CHUNK_OVERLAP = 50

TEST_RRF_LIMIT = 10
TEST_PREFETCH_LIMIT = 15
