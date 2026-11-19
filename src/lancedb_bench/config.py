"""Paths and per-dataset settings.

The db container mounts a read-only `/data` (DATA_DIR) with one folder per dataset, exactly as in the Hugging Face
repo HF_REPO: `<dataset>/{train.csv, test.csv, ground_truth.csv, manifest.json}` (scripts/download.py fetches them
into ./data). The scripts and the benchmark run in the db container; the client container only sends goals
(`db <goal> <params...>`) to it over HTTP and has no data.
"""
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DB_URI = os.getenv("DB_URI", "./db")
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./results"))
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
HF_REPO = os.getenv("HF_REPO", "zuzaj/ADB")

# Dataset name (= folder name in the Hugging Face repo) -> LanceDB distance type. To add a dataset, add a line here.
# The CSVs carry no metric, so this is the single place that decides it (scripts/check_setup.py verifies it).
DATASETS = {
    "cohere-small-100k": "cosine",
    "nytimes-256-angular": "cosine",  # angular = cosine
    "pubmed-arxiv": "cosine",
    "uform-wiki-1m": "cosine",  # unit-norm vectors, so cosine and inner product rank alike; ground truth distances are cosine
    "glove-200-angular": "cosine",  # angular = cosine
    "ccnews-nomic-768-normalized": "cosine",  # normalized vectors: cosine, dot and L2 rank alike
}
DEFAULT_DATASET = "glove-200-angular"  # the common default of the class baseline, so `db 1 443321 4` is valid

DATASET_FILES = ["train.csv", "test.csv", "ground_truth.csv", "manifest.json"]

# Identity and results (common baseline, section 7: /results/build.csv and /results/search.csv)
GROUP = os.getenv("GROUP", "07-strycharz-jarlaczynska")
SYSTEM = "lancedb"

# Search settings of the benchmark goal (db 8): nprobes is the accuracy setting that is swept (at least 3 values),
# refine_factor stays fixed. Always set explicitly: LanceDB auto-tunes nprobes otherwise.
NPROBES_SETTINGS = [int(x) for x in os.getenv("NPROBES_SETTINGS", "5,10,20,50,100,200").split(",")]
REFINE_FACTOR = int(os.getenv("REFINE_FACTOR", 5))
DEFAULT_NPROBES = int(os.getenv("NPROBES", 20))  # goals 1-7
WARMUP = 200  # queries discarded before timing (baseline section 3)
PASSES = 3  # timed passes over the first n test queries


@dataclass(frozen=True)
class Dataset:
    name: str

    @property
    def metric(self) -> str:
        if self.name not in DATASETS:
            raise SystemExit(f"unknown dataset {self.name!r}; known: {', '.join(DATASETS)} (add new ones to config.DATASETS)")
        return DATASETS[self.name]

    @property
    def dir(self) -> Path:
        return DATA_DIR / self.name

    @property
    def train_path(self) -> Path:
        return self.dir / "train.csv"

    @property
    def test_path(self) -> Path:
        return self.dir / "test.csv"

    @property
    def truth_path(self) -> Path:
        return self.dir / "ground_truth.csv"

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    @property
    def table(self) -> str:
        return re.sub(r"\W", "_", self.name)

    @property
    def table_dir(self) -> Path:
        return Path(DB_URI) / f"{self.table}.lance"  # local DB_URI only



def get_dataset(name: str | None = None) -> Dataset:
    ds = Dataset(name or os.getenv("DATASET", DEFAULT_DATASET))
    ds.metric  # fail early on unknown names
    return ds


def select_datasets(name: str | None) -> list[Dataset]:
    """`all` selects every dataset in the registry; otherwise the named one (default: $DATASET or DEFAULT_DATASET)."""
    if (name or os.getenv("DATASET")) == "all":
        return [Dataset(n) for n in DATASETS]
    return [get_dataset(name)]


def add_dataset_arg(parser) -> None:
    parser.add_argument(
        "--dataset", default=None,
        help=f"dataset name or 'all' (default: $DATASET or {DEFAULT_DATASET}); known: {', '.join(DATASETS)}",
    )


def index_defaults(n: int, dim: int) -> tuple[int, int]:
    """Starting IVF_PQ parameters: about 2300 vectors per partition (power of two), dim/8 sub-vectors
    (reduced until it divides dim). Override on the CLI or with INDEX_PARAMS."""
    partitions = 2 ** max(0, round(math.log2(max(n / 2300, 1))))
    sub_vectors = max(1, dim // 8)
    while dim % sub_vectors:
        sub_vectors -= 1
    return partitions, sub_vectors


# Tuned (partitions, sub-vectors) for datasets where the defaults do not reach recall@10 >= 0.95 at any nprobes <= 200
# (the baseline's requirement). Measured with the populate check on 200 test queries, refine_factor=5:
#   glove-200-angular   defaults (512, 25): 0.63 at nprobes=50; (256, 100): 0.97 at nprobes=100
#   nytimes-256-angular defaults (128, 32): 0.84 at nprobes=50; (128, 64): 0.97 at nprobes=100
#   pubmed-arxiv        defaults (128, 96): 0.94 at nprobes=50; (128, 192): 0.97 at nprobes=10
INDEX_PARAMS = {
    "glove-200-angular": (256, 100),
    "nytimes-256-angular": (128, 64),
    "pubmed-arxiv": (128, 192),
}


def index_params(name: str, n: int, dim: int) -> tuple[int, int]:
    return INDEX_PARAMS.get(name) or index_defaults(n, dim)
