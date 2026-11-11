"""Paths and per-dataset settings.

Each container sees `DATA_DIR/<dataset>/...` but only its own files (see README):
  db container:     train.csv
  client container: test.csv, ground_truth.csv, manifest.json
A dataset is a folder of the Hugging Face repo HF_REPO with those four files; scripts/download.py fetches them.
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
DEFAULT_DATASET = "cohere-small-100k"  # the smallest one, for quick tests

ROLE_FILES = {
    "db": ["train.csv"],
    "client": ["test.csv", "ground_truth.csv", "manifest.json"],
}


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

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR / self.name


def get_dataset(name: str | None = None) -> Dataset:
    ds = Dataset(name or os.getenv("DATASET", DEFAULT_DATASET))
    ds.metric  # fail early on unknown names
    return ds


def select_datasets(name: str | None) -> list[Dataset]:
    """`all` selects every dataset in the registry; otherwise the named one (default: $DATASET or DEFAULT_DATASET)."""
    if name == "all":
        return [Dataset(n) for n in DATASETS]
    return [get_dataset(name)]


def add_dataset_arg(parser) -> None:
    parser.add_argument(
        "--dataset", default=None,
        help=f"dataset name or 'all' (default: $DATASET or {DEFAULT_DATASET}); known: {', '.join(DATASETS)}",
    )


def index_defaults(n: int, dim: int) -> tuple[int, int]:
    """Starting IVF_PQ parameters: about 2300 vectors per partition (power of two), dim/8 sub-vectors
    (reduced until it divides dim). Override on the CLI."""
    partitions = 2 ** max(0, round(math.log2(max(n / 2300, 1))))
    sub_vectors = max(1, dim // 8)
    while dim % sub_vectors:
        sub_vectors -= 1
    return partitions, sub_vectors
