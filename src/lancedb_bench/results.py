"""The two shared result files of the class baseline (section 7), written to RESULTS_DIR (/results):

  build.csv   one row per populate run (and per K-means run, which fills kmeans_s of the latest row of the dataset)
  search.csv  one row per (dataset, k, accuracy setting) of the benchmark goal (db 8)
"""
import csv
import os
import platform
import time
from importlib.metadata import version

import psutil

from lancedb_bench.config import GROUP, RESULTS_DIR, SYSTEM

ENV_COLUMNS = ["group", "system", "version", "dataset"]
BUILD_COLUMNS = ENV_COLUMNS + ["ingest_s", "db_mb", "peak_mem_mb", "cpu", "cores", "ram_gb", "date", "kmeans_s"]
SEARCH_COLUMNS = ENV_COLUMNS + ["k", "setting_name", "setting_value", "recall", "p50_ms", "p95_ms", "p99_ms"]


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def identity(dataset: str) -> dict:
    return {"group": GROUP, "system": SYSTEM, "version": version("lancedb"), "dataset": dataset}


def machine() -> dict:
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return {"cpu": cpu_model(), "cores": cores, "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "date": time.strftime("%Y-%m-%d")}


def append_rows(name: str, columns: list[str], rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    new_file = not path.exists()
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def set_last_value(name: str, dataset: str, column: str, value) -> bool:
    """Fill `column` in the latest row of `dataset`. Returns False if the file has no row for the dataset."""
    path = RESULTS_DIR / name
    if not path.exists():
        return False
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in reversed(rows):
        if row["dataset"] == dataset:
            row[column] = value
            break
    else:
        return False
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=BUILD_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return True
