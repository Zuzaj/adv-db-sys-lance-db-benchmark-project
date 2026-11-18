"""Measurements shared by the db-side scripts and the server: latency percentiles, recall, sizes, peak memory."""
import threading
from pathlib import Path

import numpy as np
import psutil


def latency_percentiles(latencies_ms: list[float]) -> dict:
    """p50, p95 and p99 in ms over all samples (baseline section 3)."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    p50, p95, p99 = (round(float(np.percentile(arr, p)), 3) for p in (50, 95, 99))
    return {"p50_ms": p50, "p95_ms": p95, "p99_ms": p99}


def recall_at_k(predicted_ids: list[list[int]], ground_truth_ids: np.ndarray, k: int) -> float:
    """Overlap between the returned ids and the file's first k neighbors, divided by k, averaged over queries."""
    recalls = [len(set(pred[:k]) & set(true[:k].tolist())) / k for pred, true in zip(predicted_ids, ground_truth_ids)]
    return round(float(np.mean(recalls)), 4)


def dir_size_mb(path) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024), 2)


class PeakRss:
    """Peak resident memory (RSS, MB) of this process (and its children) while the `with` block runs.

    Samples every `interval` seconds with psutil (baseline section 3). The page cache is not counted.
    The database is embedded, so this process is the database process.
    """

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak_mb = 0.0
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        rss = self._process.memory_info().rss
        for child in self._process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        self.peak_mb = max(self.peak_mb, rss / (1024 * 1024))

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self) -> "PeakRss":
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()
