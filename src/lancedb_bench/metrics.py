import resource
import sys
from pathlib import Path
import numpy as np

def compute_latency_stats(latencies_ms: list[float]) -> dict:
    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "mean_ms": round(float(np.mean(arr)), 3),
        "qps": round(float(1000.0 / np.mean(arr)), 2) if np.mean(arr) > 0 else 0.0,
    }

def calculate_recall_at_k(predicted_ids: list[list[int]], ground_truth_ids: np.ndarray, k: int) -> float:
    recalls = []
    for pred, true in zip(predicted_ids, ground_truth_ids):
        true_set = set(true[:k].tolist())
        pred_set = set(pred[:k])
        recalls.append(len(pred_set & true_set) / k)
    return round(float(np.mean(recalls)), 4)

def get_peak_rss_mb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor, 2)

def get_dir_size_mb(path: str) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024), 2)