"""Client side benchmark over HTTP: warm-up, timed passes, recall@k, one CSV row per (dataset, configuration, k).

Latency is measured twice: total_* is the client wall-clock for the whole HTTP call, engine_* is the server's own
time around the LanceDB query, overhead_* is the difference (serialization + network + parsing).
Rows go to results/<dataset>/<--out>; each row carries the client and server environment.
"""
import argparse
import csv
import platform
import subprocess
import time

import numpy as np
import psutil

from lancedb_bench.client import Client
from lancedb_bench.config import add_dataset_arg, select_datasets
from lancedb_bench.data import read_ground_truth, read_queries
from lancedb_bench.metrics import compute_latency_stats

WARMUP = 200
PASSES = 3


def prefixed(prefix: str, stats: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def environment(client: Client) -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
        except OSError:  # no git in the container
            return ""

    server = client.info()
    return {
        "client_machine": platform.platform(),
        "client_cores": psutil.cpu_count(logical=True),
        "client_python": platform.python_version(),
        "git_commit": git("rev-parse", "--short", "HEAD") + ("+dirty" if git("status", "--porcelain") else ""),
        "server_machine": server["machine"],
        "server_cores": server["cores"],
        "server_cores_allowed": server["cores_allowed"],
        "server_ram_gb": server["ram_gb"],
        "server_lancedb": server["lancedb"],
        "server_peak_rss_mb": server["peak_rss_mb"],
    }


def run_config(client, ds, queries, truth, k, nprobes, refine, exact, warmup, passes) -> dict:
    def search(vec):
        return client.search(ds.name, vec, k=k, nprobes=nprobes, refine_factor=refine, bypass_index=exact)

    for vec in queries[:warmup]:
        search(vec)

    total_ms, engine_ms, recalls = [], [], []
    for p in range(passes):
        for i, vec in enumerate(queries):
            data, ms = search(vec)
            total_ms.append(ms)
            engine_ms.append(data["engine_latency_ms"])
            if p == 0:
                recalls.append(len(set(data["ids"]) & set(truth[i, :k].tolist())) / k)

    return {
        "dataset": ds.name, "metric": ds.metric, "k": k, "nprobes": nprobes, "refine_factor": refine, "exact": exact,
        "n_queries": len(queries), "recall": round(float(np.mean(recalls)), 4),
        **prefixed("total", compute_latency_stats(total_ms)),
        **prefixed("engine", compute_latency_stats(engine_ms)),
        **prefixed("overhead", compute_latency_stats([t - e for t, e in zip(total_ms, engine_ms)])),
        "n_samples": len(total_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(parser)
    parser.add_argument("--n-queries", type=int, default=1000, help="first N test queries")
    parser.add_argument("--nprobes", type=int, nargs="+", default=[20])
    parser.add_argument("--refine", type=int, nargs="+", default=[5])
    parser.add_argument("-k", type=int, nargs="+", default=[10, 100])
    parser.add_argument("--exact", action="store_true", help="also run the brute-force baseline")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--passes", type=int, default=PASSES)
    parser.add_argument("--out", default="bench.csv", help="CSV file name inside results/<dataset>/")
    args = parser.parse_args()

    client = Client()
    client.reload()
    for ds in select_datasets(args.dataset):
        queries = read_queries(ds.test_path, args.n_queries)[1]
        truth, _ = read_ground_truth(ds.truth_path, len(queries))
        rows = []
        for k in args.k:
            if k > truth.shape[1]:
                print(f"{ds.name}: skipping k={k}, ground truth has only {truth.shape[1]} neighbors")
                continue
            configs = [(n, r, False) for n in args.nprobes for r in args.refine] + ([(0, 0, True)] if args.exact else [])
            for nprobes, refine, exact in configs:
                row = run_config(client, ds, queries, truth, k, nprobes, refine, exact, args.warmup, args.passes)
                rows.append(row)
                print({key: row[key] for key in ("dataset", "k", "nprobes", "refine_factor", "exact", "recall", "total_p50_ms", "engine_p50_ms", "total_qps")}, flush=True)
        if not rows:
            continue

        env = environment(client)  # after the runs, so server peak RSS covers them
        env["server_table_size_mb"] = next(d for d in client.datasets() if d["dataset"] == ds.name)["size_mb"]
        fieldnames = list(rows[0]) + list(env)
        ds.results_dir.mkdir(parents=True, exist_ok=True)
        path = ds.results_dir / args.out
        if path.exists():
            with open(path, newline="") as old:
                if next(csv.reader(old), None) != fieldnames:  # columns changed: never mix them in one file
                    path = path.with_name(f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}")
                    print(f"existing file has different columns, writing to {path} instead")
        new_file = not path.exists()
        with open(path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if new_file:
                writer.writeheader()
            for row in rows:
                writer.writerow({**row, **env})
        print(f"appended {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
