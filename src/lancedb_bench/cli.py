"""`db <goal> <params...>` command line (client side, talks to the db container over HTTP).

Goals 1 and 8 are defined so far; 2-7 are still TBD with the class. The dataset comes from $DATASET
(default cohere-small-100k), the server from $SERVER_URL.
"""
import os
import sys
import time

from lancedb_bench.client import Client
from lancedb_bench.config import get_dataset
from lancedb_bench.data import read_ground_truth, read_queries
from lancedb_bench.metrics import calculate_recall_at_k, compute_latency_stats

# Query settings for the CLI (the reference setting; override with the environment)
NPROBES = int(os.getenv("NPROBES", 20))
REFINE_FACTOR = int(os.getenv("REFINE_FACTOR", 5))

USAGE = """Usage (dataset from $DATASET, default cohere-small-100k):
  db 1 <id> <k>           KNN of a stored vector (the vector itself is excluded)
  db 8 <n_queries> <k>    batch KNN over the first n test queries, with recall@k and latency"""


def goal_1(client: Client, ds, row_id: int, k: int) -> None:
    data, _ = client.search(ds.name, id=row_id, k=k, nprobes=NPROBES, refine_factor=REFINE_FACTOR)
    for rank, (i, d) in enumerate(zip(data["ids"], data["distances"]), 1):
        print(f"{rank:>3}  id={i}  {ds.metric}_distance={d:.6f}")


def goal_8(client: Client, ds, n_queries: int, k: int) -> None:
    queries = read_queries(ds.test_path, n_queries)[1]
    truth, _ = read_ground_truth(ds.truth_path, len(queries))
    if k > truth.shape[1]:
        sys.exit(f"ground truth of {ds.name} has only {truth.shape[1]} neighbors per query, use k <= {truth.shape[1]}")
    preds, total_ms, engine_ms = [], [], []
    for vec in queries:
        data, ms = client.search(ds.name, vec, k=k, nprobes=NPROBES, refine_factor=REFINE_FACTOR)
        preds.append(data["ids"])
        total_ms.append(ms)
        engine_ms.append(data["engine_latency_ms"])
    print(f"{ds.name}: queries={len(queries)} k={k} nprobes={NPROBES} refine_factor={REFINE_FACTOR}")
    print(f"recall@{k}: {calculate_recall_at_k(preds, truth, k)}")
    print(f"client total latency: {compute_latency_stats(total_ms)}")
    print(f"server engine latency: {compute_latency_stats(engine_ms)}")


def main() -> None:
    args = sys.argv[1:]
    if not args or not args[0].isdigit():
        sys.exit(USAGE)
    goal = int(args[0])
    if goal not in (1, 8):
        sys.exit(f"goal {goal} is not defined yet\n{USAGE}")
    try:
        if len(args) != 3:
            raise ValueError
        params = (int(args[1]), int(args[2]))
    except ValueError:
        sys.exit(USAGE)

    ds = get_dataset()
    t0 = time.perf_counter()
    (goal_1 if goal == 1 else goal_8)(Client(), ds, *params)
    print(f"[goal {goal} took {(time.perf_counter() - t0) * 1000:.1f} ms including Python-side work]")


if __name__ == "__main__":
    main()
