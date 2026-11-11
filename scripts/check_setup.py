"""Client side: check that the whole setup works, per dataset, using only HTTP and the client's own files.

What it verifies (in order): server reachable; the client's test.csv / ground_truth.csv / manifest.json exist;
the server's table matches the manifest (row count, dimension); the vector index exists, covers all rows and
uses the metric from config.DATASETS; exact search reproduces the ground truth (this catches a wrong metric
or id mapping); approximate search returns sensible results; lookup by id works.
Exit code is 1 if anything FAILs.
"""
import argparse
import json
import sys

import numpy as np

from lancedb_bench.client import Client
from lancedb_bench.config import SERVER_URL, add_dataset_arg, select_datasets
from lancedb_bench.data import read_ground_truth, read_queries

EXACT_RECALL_MIN = 0.98  # float32 ties and rounding can cost a little against the float64 ground truth


def check(client: Client, ds, state: dict, n_queries: int, nprobes: int, refine: int) -> bool:
    statuses: list[str] = []

    def report(status: str, label: str, detail: str = "") -> bool:
        statuses.append(status)
        print(f"  [{status:4}] {label}" + (f": {detail}" if detail else ""))
        return status != "FAIL"

    # 1. client files
    missing = [p.name for p in (ds.test_path, ds.truth_path, ds.manifest_path) if not p.exists()]
    if missing:
        report("FAIL", "client files", f"missing in {ds.dir}: {', '.join(missing)} (scripts/download.py --dataset {ds.name})")
        return False
    report("PASS", "client files", "test.csv, ground_truth.csv, manifest.json present")
    if ds.train_path.exists():
        report("WARN", "separation", f"{ds.train_path} is on the client side, it should only be in the db container")
    manifest = json.loads(ds.manifest_path.read_text())

    # 2. server table vs manifest
    entry = state.get(ds.name)
    if entry is None or not entry["exists"]:
        return report("FAIL", "table", "not on the server: run scripts/ingest.py and scripts/build_index.py in the db container")
    ok = True
    if entry["rows"] == manifest["train_rows"]:
        report("PASS", "rows", f"{entry['rows']} = manifest train_rows")
    else:
        ok = report("FAIL", "rows", f"server has {entry['rows']}, manifest says {manifest['train_rows']}")
    if entry["dim"] == manifest["dimension"]:
        report("PASS", "dimension", str(entry["dim"]))
    else:
        ok = report("FAIL", "dimension", f"server has {entry['dim']}, manifest says {manifest['dimension']}")

    # 3. indexes
    vector_index = next((i for i in entry["indexes"] if i["columns"] == ["vector"]), None)
    if vector_index is None:
        report("WARN", "vector index", "none: searches are brute force; run scripts/build_index.py")
    else:
        detail = f"{vector_index['type']}, {vector_index['size_mb']} MB, metric {vector_index['metric']}"
        if vector_index["num_unindexed_rows"]:
            report("WARN", "vector index", f"{detail}; {vector_index['num_unindexed_rows']} rows not indexed yet")
        elif (vector_index["metric"] or "").lower() != ds.metric:
            ok = report("FAIL", "vector index metric", f"{detail}, but config.DATASETS says {ds.metric}")
        else:
            report("PASS", "vector index", detail)
    if not any(i["columns"] == ["id"] for i in entry["indexes"]):
        report("WARN", "id index", "no BTree on id: lookups by id scan the table")

    # 4. exact search against the ground truth
    n = min(n_queries, manifest["queries"])
    ids, queries = read_queries(ds.test_path, n)
    truth, truth_dists = read_ground_truth(ds.truth_path, n)
    if queries.shape[1] != entry["dim"]:
        return report("FAIL", "queries", f"test.csv has dimension {queries.shape[1]}, table has {entry['dim']}")
    k = min(10, truth.shape[1])
    recalls, dist_diffs, total_ms, engine_ms = [], [], [], []
    for q in range(n):
        data, ms = client.search(ds.name, queries[q], k=k, bypass_index=True)
        recalls.append(len(set(data["ids"]) & set(truth[q, :k].tolist())) / k)
        have = ~np.isnan(truth_dists[q, :k])
        if have.any() and len(data["distances"]) == k:
            dist_diffs.append(np.abs(np.array(data["distances"])[have] - truth_dists[q, :k][have]).max())
    exact_recall = float(np.mean(recalls))
    if exact_recall >= EXACT_RECALL_MIN:
        report("PASS", f"exact search vs ground truth ({n} queries)", f"recall@{k} = {exact_recall:.4f}")
    else:
        ok = report("FAIL", f"exact search vs ground truth ({n} queries)",
                    f"recall@{k} = {exact_recall:.4f}: wrong metric ({ds.metric}) or ids not matching row positions?")
    if dist_diffs:
        worst = max(dist_diffs)
        report("INFO", "distance values", f"max |server - ground truth| = {worst:.2e}" + ("" if worst < 1e-3 else " (different distance convention)"))
    else:
        report("INFO", "distance values", "ground truth has no distances")

    # 5. approximate search
    recalls = []
    for q in range(n):
        data, ms = client.search(ds.name, queries[q], k=k, nprobes=nprobes, refine_factor=refine)
        recalls.append(len(set(data["ids"]) & set(truth[q, :k].tolist())) / k)
        total_ms.append(ms)
        engine_ms.append(data["engine_latency_ms"])
    detail = (f"nprobes={nprobes} refine={refine}: recall@{k} = {np.mean(recalls):.4f}, "
              f"client {np.mean(total_ms):.1f} ms, engine {np.mean(engine_ms):.1f} ms per query")
    if vector_index is None:
        report("INFO", "approximate search", detail)
    elif np.mean(recalls) > 0:
        report("PASS", "approximate search", detail)
    else:
        ok = report("FAIL", "approximate search", detail)

    # 6. lookup by id (the CLI's goal 1)
    data, _ = client.search(ds.name, id=0, k=5, nprobes=nprobes, refine_factor=refine)
    if len(data["ids"]) == 5 and 0 not in data["ids"]:
        report("PASS", "lookup by id", f"neighbors of id 0: {data['ids']}")
    else:
        ok = report("FAIL", "lookup by id", f"expected 5 neighbors excluding id 0, got {data['ids']}")
    return ok and "FAIL" not in statuses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(parser)
    parser.add_argument("--n-queries", type=int, default=50, help="test queries used for the recall checks")
    parser.add_argument("--nprobes", type=int, default=20)
    parser.add_argument("--refine", type=int, default=5)
    args = parser.parse_args()

    client = Client()
    try:
        client.health()
    except Exception as exc:
        sys.exit(f"cannot reach the server at {SERVER_URL}: {exc}\n(start it with: docker compose up -d db)")
    client.reload()  # pick up ingest / index changes made since the server last opened its tables
    info = client.info()
    print(f"server {SERVER_URL}: lancedb {info['lancedb']}, {info['cores']} cores "
          f"({info['cores_allowed']} allowed), {info['ram_gb']} GB RAM, {info['machine']}")
    state = {d["dataset"]: d for d in client.datasets()}

    outcome = {}
    for ds in select_datasets(args.dataset):
        print(f"\n{ds.name} (metric {ds.metric})")
        try:
            outcome[ds.name] = check(client, ds, state, args.n_queries, args.nprobes, args.refine)
        except Exception as exc:  # keep going so one broken dataset does not hide the others
            print(f"  [FAIL] unexpected error: {type(exc).__name__}: {exc}")
            outcome[ds.name] = False

    print("\nsummary")
    for name, ok in outcome.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    sys.exit(0 if all(outcome.values()) else 1)


if __name__ == "__main__":
    main()
