"""DB side: check that the whole setup works, per dataset, directly against the database in the db container:

  docker compose run --rm populate python scripts/check_setup.py        (dataset from $DATASET, or --dataset all)

Verifies, in order: the dataset files exist; the table matches the manifest (row count, dimension); the vector index
exists, covers all rows and uses the metric from config.DATASETS; exact search reproduces the ground truth (this
catches a wrong metric or id mapping); approximate search works; goals 1, 3 and 7 give sensible answers.
Exit code is 1 if anything FAILs.
"""
import argparse
import json
import sys

import lancedb
import numpy as np

from lancedb_bench.config import DB_URI, add_dataset_arg, select_datasets
from lancedb_bench.data import read_ground_truth, read_queries
from lancedb_bench.goals import Context, goal_1, goal_3, goal_7, search

EXACT_RECALL_MIN = 0.98  # float32 ties and rounding can cost a little against the float64 ground truth


def check(db, ds, n_queries: int) -> bool:
    ok = True

    def report(status: str, label: str, detail: str = "") -> bool:
        print(f"  [{status:4}] {label}" + (f": {detail}" if detail else ""))
        return status != "FAIL"

    missing = [p.name for p in (ds.train_path, ds.test_path, ds.truth_path, ds.manifest_path) if not p.exists()]
    if missing:
        return report("FAIL", "dataset files", f"missing in {ds.dir}: {', '.join(missing)} (scripts/download.py --dataset {ds.name})")
    report("PASS", "dataset files", "train.csv, test.csv, ground_truth.csv, manifest.json present")
    manifest = json.loads(ds.manifest_path.read_text())

    if ds.table not in list(db.list_tables().tables):
        return report("FAIL", "table", "not in the database: run `docker compose run --rm populate`")
    ctx = Context.open(db, ds)
    ok &= report(*(("PASS", "rows", str(ctx.rows)) if ctx.rows == manifest["train_rows"] else ("FAIL", "rows", f"table has {ctx.rows}, manifest says {manifest['train_rows']}")))
    ok &= report(*(("PASS", "dimension", str(ctx.dim)) if ctx.dim == manifest["dimension"] else ("FAIL", "dimension", f"table has {ctx.dim}, manifest says {manifest['dimension']}")))

    indexes = ctx.table.list_indices()
    vector_index = next((i for i in indexes if list(i.columns) == ["vector"]), None)
    if vector_index is None:
        report("WARN", "vector index", "none: searches are brute force; run populate")
    else:
        metric = ((vector_index.index_details or {}).get("metric_type") or "").lower()
        detail = f"{vector_index.index_type}, {vector_index.size_bytes / 1048576:.1f} MB, metric {metric}"
        if vector_index.num_unindexed_rows:
            report("WARN", "vector index", f"{detail}; {vector_index.num_unindexed_rows} rows not indexed yet")
        elif metric != ds.metric:
            ok &= report("FAIL", "vector index metric", f"{detail}, but config.DATASETS says {ds.metric}")
        else:
            report("PASS", "vector index", detail)

    n = min(n_queries, manifest["queries"])
    queries = read_queries(ds.test_path, n)[1]
    truth, truth_dists = read_ground_truth(ds.truth_path, n)
    k = min(10, ctx.max_k)
    recalls, diffs = [], []
    for q in range(n):
        res = search(ctx, queries[q], k, exact=True)
        recalls.append(len(set(res["id"].to_pylist()) & set(truth[q, :k].tolist())) / k)
        have = ~np.isnan(truth_dists[q, :k])
        if have.any():
            diffs.append(np.abs(np.array(res["_distance"].to_pylist())[have] - truth_dists[q, :k][have]).max())
    if np.mean(recalls) >= EXACT_RECALL_MIN:
        report("PASS", f"exact search vs ground truth ({n} queries)", f"recall@{k} = {np.mean(recalls):.4f}")
    else:
        ok &= report("FAIL", f"exact search vs ground truth ({n} queries)",
                     f"recall@{k} = {np.mean(recalls):.4f}: wrong metric ({ds.metric}) or ids not matching row positions?")
    report("INFO", "distance values", f"max |db - ground truth| = {max(diffs):.2e}" if diffs else "ground truth has no distances")

    approx = [len(set(search(ctx, queries[q], k)["id"].to_pylist()) & set(truth[q, :k].tolist())) / k for q in range(n)]
    ok &= report("PASS" if np.mean(approx) > 0 or vector_index is None else "FAIL", "approximate search", f"recall@{k} = {np.mean(approx):.4f}")

    lines = goal_1(ctx, 0, 5)["lines"]
    ok &= report(*(("PASS", "goal 1 (id 0, k=5)", f"{len(lines)} neighbors, first: {lines[0]}") if len(lines) == 5 and not lines[0].startswith("0 ") else ("FAIL", "goal 1", f"got {lines}")))
    ids = [int(x.split()[0]) for x in goal_3(ctx, 0, 5, 10)["lines"]]
    ok &= report(*(("PASS", "goal 3 (percent 10)", f"ids {ids}") if ids and all(i % 100 < 10 for i in ids) else ("FAIL", "goal 3", f"ids {ids} break the filter")))
    d = float(goal_7(ctx, 0, ctx.rows - 1)["lines"][0])
    ok &= report(*(("PASS", "goal 7 (0, last id)", f"cosine distance {d:.6f}") if 0 <= d <= 2 else ("FAIL", "goal 7", f"distance {d}")))
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(parser)
    parser.add_argument("--n-queries", type=int, default=50, help="test queries used for the recall checks")
    args = parser.parse_args()

    db = lancedb.connect(DB_URI)
    outcome = {}
    for ds in select_datasets(args.dataset):
        print(f"\n{ds.name} (metric {ds.metric})")
        try:
            outcome[ds.name] = check(db, ds, args.n_queries)
        except Exception as exc:  # keep going so one broken dataset does not hide the others
            print(f"  [FAIL] unexpected error: {type(exc).__name__}: {exc}")
            outcome[ds.name] = False
    print("\nsummary")
    for name, ok in outcome.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    sys.exit(0 if all(outcome.values()) else 1)


if __name__ == "__main__":
    main()
