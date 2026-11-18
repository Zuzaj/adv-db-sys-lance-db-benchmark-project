"""DB side: upload everything into the database in one command (`docker compose run --rm populate`).

For each dataset, in the order of the baseline (section 6): load train.csv, build the vector index and the id index,
check. It records ONE ingestion + index time (loading and indexing are not split), the database size and the peak
memory (RSS sampled every 100 ms) in /results/build.csv, and checks that the index reaches recall@10 of at least 0.95
at some nprobes setting (baseline section 3).

Dataset: $DATASET (default glove-200-angular), or `--dataset NAME`, or `all`.
It continues with the next dataset if one fails and exits with 1 at the end, so an automated run can detect the failure.
At the end it asks the running server (SERVER_URL) to reopen its tables.
"""
import argparse
import sys
import urllib.request

import lancedb

from build_index import build
from ingest import ingest
from lancedb_bench import results
from lancedb_bench.config import DB_URI, NPROBES_SETTINGS, SERVER_URL, add_dataset_arg, select_datasets
from lancedb_bench.goals import RECALL_TARGET, Context, run_setting
from lancedb_bench.data import read_ground_truth, read_queries
from lancedb_bench.metrics import PeakRss, dir_size_mb

RECALL_CHECK_QUERIES = 200


def check_index_recall(db, ds) -> None:
    """The index must reach recall@10 >= 0.95 at some nprobes setting; otherwise tune it and measure again."""
    ctx = Context.open(db, ds)
    queries = read_queries(ds.test_path, RECALL_CHECK_QUERIES)[1]
    truth = read_ground_truth(ds.truth_path, len(queries))[0]
    recalls = {n: run_setting(ctx, queries, truth, 10, n, warmup=0, passes=1)[0] for n in NPROBES_SETTINGS}
    print("  index recall@10 on {} test queries: ".format(len(queries)) + ", ".join(f"nprobes={n}: {r}" for n, r in recalls.items()))
    if max(recalls.values()) < RECALL_TARGET:
        print(f"  WARNING: no setting reaches recall@10 >= {RECALL_TARGET}: tune the index (--partitions, --sub-vectors) and populate again")


def populate(db, ds, partitions: int | None, sub_vectors: int | None) -> None:
    with PeakRss() as peak:
        seconds = ingest(db, ds) + build(db, ds, partitions, sub_vectors)
    check_index_recall(db, ds)
    row = {**results.identity(ds.name), "ingest_s": round(seconds, 2), "db_mb": dir_size_mb(ds.table_dir),
           "peak_mem_mb": round(peak.peak_mb), **results.machine(), "kmeans_s": ""}
    results.append_rows("build.csv", results.BUILD_COLUMNS, [row])
    print(f"  ingestion + index {seconds:.1f}s, database {row['db_mb']} MB, peak memory {row['peak_mem_mb']} MB -> build.csv")


def reload_server() -> None:
    """Ask the server to reopen its tables."""
    try:
        urllib.request.urlopen(urllib.request.Request(f"{SERVER_URL}/reload", method="POST"), timeout=10)
        print("server reloaded")
    except OSError:
        print(f"no server answered on {SERVER_URL}, nothing to reload")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(parser)
    parser.add_argument("--partitions", type=int, help="IVF partitions (default: see config.index_defaults)")
    parser.add_argument("--sub-vectors", type=int, help="PQ sub-vectors (default: see config.index_defaults)")
    args = parser.parse_args()

    db = lancedb.connect(DB_URI)
    done, failed = [], []
    for ds in select_datasets(args.dataset):
        try:
            populate(db, ds, args.partitions, args.sub_vectors)
            done.append(ds.name)
        except (Exception, SystemExit) as exc:  # ingest/build stop with SystemExit on missing files or bad parameters
            print(f"{ds.name}: FAILED: {exc}")
            failed.append(ds.name)

    print(f"\npopulated: {', '.join(done) or 'none'}")
    if failed:
        print(f"failed: {', '.join(failed)}")
    reload_server()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
