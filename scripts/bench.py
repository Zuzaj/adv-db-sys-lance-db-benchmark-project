"""Client side: run the benchmark goal (db 8) for k = 10 and 100 on one or all datasets (`docker compose run --rm benchmark`).

The timing happens in the db container, next to the database (baseline section 3); this script only starts it and prints
the summary. Rows are appended to /results/search.csv in the db container: one per (dataset, k, accuracy setting).
Dataset: $DATASET (default glove-200-angular), or `--dataset NAME`, or `all`. A k above the true neighbors per
query of a dataset (10 for uform-wiki-1m) is rejected by the server and reported as skipped.
"""
import argparse
import sys

from lancedb_bench.client import Client, InvalidParameters
from lancedb_bench.config import add_dataset_arg, select_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(parser)
    parser.add_argument("--n-queries", type=int, default=1000, help="first N test queries (default 1000)")
    parser.add_argument("-k", type=int, nargs="+", default=[10, 100])
    args = parser.parse_args()

    client = Client()
    client.reload()
    failed = False
    for ds in select_datasets(args.dataset):
        for k in args.k:
            print(f"\n{ds.name}, k={k}")
            try:
                out = client.goal(8, ds.name, [args.n_queries, k])
            except InvalidParameters as exc:
                print(f"  skipped: {exc}")
                continue
            except RuntimeError as exc:
                print(f"  FAILED: {exc}")
                failed = True
                continue
            print("\n".join(f"  {line}" for line in out["lines"]))
            print(f"  ({out['note']})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
