"""DB side: build the IVF_PQ vector index (metric from config.DATASETS) and a BTree on id, then optimize and report sizes."""
import argparse
import time
from datetime import timedelta

import lancedb
from lancedb.index import BTree, IvfPq

from lancedb_bench.config import DB_URI, add_dataset_arg, index_defaults, select_datasets
from lancedb_bench.metrics import get_dir_size_mb, get_peak_rss_mb


def build(db, ds, partitions: int | None, sub_vectors: int | None) -> None:
    table = db.open_table(ds.table)
    n, dim = table.count_rows(), table.schema.field("vector").type.list_size
    default_partitions, default_sub_vectors = index_defaults(n, dim)
    partitions, sub_vectors = partitions or default_partitions, sub_vectors or default_sub_vectors
    if dim % sub_vectors:
        raise SystemExit(f"num_sub_vectors={sub_vectors} must divide dim={dim}")

    print(f"{ds.name}: {n} rows, dim {dim}, metric {ds.metric}; table size before index {get_dir_size_mb(ds.table_dir):.0f} MB")
    t0 = time.perf_counter()
    table.create_index(
        "vector",
        config=IvfPq(distance_type=ds.metric, num_partitions=partitions, num_sub_vectors=sub_vectors),
        replace=True,
    )
    print(f"  IVF_PQ built in {time.perf_counter() - t0:.1f}s (partitions={partitions}, sub_vectors={sub_vectors})")
    t0 = time.perf_counter()
    table.create_index("id", config=BTree(), replace=True)
    print(f"  BTree on id built in {time.perf_counter() - t0:.1f}s")

    table.optimize(cleanup_older_than=timedelta(0))  # also drops old versions, so the size below is the live size
    print(f"  table size after optimize: {get_dir_size_mb(ds.table_dir):.0f} MB, peak RSS {get_peak_rss_mb():.0f} MB")
    for i in table.list_indices():
        print(f"  index {i.name}: {i.index_type}, {i.size_bytes / 1048576:.1f} MB, indexed rows {i.num_indexed_rows}")
    print("  the running server keeps old table handles until POST /reload (scripts/check_setup.py and bench.py call it)")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_dataset_arg(parser)
    parser.add_argument("--partitions", type=int, help="default: see config.index_defaults")
    parser.add_argument("--sub-vectors", type=int, help="default: see config.index_defaults")
    args = parser.parse_args()
    db = lancedb.connect(DB_URI)
    for ds in select_datasets(args.dataset):
        build(db, ds, args.partitions, args.sub_vectors)


if __name__ == "__main__":
    main()
