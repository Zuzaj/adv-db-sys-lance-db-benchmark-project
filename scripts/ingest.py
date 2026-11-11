"""DB side: stream DATA_DIR/<dataset>/train.csv into its LanceDB table, then check row count and stored vectors."""
import argparse
import itertools
import time

import lancedb
import numpy as np
import pyarrow as pa

from lancedb_bench.config import DB_URI, add_dataset_arg, select_datasets
from lancedb_bench.data import stream_vectors
from lancedb_bench.metrics import get_dir_size_mb, get_peak_rss_mb


def ingest(db, ds) -> None:
    if not ds.train_path.exists():
        raise SystemExit(f"{ds.train_path} not found: run scripts/download.py --dataset {ds.name} (or put train.csv there)")

    stream = stream_vectors(ds.train_path)
    first = next(stream)
    dim = first[1].shape[1]
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])

    rows = 0
    samples: dict[int, np.ndarray] = {}  # a few vectors kept for the spot check
    last: tuple[int, np.ndarray] | None = None

    def batches():
        nonlocal rows, last
        for i, (ids, vecs) in enumerate(itertools.chain([first], stream)):
            if not np.array_equal(ids, np.arange(rows, rows + len(ids))):
                raise ValueError(f"ids must be the 0-based row position (ground truth refers to them); the batch starting at row {rows} differs")
            if i == 0 or (i % 7 == 3 and len(samples) < 4):
                samples[int(ids[0])] = vecs[0].copy()
            last = (int(ids[-1]), vecs[-1].copy())
            rows += len(ids)
            yield pa.RecordBatch.from_arrays(
                [pa.array(ids), pa.FixedSizeListArray.from_arrays(pa.array(vecs.ravel(), type=pa.float32()), dim)],
                schema=schema,
            )

    t0 = time.perf_counter()
    table = db.create_table(ds.table, data=batches(), schema=schema, mode="overwrite")
    seconds = time.perf_counter() - t0

    samples[last[0]] = last[1]
    assert table.count_rows() == rows, f"table has {table.count_rows()} rows, CSV had {rows}"
    for row_id, expected in samples.items():
        got = table.search().where(f"id = {row_id}").select(["vector"]).limit(1).to_arrow()
        assert np.array_equal(np.array(got["vector"][0].as_py(), dtype=np.float32), expected), f"vector mismatch for id {row_id}"

    print(f"{ds.name}: {rows} rows x {dim} dims into table {ds.table} in {seconds:.1f}s "
          f"(peak RSS {get_peak_rss_mb():.0f} MB, table size {get_dir_size_mb(ds.table_dir):.0f} MB)")
    print(f"  checks passed: row count, ids are row positions, {len(samples)} stored vectors equal the CSV")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_dataset_arg(parser)
    args = parser.parse_args()
    db = lancedb.connect(DB_URI)
    for ds in select_datasets(args.dataset):
        ingest(db, ds)


if __name__ == "__main__":
    main()
