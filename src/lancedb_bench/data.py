"""Streaming readers for the dataset CSVs (never loads a whole file: train.csv files are up to 4 GB).

train.csv / test.csv:  id,"[v0,v1,...]"         (vector as one bracketed string)
ground_truth.csv:      query_id,rank,neighbor_id,distance
"""
from collections.abc import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

BLOCK_SIZE = 8 << 20  # bytes per CSV read block


def parse_vectors(cells: list[str], dim: int) -> np.ndarray:
    """['[0.1,0.2]', ...] -> float32 array of shape (len(cells), dim). Fails loudly on a size mismatch."""
    flat = np.fromstring(",".join(c[1:-1] for c in cells), dtype=np.float32, sep=",")
    if flat.size != len(cells) * dim:
        raise ValueError(f"expected {len(cells)} vectors x {dim} values = {len(cells) * dim}, parsed {flat.size}")
    return flat.reshape(len(cells), dim)


def stream_vectors(path, batch_rows: int = 10_000, max_rows: int | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (ids int64[n], vectors float32[n, dim]) from an `id,vector` CSV, about batch_rows rows at a time."""
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=BLOCK_SIZE),
        convert_options=pacsv.ConvertOptions(column_types={"id": pa.int64(), "vector": pa.string()}),
    )
    ids: list[np.ndarray] = []
    cells: list[str] = []
    dim = None
    emitted = 0

    def flush():
        nonlocal dim, emitted
        if dim is None:
            dim = cells[0].count(",") + 1  # dimension = number of values in the first vector
        out = np.concatenate(ids), parse_vectors(cells, dim)
        emitted += len(cells)
        ids.clear()
        cells.clear()
        return out

    for batch in reader:
        b_ids = batch.column("id").to_numpy(zero_copy_only=False)
        b_cells = batch.column("vector").to_pylist()
        if max_rows is not None:
            room = max_rows - emitted - len(cells)
            b_ids, b_cells = b_ids[:room], b_cells[:room]
        ids.append(b_ids)
        cells.extend(b_cells)
        if len(cells) >= batch_rows:
            yield flush()
        if max_rows is not None and emitted + len(cells) >= max_rows:
            break
    if cells:
        yield flush()


def read_queries(path, n: int) -> tuple[np.ndarray, np.ndarray]:
    """First n rows of test.csv as (ids, vectors)."""
    parts = list(stream_vectors(path, max_rows=n))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def read_ground_truth(path, n_queries: int) -> tuple[np.ndarray, np.ndarray]:
    """Ground truth of the first n_queries queries as (neighbors int64[n, K], distances float64[n, K]).

    neighbors[q, r] is the id of the rank-r neighbor of query q (-1 if missing); distances is NaN where the
    file has an empty cell (some datasets have no distances).
    """
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=BLOCK_SIZE),
        convert_options=pacsv.ConvertOptions(column_types={
            "query_id": pa.int64(), "rank": pa.int64(), "neighbor_id": pa.int64(), "distance": pa.float64(),
        }),
    )
    neighbors = np.full((n_queries, 0), -1, dtype=np.int64)
    distances = np.full((n_queries, 0), np.nan)
    for batch in reader:
        q = batch.column("query_id").to_numpy(zero_copy_only=False)
        keep = q < n_queries
        if not keep.any():
            continue
        q = q[keep]
        r = batch.column("rank").to_numpy(zero_copy_only=False)[keep]
        if r.max() >= neighbors.shape[1]:
            width = int(r.max()) + 1
            neighbors = np.pad(neighbors, ((0, 0), (0, width - neighbors.shape[1])), constant_values=-1)
            distances = np.pad(distances, ((0, 0), (0, width - distances.shape[1])), constant_values=np.nan)
        neighbors[q, r] = batch.column("neighbor_id").to_numpy(zero_copy_only=False)[keep]
        distances[q, r] = batch.column("distance").to_numpy(zero_copy_only=False)[keep]
    if neighbors.shape[1] == 0 or (neighbors[:, 0] < 0).any():
        raise ValueError(f"{path}: ground truth is missing for some of the first {n_queries} queries")
    return neighbors, distances
