"""Goals 1 to 8 of the class baseline (section 8), run inside the db container next to the embedded LanceDB.

Every goal returns {"lines": [...], "note": "..."}: `lines` are the result (printed to stdout by the CLI, nearest
first, as `id distance` unless stated otherwise), `note` is extra information for stderr.
Bad parameters raise InvalidParameters (CLI exit code 2); a missing precondition raises NotReady (exit code 1).
"""
import json
import time
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from lancedb.index import Bitmap

from lancedb_bench import results
from lancedb_bench.config import (DEFAULT_NPROBES, NPROBES_SETTINGS, PASSES, REFINE_FACTOR, WARMUP, Dataset)
from lancedb_bench.data import read_ground_truth, read_queries
from lancedb_bench.metrics import PeakRss, latency_percentiles, recall_at_k

MAX_RESULTS = 100  # a goal returns at most 100 results
CLUSTER_COLUMN = "cluster_id"
RECALL_TARGET = 0.95  # recall@10 every index has to reach at some setting (baseline section 3)


class InvalidParameters(ValueError):
    """The caller gave parameters outside the allowed ranges."""


class NotReady(RuntimeError):
    """A precondition is missing (for example no clustering stored yet)."""


@dataclass
class Context:
    ds: Dataset
    table: object
    rows: int
    dim: int
    max_k: int  # not more than the true neighbors per query of the dataset, and not more than 100
    test_queries: int | None  # None: unknown (no manifest)

    @classmethod
    def open(cls, db, ds: Dataset) -> "Context":
        table = db.open_table(ds.table)
        manifest = json.loads(ds.manifest_path.read_text()) if ds.manifest_path.exists() else {}
        return cls(ds, table, table.count_rows(), table.schema.field("vector").type.list_size,
                   min(MAX_RESULTS, manifest.get("neighbors_per_query", MAX_RESULTS)), manifest.get("queries"))

    def check_id(self, row_id: int, name: str = "id") -> None:
        if not 0 <= row_id < self.rows:
            raise InvalidParameters(f"{name} must be from 0 to {self.rows - 1} for {self.ds.name}, got {row_id}")

    def check_k(self, k: int) -> None:
        if not 1 <= k <= self.max_k:
            raise InvalidParameters(f"k must be from 1 to {self.max_k} for {self.ds.name}, got {k}")


def fetch_vector(ctx: Context, row_id: int) -> np.ndarray:
    ctx.check_id(row_id)
    rows = ctx.table.search().where(f"id = {row_id}").select(["vector"]).limit(1).to_arrow()
    return np.array(rows["vector"][0].as_py(), dtype=np.float32)


def search(ctx: Context, vec: np.ndarray, k: int, nprobes: int = DEFAULT_NPROBES, where: str | None = None,
           exact: bool = False):
    """One KNN search returning an Arrow table of (id, _distance). `_distance` is selected explicitly because LanceDB
    adds it anyway (and warns on every call if it is not selected)."""
    q = ctx.table.search(vec).distance_type(ctx.ds.metric).limit(k).select(["id", "_distance"])
    if where:
        q = q.where(where, prefilter=True)
    q = q.bypass_vector_index() if exact else q.nprobes(nprobes).refine_factor(REFINE_FACTOR)
    return q.to_arrow()


def neighbors_of(ctx: Context, row_id: int, k: int, where: str | None = None) -> tuple[list[int], list[float]]:
    """k nearest neighbors of a stored vector without the vector itself (ask for k + 1)."""
    res = search(ctx, fetch_vector(ctx, row_id), k + 1, where=where)
    pairs = [(i, d) for i, d in zip(res["id"].to_pylist(), res["_distance"].to_pylist()) if i != row_id]
    return [p[0] for p in pairs[:k]], [p[1] for p in pairs[:k]]


def as_lines(ids: list[int], dists: list[float]) -> list[str]:
    return [f"{i} {d:.6f}" for i, d in zip(ids, dists)]


def goal_1(ctx: Context, row_id: int, k: int) -> dict:
    """db 1 <id> <k>: the k nearest neighbors of a stored vector."""
    ctx.check_k(k)
    return {"lines": as_lines(*neighbors_of(ctx, row_id, k))}


def goal_2(ctx: Context, row_id: int, radius: float) -> dict:
    """db 2 <id> <radius>: all vectors within the distance radius of a stored vector (at most 100)."""
    if not 0 <= radius <= 2:
        raise InvalidParameters(f"radius is a cosine distance from 0 to 2, got {radius}")
    ids, dists = neighbors_of(ctx, row_id, MAX_RESULTS)
    inside = [(i, d) for i, d in zip(ids, dists) if d <= radius]
    note = f"{len(inside)} results within radius {radius}" + (" (limit of 100 reached)" if len(inside) == MAX_RESULTS else "")
    return {"lines": as_lines([p[0] for p in inside], [p[1] for p in inside]), "note": note}


def goal_3(ctx: Context, row_id: int, k: int, percent: int) -> dict:
    """db 3 <id> <k> <percent>: the k nearest neighbors among the vectors with id mod 100 < percent."""
    ctx.check_k(k)
    if not 1 <= percent <= 100:
        raise InvalidParameters(f"percent must be from 1 to 100, got {percent}")
    return {"lines": as_lines(*neighbors_of(ctx, row_id, k, where=f"id % 100 < {percent}"))}


def load_unit_vectors(ctx: Context) -> np.ndarray:
    """All vectors ordered by id (= row position) and scaled to unit length, as float32 (n x dim)."""
    arrow = ctx.table.to_arrow()
    ids = arrow["id"].to_numpy()
    vectors = arrow["vector"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(ctx.rows, ctx.dim)
    vectors = vectors[np.argsort(ids)].astype(np.float32, copy=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.where(norms == 0, 1, norms)
    return vectors


def kmeans(vectors: np.ndarray, clusters: int, iterations: int, chunk: int = 20_000) -> np.ndarray:
    """Spherical K-means: the vectors with ids 0..clusters-1 are the starting centers, the distance is cosine, each
    center becomes the mean of its members, exactly `iterations` iterations, no early stop, no random numbers.
    An empty cluster keeps its center. Returns the cluster of every vector from the last assignment."""
    n = len(vectors)
    centers = vectors[:clusters].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iterations):
        unit = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
        sums = np.zeros_like(centers)
        counts = np.zeros(clusters, dtype=np.float32)
        for s in range(0, n, chunk):
            block = vectors[s:s + chunk]
            labels[s:s + chunk] = (block @ unit.T).argmax(axis=1)  # largest cosine similarity = smallest cosine distance
            onehot = np.zeros((len(block), clusters), dtype=np.float32)
            onehot[np.arange(len(block)), labels[s:s + chunk]] = 1
            sums += onehot.T @ block
            counts += onehot.sum(axis=0)
        filled = counts > 0
        centers[filled] = sums[filled] / counts[filled, None]
    return labels


def goal_4(ctx: Context, clusters: int, iterations: int) -> dict:
    """db 4 <clusters> <iterations>: K-means outside the engine (numpy, in the db container); the cluster id of
    every vector is stored in the table (column cluster_id, bitmap index). Prints `cluster_id size` per cluster."""
    if not 1 <= clusters <= ctx.rows:
        raise InvalidParameters(f"clusters must be from 1 to {ctx.rows}, got {clusters}")
    if iterations < 1:
        raise InvalidParameters(f"iterations must be at least 1, got {iterations}")
    t0 = time.perf_counter()
    with PeakRss() as peak:
        labels = kmeans(load_unit_vectors(ctx), clusters, iterations)
        if CLUSTER_COLUMN not in ctx.table.schema.names:
            ctx.table.add_columns(pa.field(CLUSTER_COLUMN, pa.int32()))
        ctx.table.merge_insert("id").when_matched_update_all().execute(
            pa.table({"id": np.arange(ctx.rows, dtype=np.int64), CLUSTER_COLUMN: labels}))
        ctx.table.create_index(CLUSTER_COLUMN, config=Bitmap(), replace=True)
        ctx.table.optimize()  # re-indexes the rows touched by the update (the id BTree, the vector index)
    seconds = time.perf_counter() - t0
    results.set_last_value("build.csv", ctx.ds.name, "kmeans_s", round(seconds, 2))
    sizes = np.bincount(labels, minlength=clusters)
    return {"lines": [f"{c} {s}" for c, s in enumerate(sizes)],
            "note": f"K-means took {seconds:.1f} s, peak memory {peak.peak_mb:.0f} MB (numpy in the db container, result stored in the table)"}


def cluster_of(ctx: Context, row_id: int) -> int:
    ctx.check_id(row_id)
    if CLUSTER_COLUMN not in ctx.table.schema.names:
        raise NotReady("no clustering stored yet: run `db 4 <clusters> <iterations>` first")
    rows = ctx.table.search().where(f"id = {row_id}").select([CLUSTER_COLUMN]).limit(1).to_arrow()
    cluster = rows[CLUSTER_COLUMN][0].as_py()
    if cluster is None:
        raise NotReady("no clustering stored yet: run `db 4 <clusters> <iterations>` first")
    return cluster


def goal_5(ctx: Context, row_id: int) -> dict:
    """db 5 <id>: the cluster of a stored vector."""
    return {"lines": [str(cluster_of(ctx, row_id))]}


def goal_6(ctx: Context, row_id: int, k: int) -> dict:
    """db 6 <id> <k>: the k nearest neighbors inside the cluster of the vector (all of them if fewer)."""
    ctx.check_k(k)
    cluster = cluster_of(ctx, row_id)
    return {"lines": as_lines(*neighbors_of(ctx, row_id, k, where=f"{CLUSTER_COLUMN} = {cluster}"))}


def goal_7(ctx: Context, id_a: int, id_b: int) -> dict:
    """db 7 <id_a> <id_b>: the cosine distance between two stored vectors, computed from the stored vectors."""
    a, b = fetch_vector(ctx, id_a).astype(np.float64), fetch_vector(ctx, id_b).astype(np.float64)
    distance = 1 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return {"lines": [f"{distance:.8f}"]}


def run_setting(ctx: Context, queries: np.ndarray, truth: np.ndarray, k: int, nprobes: int,
                warmup: int = WARMUP, passes: int = PASSES) -> tuple[float, list[float]]:
    """Warm up, then `passes` timed passes over the queries, one query at a time. The timed region is the search call
    (building the query, running it, getting the Arrow result); converting ids to Python happens outside it.
    Returns (recall@k from the first pass, all latencies in ms)."""
    for vec in queries[:warmup]:
        search(ctx, vec, k, nprobes)
    latencies, predicted = [], []
    for p in range(passes):
        for vec in queries:
            t0 = time.perf_counter_ns()
            res = search(ctx, vec, k, nprobes)
            latencies.append((time.perf_counter_ns() - t0) / 1e6)
            if p == 0:
                predicted.append(res["id"].to_pylist())
    return recall_at_k(predicted, truth, k), latencies


def goal_8(ctx: Context, n_queries: int, k: int) -> dict:
    """db 8 <n_queries> <k>: the benchmark query. For every accuracy setting (nprobes) it runs the first n test
    queries with the protocol of baseline section 3 and appends one row per setting to /results/search.csv."""
    ctx.check_k(k)
    if not 1 <= n_queries or (ctx.test_queries and n_queries > ctx.test_queries):
        raise InvalidParameters(f"n_queries must be from 1 to {ctx.test_queries or 'the number of test queries'}, got {n_queries}")
    queries = read_queries(ctx.ds.test_path, n_queries)[1]
    truth = read_ground_truth(ctx.ds.truth_path, len(queries))[0]

    rows, lines = [], []
    with PeakRss() as peak:
        for nprobes in NPROBES_SETTINGS:
            recall, latencies = run_setting(ctx, queries, truth, k, nprobes)
            stats = latency_percentiles(latencies)
            rows.append({**results.identity(ctx.ds.name), "k": k, "setting_name": "nprobes", "setting_value": nprobes,
                         "recall": recall, **stats})
            lines.append(f"nprobes={nprobes} recall@{k}={recall} p50={stats['p50_ms']} ms p95={stats['p95_ms']} ms p99={stats['p99_ms']} ms")
    results.append_rows("search.csv", results.SEARCH_COLUMNS, rows)

    note = f"{len(queries)} queries x {PASSES} passes per setting after {WARMUP} warm-up queries, refine_factor={REFINE_FACTOR}; peak memory {peak.peak_mb:.0f} MB; rows appended to search.csv"
    if k == 10:
        reaching = [r for r in rows if r["recall"] >= RECALL_TARGET]
        note += (f"; headline: p50 {reaching[0]['p50_ms']} ms at nprobes={reaching[0]['setting_value']} (lowest setting with recall@10 >= {RECALL_TARGET})"
                 if reaching else f"; no setting reaches recall@10 >= {RECALL_TARGET}")
    return {"lines": lines, "note": note}


GOAL_FUNCTIONS = {1: goal_1, 2: goal_2, 3: goal_3, 4: goal_4, 5: goal_5, 6: goal_6, 7: goal_7, 8: goal_8}
