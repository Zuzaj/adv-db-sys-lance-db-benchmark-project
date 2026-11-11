"""HTTP wrapper around the embedded LanceDB database: the service of the db container.

The database container holds train.csv and the LanceDB directory (DB_URI); clients only send queries.
Ingest and index building are not endpoints: run scripts/ingest.py and scripts/build_index.py in this
container, then call POST /reload (scripts/check_setup.py does) so cached table handles see the new data.
"""
import os
import platform
import time

import lancedb
import numpy as np
import psutil
import pyarrow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from lancedb_bench.config import DATASETS, DB_URI, get_dataset
from lancedb_bench.metrics import get_dir_size_mb, get_peak_rss_mb

app = FastAPI(title="LanceDB wrapper")
db = lancedb.connect(DB_URI)
_tables: dict = {}  # dataset name -> (table, dim)


def table_exists(name: str) -> bool:
    return get_dataset(name).table in list(db.list_tables().tables)


def get_table(name: str):
    if name not in DATASETS:
        raise HTTPException(404, f"unknown dataset {name!r}; known: {', '.join(DATASETS)}")
    if name not in _tables:
        if not table_exists(name):
            raise HTTPException(503, f"table for {name!r} not found: run scripts/ingest.py --dataset {name} in the db container")
        table = db.open_table(get_dataset(name).table)
        _tables[name] = (table, table.schema.field("vector").type.list_size)
    return _tables[name]


class SearchRequest(BaseModel):
    dataset: str
    vector: list[float] | None = None
    id: int | None = None  # alternative to `vector`: search around a stored vector, excluding itself
    k: int = 10
    nprobes: int = 20
    refine_factor: int | None = None
    bypass_index: bool = False  # exact (brute force) search


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "lancedb": lancedb.__version__,
        "pyarrow": pyarrow.__version__,
        "python": platform.python_version(),
        "machine": platform.platform(),
        "cores": psutil.cpu_count(logical=True),
        "cores_allowed": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "rss_mb": round(psutil.Process().memory_info().rss / 1e6, 1),
        "peak_rss_mb": get_peak_rss_mb(),
    }


@app.get("/datasets")
def datasets():
    """State of every known dataset in this database (opens fresh handles, so it always shows the latest version)."""
    out = []
    for name in DATASETS:
        ds = get_dataset(name)
        entry = {"dataset": name, "metric": ds.metric, "table": ds.table, "exists": table_exists(name)}
        if entry["exists"]:
            table = db.open_table(ds.table)
            entry["rows"] = table.count_rows()
            entry["dim"] = table.schema.field("vector").type.list_size
            entry["size_mb"] = get_dir_size_mb(ds.table_dir)
            entry["indexes"] = [
                {
                    "name": i.name,
                    "type": i.index_type,
                    "columns": list(i.columns),
                    "num_indexed_rows": i.num_indexed_rows,
                    "num_unindexed_rows": i.num_unindexed_rows,
                    "size_mb": round(i.size_bytes / (1024 * 1024), 1),
                    "metric": (i.index_details or {}).get("metric_type"),
                }
                for i in table.list_indices()
            ]
        out.append(entry)
    return out


@app.post("/reload")
def reload():
    """Forget cached table handles; the next search reopens the latest version."""
    _tables.clear()
    return {"status": "reloaded"}


@app.post("/search")
def search(req: SearchRequest):
    for attempt in (1, 2):
        try:
            return run_search(req)
        except HTTPException:
            raise
        except Exception:
            if attempt == 2:
                raise
            _tables.pop(req.dataset, None)  # stale handle (table rebuilt or old versions cleaned up since it was opened): reopen once


def run_search(req: SearchRequest) -> dict:
    table, dim = get_table(req.dataset)
    metric = get_dataset(req.dataset).metric
    k = req.k

    if req.vector is not None:
        vec = np.asarray(req.vector, dtype=np.float32)
    elif req.id is not None:
        rows = table.search().where(f"id = {req.id}").select(["vector"]).limit(1).to_arrow()
        if rows.num_rows == 0:
            raise HTTPException(404, f"id {req.id} not found in {req.dataset}")
        vec = np.array(rows["vector"][0].as_py(), dtype=np.float32)
        k += 1  # the stored vector finds itself at distance 0
    else:
        raise HTTPException(400, "provide `vector` or `id`")
    if vec.shape != (dim,):
        raise HTTPException(400, f"vector has {vec.size} values, {req.dataset} has dimension {dim}")

    q = table.search(vec).distance_type(metric).limit(k).select(["id", "_distance"])
    q = q.bypass_vector_index() if req.bypass_index else q.nprobes(req.nprobes)
    if req.refine_factor is not None and not req.bypass_index:
        q = q.refine_factor(req.refine_factor)

    t0 = time.perf_counter_ns()
    res = q.to_arrow()
    engine_ms = (time.perf_counter_ns() - t0) / 1e6

    ids, dists = res["id"].to_pylist(), res["_distance"].to_pylist()
    if req.id is not None:
        pairs = [(i, d) for i, d in zip(ids, dists) if i != req.id][:req.k]
        ids, dists = [p[0] for p in pairs], [p[1] for p in pairs]
    return {"engine_latency_ms": engine_ms, "ids": ids, "distances": dists}
