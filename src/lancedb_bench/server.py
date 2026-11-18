"""HTTP wrapper around the embedded LanceDB database: the service of the db container.

The db container holds all dataset files (mounted read-only as /data), the LanceDB directory (DB_URI) and the results
directory. The client container sends goals to POST /goal/{n} (`db <goal> <params...>`); the goals, including the
timed benchmark (goal 8), run here next to the database, so the measured times do not include the client-server hop.
Populating (scripts/populate.py) runs in the db image, not through HTTP; it calls POST /reload afterwards so cached
table handles see the new data.
"""
import os
import platform
from typing import Any

import lancedb
import psutil
import pyarrow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from lancedb_bench.config import DATASETS, DB_URI, get_dataset
from lancedb_bench.goals import GOAL_FUNCTIONS, Context, InvalidParameters, NotReady
from lancedb_bench.metrics import dir_size_mb
from lancedb_bench.spec import parse_params

app = FastAPI(title="LanceDB wrapper")
db = lancedb.connect(DB_URI)
_contexts: dict[str, Context] = {}  # dataset name -> open table and its facts


def table_exists(name: str) -> bool:
    return get_dataset(name).table in list(db.list_tables().tables)


def get_context(name: str) -> Context:
    if name not in DATASETS:
        raise HTTPException(422, f"unknown dataset {name!r}; known: {', '.join(DATASETS)}")
    if name not in _contexts:
        if not table_exists(name):
            raise HTTPException(503, f"table for {name!r} not found: run `docker compose run --rm populate` with DATASET={name}")
        _contexts[name] = Context.open(db, get_dataset(name))
    return _contexts[name]


class GoalRequest(BaseModel):
    dataset: str
    params: list[Any] = []


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
            entry["size_mb"] = dir_size_mb(ds.table_dir)
            entry["indexes"] = [
                {"name": i.name, "type": i.index_type, "columns": list(i.columns), "num_unindexed_rows": i.num_unindexed_rows,
                 "size_mb": round(i.size_bytes / (1024 * 1024), 1), "metric": (i.index_details or {}).get("metric_type")}
                for i in table.list_indices()
            ]
        out.append(entry)
    return out


@app.post("/reload")
def reload():
    """Forget cached table handles; the next goal reopens the latest version."""
    _contexts.clear()
    return {"status": "reloaded"}


@app.post("/goal/{goal}")
def run_goal(goal: int, req: GoalRequest):
    """Run one goal. 422: invalid parameters, 409: a precondition is missing (no clustering yet), 503: no table."""
    try:
        values = parse_params(goal, req.params)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    for attempt in (1, 2):
        ctx = get_context(req.dataset)
        try:
            out = GOAL_FUNCTIONS[goal](ctx, *values)
            break
        except InvalidParameters as exc:
            raise HTTPException(422, str(exc))
        except NotReady as exc:
            raise HTTPException(409, str(exc))
        except Exception:
            if attempt == 2:
                raise
            _contexts.pop(req.dataset, None)  # stale handle (table rebuilt or old versions cleaned up since it was opened): reopen once
    if goal == 4:
        _contexts.pop(req.dataset, None)  # the clustering added a column and a table version
    return out
