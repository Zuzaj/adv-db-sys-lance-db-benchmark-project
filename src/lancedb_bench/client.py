"""HTTP client for the db container's wrapper (server.py)."""
import time

import httpx
import numpy as np

from lancedb_bench.config import SERVER_URL


class Client:
    def __init__(self, base_url: str = SERVER_URL, timeout: float = 300.0):
        self.http = httpx.Client(base_url=base_url, timeout=timeout)

    def _get(self, path: str) -> dict:
        resp = self.http.get(path)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        return self._get("/health")

    def info(self) -> dict:
        return self._get("/info")

    def datasets(self) -> list[dict]:
        return self._get("/datasets")

    def reload(self) -> dict:
        resp = self.http.post("/reload")
        resp.raise_for_status()
        return resp.json()

    def search(self, dataset: str, vector: np.ndarray | None = None, *, id: int | None = None, k: int = 10,
               nprobes: int = 20, refine_factor: int | None = None, bypass_index: bool = False) -> tuple[dict, float]:
        """One search. Returns (response, total_ms). total_ms is the client wall-clock for the whole call:
        request serialization, network round trip, and response parsing. response["engine_latency_ms"] is the
        server's own time around the LanceDB query, so total_ms - engine_latency_ms is the wrapper overhead."""
        payload = {"dataset": dataset, "k": k, "nprobes": nprobes, "refine_factor": refine_factor,
                   "bypass_index": bypass_index, "id": id}
        t0 = time.perf_counter_ns()
        if vector is not None:
            payload["vector"] = vector.tolist()
        resp = self.http.post("/search", json=payload)
        data = resp.json()
        total_ms = (time.perf_counter_ns() - t0) / 1e6
        if resp.status_code != 200:
            raise RuntimeError(f"search failed ({resp.status_code}): {data.get('detail', data)}")
        return data, total_ms
