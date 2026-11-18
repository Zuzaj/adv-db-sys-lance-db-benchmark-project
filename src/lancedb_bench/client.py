"""HTTP client for the db container's wrapper (server.py). It only sends goals; it holds no data."""
import httpx

from lancedb_bench.config import SERVER_URL


class InvalidParameters(ValueError):
    """The server rejected the parameters of a goal (HTTP 422)."""


class Client:
    def __init__(self, base_url: str = SERVER_URL):
        self.http = httpx.Client(base_url=base_url, timeout=None)  # goal 8 runs for minutes

    @staticmethod
    def _check(resp: httpx.Response) -> dict:
        if resp.status_code == 200:
            return resp.json()
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        if resp.status_code == 422:
            raise InvalidParameters(detail)
        raise RuntimeError(f"{resp.request.method} {resp.request.url.path} failed ({resp.status_code}): {detail}")

    def health(self) -> dict:
        return self._check(self.http.get("/health"))

    def info(self) -> dict:
        return self._check(self.http.get("/info"))

    def datasets(self) -> list[dict]:
        return self._check(self.http.get("/datasets"))

    def reload(self) -> dict:
        return self._check(self.http.post("/reload"))

    def goal(self, goal: int, dataset: str, params: list) -> dict:
        """Run one goal in the db container. Returns {"lines": [...], "note": "..."}."""
        return self._check(self.http.post(f"/goal/{goal}", json={"dataset": dataset, "params": params}))
