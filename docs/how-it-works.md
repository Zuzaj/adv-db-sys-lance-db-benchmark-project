# How the code works

This document walks through the whole project: what each file does, what happens when you run a command, and why it
was written that way. It describes the code as it is now (two containers, six datasets, CSV import from Hugging Face,
goals 1-8 of the class baseline). The README has the commands; this file explains what is behind them.

## 1. The big picture

```
   Hugging Face repo zuzaj/ADB: <dataset>/{train.csv, test.csv, ground_truth.csv, manifest.json}
                |
                |  scripts/download.py  (on the host, or the files are already on the server)
                v
        data/<dataset>/{train.csv, test.csv, ground_truth.csv, manifest.json}
                |
                | mounted read-only as /data
                v
 +----------------------------------------+   HTTP :8000   +--------------------------------+
 | db container (lancedb_server)          | <------------- | benchmark container (client)   |
 |  server.py (FastAPI) -> goals.py       |   one goal     |  db <goal> <params...> (cli.py)|
 |  scripts/populate.py (same image, once)| -------------> |  scripts/bench.py              |
 |  LanceDB in /db, results in /results   |   result lines |  no data, no database engine   |
 +----------------------------------------+                +--------------------------------+
```

Four ideas shape the design:

1. **The database and the client are separate containers**, even though LanceDB is an embedded library. LanceDB only runs
   inside some process, so that process is a small HTTP service (`server.py`). This was agreed in class for all databases.
2. **All dataset files are in the db container** (read-only), so every group works with identical train, test and
   ground-truth files. The test queries and the ground truth stay in files; only the train vectors are loaded into the
   database. The client holds no data and no database engine: it only sends goals.
3. **The goals and the benchmark run inside the db container**, next to the database. Each search call is timed there, so the
   measured latency does not include the client-server hop (baseline section 3).
4. **Everything about a dataset comes from its files plus one line in a registry.** Dimension and row count are read from
   the files; only the distance metric must be declared, because the CSVs do not say which one the ground truth uses.

## 2. Repository map

| File | Runs in | What it does |
|---|---|---|
| `src/lancedb_bench/config.py` | both | Paths from environment variables, the dataset registry (name to metric), `Dataset` paths, search settings, `index_defaults()`, `INDEX_PARAMS` |
| `src/lancedb_bench/spec.py` | both | The eight goals, their parameters and types, the usage text |
| `src/lancedb_bench/data.py` | db | Streaming readers for `train.csv`, `test.csv`, `ground_truth.csv` |
| `src/lancedb_bench/metrics.py` | db | Recall@k, latency percentiles, folder size, `PeakRss` (memory sampling) |
| `src/lancedb_bench/results.py` | db | The two result files `build.csv` and `search.csv` |
| `src/lancedb_bench/goals.py` | db | Goals 1-8: queries, K-means, the timed benchmark |
| `src/lancedb_bench/server.py` | db | FastAPI service around LanceDB |
| `src/lancedb_bench/client.py` | client | HTTP client |
| `src/lancedb_bench/cli.py` | client | The `db <goal> <params...>` command with exit codes |
| `scripts/download.py` | host | Downloads the datasets from Hugging Face into `data/<dataset>/` |
| `scripts/populate.py` | db | Uploads everything: `ingest.py`, `build_index.py`, recall check, one row in `build.csv` |
| `scripts/ingest.py` | db | `train.csv` into a LanceDB table |
| `scripts/build_index.py` | db | IVF_PQ vector index and BTree index |
| `scripts/check_setup.py` | db | Verifies the whole setup, per dataset |
| `scripts/bench.py` | client | Runs goal 8 for k = 10 and 100 and prints the summary |
| `Dockerfile`, `docker-compose.yml` | host | The images and the three services |

## 3. Configuration and the dataset registry (`config.py`)

**Paths** come from environment variables with defaults for running on the host:

| Variable | Default | In the containers |
|---|---|---|
| `DATA_DIR` | `./data` | `/data` |
| `DB_URI` | `./db` | `/db` |
| `RESULTS_DIR` | `./results` | `/results` |
| `SERVER_URL` | `http://localhost:8000` | `http://db:8000` (client container and populate) |
| `DATASET` | `glove-200-angular` | passed through from the host if set; `all` means every dataset |
| `GROUP` | `07-strycharz-jarlaczynska` | name of the group in the result files |
| `NPROBES_SETTINGS` | `5,10,20,50,100,200` | the `nprobes` values swept by goal 8 |
| `REFINE_FACTOR`, `NPROBES` | 5, 20 | fixed `refine_factor` (all goals) and `nprobes` for goals 1-7 |

**The registry** `DATASETS` maps a dataset name (the folder name) to a LanceDB distance type: `cosine`, `dot` or `l2`.
The CSVs carry no metric, and using the wrong one gives wrong answers, so this is the one place that decides it. The
current entries are all cosine (the other types are supported but unused). `check_setup.py` verifies each entry against
the ground truth.

**`Dataset`** is a small frozen dataclass built from a name. It derives everything else:

- `train_path`, `test_path`, `truth_path`, `manifest_path` under `DATA_DIR/<name>/`
- `table`: the LanceDB table name, the dataset name with every non-word character replaced by `_`
  (`cohere-small-100k` becomes `cohere_small_100k`)
- `table_dir`: where that table lives on disk, `DB_URI/<table>.lance` (used to report the size of one table)
- `metric`: looked up in the registry; an unknown name stops with an error listing the known datasets

`get_dataset(name)` returns a `Dataset` (the name defaults to `$DATASET`, then to `glove-200-angular`, the common default of the baseline).
`select_datasets("all")` (or `DATASET=all`) returns every registry dataset. Every script offers `--dataset <name or all>`.

**`index_defaults(n, dim)`** picks the starting IVF_PQ parameters:

- partitions: about 2,300 vectors per partition, rounded to a power of two (`2 ** round(log2(n / 2300))`)
- sub-vectors: `dim // 8`, lowered until it divides `dim` (product quantization needs that)

Examples: 100,000 rows with 768 dimensions give 32 partitions and 96 sub-vectors; 1,183,514 rows with 200 dimensions give
512 and 25. These are starting points, not tuned values; `populate.py --partitions N --sub-vectors M` overrides them.

## 4. Reading the CSV files (`data.py`)

`ingest.py` uses these readers for `train.csv`; goal 8, `populate.py` and `check_setup.py` use them for `test.csv` and
`ground_truth.csv`. All of them run in the db container.

The files are large (up to 5.2 GB), so nothing is read whole. All readers use `pyarrow.csv.open_csv`, which returns the
file in blocks of 8 MiB.

**Formats**

- `train.csv`, `test.csv`: `id,"[v0,v1,...]"`, one vector per row as a bracketed string in a quoted cell.
- `ground_truth.csv`: `query_id,rank,neighbor_id,distance`, one row per (query, rank).

**`parse_vectors(cells, dim)`** turns a list of strings like `"[0.1,0.2,...]"` into a float32 matrix. It strips the
brackets, joins all rows with commas into one string and parses it with `numpy.fromstring(..., sep=",")`, which is much
faster than parsing row by row. It then checks that exactly `rows x dim` numbers came out and raises an error if not,
so a malformed row cannot silently shift the data.

**`stream_vectors(path, batch_rows=10_000, max_rows=None)`** is a generator that yields `(ids, vectors)` batches of about
10,000 rows:

1. It reads blocks, forcing the `id` column to int64 and the `vector` column to a string.
2. It collects rows until it has `batch_rows`, then parses them (`flush`).
3. The dimension is not configured anywhere: it is the number of commas in the first vector plus one.
4. With `max_rows` it stops early, so reading the first 50 queries of a 350 MB file takes a moment.

**`read_queries(path, n)`** returns the first `n` test queries (ids and vectors).

**`read_ground_truth(path, n_queries)`** returns two matrices of shape `(n_queries, K)`:
`neighbors[q, r]` is the id of the rank-`r` neighbor of query `q`, and `distances[q, r]` its distance. `K` is not known in
advance (1000, 100 or 10 depending on the dataset), so the matrices start with zero columns and grow when a larger rank
appears. Empty distance cells become `NaN` (some datasets have no distances, and 9 `nytimes` queries have missing ones).
Only rows of the first `n_queries` queries are kept, although the whole file is still scanned. It raises an error if any
of those queries has no neighbors.

## 5. Getting the data (`scripts/download.py`)

The six datasets were prepared once and published in the Hugging Face dataset repo `zuzaj/ADB` (`HF_REPO` in
`config.py`, overridable by an environment variable): one folder per dataset with `train.csv`, `test.csv`,
`ground_truth.csv` and `manifest.json` (`DATASET_FILES` in `config.py`). The script puts them in the same layout under
`--dest` (default `./data`):

```
data/<dataset>/{train.csv, test.csv, ground_truth.csv, manifest.json}
```

For each file it skips the download if the file already exists; otherwise it calls
`huggingface_hub.hf_hub_download(..., local_dir=<dest>)`, which keeps the repo's `<dataset>/<file>` path, resumes
interrupted downloads and never leaves a partial file at the final path. `--dataset all` downloads all six (about 18 GB).

The script runs on the host. The db container mounts `./data` **read-only** as `/data`, so nothing inside the container can
change the data. On the test server the files are already in its filesystem and this step is not needed.

## 6. The database side

In practice you run `docker compose run --rm populate` (the db image, run once with the same limits). It runs
`scripts/populate.py` for `$DATASET` (default `glove-200-angular`, or `all`). For each dataset it:

1. calls `ingest()` (6.1) and `build()` (6.2) inside a `PeakRss` block and adds the two times: loading and indexing are
   **one number**, `ingest_s`, because some systems index while loading and the baseline does not split them;
2. checks that the index reaches recall@10 of at least 0.95 at some `nprobes` setting on the first 200 test queries
   (printed; a warning if no setting reaches it, it does not fail the run);
3. appends one row to `/results/build.csv` (`ingest_s`, `db_mb` after `optimize()`, `peak_mem_mb` from the sampler, CPU,
   cores, RAM, date; `kmeans_s` stays empty until goal 4 runs).

If a dataset fails (a missing file, ids that are not the row position, a bad parameter) it prints the reason and goes on with
the next one. At the end it prints which datasets were populated and which failed, asks the running server
(`SERVER_URL`) to reload its tables (`POST /reload`), and exits with 1 if anything failed, so an automated run can detect it.
`ingest.py` and `build_index.py` can still be run separately, for example to rebuild only an index with other parameters.

**Peak memory** (`metrics.PeakRss`): a thread samples the RSS of the process (and its children) every 100 ms with `psutil`
and keeps the maximum. The page cache is not counted. Because LanceDB is embedded, this process is the database process.

### 6.1 Ingest (`scripts/ingest.py`)

`python scripts/ingest.py --dataset <name or all>` (or through populate).

1. **Stream.** `stream_vectors` yields batches of 10,000 vectors. The first batch is read up front to learn the dimension.
2. **Schema (the "DDL as code").** Two columns: `id` as int64 and `vector` as a fixed-size list of float32 values of length
   `dim`. Vectors are always float32, because LanceDB cannot index other types.
3. **Arrow batches.** Each batch becomes a `pyarrow.RecordBatch` (the flat float array is wrapped as a
   `FixedSizeListArray`). A generator of batches goes to `db.create_table(..., mode="overwrite")`, which writes the table
   while consuming the stream, so the whole file is never in memory.
4. **Id check.** While streaming, every batch must have ids equal to the row position (`0, 1, 2, ...`). The ground truth
   refers to these ids, so if they differed the ingest stops with an error.
5. **Spot check.** It remembers a few vectors (the first of the first batch, one from every seventh batch up to a few, and
   the last row). After writing, it asserts that the row count equals the rows read, and reads each remembered
   vector back from the table by `id` and compares it exactly with the CSV value.
6. It prints rows, dimension, time and the size of the table, and returns the loading time.

Re-running ingest overwrites the table (a new table version); the index must then be built again.

### 6.2 Indexes (`scripts/build_index.py`)

For each dataset it opens the table, reads the row count and dimension, and takes the default partitions and
sub-vectors from `config.INDEX_PARAMS` (tuned values for glove-200, nytimes and pubmed) or `index_defaults` unless overridden (it refuses a sub-vector count that does not divide the dimension).

- **Vector index:** `IvfPq(distance_type=<registry metric>, num_partitions=..., num_sub_vectors=...)` on the `vector` column.
  The index must use the same metric as the queries.
- **Scalar index:** a `BTree` on `id`, so lookups by id (goals 1-3, 5-7) do not scan the table.
- **`optimize(cleanup_older_than=0)`** compacts the table and deletes every old version. This makes the printed table
  size the live size. LanceDB warns that readers still using an older version will fail, which is why the server
  retries (6.4). It returns the time of all three steps.

IVF_PQ in short: k-means splits the vectors into `num_partitions` groups; each vector is stored as a short code
(`num_sub_vectors` bytes) instead of its full 4 x `dim` bytes. This is why the index is only 10 to 48 MB for tables of
hundreds of MB, and why it is approximate.

### 6.3 The goals (`src/lancedb_bench/goals.py`)

Each goal is a function of a `Context` (the open table, the dataset, row count, dimension and `max_k`, the largest allowed
k: 100, or the true neighbors per query of the dataset if fewer). It returns `{"lines": [...], "note": "..."}`: the result
lines and an information line for stderr. Bad parameters raise `InvalidParameters` (a k above `max_k`, an id outside
`0..rows-1`, a radius outside 0 to 2, a percent outside 1 to 100); a missing precondition raises `NotReady`.

The vector query is `search(vec).distance_type(metric).limit(k).select(["id", "_distance"])` with either
`nprobes(...).refine_factor(5)` or `bypass_vector_index()` (exact search, used by `check_setup.py`), and an optional
`where(..., prefilter=True)`. `_distance` is selected explicitly because LanceDB adds it anyway and warns on every call
otherwise. Neighbor goals use `neighbors_of`: look the stored vector up by id, ask for k + 1 neighbors, drop the vector itself.

| Goal | How it works |
|---|---|
| 1 | `neighbors_of(id, k)` |
| 2 | `neighbors_of(id, 100)`, keep distance <= radius; the count goes to the note |
| 3 | `neighbors_of(id, k, where="id % 100 < percent")`: the prefilter is applied before the vector search |
| 4 | `load_unit_vectors` reads the table (`to_arrow()`), orders it by id and scales to unit length; `kmeans()` takes ids 0..clusters-1 as start centers and runs exactly `iterations` rounds of: assign every vector to the center with the largest dot product (= cosine), set each center to the mean of its members (an empty cluster keeps its center), in chunks of 20,000 rows. No random numbers. The labels are those of the last assignment. They are stored with `add_columns(cluster_id int32)` and `merge_insert("id")`, a bitmap index on `cluster_id` and `optimize()`. The time goes to `kmeans_s` in `build.csv` |
| 5 | Reads `cluster_id` of the row |
| 6 | `neighbors_of(id, k, where="cluster_id = <cluster of id>")` |
| 7 | Fetches both vectors and computes `1 - a.b / (|a||b|)` in float64 |
| 8 | `run_setting` for each `nprobes` in `NPROBES_SETTINGS`, see below |

**Goal 8, the benchmark** (`run_setting`). For one accuracy setting it first runs the 200 warm-up queries (discarded), then
3 passes over the first n test queries, one at a time and one thread. The timed region is the whole search call (building
the query, running it, getting the Arrow result) measured with `perf_counter_ns`; turning the ids into Python lists happens
outside it, and only in the first pass, from which recall@k comes (overlap with the first k ground-truth ids, divided by k,
averaged). p50, p95 and p99 are taken over all 3 x n latencies. One row per (dataset, k, setting) is appended to
`/results/search.csv`; the note shows the peak memory of the run and the lowest `nprobes` with recall@10 of at least 0.95 (the
baseline's headline number).

### 6.4 The HTTP wrapper (`src/lancedb_bench/server.py`)

A FastAPI app, started with `uvicorn lancedb_bench.server:app` (one worker). At import it opens the LanceDB folder
`DB_URI`. It keeps a cache `dataset name -> Context`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Used by the Docker healthcheck |
| `GET /info` | Server environment: LanceDB and pyarrow versions, cores (and how many are allowed), RAM |
| `GET /datasets` | For each registry dataset: whether its table exists, rows, dimension, table size, and every index (name, type, columns, unindexed rows, size, metric) |
| `POST /reload` | Clears the cache, so the next goal opens the latest version |
| `POST /goal/{n}` | Runs goal n with `{"dataset": ..., "params": [...]}` and returns `{"lines": [...], "note": ...}` |

`POST /goal/{n}` converts the parameters with `spec.parse_params` and calls the function in `goals.py`. Status codes: 422
invalid parameters (including an unknown dataset), 409 a precondition is missing (goal 5 or 6 before goal 4), 503 the table
does not exist (run populate).

**`nprobes`** is how many partitions the index looks into; **`refine_factor`** makes the index fetch
`k x refine_factor` candidates and re-rank them with their exact vectors. Both trade speed for recall.

**Stale handles.** A table handle points at one table version. If populate runs while the server is up (and deletes old
versions), or goal 4 adds a column, the cached handle can fail. The endpoint therefore catches an error, drops that cached
handle and tries once more with a freshly opened one; after goal 4 it drops the handle itself.

## 7. The client side

### 7.1 `client.py` and `cli.py` (the `db` command)

`Client` is a thin wrapper around `httpx.Client` (no timeout, because goal 8 runs for minutes) with `goal(n, dataset,
params)`, `reload()`, `info()`, `datasets()` and `health()`. A 422 becomes `InvalidParameters`, any other error a `RuntimeError`.

`db <goal> <params...>` takes the dataset from `$DATASET`. It first checks the parameters with the same
`spec.parse_params` as the server (a typo needs no round trip), sends the goal, prints the result lines to stdout and the
note to stderr. **Exit codes:** 0 success, 2 invalid parameters (wrong count or type, unknown goal or dataset, a value out of
range), 1 any other failure (server not reachable, no table, goal 5 before goal 4).

### 7.2 `scripts/bench.py`

`docker compose run --rm benchmark` (default command): for each selected dataset and each k in 10 and 100 it calls goal 8 with
n = 1,000 and prints the summary. A k that the dataset cannot serve (`uform-wiki-1m` has 10 neighbors) is reported as skipped.
Exit code 1 if a run failed. The measurements themselves are made inside the db container.

### 7.3 `scripts/check_setup.py` (the setup test)

Runs in the db image: `docker compose run --rm populate python scripts/check_setup.py` (`--dataset <name or all>`). It talks to
the database directly, not through HTTP. For each dataset it runs these checks, printing `PASS`, `FAIL`, `WARN` or `INFO`:

| # | Check | Fails when |
|---|---|---|
| 1 | The four dataset files exist | A file is missing |
| 2 | Row count and dimension equal the manifest's `train_rows` and `dimension` | The table is missing, incomplete, or has another dimension |
| 3 | A vector index exists, covers every row and has the registry's metric | Metric differs (warns if there is no index or rows are unindexed) |
| 4 | **Exact search** (`bypass_index`) on 50 queries reproduces the ground truth, recall@10 at least 0.98 | Recall is lower: the metric or the id mapping is wrong |
| 4b | Compares returned distances with the ground truth distances | Information only (shows the distance convention) |
| 5 | **Approximate search** through the index | Recall is zero (any recall above zero passes, so read the printed value) |
| 6 | Goal 1 (`id=0, k=5`) returns 5 neighbors without id 0; goal 3 returns only ids with `id % 100 < 10`; goal 7 gives a distance between 0 and 2 | Wrong count, the vector itself, a broken filter |

Check 4 is the most important one: brute-force search is correct by construction, so if it does not match the
ground truth, the data, ids or metric are wrong. Any exception for one dataset is reported as a FAIL and the script
continues with the next dataset. The exit code is 1 if any dataset failed, 0 otherwise.

## 8. The containers

**`Dockerfile`** builds both images from one definition. The build argument `ROLE` is `db` or `client`:

- It installs `uv`, copies `pyproject.toml` and `uv.lock`, and runs `uv sync --frozen --no-dev` (the lock file pins every
  version). With `ROLE=db` it adds the `db` extra (LanceDB, FastAPI, uvicorn); with `ROLE=client` it does not, so the
  client image has no database engine. The dependencies are installed before the source is copied, so source changes
  do not reinstall them.
- It sets `DATA_DIR=/data`, `DB_URI=/db`, `RESULTS_DIR=/results` and `OMP_NUM_THREADS=1`.

**`docker-compose.yml`** defines three services, the standard commands of the baseline (`docker compose build`,
`up -d db`, `run --rm populate`, `run --rm benchmark db 1 443321 4`):

| | `db` (container `lancedb_server`) | `populate` | `benchmark` |
|---|---|---|---|
| Image | `lancedb-bench-db` (`ROLE=db`) | the same | `lancedb-bench-client` (`ROLE=client`) |
| Command | `uvicorn lancedb_bench.server:app` on port 8000 | `python scripts/populate.py` | `python scripts/bench.py`, or any `db <goal> ...` |
| Mounts | `./data` as `/data` (read-only), `./db` as `/db`, `./results` as `/results` | the same | none |
| Limits | 4 pinned cores (0-3), 8 GB RAM, swap off | the same | none |
| Other | healthcheck calls `/health` | waits until `db` is healthy; `SERVER_URL=http://db:8000` | waits until `db` is healthy; `SERVER_URL=http://db:8000` |

The database is the system under test, so it and the populate step share the limits. The benchmark container is not
CPU-pinned here; on the test server the baseline gives it its own core.

## 9. One goal from start to end

`DATASET=cohere-small-100k docker compose run --rm benchmark db 1 42 5`:

1. `cli.py` checks the parameters (`spec.parse_params`) and calls `Client.goal(1, "cohere-small-100k", ["42", "5"])`, that is
   `POST http://db:8000/goal/1` with `{"dataset": "cohere-small-100k", "params": ["42", "5"]}`.
2. `server.py` converts the parameters, opens the cached `Context` and calls `goal_1`.
3. `goal_1` checks k, finds vector 42 by id through the BTree and builds a LanceDB query for `k + 1 = 6` neighbors with metric
   cosine, 20 probes and refine factor 5.
4. LanceDB compares the query vector with the 32 partition centers, scans the compressed vectors of the 20 closest
   partitions, keeps the `6 x 5 = 30` best candidates, fetches their exact vectors and re-ranks them.
5. The goal removes id 42 from the six results and returns five lines `id distance`.
6. The client prints the lines and exits with 0.

## 10. Design decisions and limits

- **Why a registry line for the metric.** Dimension and size can be read from files, but the metric cannot, and a wrong
  one only shows up as bad recall. Declaring it once and verifying it by exact search is the cheapest safe approach.
- **Why all files are in the db container and the benchmark runs there.** It follows the baseline: identical files for every
  group (no accidental swap), and the timing excludes the client-server hop. The cost is that the database container can in
  principle see the test queries; we only read them from the files in the benchmark.
- **Why the check is exact search first.** It tests ids, metric and data without involving the approximate index, so a
  failure there points to the import, not to index tuning.
- **Populate is a script, not an endpoint.** It takes minutes; HTTP requests for that would need background jobs and status
  polling. Running it as a one-off container of the same image is simpler.
- **K-means is outside the engine.** LanceDB has no clustering query, so goal 4 runs numpy in the db container and stores
  the result in the table. It loads all vectors into memory (about 0.3 GB for `cohere-small-100k`).
- **Known limits:**
  - Recall is not tuned: one `refine_factor` and four `nprobes` values for all datasets. The recall check in populate only warns.
  - The approximate-search check in `check_setup.py` passes with any recall above zero.
  - The goals follow baseline proposals the class has not approved. The goal 2 radius gives 0 to 100 results for stored vectors.
  - The goals were run for `cohere-small-100k` only (natively; goals 1, 4 and 8 also in Docker). All six datasets were populated in Docker, but the other five were not benchmarked, and goals 2, 3, 5, 6 and 7 are untested in Docker.
  - Table versions: indexing deletes old versions, so do not run searches while `build_index.py` runs.

## 11. How to extend it

- **Add a dataset:** add its folder (the same four files) to the Hugging Face repo, add `"<name>": "<metric>"` to
  `DATASETS` in `config.py`, run `download.py --dataset <name>`, then populate and check it with the same commands. Always read
  the `check_setup.py` result before trusting a new metric (LanceDB's L2 distance is the squared distance, so an L2
  dataset's distances will differ from a plain Euclidean ground truth, which the check reports as information).
- **Change a goal or add one:** add its parameters to `GOALS` in `spec.py`, write the function in `goals.py` and add it to
  `GOAL_FUNCTIONS`. The CLI, the server and the usage text pick it up.
- **Tune an index:** `populate.py --dataset <name> --partitions N --sub-vectors M`, then `db 8 1000 10` and compare the
  rows in `results/search.csv`; change the swept settings with `NPROBES_SETTINGS`.
