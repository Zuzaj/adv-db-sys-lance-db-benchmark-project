# How the code works

This document walks through the whole project: what each file does, what happens when you run a command, and why it
was written that way. It describes the code as it is now (two containers, six datasets, CSV import from Hugging Face). The README has
the commands; this file explains what is behind them.

## 1. The big picture

```
         Hugging Face repo zuzaj/ADB: <dataset>/{train.csv, test.csv, ground_truth.csv, manifest.json}
                      |
                      |  scripts/download.py  (on the host)
                      v
   data/db/<dataset>/train.csv            data/client/<dataset>/{test.csv, ground_truth.csv, manifest.json}
          |                                             |
          | mounted read-only as /data                  | mounted read-only as /data
          v                                             v
 +-------------------------------+   HTTP   +----------------------------------+
 | db container (lancedb_server) | <------- | client container (lancedb_client)|
 |  scripts/ingest.py            |  :8000   |  scripts/check_setup.py          |
 |  scripts/build_index.py       |          |  scripts/bench.py                |
 |  server.py (FastAPI)          |          |  db CLI (cli.py)                 |
 |  LanceDB in ./db  (/db)       |          |  results written to ./results    |
 +-------------------------------+          +----------------------------------+
```

Three ideas shape the design:

1. **The database and the benchmark client are separate processes in separate containers**, even though LanceDB is an
   embedded library. LanceDB only runs inside some process, so that process is a small HTTP service (`server.py`).
   This was agreed in class for all databases.
2. **Each container only sees its own data.** The database gets the training vectors; the client gets the test queries and
   the correct answers (ground truth). The client never sees `train.csv`, and the client image does not even contain
   LanceDB.
3. **Everything about a dataset comes from its files plus one line in a registry.** Dimension and row count are read from
   the files; only the distance metric must be declared, because the CSVs do not say which one the ground truth uses.

## 2. Repository map

| File | Runs in | What it does |
|---|---|---|
| `src/lancedb_bench/config.py` | both | Paths from environment variables, the dataset registry (name to metric), `Dataset` paths, `index_defaults()` |
| `src/lancedb_bench/data.py` | both | Streaming readers for `train.csv`, `test.csv`, `ground_truth.csv` |
| `src/lancedb_bench/metrics.py` | both | Recall@k, latency percentiles, peak memory, folder size |
| `src/lancedb_bench/server.py` | db | FastAPI service around LanceDB |
| `src/lancedb_bench/client.py` | client | HTTP client with client-side timing |
| `src/lancedb_bench/cli.py` | client | The `db <goal> <params>` command |
| `scripts/download.py` | host | Downloads the datasets from Hugging Face into `data/db` and `data/client` |
| `scripts/ingest.py` | db | `train.csv` into a LanceDB table |
| `scripts/build_index.py` | db | IVF_PQ vector index and BTree index |
| `scripts/check_setup.py` | client | Verifies the whole setup, per dataset |
| `scripts/bench.py` | client | The benchmark: latency, throughput, recall |
| `Dockerfile`, `docker-compose.yml` | host | The two containers |

## 3. Configuration and the dataset registry (`config.py`)

**Paths** come from environment variables with defaults for running on the host:

| Variable | Default | In the containers |
|---|---|---|
| `DATA_DIR` | `./data` | `/data` |
| `DB_URI` | `./db` | `/db` |
| `RESULTS_DIR` | `./results` | `/results` |
| `SERVER_URL` | `http://localhost:8000` | `http://db:8000` (client container) |

**The registry** `DATASETS` maps a dataset name (the folder name) to a LanceDB distance type: `cosine`, `dot` or `l2`.
The CSVs carry no metric, and using the wrong one gives wrong answers, so this is the one place that decides it. The
current entries are all cosine (the other types are supported but unused). `check_setup.py` verifies each entry against
the ground truth.

**`Dataset`** is a small frozen dataclass built from a name. It derives everything else:

- `train_path`, `test_path`, `truth_path`, `manifest_path` under `DATA_DIR/<name>/`
- `table`: the LanceDB table name, the dataset name with every non-word character replaced by `_`
  (`cohere-small-100k` becomes `cohere_small_100k`)
- `table_dir`: where that table lives on disk, `DB_URI/<table>.lance` (used to report the size of one table)
- `results_dir`: `RESULTS_DIR/<name>`
- `metric`: looked up in the registry; an unknown name stops with an error listing the known datasets

`get_dataset(name)` returns a `Dataset` (the name defaults to `$DATASET`, then to `cohere-small-100k`).
`select_datasets("all")` returns every registry dataset. Every script offers `--dataset <name or all>`.

**`index_defaults(n, dim)`** picks the starting IVF_PQ parameters:

- partitions: about 2,300 vectors per partition, rounded to a power of two (`2 ** round(log2(n / 2300))`)
- sub-vectors: `dim // 8`, lowered until it divides `dim` (product quantization needs that)

Examples: 100,000 rows with 768 dimensions give 32 partitions and 96 sub-vectors; 1,183,514 rows with 200 dimensions give
512 and 25. These are starting points, not tuned values; `build_index.py --partitions N --sub-vectors M` overrides them.

## 4. Reading the CSV files (`data.py`)

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
`ground_truth.csv` and `manifest.json`. The containers must not see each other's files, so the script puts them into two
folders:

```
data/db/<dataset>/train.csv
data/client/<dataset>/{test.csv, ground_truth.csv, manifest.json}
```

`ROLE_FILES` in `config.py` defines which file belongs to which side. For each selected dataset and each file the script
calls `huggingface_hub.hf_hub_download(..., local_dir=<dest>/<db or client>)`; the library keeps the repo's
`<dataset>/<file>` path, which gives exactly the layout above, resumes interrupted downloads and skips files that are
already complete. `--dataset all` downloads all six (about 18 GB), `--dest` changes the base folder (default `./data`).

The script runs on the host. The containers mount the two folders **read-only**, so nothing inside a container can
change the data.

## 6. The database side

### 6.1 Ingest (`scripts/ingest.py`)

Runs in the db container: `python scripts/ingest.py --dataset <name or all>`.

1. **Stream.** `stream_vectors` yields batches of 10,000 vectors. The first batch is read up front to learn the dimension.
2. **Schema (the "DDL as code").** Two columns: `id` as int64 and `vector` as a fixed-size list of float32 values of length
   `dim`. Vectors are always float32, because LanceDB cannot index other types.
3. **Arrow batches.** Each batch becomes a `pyarrow.RecordBatch` (the flat float array is wrapped as a
   `FixedSizeListArray`). A generator of batches goes to `db.create_table(..., mode="overwrite")`, which writes the table
   while consuming the stream, so the whole file is never in memory (peak process memory was 0.8 to 1.3 GB).
4. **Id check.** While streaming, every batch must have ids equal to the row position (`0, 1, 2, ...`). The ground truth
   refers to these ids, so if they differed the ingest stops with an error.
5. **Spot check.** It remembers a few vectors (the first of the first batch, one from every seventh batch up to a few, and
   the last row). After writing, it asserts that the row count equals the rows read, and reads each remembered
   vector back from the table by `id` and compares it exactly with the CSV value.
6. It prints rows, dimension, time, peak memory and the size of the table.

Re-running ingest overwrites the table (a new table version); the index must then be built again.

### 6.2 Indexes (`scripts/build_index.py`)

For each dataset it opens the table, reads the row count and dimension, and takes the default partitions and
sub-vectors from `index_defaults` unless overridden (it refuses a sub-vector count that does not divide the dimension).

- **Vector index:** `IvfPq(distance_type=<registry metric>, num_partitions=..., num_sub_vectors=...)` on the `vector` column.
  The index must use the same metric as the queries.
- **Scalar index:** a `BTree` on `id`, so lookups by id (the CLI's goal 1) do not scan the table.
- **`optimize(cleanup_older_than=0)`** compacts the table and deletes every old version. This makes the printed table
  size the live size. LanceDB warns that readers still using an older version will fail, which is why the server
  retries (section 7).

IVF_PQ in short: k-means splits the vectors into `num_partitions` groups; each vector is stored as a short code
(`num_sub_vectors` bytes) instead of its full 4 x `dim` bytes. This is why the index is only 10 to 48 MB for tables of
hundreds of MB, and why it is approximate.

### 6.3 The HTTP wrapper (`src/lancedb_bench/server.py`)

A FastAPI app, started with `uvicorn lancedb_bench.server:app` (one worker). At import it opens the LanceDB folder
`DB_URI`. It keeps a cache `dataset name -> (table handle, dimension)`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Used by the Docker healthcheck |
| `GET /info` | Server environment: LanceDB and pyarrow versions, cores (and how many are allowed), RAM, current and peak memory |
| `GET /datasets` | For each registry dataset: whether its table exists, rows, dimension, table size, and every index (name, type, columns, indexed and unindexed rows, size, metric) |
| `POST /reload` | Clears the table cache, so the next search opens the latest version |
| `POST /search` | The query endpoint |

**`POST /search`** takes `dataset`, `k`, `nprobes`, `refine_factor`, `bypass_index` and either `vector` or `id`:

1. Opens (or takes from the cache) the table. An unknown dataset gives 404; a dataset that is not ingested gives 503 with
   the command to run.
2. With `vector` it uses it. With `id` it looks the stored vector up (`where id = N`, served by the BTree) and asks for
   `k + 1` neighbors, because the vector finds itself at distance 0 and is removed from the answer afterwards.
3. A vector of the wrong length gives 400.
4. It builds the LanceDB query: `search(vec).distance_type(metric).limit(k).select(["id", "_distance"])`. With
   `bypass_index` it adds `bypass_vector_index()` (exact brute force), otherwise `nprobes(...)` and, if given,
   `refine_factor(...)`.
5. **It times only the LanceDB call** (`to_arrow()`) with `perf_counter_ns` and returns that as `engine_latency_ms`.
6. The response is only `ids`, `distances` and the engine time, no vectors.

**`nprobes`** is how many partitions the index looks into; **`refine_factor`** makes the index fetch
`k x refine_factor` candidates and re-rank them with their exact vectors. Both trade speed for recall.

**Stale handles.** A table handle points at one table version. If ingest or `build_index.py` run while the server is up
(and delete old versions), the cached handle can fail. `/search` therefore catches an error, drops that cached handle,
and tries once more with a freshly opened one.

## 7. The client side

### 7.1 `client.py`

A thin wrapper around `httpx.Client` with methods for each endpoint. `search()` returns `(response, total_ms)`, where
`total_ms` is the wall-clock time on the client around the whole call: building the JSON (the vector list), the network
round trip and parsing the response. The difference `total_ms - engine_latency_ms` is the overhead of the HTTP wrapper
(about 1 ms in the Docker run).

### 7.2 `scripts/check_setup.py` (the setup test)

`python scripts/check_setup.py --dataset <name or all>` first checks that the server answers, calls `POST /reload`,
prints the server environment, and fetches `/datasets`. Then for each dataset it runs these checks, printing
`PASS`, `FAIL`, `WARN` or `INFO`:

| # | Check | Fails when |
|---|---|---|
| 1 | The client has `test.csv`, `ground_truth.csv`, `manifest.json` | A file is missing. It warns if `train.csv` is on the client side. |
| 2 | Row count and dimension equal the manifest's `train_rows` and `dimension` | The table is missing, incomplete, or has another dimension |
| 3 | A vector index exists, covers every row and has the registry's metric | Metric differs (warns if there is no index or rows are unindexed) |
| 4 | **Exact search** (`bypass_index`) on 50 queries reproduces the ground truth, recall@10 at least 0.98 | Recall is lower: the metric or the id mapping is wrong |
| 4b | Compares returned distances with the ground truth distances | Information only (shows the distance convention) |
| 5 | **Approximate search** through the index with `nprobes=20`, `refine_factor=5` | Recall is zero (any recall above zero passes, so read the printed value) |
| 6 | Lookup by id: `id=0, k=5` returns 5 neighbors without id 0 | Wrong count or the vector itself is returned |

Check 4 is the most important one: brute-force search is correct by construction, so if it does not match the
ground truth, the data, ids or metric are wrong. Any exception for one dataset is reported as a FAIL and the script
continues with the next dataset. The exit code is 1 if any dataset failed, 0 otherwise.

### 7.3 `scripts/bench.py` (the benchmark)

`python scripts/bench.py --dataset <name> [--nprobes ...] [--refine ...] [-k ...] [--exact] [--n-queries N]`

Protocol, per dataset and per configuration (k, nprobes, refine_factor):

1. Read the first `N` test queries (default 1,000) and their ground truth. Values of `k` larger than the ground truth
   has neighbors (`uform-wiki-1m` has 10) are skipped.
2. **Warm-up:** 200 queries, not counted.
3. **Timed passes:** 3 passes over all queries, each query one HTTP call. Latency statistics use all samples.
4. **Recall@k** is computed on the first pass: the overlap between the returned ids and the first `k` ground-truth ids,
   divided by `k`, averaged over queries.
5. `--exact` adds a brute-force run per `k` as a reference.

Each configuration produces one CSV row in `results/<dataset>/<--out>` (default `bench.csv`) with: parameters, recall,
and for three measurements (`total_*`: client wall clock; `engine_*`: server time; `overhead_*`: the difference) the
p50, p95, p99, mean and QPS. Every row also carries the environment: client and server machine, cores (and how many
are allowed), RAM, Python and LanceDB versions, git commit, the server's peak memory and the table size.
If an existing file has other columns, results go to a new timestamped file instead of mixing columns.

### 7.4 `cli.py` (the `db` command)

`db <goal> <params>` with the dataset taken from `$DATASET`. Only two goals exist so far:

- `db 1 <id> <k>`: the neighbors of a stored vector, excluding itself (the wrapper's `id` path).
- `db 8 <n_queries> <k>`: the first `n` test queries; prints recall@k and latency (client total and server engine).

Both use `nprobes=20` and `refine_factor=5` unless `NPROBES` and `REFINE_FACTOR` are set. Other goal numbers print
"not defined yet".

### 7.5 `metrics.py`

`calculate_recall_at_k` (set overlap divided by `k`), `compute_latency_stats` (p50, p95, p99, mean, QPS as
1000 / mean ms), `get_peak_rss_mb` (peak memory of the process, converted from the platform's unit) and
`get_dir_size_mb` (total file size of a folder).

## 8. The containers

**`Dockerfile`** builds both images from one definition. The build argument `ROLE` is `db` or `client`:

- It installs `uv`, copies `pyproject.toml` and `uv.lock`, and runs `uv sync --frozen --no-dev` (the lock file pins every
  version). With `ROLE=db` it adds the `db` extra (LanceDB, FastAPI, uvicorn); with `ROLE=client` it does not, so the
  client image has no database engine. The dependencies are installed before the source is copied, so source changes
  do not reinstall them.
- It sets `DATA_DIR=/data`, `DB_URI=/db`, `RESULTS_DIR=/results` and `OMP_NUM_THREADS=1`.

**`docker-compose.yml`** defines two services:

| | `db` (container `lancedb_server`) | `client` (container `lancedb_client`) |
|---|---|---|
| Image | `ROLE=db` | `ROLE=client` |
| Command | `uvicorn lancedb_bench.server:app` on port 8000 | none: you run scripts with `docker compose run` |
| Mounts | `./data/db` as `/data` (read-only), `./db` as `/db` | `./data/client` as `/data` (read-only), `./results` as `/results` |
| Limits | 4 pinned cores (0-3), 8 GB RAM, swap off | none |
| Other | healthcheck calls `/health` | waits until `db` is healthy; `SERVER_URL=http://db:8000` |

The database container is the system under test, so only it is limited. Scripts for the db side run with
`docker compose exec db ...` (inside the running server container, sharing its CPU and memory limits), the client
side with `docker compose run --rm client ...`.

## 9. One search from start to end

`docker compose run --rm client db 1 42 5` (dataset `cohere-small-100k`):

1. `cli.py` builds a `Client` and calls `search("cohere-small-100k", id=42, k=5, nprobes=20, refine_factor=5)`.
2. The client sends `POST http://db:8000/search` with that JSON and starts its timer.
3. `server.py` opens the cached table, finds vector 42 by id through the BTree, and builds a LanceDB query for `k + 1 = 6`
   neighbors with metric cosine, 20 probes and refine factor 5.
4. LanceDB compares the query vector with the 32 partition centers, scans the compressed vectors of the 20 closest
   partitions, keeps the `6 x 5 = 30` best candidates, fetches their exact vectors and re-ranks them.
5. The server removes id 42 from the six results, and returns the 5 ids, their distances and the time spent in LanceDB.
6. The client stops its timer, parses the answer and prints the ids with their cosine distances.

## 10. Design decisions and limits

- **Why a registry line for the metric.** Dimension and size can be read from files, but the metric cannot, and a wrong
  one only shows up as bad recall. Declaring it once and verifying it by exact search is the cheapest safe approach.
- **Why two data folders.** The client must not be able to read the training data and the database must not read the test
  data, so each container mounts only its own folder, read-only.
- **Why the check is exact search first.** It tests ids, metric and data without involving the approximate index, so a
  failure there points to the import, not to index tuning.
- **Ingest and indexing are scripts, not endpoints.** They take minutes; HTTP requests for that would need background
  jobs and status polling. Running them inside the container with `docker compose exec` is simpler.
- **Known limits:**
  - Recall is not tuned: one `nprobes` and `refine_factor` for all datasets.
  - The approximate-search check in `check_setup.py` passes with any recall above zero.
  - Only goals 1 and 8 exist.
  - The client container is not CPU-pinned, and all measurements so far are warm, single runs on a laptop.
  - Table versions: indexing deletes old versions, so do not run searches while `build_index.py` runs.

## 11. How to extend it

- **Add a dataset:** add its folder (the same four files) to the Hugging Face repo, add `"<name>": "<metric>"` to
  `DATASETS` in `config.py`, run `download.py --dataset <name>`, then ingest, index and check it with the same commands. Always read
  the `check_setup.py` result before trusting a new metric (LanceDB's L2 distance is the squared distance, so an L2
  dataset's distances will differ from a plain Euclidean ground truth, which the check reports as information).
- **Add a goal to the `db` command:** add a function in `cli.py` and a branch in `main()`. If it needs a new kind of
  query, add a field to `SearchRequest` and handle it in `run_search` in `server.py`.
- **Tune an index:** `build_index.py --dataset <name> --partitions N --sub-vectors M`, then `bench.py` with a sweep such as
  `--nprobes 1 5 20 100 --refine 1 5 20 --exact` and compare the CSV rows.
