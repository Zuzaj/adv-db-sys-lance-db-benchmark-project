# Advanced Database Systems 2026: LanceDB benchmark

Authors: Zuzanna Jarlaczyńska, Tomasz Strycharz
Course: Advanced Database Systems 2026

Vector-search benchmark of **LanceDB** (open-source, embedded) on six datasets
(`cohere-small-100k`, `nytimes-256-angular`, `pubmed-arxiv`, `uform-wiki-1m`, `glove-200-angular`,
`ccnews-nomic-768-normalized`), prepared once and published in the Hugging Face dataset repo `zuzaj/ADB`.
The database runs in its own container behind a small HTTP wrapper and a separate client container sends the
goals, as agreed in class for all databases, embedded ones included. The commands, goals, parameters and result files follow
the class document "Common Baseline for the Vector Database Project" (proposals, not yet approved by the class). Adding a
dataset needs one line of configuration (section 4).

> Status: work in progress. Goals 1-8 follow the baseline proposals; the class has not approved them yet.
> The results in section 8 are first runs (a native run of the new goals and an earlier Docker run), not the final benchmark.

## 1. Choice of technology

LanceDB is an embedded vector database built on the Lance columnar format and Apache Arrow. It runs inside the
calling process and stores tables and indexes in a local directory, like SQLite or DuckDB. There is no server to
install, which fits the "as light as possible" requirement.

- Schema is defined in code (Arrow schema) instead of SQL `CREATE TABLE`.
- Vector index used here: IVF-PQ (k-means partitions, product-quantized vectors). Scalar index used here: BTree.

### How data is stored

A database is a plain folder (`DB_URI`); each table is a subfolder `<name>.lance/` (observed in this project):

| Folder | Contents |
|---|---|
| `data/` | The rows, in Lance's columnar file format. Fragment files are immutable. |
| `_versions/` | One manifest per table version: schema, fragment list, row counts, indexes. |
| `_transactions/` | One small file per commit. |
| `_indices/<uuid>/` | One folder per index build: IVF_PQ is `index.idx` (centroids, codebook) plus `auxiliary.idx` (compressed vectors); BTree is `page_data.lance` plus `page_lookup.lance`. |

- **Versioning:** every write, index build or `optimize()` creates a new version, and old files stay on disk until cleaned up
  (`optimize(cleanup_older_than=...)`). Re-running ingest or indexing can therefore leave the folder about twice the live size,
  so clean up before measuring disk size.
- **Metadata:** our tables hold only `id` and `vector`. LanceDB keeps the schema, fragments, row counts and index settings
  (including the metric) in the manifests. The dataset name and its distance metric are not stored in the database; the metric comes from
  `DATASETS` in `src/lancedb_bench/config.py`, and the query's `distance_type` must match the index's metric.
- **Concurrent access:** there is no server, so each client opens the folder in its own process. Several readers at once work
  (tested with the HTTP server and a second process on the same table). Writers commit atomically and each reader sees one version;
  concurrent writers can conflict and retry. This is from the LanceDB documentation and was not tested here.
- **Storage location:** local disk in this project. LanceDB can also use S3, GCS or Azure (documentation, not tested).

### Available indexes

Names available in `lancedb.index` (version 0.40.0); only `IvfPq` and `BTree` were built and tested here, the descriptions of the others come from the documentation:

| Kind | Index | Idea |
|---|---|---|
| Vector | `IvfPq` (used) | Partitions plus 1-byte codes per sub-vector: small and fast, lossy. |
| Vector | `IvfFlat` | Partitions with raw vectors: larger, no compression loss. |
| Vector | `IvfSq` | Partitions with 8-bit scalar quantization. |
| Vector | `IvfRq` | Partitions with binary quantization. |
| Vector | `IvfHnswPq`, `IvfHnswSq`, `IvfHnswFlat` (also `HnswPq`, `HnswSq`, `HnswFlat`) | A graph inside each partition. |
| Scalar | `BTree` (used), `Bitmap`, `LabelList` | Lookups and filters on id, low-cardinality and list columns. |
| Text | `FTS` | Full-text search. |

Without a vector index, search scans the whole table (exact, about 109 ms per query on glove-200).

How IVF-PQ answers a query, with the glove-200 numbers: k-means splits the vectors into 512 partitions; each 200-dimension
vector is cut into 25 chunks of 8 dimensions, and each chunk is replaced by a 1-byte code (800 bytes become 25, about 30 MB
for 1.19M vectors). A query is compared with the partition centroids, the compressed vectors of the `nprobes` nearest
partitions are scanned, `k x refine_factor` candidates are kept, and their full vectors are fetched from `data/` and re-ranked
with exact distances. Because compression is lossy, recall stops improving with `nprobes` at some point; more sub-vectors
or a larger `refine_factor` reduce the loss (section 8).

## 2. Architecture

```
 db container (lancedb_server)                        benchmark container (client)
 +---------------------------------------+  HTTP :8000  +------------------------------+
 | /data: all dataset files (read-only)  | <----------- | db <goal> <params...>        |
 | /db: LanceDB directory, /results      |  one goal    | no data, no database engine  |
 | FastAPI wrapper (server.py)           | -----------> | prints the result lines      |
 | goals 1-8 and their timing (goals.py) |  result      +------------------------------+
 | embedded LanceDB                      |
 +---------------------------------------+
```

- The **db container** holds everything about the data: all dataset files (`train.csv`, `test.csv`, `ground_truth.csv`,
  `manifest.json`, mounted read-only as `/data`), the LanceDB directory and `/results`. `docker compose run --rm populate`
  (the same image, run once) loads and indexes the data; the wrapper runs the goals, including the timed benchmark.
- The **benchmark (client) container** runs the `db <goal> <params...>` command, sends the goal over HTTP and prints the
  answer. It has no data and no database engine (`lancedb` is not even installed there).
- All dataset files are in the db container, so every group using the same files works with identical train, test and
  ground-truth data. The test queries and the ground truth are read from files by the benchmark and are not stored in the database.
- The benchmark (goal 8) runs inside the db container and times each search call there, so the measured latency does not
  include the client-server hop (baseline section 3).
- Both containers are built from the same `Dockerfile` with a `ROLE` argument (`db` or `client`); `docker-compose.yml`
  limits the db container to 4 pinned cores and 8 GB RAM with swap off.

| Path | Purpose |
|---|---|
| `scripts/download.py` | Download the datasets from Hugging Face into `data/<dataset>/` (mounted into the db container) |
| `scripts/populate.py` | db side: upload everything in one command (load, indexes, recall check, one row in `build.csv`); exits with 1 if a dataset fails |
| `scripts/ingest.py` | db side: stream `train.csv` into a LanceDB table and verify it (used by `populate.py`) |
| `scripts/build_index.py` | db side: build IVF_PQ and BTree indexes, `optimize()`, print sizes (used by `populate.py`) |
| `scripts/check_setup.py` | db side: check that the whole setup works for each dataset (section 9) |
| `scripts/bench.py` | client side: runs goal 8 for k = 10 and 100 and prints the summary |
| `src/lancedb_bench/goals.py` | db side: goals 1-8 (queries, K-means, the timed benchmark) |
| `src/lancedb_bench/server.py` | db side: the FastAPI wrapper (`/goal/{n}`, `/datasets`, `/info`, `/health`, `/reload`) |
| `src/lancedb_bench/spec.py` | The goals and their parameters (shared by CLI and server) |
| `src/lancedb_bench/client.py` | client side: HTTP client |
| `src/lancedb_bench/cli.py` | The `db` command (client side), exit codes 0, 1, 2 |
| `src/lancedb_bench/results.py` | The two result files, `build.csv` and `search.csv` |
| `src/lancedb_bench/config.py` | Paths, the dataset registry (name -> metric) and index defaults |
| `src/lancedb_bench/data.py` | Streaming readers for the CSV formats |
| `src/lancedb_bench/metrics.py` | Recall, latency percentiles, sizes, peak-memory sampling |
| `results/` | `build.csv` and `search.csv` (small result files, committed), older per-dataset folders |

A detailed walk-through of how the code works is in [`docs/how-it-works.md`](docs/how-it-works.md).

## 3. Prerequisites

- Linux (macOS also works natively; numbers differ, see section 8)
- Docker or Podman with Compose for the two-container setup, or Python 3.12 and [`uv`](https://docs.astral.sh/uv/) to run natively
- Network access to Hugging Face, and disk: the six `train.csv` files are about 17.8 GB in total (0.96, 1.0, 4.2, 3.5, 3.0
  and 5.2 GB), plus the LanceDB directory, roughly the size of the raw vectors per dataset (about 5.3 GB for all six,
  more while old table versions exist)
- Docker: the db container is pinned to cores 0-3 with an 8 GB limit, so Docker Desktop needs at least 4 CPUs and 8 GB RAM
  (Settings, Resources)
- Memory: ingest streams the CSV in batches of 10,000 rows, so it needs far less than the file size

Pinned versions (`pyproject.toml`, `uv.lock`): lancedb 0.40.0 (db container only), pyarrow 26.0.0, numpy 2.4.4,
psutil 7.2.2.

## 4. Installation and setup

With Docker nothing needs to be installed on the host (section 9). Natively:

```bash
uv sync --extra db      # db side (LanceDB, FastAPI); plain `uv sync` installs the client side only
```

Paths default to `./data`, `./db`, `./results` and can be overridden with `DATA_DIR`, `DB_URI`, `RESULTS_DIR`;
`SERVER_URL` (default `http://localhost:8000`) tells the client where the wrapper is, `GROUP` names the group in the result files.

### Datasets: layout, selecting and adding

Each dataset is a folder with four files (`manifest.json` of each dataset gives the numbers below):

| File | Content |
|---|---|
| `train.csv` | `id,"[v0,v1,...]"`, one vector per row, ids are 0-based row positions |
| `test.csv` | same format, the query vectors |
| `ground_truth.csv` | `query_id,rank,neighbor_id,distance`, true neighbors of each query |
| `manifest.json` | row count, query count, dimension, how the files were made |

All four files of a dataset live in `data/<dataset>/`, which the db container mounts read-only as `/data`.

(`ccnews-nomic-768-normalized` also has an `avg_distances.csv`, which is not used.)

| Dataset | Train vectors | Dim | Queries | Neighbors per query | `train.csv` | Metric used |
|---|---|---|---|---|---|---|
| `cohere-small-100k` | 100,000 | 768 | 1,000 | 1000 | 0.96 GB | cosine |
| `nytimes-256-angular` | 290,000 | 256 | 10,000 | 1000 | 1.0 GB | cosine (angular) |
| `pubmed-arxiv` | 400,335 | 768 | 1,000 | 1000 | 4.2 GB | cosine |
| `uform-wiki-1m` | 1,000,000 | 256 | 100,000 | 10 | 3.5 GB | cosine (unit-norm vectors; ground-truth distances are cosine) |
| `glove-200-angular` | 1,183,514 | 200 | 10,000 | 100 | 3.0 GB | cosine (angular) |
| `ccnews-nomic-768-normalized` | 495,328 | 768 | 1,000 | 100 | 5.2 GB | cosine (normalized vectors) |

(For the first three, the value of 1000 comes from the notes of the conversion that prepared the files; we checked it
for `cohere-small-100k`.)

The CSVs carry no metric, so the single place that decides it is the `DATASETS` dictionary in
`src/lancedb_bench/config.py`. The metrics are our reading of the dataset names and manifests;
`scripts/check_setup.py` verifies them, because exact search with a wrong metric does not reproduce the ground truth.
In the Docker run (section 8) exact search reproduced the ground truth for all six datasets, which confirms them.

**Getting the data into place.** The six datasets were prepared once and are published in the Hugging Face repo
`zuzaj/ADB` (override with `HF_REPO`), one folder per dataset with the four files above. `scripts/download.py` fetches
them into `data/<dataset>/`, the same layout as in the repo (on the test server the files are already in its filesystem).

```bash
uv run python scripts/download.py --dataset cohere-small-100k
```

`--dataset all` downloads all six (about 18 GB). Files that are already complete are skipped.

Every script takes `--dataset <name>` (or `all`) and otherwise reads the `DATASET` environment variable, like the `db`
command. The default is `glove-200-angular`, the common default of the baseline, so the assignment's example `db 1 443321 4`
is valid. For quick tests use `DATASET=cohere-small-100k`, the smallest one.

**To add a dataset:** add a folder with the same four files to the Hugging Face repo, add one line to `DATASETS` in
`config.py` (name and metric), and run the same commands with the new name. Each dataset gets its own table in the
database and its own folder in `results/`. Dimension and row count are read from the files, and index parameters default
to about 2,300 vectors per partition and dimension/8 sub-vectors (`config.index_defaults`); datasets where that does not
reach recall@10 of 0.95 have tuned values in `config.INDEX_PARAMS` (section 8.2); override both with
`populate.py --partitions N --sub-vectors M`. Ground truth may have fewer neighbors than 100 (`uform-wiki-1m` has 10);
the goals reject a k that is too large for the dataset (exit code 2).

Limits: LanceDB's L2 `_distance` is the squared distance, so if you add an L2 dataset, compare against the ground
truth with `check_setup.py` (it prints the difference between the two distance conventions). The goals 1-7 use fixed
`nprobes=20` and `refine_factor=5` (override with `NPROBES` and `REFINE_FACTOR` in the db container); goal 8 sweeps
`nprobes` over 5, 10, 20, 50, 100 and 200 (`NPROBES_SETTINGS`) with `refine_factor=5`. These settings are the same for every dataset.

## 5. Design and implementation process

1. **Early single-container test.** First the whole pipeline was built and measured with LanceDB called in-process on
   VIBE `glove-200-cosine` (HDF5): exact search reproduced the ground truth, and the IVF_PQ index and the
   `nprobes` / `refine_factor` trade-offs were measured (end of section 8). That code was replaced by the CSV import below
   and is only in the first commit (`33e2d9c`).
2. **Two-container split.** As agreed in class, the database and the client run in separate containers, also for
   embedded databases. The db container holds all dataset files, the database, the goals and the benchmark; the client
   container only sends goals over HTTP. Both build from one `Dockerfile` (`ROLE=db` installs LanceDB, `ROLE=client` does not).
3. **Schema (DDL as code).** `id: int64` (row position in `train.csv`), `vector: fixed_size_list<float32>[dim]`, one
   table per dataset. Vectors must be float32: LanceDB cannot index uint8 vectors with L2. Goal 4 adds the column
   `cluster_id` (int32, bitmap index).
4. **Ingest.** `train.csv` is streamed in batches of 10,000 rows (up to 4 GB files, never loaded whole). Checks: row count,
   ids equal the 0-based row position (the ground truth refers to them), and a few stored vectors equal the CSV.
5. **Index.** IVF_PQ with the dataset's metric plus a BTree on `id`, then `optimize()`. Loading and indexing are timed
   together as one number (`ingest_s`), and the index must reach recall@10 of at least 0.95 at some `nprobes` setting
   (`populate.py` checks this on 200 test queries and warns otherwise).
6. **Setup check.** `scripts/check_setup.py` verifies the whole chain per dataset in the db container (section 9).
7. **Goals and benchmark.** Goals 1-7 are queries and K-means; goal 8 runs the benchmark protocol of the baseline:
   200 warm-up queries, 3 passes over the first n test queries, recall@k against `ground_truth.csv`, p50, p95 and p99 of
   the search call measured inside the db container, one row per accuracy setting in `search.csv`.

## 6. Goals and how they are addressed

The goals and their parameters follow the baseline document (proposals). The `db` command runs on the client side and calls
the wrapper (dataset from `DATASET`, default `glove-200-angular`). A goal prints one result per line to stdout (`id distance`,
nearest first, unless stated otherwise), notes go to stderr. Exit code: 0 success, 2 invalid parameters (including a k above
the dataset's true neighbors per query), 1 any other failure. Every neighbor goal leaves out the vector itself (it asks for
k + 1 and drops it). The vector search inside the db container is:

```python
table.search(vec).distance_type(metric).nprobes(20).refine_factor(5).limit(k).select(["id", "_distance"])
```

| Goal | Command | Logic |
|---|---|---|
| 1 | `db 1 <id> <k>` | Looks up the vector by id (BTree), searches k + 1 neighbors with the dataset's metric, drops the vector itself. |
| 2 | `db 2 <id> <radius>` | Searches the 100 nearest neighbors (at most 100 results) and keeps those with distance <= radius; the number of results goes to stderr. |
| 3 | `db 3 <id> <k> <percent>` | Same as goal 1 with the prefilter `id % 100 < percent` (`where(..., prefilter=True)`). |
| 4 | `db 4 <clusters> <iterations>` | K-means in numpy inside the db container (outside the engine): unit-length vectors, cosine distance, ids 0 to clusters-1 as start centers, centers become member means, exactly `iterations` iterations, no early stop, no random numbers. The cluster id of every vector is stored in the column `cluster_id` (`merge_insert`, bitmap index). Prints `cluster_id size` per cluster; time goes to `kmeans_s` in `build.csv`. |
| 5 | `db 5 <id>` | Reads `cluster_id` of the row; exit code 1 if goal 4 has not run yet. |
| 6 | `db 6 <id> <k>` | Same as goal 1 with the prefilter `cluster_id = <cluster of id>`. |
| 7 | `db 7 <id_a> <id_b>` | Fetches both stored vectors and prints the cosine distance (1 - cosine similarity, float64). |
| 8 | `db 8 <n_queries> <k>` | The benchmark: for each `nprobes` in 5, 10, 20, 50, 100, 200 it runs 200 warm-up queries and 3 passes over the first n test queries (one at a time, ids only), prints recall@k and p50, p95, p99, and appends the rows to `/results/search.csv`. |

Result files (columns as in the baseline): `build.csv` has one row per populate run (`ingest_s` is loading plus indexing,
`peak_mem_mb` is the process RSS sampled every 100 ms with psutil); `search.csv` has one row per dataset, k and `nprobes`.

## 7. Roles

TBA


## 8. Results

### 8.1 Native run of the baseline goals (`cohere-small-100k`)

**Environment.** Apple M3 Mac (8 cores, 17.2 GB RAM), native (no Docker), LanceDB 0.40.0, Python 3.12, warm, one run. The
laptop was in normal use, so treat the timings as indicative. Native macOS is not the Linux test server. The files are the
real `cohere-small-100k` files; the commands were `populate.py`, the `db` CLI against a local server, and `check_setup.py`.
This run was made before the Docker run in section 8.2.

| Measure | Value |
|---|---|
| Loading + IVF_PQ (32 partitions, 96 sub-vectors) + BTree + optimize | 151.7 s in total (load 16.1 s, index about 135 s) |
| Database size after `optimize()` | 314 MB |
| Peak memory (RSS, 100 ms sampling) while loading and indexing | 1986 MB |
| Index recall@10 on 200 test queries, `nprobes` 5 / 10 / 20 / 50 | 0.9435 / 0.9755 / 0.982 / 0.9825 |
| Goal 4 (`db 4 100 20`): time, peak memory | 1.7 s, 1659 MB |
| Cluster sizes (100 clusters of 100,000 vectors) | min 308, median 941, max 1667 |

Goal 8 (1,000 queries, 3 passes, 200 warm-up queries, `refine_factor=5`), inside the db process, ids only:

| k | `nprobes` | recall | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|
| 10 | 5 | 0.9302 | 3.97 | 6.14 | 9.80 |
| 10 | 10 | 0.9665 | 4.95 | 7.03 | 9.73 |
| 10 | 20 | 0.9772 | 6.96 | 8.72 | 11.56 |
| 10 | 50 | 0.9775 | 8.03 | 11.84 | 15.44 |
| 100 | 5 | 0.8944 | 7.19 | 10.43 | 12.52 |
| 100 | 10 | 0.9582 | 8.52 | 10.95 | 13.14 |
| 100 | 20 | 0.9804 | 10.65 | 12.70 | 15.17 |
| 100 | 50 | 0.9812 | 11.54 | 14.16 | 16.00 |

At k = 10 the lowest setting with recall@10 of at least 0.95 is `nprobes=10` (p50 4.95 ms). `check_setup.py` passed for this
dataset: exact search reproduces the ground truth (recall@10 = 1.0, distances differ by at most 2.3e-07), and goals 1, 3 and 7
gave valid answers. Error handling was tried by hand: invalid ids, k, radius, a non-number and an unknown goal gave exit code 2,
goal 5 before goal 4 gave exit code 1. Goals 2 and 6 were run by hand and goal 6 returned 10 neighbors for all 100 test ids
(a cluster is never smaller than 10 here); other datasets and Linux were not tested.

**What this run shows about the proposed parameters.**

- Goal 2 with the proposed radius for this dataset (0.142, the median distance of the 10th true neighbor of the *test
  queries*) returned a median of 13.5 and a mean of 30 results for the 100 test *ids* (stored vectors), from 0 up to the limit
  of 100. The "about 10 results" holds for test queries but not for stored vectors, so the result counts vary a lot
  between ids.
- Goals 3 and 6 returned 10 neighbors for all tested ids and percentages (1, 10, 50).
- The proposed `db 4 100 20` runs in 1.7 s on this dataset; the 1,000 vectors per cluster are on average, with sizes between 308 and 1667.

### 8.2 Docker run of the baseline setup (`cohere-small-100k`)

**Environment.** Docker Desktop on the same Apple M3 Mac: Linux VM (aarch64) with 8.3 GB RAM, the db container pinned to 4
cores (0-3) with an 8 GB limit, the benchmark container not pinned, LanceDB 0.40.0, one run, warm. Commands: `docker compose
run --rm populate`, then `db 1 42 5`, `db 4 100 20` and the default `benchmark` command, all with
`DATASET=cohere-small-100k`. The rows are in `results/build.csv` and `results/search.csv`.

| Measure | Docker | Native (8.1) |
|---|---|---|
| Loading + indexing (`ingest_s`) | 118.2 s | 151.7 s |
| Database size | 314.6 MB | 314 MB |
| Peak memory loading + indexing | 1718 MB | 1986 MB |
| Goal 4 (`db 4 100 20`): time, peak memory | 8.8 s, 1482 MB | 1.7 s, 1659 MB |
| Cluster sizes | min 308, max 1665 | min 308, max 1667 |

| k | `nprobes` | recall | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|
| 10 | 5 | 0.9298 | 6.47 | 10.24 | 17.83 |
| 10 | 10 | 0.9643 | 7.42 | 9.64 | 17.76 |
| 10 | 20 | 0.9744 | 10.01 | 11.92 | 13.61 |
| 10 | 50 | 0.9750 | 10.94 | 12.37 | 12.94 |
| 100 | 5 | 0.8975 | 16.05 | 18.84 | 21.99 |
| 100 | 10 | 0.9588 | 16.81 | 19.31 | 24.47 |
| 100 | 20 | 0.9793 | 18.67 | 20.40 | 22.33 |
| 100 | 50 | 0.9802 | 19.79 | 21.45 | 23.18 |

**All six datasets in Docker** (`DATASET=all docker compose run --rm populate`, same environment, default index parameters
of `config.index_defaults`, `nprobes` up to 50, one run; the rows are in `results/build.csv`). The recall columns are the populate check on the
first 200 test queries, k = 10, `refine_factor=5`:

| Dataset | Loading + indexing | Index (partitions, sub-vectors) | Database size | Peak memory | Recall@10 at `nprobes` 5 / 10 / 20 / 50 |
|---|---|---|---|---|---|
| `cohere-small-100k` | 125 s | 32, 96 | 315 MB | 1810 MB | 0.916 / 0.968 / 0.977 / 0.977 |
| `nytimes-256-angular` | 46 s | 128, 32 | 306 MB | 1652 MB | 0.609 / 0.683 / 0.767 / **0.838** |
| `pubmed-arxiv` | 151 s | 128, 96 | 1255 MB | 2519 MB | 0.872 / 0.916 / 0.934 / **0.939** |
| `uform-wiki-1m` | 92 s | 512, 32 | 1056 MB | 2175 MB | 0.931 / 0.951 / 0.956 / 0.958 |
| `glove-200-angular` | 73 s | 512, 25 | 982 MB | 2205 MB | 0.518 / 0.567 / 0.600 / **0.628** |
| `ccnews-nomic-768-normalized` | 189 s | 256, 96 | 1553 MB | 3252 MB | 0.878 / 0.932 / 0.952 / 0.959 |

Populate worked for all six. Three datasets (bold) do not reach the baseline's requirement of recall@10 of at least 0.95
at any of the four `nprobes` settings with the default index, so populate printed its warning for them.

**Tuning the three datasets** (native runs on the Mac, `scripts/populate.py` check on 200 test queries, `refine_factor=5`;
recall only, the timings of those runs are not comparable because several builds ran at the same time). Two findings:

- **The number of probed partitions was the main limit, not only the compression.** An `IvfFlat` index (no compression)
  with the same 512 partitions gave the same recall as IVF_PQ with 100 sub-vectors on glove-200 (0.853 against 0.854 at
  `nprobes=50`) and needed `nprobes=200` to pass 0.95 (0.960). So `NPROBES_SETTINGS` now goes up to 200.
- **Too few sub-vectors cost recall.** With the default sub-vectors, glove-200 and nytimes lost most: 0.63 at
  `nprobes=50` (25 sub-vectors, 512 partitions), against 0.83 with 50 and 0.85 with 100. `refine_factor` 20 and 50
  instead of 5 changed recall by at most 0.04 and usually by less than 0.01.

The chosen values (`config.INDEX_PARAMS`) and the recall of the populate check afterwards:

| Dataset | Default (partitions, sub-vectors) | Tuned | Recall@10 at `nprobes` 5 / 10 / 20 / 50 / 100 / 200 with the tuned index | First setting >= 0.95 |
|---|---|---|---|---|
| `glove-200-angular` | 512, 25 | 256, 100 | 0.706 / 0.786 / 0.845 / 0.913 / 0.966 / 1.000 | `nprobes=100` |
| `nytimes-256-angular` | 128, 32 | 128, 64 | 0.642 / 0.729 / 0.824 / 0.927 / 0.969 / 0.969 | `nprobes=100` |
| `pubmed-arxiv` | 128, 96 | 128, 192 | 0.907 / 0.966 / 0.988 / 0.9995 / 1.000 / 1.000 | `nprobes=10` |

Only these three datasets were changed; `cohere-small-100k`, `uform-wiki-1m` and `ccnews-nomic-768-normalized` already
reach 0.95 with the defaults (`uform-wiki-1m` and `ccnews-nomic-768-normalized` only barely, at `nprobes=10` and 20). Costs
of the tuned indexes (native Mac, so not comparable with the Docker numbers above): loading + indexing 70 s, 88 s and 207 s,
database size 1148 MB, 323 MB and 1326 MB. The nytimes row reaches only 0.969 at `nprobes=200`, which is all of the
128 partitions: that is the limit of 64 sub-vectors, and 128 sub-vectors gave 0.931 at `nprobes=50` against 0.912 (tested with
the older, smaller nprobes grid). Whether the settings are the best choice (for example fewer partitions for glove-200: 128
partitions with 100 sub-vectors reached 0.965 at `nprobes=50`, but each probe scans more) was not explored further.

**All six datasets in Docker with the tuned indexes** (`DATASET=all`: `populate`, `check_setup.py` and the default
`benchmark` command, same environment, one warm run, 1000 queries x 3 passes after 200 warm-up queries, `refine_factor=5`;
rows in `results/build.csv` and `results/search.csv`). All six pass `check_setup.py` (exact search recall@10 of at least 0.998
against the ground truth) and now reach recall@10 of at least 0.95 at some `nprobes` setting:

| Dataset | Loading + indexing | Database size | Peak memory (build) | First `nprobes` with recall@10 >= 0.95 | p50 / p95 / p99 there | Best recall@10 (`nprobes=200`) |
|---|---|---|---|---|---|---|
| `cohere-small-100k` | 105 s | 314 MB | 1825 MB | 10 (0.966) | 6.0 / 8.9 / 12.7 ms | 0.977 |
| `nytimes-256-angular` | 55 s | 323 MB | 1692 MB | 100 (0.953) | 13.0 / 17.3 / 25.5 ms | 0.953 |
| `pubmed-arxiv` | 195 s | 1326 MB | 2629 MB | 10 (0.969) | 6.9 / 8.6 / 10.4 ms | 1.000 |
| `uform-wiki-1m` | 80 s | 1051 MB | 2286 MB | 10 (0.957) | 3.7 / 4.4 / 5.3 ms | 0.960 |
| `glove-200-angular` | 85 s | 1148 MB | 2421 MB | 100 (0.967) | 20.1 / 25.0 / 34.8 ms | 0.997 |
| `ccnews-nomic-768-normalized` | 163 s | 1550 MB | 3372 MB | 20 (0.954) | 8.4 / 11.0 / 13.9 ms | 0.962 |

- **The tuning worked in Docker:** the three tuned datasets now pass, with recall values close to the native check (glove
  0.9665 against 0.966, nytimes 0.953 against 0.969, pubmed 0.9685 against 0.966). nytimes is the one that moved: 0.953
  on 1000 queries against 0.969 on the first 200, so it sits only just above the target and is the weakest case.
- **The price is latency at high `nprobes`:** glove-200 and nytimes need `nprobes=100`, so their p50 (20 and 13 ms) is two to
  five times that of the datasets that pass at `nprobes=10`. At k = 100 the first setting with recall@100 of at least 0.95 is
  `nprobes=200` for glove-200 (0.938 at 100, 0.996 at 200), `nprobes=10` for pubmed
  (0.950), `nprobes=20` for cohere (0.980) and `nprobes=50` for ccnews (0.958); nytimes does not pass 0.95 at k = 100
  (0.949 at best). The 0.95 requirement of the baseline is for k = 10.
- `uform-wiki-1m` has no k = 100 run: its ground truth holds 10 neighbors, so the goal refuses k above 10.
- **`check_setup.py` prints a lower recall** for nytimes and glove (0.828 and 0.834) than the benchmark, because it checks the
  approximate search at the default `nprobes=20`, not at the setting that reaches 0.95.
- Latencies and peak memory come from one run each and were not repeated.

What it shows:

- **The whole setup works in Docker:** populate (all files read from `/data`), goal 1 through the client container, goal 4
  and the benchmark (goal 8), with the result files written to `./results`. At k = 10 the lowest setting with recall@10 of at
  least 0.95 is `nprobes=10` (p50 7.42 ms), the same setting as natively.
- **Recall matches the native run** (for example 0.9643 against 0.9665 at k = 10 and `nprobes=10`; the index is rebuilt in
  each run, so a difference of about 0.002 is expected). The cluster sizes also agree closely (min 308 in both; max 1665
  against 1667), as expected for a fixed start and 20 iterations, but not bit-identical, which we did not investigate
  (parallel floating-point sums and the different BLAS libraries of macOS and Linux are possible causes).
- **Latency is higher in Docker**: p50 6.5 to 19.8 ms against 4.0 to 11.5 ms natively, about 1.6 to 2 times. That fits a Linux
  VM on 4 pinned cores against native macOS, but we did not separate the causes. Goal 4 is also slower (8.8 s against 1.7 s,
  probably the BLAS library without the Mac's accelerated matrix routines; not tested). Indexing was faster in Docker
  (118 s against 152 s), which is not what the core count suggests, so the native number was probably disturbed by other
  work on the laptop (single runs, not repeated).
- Docker Desktop on macOS is not the Linux test server, so these numbers show that the setup works, not what the server will give.

### 8.3 Earlier Docker run (client-side timing, before the baseline alignment)

That run used the previous layout (train data in the db container, test files in the client container, latency measured by the
client over HTTP). It is kept because it covers all six datasets; its commands no longer exist in this form.

**Environment.** Docker Desktop on an Apple M3 Mac (8 cores, 17 GB RAM). The containers run in a Linux VM (aarch64) that
shows 8.3 GB RAM. The db container is pinned to 4 cores (0-3) with an 8 GB limit, the client container is not pinned.
LanceDB 0.40.0, one run per dataset, warm. The numbers come from the ingest and index scripts and from
`scripts/check_setup.py` (50 test queries, k=10, one pass, no warm-up), so they are a first end-to-end run of the
setup and not the benchmark protocol (200 warm-up queries, 3 passes), which has not been run in Docker yet.
Peak RSS there is `ru_maxrss` of the script process; the new scripts sample RSS every 100 ms as the baseline proposes.

| Dataset | Metric | Ingest | Index build (partitions, sub-vectors) | Index size | Peak RSS ingest / index | Exact recall@10 | Approx. recall@10 | Client / engine ms per query |
|---|---|---|---|---|---|---|---|---|
| `cohere-small-100k` | cosine | 5.4 s | 94.7 s (32, 96) | 10.4 MB | 1086 / 1208 MB | 1.000 | 0.976 | 8.5 / 7.2 |
| `nytimes-256-angular` | cosine | 5.8 s | 33.2 s (128, 32) | 9.9 MB | 790 / 792 MB | 0.998 | 0.802 | 5.7 / 4.7 |
| `glove-200-angular` | cosine | 17.9 s | 36.4 s (512, 25) | 33.4 MB | 760 / 912 MB | 1.000 | 0.564 | 5.8 / 4.9 |
| `uform-wiki-1m` | cosine | 20.3 s | 44.0 s (512, 32) | 33.5 MB | 771 / 879 MB | 1.000 | 0.956 | 7.0 / 6.0 |
| `pubmed-arxiv` | cosine | 23.9 s | 107.6 s (128, 96) | 38.7 MB | 1255 / 1502 MB | 1.000 | 0.942 | 9.4 / 8.2 |
| `ccnews-nomic-768-normalized` | cosine | 29.6 s | 114.7 s (256, 96) | 47.6 MB | 1136 / 1470 MB | 1.000 | 0.928 | 9.3 / 8.2 |

(The run also included a seventh dataset, `wikipedia-cohere-100k`, which the project no longer uses; it is left out here.)

Approximate recall is with `nprobes=20`, `refine_factor=5` and the default index parameters of each dataset
(`config.index_defaults`). Peak RSS is the resident memory of the script process.

**What the run shows**

- **The setup works for all six datasets.** Row counts and dimensions equal the manifests, the vector index uses the
  metric of the registry, lookup by id works, and exact search reproduces the ground truth (recall@10 of 1.0, 0.998 for
  `nytimes-256-angular`). For all six datasets the server's `_distance` matches the distances in the file to
  at most 7.2e-07, so cosine is the right metric for all of them, including the ones that were assumptions
  (angular, normalized, unit-norm vectors).
- **Ingest is fast and light.** 5 to 30 s per dataset, about 165 to 178 MB of CSV per second, with at most 1.3 GB of
  memory, although the files are up to 5.2 GB.
- **Index build takes 33 to 115 s, and the time does not follow the row count.** The 100k-row 768-dimension datasets
  (95 s) took nearly as long as the 400k and 495k-row ones (108 and 115 s), while the 1M-row 256-dimension and 1.18M-row
  200-dimension datasets took 44 and 36 s. This suggests the time is dominated by training the product-quantization
  codebooks (96 sub-vectors for 768 dimensions) rather than by the number of rows; we did not test that directly.
  Natively on the Mac the same builds were faster (72 s instead of 95 s for `cohere-small-100k`, 23 s instead of 33 s for
  `nytimes-256-angular`), consistent with the container's 4 pinned cores, but these are single runs.
- **Memory is far below the limit.** The largest peak was 1.5 GB against the 8 GB limit, so a 4 GB limit would probably
  also be enough for these datasets (not tested; RSS does not include the page cache).
- **The HTTP wrapper adds 0.9 to 1.3 ms per query** (client minus engine time). Per-query latency is 5.7 to 9.4 ms on
  the client, higher for 768 dimensions. These come from only 50 queries, so they are indicative.
- **Approximate recall at the same settings differs a lot between datasets**: 0.98, 0.96, 0.94 and 0.93 for cohere,
  uform, pubmed and ccnews, but 0.80 for nytimes and 0.56 for glove-200-angular. The settings are not tuned per
  dataset, and the index parameters differ too (for example 20 of 32 partitions are probed for cohere but 20 of 512 for
  uform), so the differences mix data and parameters. Recall of 50 queries is noisy: `nytimes-256-angular` gave 0.762
  natively and 0.802 in Docker, with a new index each time.
- **Disk.** The logs of this run printed the size of the whole `./db` folder, so the numbers grew from dataset to
  dataset (5,767 MB at the end, for all tables of that run). The scripts
  now print the size of the table itself.

**Other measurements.** A native (no Docker) run of the old client-side `scripts/bench.py` on `cohere-small-100k` (1,000 queries, 3 passes,
`results/cohere-small-100k/bench_reference.csv`) gave, at k=10 and `nprobes=20`: recall 0.976, client p50 5.8 ms, engine
p50 4.8 ms, 173 QPS; exact search took 37.4 ms (27 QPS).

**Earlier glove-200-cosine test** (single container, LanceDB called in-process, HDF5; code and result files are in the
first commit `33e2d9c`). On 1.19M vectors, the main findings were: exact search 109 ms per query against 1 to 12 ms
for the index; a larger `refine_factor` helps recall more than more `nprobes`; more sub-vectors help most (recall@10 of
0.80 with 25 sub-vectors and 0.93 with 50, at `nprobes=100` and `refine_factor=20`, for 8 s more build time and 30 MB
more index); and index builds are not exactly repeatable (about 0.01 recall between builds).

## 9. User manual and reproduction

Each step prints what it did, and the scripts are safe to re-run. The commands below follow the standard commands of the
baseline (section 2); the dataset comes from `DATASET` (default `glove-200-angular`, or `all`).

**1. Download the data** (on the host, once; needs `uv sync` and network access). The four files of each dataset go to
`./data/<dataset>/`, which the db container mounts read-only as `/data`:

```bash
uv sync
```

```bash
uv run python scripts/download.py --dataset cohere-small-100k
```

`--dataset all` downloads all six (about 18 GB); files that already exist are skipped. If you have the data in the old
layout (`data/train/<dataset>/train.csv` and `data/queries/<dataset>/...`), move the files so that every dataset folder holds
all four files.

**2. Run with Docker** (replace `docker compose` with `podman compose` if needed). Docker Desktop needs at least 4 CPUs
and 8 GB RAM. Delete `./db` first if you want to start from an empty database.

```bash
docker compose build
docker compose up -d db
DATASET=cohere-small-100k docker compose run --rm populate                  # load + index, one row in results/build.csv
DATASET=cohere-small-100k docker compose run --rm benchmark db 1 42 10      # goal 1: neighbors of vector 42
DATASET=cohere-small-100k docker compose run --rm benchmark                 # goal 8 for k = 10 and 100, rows in results/search.csv
```

The other goals: `db 2 <id> <radius>`, `db 3 <id> <k> <percent>`, `db 4 100 20`, `db 5 <id>`, `db 6 <id> <k>`,
`db 7 <id_a> <id_b>`, `db 8 1000 10`. `docker compose run --rm benchmark db` prints the list.

To populate and check every dataset:

```bash
DATASET=all docker compose run --rm populate
```

```bash
DATASET=all docker compose run --rm populate python scripts/check_setup.py
```

`populate.py` goes on with the next dataset if one fails and exits with 1 at the end; `check_setup.py` also exits with 1
if any dataset fails, and a goal exits with 2 for invalid parameters, so all can be used in an automated run.

Watch memory and CPU with `docker stats` in a second terminal. If a command prints `Killed` or exits with 137, the
container ran out of memory: raise Docker's memory limit. `docker compose down` stops the containers and keeps `./db`.

**What `check_setup.py` verifies** (it prints PASS, FAIL, WARN or INFO per line, and exits with 1 on any FAIL):
the dataset files exist; the table's row count and dimension equal the manifest; the vector index exists, covers every row
and uses the dataset's metric; exact (brute-force) search reproduces the ground truth (recall@10 at least 0.98 over 50
queries), which catches a wrong metric or id mapping; approximate search through the index returns results (it passes with
any recall above zero, so read the printed recall); and goals 1, 3 and 7 give sensible answers.

**Without Docker** (after step 1; two terminals, same host):

```bash
uv sync --extra db
DATASET=cohere-small-100k uv run python scripts/populate.py
DATASET=cohere-small-100k uv run uvicorn lancedb_bench.server:app --port 8000     # terminal 1
DATASET=cohere-small-100k uv run db 1 42 10                                       # terminal 2
```

Notes:

- The running server keeps table handles open. `populate.py` and `scripts/bench.py` call `POST /reload`, goal 4 refreshes
  the handle itself, and a goal on an outdated handle is retried once on a fresh one.
- `build_index.py` removes old table versions (so the reported size is the live size). LanceDB warns that this can
  break other processes still reading an older version, so do not benchmark while it runs.
- Goal 8 runs for many minutes (6 settings x (200 + 3 x n) searches, the high `nprobes` are slower), and the client waits without a timeout. Change the swept
  settings with `NPROBES_SETTINGS=5,10,20,50,100,200` (db container) and the name of the group with `GROUP`.

## 10. Self-evaluation

What worked: all six datasets can be imported (streamed, even the 5 GB files, with at most 1.5 GB of memory), indexed
and queried through the two-container setup in Docker; exact search reproduces every dataset's ground truth, which
also confirmed the metric of each dataset; and `scripts/check_setup.py` gives a single PASS/FAIL answer per dataset.
Adding a dataset needs one line of configuration.

Shortcomings:

- **Recall is tuned only enough to pass 0.95.** The default index parameters gave recall@10 from 0.98 down to 0.56 at
  `nprobes=20` in the earlier Docker run (section 8.3). Three datasets got tuned parameters (section 8.2), checked natively with
  200 test queries; the other three keep the defaults. The goals use fixed settings, the populate check only warns, and we
  did not explore the best trade-off between index size, build time and speed.
- **The measurements are first runs, not the benchmark.** The new setup was run for one dataset (`cohere-small-100k`),
  natively and in Docker Desktop on a Mac, once each; the six-dataset run is from the earlier Docker layout, with 50
  queries per dataset and no cold-cache run. Differences of about 0.01 to 0.04 recall between index builds are noise.
  The other five datasets were populated with the new setup but not benchmarked (goal 8) or checked with `check_setup.py`,
  and the benchmark client container is not CPU-pinned.
- Peak memory is the RSS of the database process sampled every 100 ms (page cache not counted), not a cgroup measurement.
  For goal 4 it includes the numpy copy of all vectors.
- `check_setup.py` passes the approximate-search check with any recall above zero, so a poor index still shows PASS; read the printed recall.
- Goals 1 to 8 exist but follow proposals the class has not approved. K-means runs outside the engine (numpy) and loads
  all vectors into memory (about 1 GB for glove-200, not tested there); range search (goal 2) takes the 100 nearest
  neighbors and filters them; the index-build sweep of the early glove test has not been rebuilt for the two-container setup.
- The goal 2 radius of the baseline (about 10 results per test query) gives 0 to 100 results for stored vectors (section 8.1).

Future work: run the full benchmark protocol and sweeps in Docker for all datasets,
ideally on native Linux with cold-cache runs; choose `nprobes`, `refine_factor` and index parameters per dataset
for target recalls of 0.90, 0.95 and 0.99; and adapt the goals to what the class approves.
