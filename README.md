# Advanced Database Systems 2026: LanceDB benchmark

Authors: Zuzanna Jarlaczyńska, Tomasz Strycharz
Course: Advanced Database Systems 2026

Vector-search benchmark of **LanceDB** (open-source, embedded) on six datasets
(`cohere-small-100k`, `nytimes-256-angular`, `pubmed-arxiv`, `uform-wiki-1m`, `glove-200-angular`,
`ccnews-nomic-768-normalized`), prepared once and published in the Hugging Face dataset repo `zuzaj/ADB`.
The database runs in its own container behind a small HTTP wrapper and a separate client container sends the
queries, as agreed in class for all databases, embedded ones included. Adding a dataset needs one line of
configuration (section 4).

> Status: work in progress. Goals 2-7 and the common performance protocol are still being defined in class.
> The results in section 8 are a first end-to-end run in Docker, not the final benchmark.

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
 db container (lancedb_server)                       client container (lancedb_client)
 +-----------------------------------+   HTTP :8000   +----------------------------------+
 | FastAPI wrapper (server.py)       | <------------- | check_setup.py, bench.py, db CLI |
 | embedded LanceDB, directory ./db  |  ids, distances| test.csv, ground_truth.csv,      |
 | train.csv  --ingest--> table      |  engine time   | manifest.json                    |
 +-----------------------------------+                +----------------------------------+
```

- The **db container** holds `train.csv` and the LanceDB directory. The `ingest` and `build_index` scripts run
  inside it, and the wrapper answers search requests. It never sees the test data.
- The **client container** holds only `test.csv`, `ground_truth.csv` and `manifest.json`, sends each query over HTTP and
  computes recall itself. It does not contain the database engine (`lancedb` is not even installed there) and never
  sees `train.csv`.
- The wrapper returns only `id` and `_distance` plus its own engine time (`engine_latency_ms`, measured around the
  LanceDB query), so the client can report total latency, engine latency, and their difference (HTTP overhead).
- Both containers are built from the same `Dockerfile` with a `ROLE` argument (`db` or `client`); `docker-compose.yml`
  limits the db container to 4 pinned cores and 8 GB RAM with swap off.

| Path | Purpose |
|---|---|
| `scripts/download.py` | Download the datasets from Hugging Face: train files into `data/db`, test files into `data/client` |
| `scripts/ingest.py` | db side: stream `train.csv` into a LanceDB table and verify it |
| `scripts/build_index.py` | db side: build IVF_PQ and BTree indexes, `optimize()`, print sizes |
| `scripts/check_setup.py` | client side: check that the whole setup works for each dataset (section 9) |
| `scripts/bench.py` | client side: latency, QPS and recall over HTTP, one CSV row per configuration and k |
| `src/lancedb_bench/server.py` | db side: the FastAPI wrapper (`/search`, `/datasets`, `/info`, `/health`, `/reload`) |
| `src/lancedb_bench/client.py` | client side: HTTP client with client-side timing |
| `src/lancedb_bench/cli.py` | The `db` command (client side) |
| `src/lancedb_bench/config.py` | Paths, the dataset registry (name -> metric) and index defaults |
| `src/lancedb_bench/data.py` | Streaming readers for the CSV formats |
| `src/lancedb_bench/metrics.py` | Recall, latency and size helpers |
| `results/<dataset>/` | Result CSVs, one folder per dataset |

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
`SERVER_URL` (default `http://localhost:8000`) tells the client where the wrapper is.

### Datasets: layout, selecting and adding

Each dataset is a folder with four files (`manifest.json` of each dataset gives the numbers below):

| File | Content | Goes to |
|---|---|---|
| `train.csv` | `id,"[v0,v1,...]"`, one vector per row, ids are 0-based row positions | db container (`./data/db/<dataset>/`) |
| `test.csv` | same format, the query vectors | client container (`./data/client/<dataset>/`) |
| `ground_truth.csv` | `query_id,rank,neighbor_id,distance`, true neighbors of each query | client container |
| `manifest.json` | row count, query count, dimension, how the files were made | client container |

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
them into the layout the containers need, so that each container sees only its own files
(`data/db/<dataset>/train.csv` and `data/client/<dataset>/{test.csv,ground_truth.csv,manifest.json}`):

```bash
uv run python scripts/download.py --dataset cohere-small-100k
```

`--dataset all` downloads all six (about 18 GB). Files that are already complete are skipped.

Every script takes `--dataset <name>` (or `all`); the `db` command reads the `DATASET` environment variable.
The default is `cohere-small-100k`, the smallest one.

**To add a dataset:** add a folder with the same four files to the Hugging Face repo, add one line to `DATASETS` in
`config.py` (name and metric), and run the same commands with the new name. Each dataset gets its own table in the
database and its own folder in `results/`. Dimension and row count are read from the files, and index parameters default
to about 2,300 vectors per partition and dimension/8 sub-vectors (`config.index_defaults`); override them with
`build_index.py --partitions N --sub-vectors M`. Ground truth may have fewer neighbors than 100 (`uform-wiki-1m` has 10);
`bench.py` skips values of k that are too large for it.

Limits: LanceDB's L2 `_distance` is the squared distance, so if you add an L2 dataset, compare against the ground
truth with `check_setup.py` (it prints the difference between the two distance conventions). The CLI's `nprobes` and
`refine_factor` are fixed (20 and 5, override with `NPROBES` and `REFINE_FACTOR`), not tuned per dataset.

## 5. Design and implementation process

1. **Early single-container test.** First the whole pipeline was built and measured with LanceDB called in-process on
   VIBE `glove-200-cosine` (HDF5): exact search reproduced the ground truth, and the IVF_PQ index and the
   `nprobes` / `refine_factor` trade-offs were measured (end of section 8). That code was replaced by the CSV import below
   and is only in the first commit (`33e2d9c`).
2. **Two-container split.** As agreed in class, the database and the client run in separate containers, also for
   embedded databases. The db container gets `train.csv` and the wrapper; the client gets only the test files and talks
   HTTP. Both build from one `Dockerfile` (`ROLE=db` installs LanceDB, `ROLE=client` does not).
3. **Schema (DDL as code).** `id: int64` (row position in `train.csv`), `vector: fixed_size_list<float32>[dim]`, one
   table per dataset. Vectors must be float32: LanceDB cannot index uint8 vectors with L2.
4. **Ingest.** `train.csv` is streamed in batches of 10,000 rows (up to 4 GB files, never loaded whole). Checks: row count,
   ids equal the 0-based row position (the ground truth refers to them), and a few stored vectors equal the CSV.
5. **Index.** IVF_PQ with the dataset's metric plus a BTree on `id`, then `optimize()`.
6. **Setup check.** `scripts/check_setup.py` verifies the whole chain per dataset from the client side (section 9).
7. **Benchmark.** `scripts/bench.py`: 200 warm-up queries, then 3 passes over the first N test queries (default 1,000),
   recall@k against `ground_truth.csv`, latency measured on the client (total) and in the server (engine).

## 6. Goals and how they are addressed

The goals are not yet defined; we implemented two as examples. The `db` command runs on the client side and calls the
wrapper (dataset from `DATASET`, default `cohere-small-100k`).

| Goal | Command | Logic |
|---|---|---|
| 1 | `db 1 <id> <k>` | The wrapper looks up the vector by id (`where id = <id>`, served by the BTree), searches `k+1` neighbors with the dataset's metric, drops the vector itself, returns `k`. |
| 8 | `db 8 <n_queries> <k>` | The client sends the first `n` test queries, computes recall@k against the ground truth, and prints client-total and server-engine latency percentiles. |

The vector query inside the wrapper is:

```python
table.search(vec).distance_type(metric).nprobes(20).refine_factor(5).limit(k).select(["id", "_distance"])
```

## 7. Roles

TBA


## 8. Results

**Environment.** Docker Desktop on an Apple M3 Mac (8 cores, 17 GB RAM). The containers run in a Linux VM (aarch64) that
shows 8.3 GB RAM. The db container is pinned to 4 cores (0-3) with an 8 GB limit, the client container is not pinned.
LanceDB 0.40.0, one run per dataset, warm. The numbers come from the ingest and index scripts and from
`scripts/check_setup.py` (50 test queries, k=10, one pass, no warm-up), so they are a first end-to-end run of the
setup and not the benchmark protocol (200 warm-up queries, 3 passes), which has not been run in Docker yet.

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

**Other measurements.** A native (no Docker) run of `scripts/bench.py` on `cohere-small-100k` (1,000 queries, 3 passes,
`results/cohere-small-100k/bench_reference.csv`) gave, at k=10 and `nprobes=20`: recall 0.976, client p50 5.8 ms, engine
p50 4.8 ms, 173 QPS; exact search took 37.4 ms (27 QPS).

**Earlier glove-200-cosine test** (single container, LanceDB called in-process, HDF5; code and result files are in the
first commit `33e2d9c`). On 1.19M vectors, the main findings were: exact search 109 ms per query against 1 to 12 ms
for the index; a larger `refine_factor` helps recall more than more `nprobes`; more sub-vectors help most (recall@10 of
0.80 with 25 sub-vectors and 0.93 with 50, at `nprobes=100` and `refine_factor=20`, for 8 s more build time and 30 MB
more index); and index builds are not exactly repeatable (about 0.01 recall between builds).

## 9. User manual and reproduction

Each step prints what it did, and the scripts are safe to re-run. `--dataset all` runs a script for every dataset
in the registry; the commands below use the smallest one.

**1. Download the data** (on the host, once; needs `uv sync` and network access). The train files go to `./data/db`,
the test files to `./data/client`, which the containers mount read-only:

```bash
uv sync
```

```bash
uv run python scripts/download.py --dataset cohere-small-100k
```

`--dataset all` downloads all six (about 18 GB); files that are already complete are skipped.

**2. Run with Docker** (replace `docker compose` with `podman compose` if needed). Docker Desktop needs at least 4 CPUs
and 8 GB RAM. Delete `./db` first if you want to start from an empty database.

```bash
docker compose up -d --build db                                                       # start the database container
docker compose exec -T db python scripts/ingest.py --dataset cohere-small-100k        # build the table
docker compose exec -T db python scripts/build_index.py --dataset cohere-small-100k   # IVF_PQ + BTree
docker compose run --rm -T client python scripts/check_setup.py --dataset cohere-small-100k    # is everything working?
docker compose run --rm -T client python scripts/bench.py --dataset cohere-small-100k --exact  # results/<dataset>/bench.csv
docker compose run --rm -T -e DATASET=cohere-small-100k client db 1 42 10             # goal 1: neighbors of vector 42
```

To test every dataset (small to large; stops at the first failure):

```bash
for ds in cohere-small-100k nytimes-256-angular glove-200-angular uform-wiki-1m pubmed-arxiv ccnews-nomic-768-normalized; do docker compose exec -T db python scripts/ingest.py --dataset $ds && docker compose exec -T db python scripts/build_index.py --dataset $ds && docker compose run --rm -T client python scripts/check_setup.py --dataset $ds || break; done
```

Watch memory and CPU with `docker stats` in a second terminal. If a command prints `Killed` or exits with 137, the
container ran out of memory: raise Docker's memory limit. `docker compose down` stops the containers and keeps `./db`.

**What `check_setup.py` verifies** (it prints PASS, FAIL, WARN or INFO per line, and exits with 1 on any FAIL):
the server is reachable; the client has its test files (and warns if `train.csv` is on the client side); the table's row
count and dimension equal the manifest; the vector index exists, covers every row and uses the dataset's metric;
exact (brute-force) search reproduces the ground truth (recall@10 at least 0.98 over 50 queries), which catches a
wrong metric or id mapping; approximate search through the index returns results (it passes with any recall above zero, so read the printed recall);
and lookup by id works.

**Without Docker** (after step 1; two terminals, same host):

```bash
uv sync --extra db
DATA_DIR=./data/db uv run python scripts/ingest.py --dataset cohere-small-100k
DATA_DIR=./data/db uv run python scripts/build_index.py --dataset cohere-small-100k
DATA_DIR=./data/db uv run uvicorn lancedb_bench.server:app --port 8000                       # terminal 1
DATA_DIR=./data/client uv run python scripts/check_setup.py --dataset cohere-small-100k      # terminal 2
```

Notes:

- The running server keeps table handles open. After a new ingest or index build, `check_setup.py` and `bench.py` call
  `POST /reload`, and a search on an outdated handle is retried once on a fresh one.
- `build_index.py` removes old table versions (so the reported size is the live size). LanceDB warns that this can
  break other processes still reading an older version, so do not benchmark while it runs.
- `bench.py` defaults to the reference setting (`nprobes=20`, `refine_factor=5`, k=10 and 100); sweep with e.g.
  `--nprobes 1 5 20 100 --refine 1 5 20`.

## 10. Self-evaluation

What worked: all six datasets can be imported (streamed, even the 5 GB files, with at most 1.5 GB of memory), indexed
and queried through the two-container setup in Docker; exact search reproduces every dataset's ground truth, which
also confirmed the metric of each dataset; and `scripts/check_setup.py` gives a single PASS/FAIL answer per dataset.
Adding a dataset needs one line of configuration.

Shortcomings:

- **Recall is not tuned.** One setting (`nprobes=20`, `refine_factor=5`) and default index parameters give recall from
  0.98 down to 0.56. The CLI uses
  the same fixed setting.
- **The measurements are a first run, not the benchmark.** They come from one run per dataset on a laptop (Docker
  Desktop's Linux VM, not native Linux), warm, with 50 queries per dataset and no cold-cache run. Differences of about
  0.01 to 0.04 recall between index builds are noise. The benchmark protocol of `scripts/bench.py` (200 warm-up queries,
  3 passes, sweeps of `nprobes`, `refine_factor` and index parameters) has not been run in Docker for the six
  datasets, and the client container is not CPU-pinned.
- Peak memory is the script process's RSS (`ru_maxrss`), not a cgroup measurement.
- `check_setup.py` passes the approximate-search check with any recall above zero, so a poor index still shows PASS; read the printed recall.
- Only goals 1 and 8 of the `db` command exist. K-means, range search, filtered search and goals 2-7 are not
  implemented, and the index-build sweep of the early glove test has not been rebuilt for the two-container setup.

Future work: run the full benchmark protocol and sweeps in Docker for all datasets,
ideally on native Linux with cold-cache runs; choose `nprobes`, `refine_factor` and index parameters per dataset
for target recalls of 0.90, 0.95 and 0.99; and implement the remaining goals once they are defined.
