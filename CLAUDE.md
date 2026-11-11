# CLAUDE.md: LanceDB benchmark project, glove-200 test

## How to use this file

This is a context handoff from a long planning conversation. Read all of it before writing code.

- Facts marked **(verified)** were tested in a Linux sandbox with Python 3.12.3 and lancedb 0.40.0.
- Facts marked **(unverified)** come from documentation or reasoning and must be checked before you rely on them.
- Never invent numbers. Every figure in a report must come from a run you actually executed, with the environment recorded.
- If something here contradicts what you observe, trust your observation, say so, and update this file.

## 1. The project (university course)

- Course: Advanced Database Systems 2026 (instructor: Igor Wojnicki). Groups of 2 people. Each group gets a GitLab project; everything relevant must be on the `main` branch.
- Topic: vector databases. Starting points given by the instructor: the Wikipedia article "Vector database" and the VIBE benchmark (https://github.com/vector-index-bench/vibe).
- Our technology: **LanceDB (open-source embedded library)**. Other groups use other systems (for example pgvector, Chroma, Qdrant, DuckDB, YDB).
- Technology rules: as light as possible (minimal setup, easy to launch and evaluate), must run under Linux, free or open-source software preferred, architectural choices must be approved by the instructor.
- Self-sufficiency: the repo must contain code to set up dependencies, create database structures and populate them. Source data is fetched by a script or included in the repo.
- Documentation: a `README.md` in the repo root covering this checklist: (1) choice of technology, (2) architecture and components, (3) prerequisites, (4) installation and setup, (5) design and implementation process step by step, (6) how each goal is addressed including queries and logic, (7) roles of both students and who did what, (8) results with example runs and timings, (9) user manual and step-by-step reproduction, (10) self-evaluation (efficiency, shortcomings, future mitigation).
- Goals/queries (final deliverable, details still TBD with the class): "ML meaningful" queries about neighborhood, distance and similarity, for example KNN and K-means. All goals are numbered with specified parameters. A single CLI runs a goal: `db <goal> <params...>`, for example `db 1 443321 4` (goal 1, vector id 443321, k=4).
- Performance: all groups must agree a common way to measure, so results are comparable (which quantities, which queries, which parameter values, whether to measure more than queries). Still being discussed in class.
- Class schedule: intro; datasets, technologies and deployment; **database design and populating (DDL, tooling)**; queries/goals; performance measures; problems and tweaks; final presentation.

## 2. Decisions so far

- Technology: LanceDB OSS, pending formal instructor approval. Open questions for the instructor: is an embedded engine acceptable, is schema-as-code (Arrow/Pydantic) acceptable instead of SQL DDL, and does K-means computed outside the database (results stored and queried in LanceDB) count as a database-side goal.
- Dataset grid agreed in class: 3 sizes (about 1, 6, 12 GB) x 3 dimension classes (128 to 256, 768, 2048). Rules we adopted: whole datasets preferred, precomputed ground truth required, no deprecated datasets. Sizes may differ somewhat.
- **Settled cell: 128 to 256 dimensions, Small = VIBE `glove-200-cosine`.** This is the dataset for the current task. Other cells are still being negotiated, do not work on them.
- Fallback technology if LanceDB is rejected: SurrealDB (not verified).

## 3. LanceDB facts

### Verified (lancedb 0.40.0, pyarrow 26.0.0, numpy 2.4.4, h5py 3.16.0, scikit-learn 1.8.0, psutil 7.2.2)

```python
import lancedb, pyarrow as pa
from lancedb.index import IvfPq, BTree

db = lancedb.connect("/db/lancedb")                       # a directory, no server
schema = pa.schema([pa.field("id", pa.int64()),
                    pa.field("vector", pa.list_(pa.float32(), 200))])
tbl = db.create_table("glove200", data=batches(), schema=schema)   # generator of RecordBatch: streaming load worked
tbl.add(more_batches())                                   # append; every write creates a new version

tbl.create_index("vector", config=IvfPq(distance_type="cosine",
                                        num_partitions=512, num_sub_vectors=25))
tbl.create_index("id", config=BTree())                    # scalar index
tbl.optimize()                                            # worked
tbl.list_indices(); tbl.list_versions()                   # worked
tbl.add_columns({"bucket": "cast(id % 10 as int)"})       # schema evolution via SQL expression worked

q = (tbl.search(vec).distance_type("cosine").nprobes(20).refine_factor(5)
        .limit(10).select(["id"]).to_arrow())             # approximate search worked
```

Also tested, but with the older `.metric("cosine")` spelling, so re-test with `.distance_type(...)`:
`.bypass_vector_index()` (exact search), `.distance_range(lower, upper)`, `.where("id < 1000", prefilter=True)`, and lookup by id with `tbl.search().where("id = 5")`.

Deprecated in 0.40.0, do not use: `create_index(metric=..., index_type="IVF_PQ")`, `create_scalar_index(...)`, `.metric(...)`.

Other verified facts:
- Pydantic schema (`LanceModel` with `Vector(d)`) produces `fixed_size_list<float>[d]`.
- **uint8 vectors cannot be indexed or searched with L2** (error "Unsupported training sample type UInt8 with metric l2"). Always store float32.
- After writes and index builds the database directory was about 2.2x the raw vector size in a small test. Old versions stay on disk. Cause not isolated (unverified), so measure disk size after `optimize()` and any version cleanup.

### Unverified (from documentation, check on our version)

- `nprobes` is auto-tuned by default and filtered queries may scan extra partitions. **Always set `nprobes` explicitly.**
- Metrics: `l2` (default), `cosine`, `dot`, and `hamming` for binary. Index and query must use the same metric.
- HNSW variants exist only combined with IVF (IVF_HNSW_*), not as a standalone index. IVF_RQ also exists.
- Result `_distance` for cosine is probably `1 - cosine_similarity` and L2 is probably squared L2. Compare against the `distances` array in the dataset file before choosing any range threshold.
- PQ requires `num_sub_vectors` to divide the dimension (200 is divisible by 1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 200). Check which values the library accepts.
- Docs suggest starting at roughly `num_partitions = rows // 4096` (about 290 for 1.19M rows) and `num_sub_vectors = dim // 8` (25 here).
- K-means is not a native LanceDB query. Plan: compute it outside (scikit-learn or Faiss), then store centroids in a table and `cluster_id` as a column.

## 4. Dataset: VIBE `glove-200-cosine`

- Source: https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/glove-200-cosine.hdf5 (use the `resolve/main` form for downloading). The VIBE README lists it as type Word, n = 1,192,514, d = 200, distance cosine.
- Raw size as float32: 1,192,514 x 200 x 4 = 0.95 GB.
- HDF5 structure (from the VIBE documentation):
  - attributes: `dimension`, `distance`, `point_type` (float, uint8 or binary)
  - `train` (n_corpus, dim): vectors to load and index
  - `test` (n_test, dim): query vectors, not part of `train` (the VIBE paper says about 1000 test queries, unverified for this file)
  - `neighbors` (n_test, 100): ids of the true 100 nearest neighbors of each query
  - `distances` (n_test, 100): their distances
  - `avg_distances` (n_test,): average distance from each query to the corpus (probably a query-difficulty indicator, unverified, not needed)
- Maximum usable k is 100. The ground truth ignores any filter.
- GloVe vectors are not unit length, so use the **cosine** metric consistently everywhere (index, query, ground-truth comparison). For future K-means, normalize to unit length first.
- Assumption to check first: ids in `neighbors` are row positions in `train` (0-based). Our table id must equal that row position.
- Alternative copy (not used): ann-benchmarks `glove-200-angular` (1,183,514 train, 10,000 test, HDF5 918 MB).
- If the HuggingFace download is blocked in your environment, tell the user to download the file manually and place it in `data/`.

## 5. Deployment (proposed to the class, not yet agreed)

Embedded engine, so one container runs the database and the benchmark code together. Docker or Podman, Linux, same flags in both. The project must also run without Docker (documented pip path).

Minimal Dockerfile (create it in the repo root):

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "db.py"]
```

`requirements.txt` (pins tested together): `lancedb==0.40.0`, `pyarrow==26.0.0`, `numpy==2.4.4`, `h5py==3.16.0`, `scikit-learn==1.8.0`, `psutil==7.2.2`.

`.dockerignore`: `data/`, `db/`, `results/`, `.git/`.

Run with limits (proposal: 4 pinned cores, 8 GB RAM, swap off):

```
docker run --rm --user $(id -u):$(id -g) \
  --cpuset-cpus="0-3" --memory=8g --memory-swap=8g -e OMP_NUM_THREADS=1 \
  -v ./data:/data:ro -v ./db:/db -v ./results:/results \
  lancedb-bench <args>
```

Paths used by the code: dataset in `/data` (read-only), LanceDB directory in `/db`, results in `/results`. Make them overridable with environment variables (`DATA_DIR`, `DB_URI`, `RESULTS_DIR`) and default to `./data`, `./db`, `./results` outside Docker.

Native Linux is preferred for final measurements. On macOS or Windows, Docker runs in a VM and memory limits and file speed differ.

## 6. Measurement protocol (proposed, adopt unless the user says otherwise)

- Queries: the dataset's `test` set. k = 10 and k = 100. Ground truth: the file's `neighbors` array.
- Timing: client-side wall-clock around the search call, measured with `time.perf_counter_ns()`. Return ids only, do not convert to Python dicts inside the timed region.
- Protocol: 200 warm-up queries (discarded), then 3 timed passes over all test queries. Report p50, p95, p99, mean latency and QPS from all samples. Single search thread for latency runs.
- Quality: recall@10 and recall@100 (overlap with the true neighbors divided by k, averaged over queries).
- Exact baseline: `bypass_vector_index()` for correctness and as a speed reference.
- Cold run: after `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` on the host. Without root, report the first pass after a container restart and say so.
- Resources: ingestion time, index build time (1 build per configuration, 3 repeats for the reference configuration), index size on disk and total table size on disk (after `optimize()`), peak memory during load, build and queries (cgroup v2 `memory.peak` on kernel 5.19 or newer is unverified, otherwise sample RSS with psutil and also record `resource.getrusage`).
- Always log: CPU model, cores available, RAM, disk type, kernel, Python and library versions, git commit, random seeds, exact parameters.
- Parameter sweep (reference index: 512 partitions, 25 sub-vectors, cosine):
  - query time: `nprobes` in {1, 2, 5, 10, 20, 50, 100, 200} x `refine_factor` in {1, 5, 20}
  - build time, one factor at a time: partitions in {256, 512, 1024}, sub-vectors in {25, 50, 100} (5 builds in total including the reference)
  - report QPS at recall@10 of at least 0.90, 0.95 and 0.99, plus the full recall versus QPS curve

## 7. Task: LanceDB test on glove-200

Goal: show end to end that LanceDB can ingest, index and search `glove-200-cosine`, and produce reproducible recall, latency and resource numbers. Work in small verified steps and report after each phase what you checked and what you found.

**Phase 0: environment.** Create the repo skeleton (section 8), `requirements.txt`, Dockerfile, `.dockerignore`, `.gitignore` (data, db, results). Confirm Python and pinned library versions import. Record the machine (cores, RAM, disk, kernel).

**Phase 1: data and assumption checks.** Script `scripts/download.py` fetches the HDF5 file into `data/` and prints its size and SHA-256. Script `scripts/inspect_hdf5.py` prints attributes and shapes and checks:
- `train` is about 1.19M x 200, `test` shape, `neighbors` shape and dtype
- vector norms of `train` and `test` (are they unit length?)
- the distribution of `neighbors` ids over the id range (histogram in 10 bins, each should be near 10% if the file order is random)
- `neighbors` ids are valid row positions (0 <= id < n_train)
Write the findings into `results/dataset_check.md`.

**Phase 2: ingest.** `scripts/ingest.py` streams `train` in chunks of about 50,000 rows into table `glove200` with schema `id: int64` (row position) and `vector: fixed_size_list<float32>[200]`. Do not load the full array into memory. Log ingestion time, peak memory and row count. Add checks: row count equals `train` rows, a few random ids return the exact stored vector.

**Phase 3: index.** `scripts/build_index.py` builds IVF_PQ (cosine) with the reference parameters and a BTree on `id`, then calls `optimize()`. Log build time, index size and total table size (before and after optimize). Verify which `num_sub_vectors` values the library accepts for dimension 200.

**Phase 4: correctness.** `scripts/exact_check.py`:
- For about 20 test queries compare the exact search (`bypass_vector_index`, cosine) with the file's `neighbors`. Their overlap should be close to 100%. If not, investigate the metric convention, id mapping or normalization before going on, and report it.
- Compare LanceDB `_distance` values with the file's `distances` and state the convention (squared or not, cosine distance or similarity).

**Phase 5: benchmark.** `scripts/bench.py` implements the protocol in section 6 and writes one CSV or JSON row per configuration and k (parameters, recall@k, p50, p95, p99, mean, QPS, plus environment) to `results/`. Run the reference configuration first and show the numbers before running the full sweep.

**Phase 6: report.** Write `results/REPORT.md`: environment, exact commands, tables, recall versus QPS plot (matplotlib is allowed for plots, add it to requirements only if used), and a short honest analysis (what worked, surprises, limits, what to try next). State clearly anything unverified or any deviation from the protocol.

**Phase 7 (optional, only if asked): CLI skeleton.** `db.py` with goal 1 (`db 1 <id> <k>`: KNN of a stored vector by id, return k neighbors excluding the vector itself by requesting k+1) and goal 8 (`db 8 <n_queries> <k>`: batch KNN over test queries with recall against ground truth). Goals 2 to 7 are not defined yet.

## 8. Repository layout

```
README.md              documentation per the checklist in section 1 (keep it updated, leave roles of the two students as placeholders)
CLAUDE.md              this file
Dockerfile
requirements.txt
.dockerignore
.gitignore
db.py                  CLI entry point (phase 7)
src/                   shared helpers (config, hdf5 reader, timing, memory sampling, recall)
scripts/               download.py, inspect_hdf5.py, ingest.py, build_index.py, exact_check.py, bench.py
data/                  not committed
db/                    not committed
results/               commit small summary files (REPORT.md, CSV), not the database
```

## 9. Things to verify early (assumptions that can break everything)

1. `neighbors` contains row positions of `train` (0-based).
2. The number of test queries and the exact shapes in this file.
3. `train` vector norms (not unit length is expected for GloVe) and that cosine is the right metric.
4. LanceDB cosine `_distance` convention versus the file's `distances`.
5. `num_sub_vectors` values accepted for dimension 200, and `num_partitions` limits.
6. That `.nprobes()` is honored and what the library does by default.
7. Whether `.distance_type()` works with `bypass_vector_index()` and `distance_range()` in 0.40.0.
8. How to take a clean disk-size measurement (versions, `optimize()`, cleanup).
9. Whether the machine can provide the proposed limits (4 cores, 8 GB). Record what you actually used.

## 10. Known pitfalls

- Use the non-deprecated API (section 3). Read deprecation warnings, they change between versions.
- Do not convert results to Python dicts or load full vectors inside timed regions.
- Query ids that exist in the corpus find themselves at distance 0. The official test queries are not in `train`, but the CLI goal 1 uses stored vectors, so exclude the vector itself there.
- Index metric, query metric and ground-truth metric must match (cosine).
- A benchmark process and the database share CPU and memory in the embedded setup. Keep the harness light, and report that limit.
- Page cache makes repeated runs faster. Always say whether a number is warm or cold.
- Do not benchmark on a laptop under load, on battery, or with other heavy processes running, and record the CPU governor if you can.
- Do not commit the dataset, the database directory or large result files.

## 11. Open class-level decisions (do not decide these yourself, ask the user)

- Which datasets fill the other cells of the grid, and whether official reduced releases (SIFT-10M, Yandex Text-to-Image 10M, wiki_all_10M) count as datasets.
- Whether deprecated or out-of-distribution datasets may fill the 2048-dimension row.
- The shared resource limits, thread counts and the final timing definition for all groups.
- The shared logical schema and synthetic metadata attributes for filtered queries (to be generated once with a fixed seed and shared).
- The definition of goals 2 to 7 and the final CLI contract.

## 12. Working agreements

- Keep the solution minimal and readable. Prefer the standard library plus the pinned packages.
- Python 3.12, type hints where cheap, no global state, clear function boundaries.
- Make every script idempotent where possible and print what it did. Fail loudly on assumption violations.
- Pin versions, fix seeds, log the environment with every result.
- Explain trade-offs and limitations plainly in the report. Do not oversell results.
- After each phase, summarize in a few lines what was verified, what was not, and what you will do next.
