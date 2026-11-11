FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9.13 /uv /uvx /bin/

WORKDIR /workspace

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# One image definition, two roles:
#   ROLE=db      installs the LanceDB engine and the HTTP wrapper (the database container)
#   ROLE=client  installs only the benchmark client libraries (no database engine)
ARG ROLE=client

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project $( [ "$ROLE" = "db" ] && echo --extra db )

# Copy source code and install the project (registers the 'db' CLI script)
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev $( [ "$ROLE" = "db" ] && echo --extra db )

ENV PATH="/workspace/.venv/bin:$PATH"

# One thread per numeric library so the engine's own thread pool is the only parallelism
ENV OMP_NUM_THREADS=1
ENV DATA_DIR=/data
ENV DB_URI=/db
ENV RESULTS_DIR=/results
