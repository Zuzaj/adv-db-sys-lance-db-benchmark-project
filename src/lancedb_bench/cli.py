"""`db <goal> <params...>`: the command line of the class baseline (client container, talks to the db container).

The result goes to stdout, one line per result (`id distance`, nearest first, unless the goal says otherwise);
notes go to stderr. Exit code: 0 success, 2 invalid parameters, 1 any other failure.
The dataset comes from $DATASET (default glove-200-angular), the server from $SERVER_URL.
"""
import sys

import httpx

from lancedb_bench.client import Client, InvalidParameters
from lancedb_bench.config import DATASETS, DEFAULT_DATASET, get_dataset
from lancedb_bench.spec import USAGE, parse_params

EXIT_OK, EXIT_FAILURE, EXIT_INVALID = 0, 1, 2


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or not args[0].isdigit():
        print(USAGE, file=sys.stderr)
        return EXIT_INVALID
    goal, raw = int(args[0]), args[1:]
    try:
        parse_params(goal, raw)  # same check as on the server, so a typo needs no round trip
        ds = get_dataset()
    except ValueError as exc:
        print(f"invalid parameters: {exc}\n{USAGE}", file=sys.stderr)
        return EXIT_INVALID
    except SystemExit:  # unknown dataset
        print(f"invalid parameters: unknown dataset in $DATASET (known: {', '.join(DATASETS)}; default {DEFAULT_DATASET})", file=sys.stderr)
        return EXIT_INVALID

    try:
        out = Client().goal(goal, ds.name, raw)
    except InvalidParameters as exc:
        print(f"invalid parameters: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except (RuntimeError, httpx.HTTPError) as exc:
        print(f"failed: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    print("\n".join(out["lines"]))
    if out.get("note"):
        print(out["note"], file=sys.stderr)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
