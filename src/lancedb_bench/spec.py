"""The goals of the class baseline (section 8) and their parameters: shared by the CLI and the server."""

# goal -> (what it does, [(parameter name, type), ...])
GOALS = {
    1: ("the k nearest neighbors of a stored vector (the vector itself is left out)", [("id", int), ("k", int)]),
    2: ("all vectors within a cosine distance radius of a stored vector (at most 100)", [("id", int), ("radius", float)]),
    3: ("the k nearest neighbors among the vectors with id mod 100 < percent", [("id", int), ("k", int), ("percent", int)]),
    4: ("K-means (cosine, ids 0..clusters-1 as start, exact iterations); stores the cluster id of every vector",
        [("clusters", int), ("iterations", int)]),
    5: ("the cluster of a stored vector", [("id", int)]),
    6: ("the k nearest neighbors inside the cluster of a stored vector", [("id", int), ("k", int)]),
    7: ("the cosine distance between two stored vectors", [("id_a", int), ("id_b", int)]),
    8: ("benchmark: the first n test queries at every accuracy setting, recall@k and latency", [("n_queries", int), ("k", int)]),
}

USAGE = "Usage: db <goal> <params...>   (dataset from $DATASET, default glove-200-angular)\n" + "\n".join(
    f"  db {goal} {' '.join(f'<{name}>' for name, _ in params):<28} {text}" for goal, (text, params) in GOALS.items()
)


def parse_params(goal: int, raw: list) -> list:
    """Convert the raw parameters of a goal to their types. Raises ValueError (with a message) if they do not fit."""
    if goal not in GOALS:
        raise ValueError(f"goal {goal} does not exist (goals are 1 to {len(GOALS)})")
    params = GOALS[goal][1]
    if len(raw) != len(params):
        raise ValueError(f"goal {goal} takes {len(params)} parameters ({', '.join(n for n, _ in params)}), got {len(raw)}")
    values = []
    for (name, kind), value in zip(params, raw):
        try:
            values.append(kind(value))
        except (TypeError, ValueError):
            raise ValueError(f"parameter {name} must be {'an integer' if kind is int else 'a number'}, got {value!r}") from None
    return values
