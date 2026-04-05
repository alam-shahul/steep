import numpy as np
from sklearn.metrics import pairwise_distances


def hopper_sketch_indices(x, num_points: int, random_seed: int = 0) -> np.ndarray:
    """Select a farthest-first sketch of ``num_points`` rows from ``x``."""
    num_rows = int(x.shape[0])
    if num_points <= 0:
        raise ValueError(f"`num_points` must be positive, got {num_points}.")
    if num_points >= num_rows:
        return np.arange(num_rows, dtype=int)

    rng = np.random.default_rng(random_seed)
    first_idx = int(rng.integers(num_rows))
    selected = [first_idx]

    min_dists = pairwise_distances(x[[first_idx]], x, metric="euclidean")[0]
    min_dists[first_idx] = -np.inf

    while len(selected) < num_points:
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        new_dists = pairwise_distances(x[[next_idx]], x, metric="euclidean")[0]
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[selected] = -np.inf

    return np.asarray(selected, dtype=int)
