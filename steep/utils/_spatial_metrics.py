"""Metrics that a sketch can only score well on by preserving spatial structure.

Every metric currently in the benchmark is computed from cell embeddings and
cell labels. Shuffling the spatial coordinates while keeping the graph would
leave all of them unchanged, so they cannot distinguish a spatially-informed
sketch from one that ignores geometry. The two metrics here read the spatial
graph directly:

- `neighbourhood_composition_divergence` asks whether each surviving cell still
  sits among the same mix of cell types it did in the full graph.
- `neighbourhood_enrichment_agreement` asks whether the tissue-level
  co-localization pattern between cell types survives.

Both compare a sketched AnnData against the original it came from, and neither
needs a trained model.

"""

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

EPS = 1e-12


def _symmetric_csr(adata: ad.AnnData, adjacency_key: str) -> sp.csr_matrix:
    adjacency = sp.csr_matrix(adata.obsp[adjacency_key])
    return adjacency.maximum(adjacency.T).tocsr()


def _composition_matrix(
    adjacency: sp.csr_matrix,
    codes: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Row-normalized count of each label among every cell's neighbours."""
    valid = codes >= 0
    onehot = sp.csr_matrix(
        (
            np.ones(int(valid.sum()), dtype=np.float32),
            (np.flatnonzero(valid), codes[valid]),
        ),
        shape=(codes.shape[0], num_classes),
    )
    counts = np.asarray((adjacency @ onehot).todense(), dtype=np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    return counts / np.maximum(totals, EPS)


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.where(p > 0, p * (np.log2(p + EPS) - np.log2(m + EPS)), 0.0).sum(axis=1)
        kl_q = np.where(q > 0, q * (np.log2(q + EPS) - np.log2(m + EPS)), 0.0).sum(axis=1)
    return np.clip(0.5 * (kl_p + kl_q), 0.0, 1.0)


def neighbourhood_composition_divergence(
    original: ad.AnnData,
    sketched: ad.AnnData,
    label_key: str,
    adjacency_key: str = "adjacency_matrix",
) -> dict[str, float | int] | None:
    """How much each surviving cell's neighbourhood composition changed.

    For every cell kept by the sketch, build the distribution of labels over its
    spatial neighbours in the original graph and in the sketched graph, then
    take the Jensen-Shannon divergence between the two (base 2, so it lies in
    [0, 1]).

    Returns the mean divergence -- lower is better -- alongside the fraction of
    evaluated cells whose neighbourhood was left completely empty by the sketch,
    which a divergence alone would hide.

    """
    warnings.simplefilter("ignore")

    if label_key not in original.obs or label_key not in sketched.obs:
        return None

    shared = sketched.obs_names.intersection(original.obs_names)
    if len(shared) < 3:
        return None

    categories = pd.Categorical(original.obs[label_key].to_numpy()).categories
    if len(categories) < 2:
        return None
    num_classes = len(categories)
    labels = pd.Categorical(original.obs.loc[shared, label_key].to_numpy(), categories=categories)

    original_adjacency = _symmetric_csr(original, adjacency_key)
    sketched_adjacency = _symmetric_csr(sketched, adjacency_key)

    original_index = original.obs_names.get_indexer(shared)
    sketched_index = sketched.obs_names.get_indexer(shared)

    # The ORIGINAL composition is taken over the full original graph, not over
    # the surviving cells. Restricting it first would delete the same neighbours
    # from both sides, so a sketch that removed half a cell's neighbourhood
    # would look like it changed nothing.
    all_codes = pd.Categorical(original.obs[label_key].to_numpy(), categories=categories).codes.astype(np.int64)
    original_composition = _composition_matrix(original_adjacency, all_codes, num_classes)[original_index]

    sketched_sub = sketched_adjacency[sketched_index][:, sketched_index]
    codes = labels.codes.astype(np.int64)
    sketched_composition = _composition_matrix(sketched_sub, codes, num_classes)

    kept_degree = np.asarray(sketched_sub.getnnz(axis=1)).ravel()
    original_degree = np.asarray(original_adjacency.getnnz(axis=1)).ravel()[original_index]
    evaluable = (original_degree > 0) & (kept_degree > 0)
    if evaluable.sum() < 3:
        return None

    divergence = _jensen_shannon(original_composition[evaluable], sketched_composition[evaluable])

    return {
        "composition_jsd_mean": float(divergence.mean()),
        "composition_jsd_median": float(np.median(divergence)),
        "emptied_neighbourhood_frac": float((kept_degree == 0).mean()),
        # Cells the divergence was averaged over, and cells the emptied
        # fraction was taken over -- different denominators, so callers that
        # weight across sections need both.
        "count": int(evaluable.sum()),
        "shared_count": int(len(shared)),
    }


def _enrichment_matrix(
    adjacency: sp.csr_matrix,
    codes: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Observed-over-expected co-occurrence of label pairs across edges.

    A self-contained stand-in for `squidpy.gr.nhood_enrichment` that needs no
    permutation test: the expected count under random labelling factorizes, so
    the log ratio is comparable between two graphs of different densities.

    """
    upper = sp.triu(adjacency, k=1).tocoo()
    src, dst = upper.row, upper.col
    keep = (codes[src] >= 0) & (codes[dst] >= 0)
    src, dst = src[keep], dst[keep]
    if src.size == 0:
        return np.zeros((num_classes, num_classes))

    observed = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(observed, (codes[src], codes[dst]), 1.0)
    observed = observed + observed.T

    degree = observed.sum(axis=1)
    total = degree.sum()
    expected = np.outer(degree, degree) / max(total, EPS)

    return np.log2((observed + 1.0) / (expected + 1.0))


def neighbourhood_enrichment_agreement(
    original: ad.AnnData,
    sketched: ad.AnnData,
    label_key: str,
    adjacency_key: str = "adjacency_matrix",
) -> dict[str, float | int] | None:
    """Does the sketch keep the tissue's cell-type co-localization pattern?

    Builds a label-by-label observed/expected co-occurrence matrix on each graph
    and correlates the two. High agreement means the sketch preserved which cell
    types sit next to which -- the tissue architecture -- not merely which cells
    survived.

    """
    warnings.simplefilter("ignore")

    if label_key not in original.obs or label_key not in sketched.obs:
        return None

    shared = sketched.obs_names.intersection(original.obs_names)
    if len(shared) < 3:
        return None

    # Categories come from the ORIGINAL section, not from the surviving cells:
    # a label the sketch deleted outright must show up as an empty row/column,
    # not vanish from the comparison and inflate the agreement.
    categories = pd.Categorical(original.obs[label_key].to_numpy()).categories
    if len(categories) < 2:
        return None
    num_classes = len(categories)
    codes = pd.Categorical(
        original.obs.loc[shared, label_key].to_numpy(),
        categories=categories,
    ).codes.astype(np.int64)

    original_index = original.obs_names.get_indexer(shared)
    sketched_index = sketched.obs_names.get_indexer(shared)
    original_sub = _symmetric_csr(original, adjacency_key)[original_index][:, original_index]
    sketched_sub = _symmetric_csr(sketched, adjacency_key)[sketched_index][:, sketched_index]

    original_enrichment = _enrichment_matrix(original_sub, codes, num_classes)
    sketched_enrichment = _enrichment_matrix(sketched_sub, codes, num_classes)

    triangle = np.triu_indices(num_classes, k=1)
    a, b = original_enrichment[triangle], sketched_enrichment[triangle]
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None

    correlation = spearmanr(a, b).statistic
    return {
        "enrichment_spearman": float(correlation),
        "enrichment_pearson": float(np.corrcoef(a, b)[0, 1]),
        "num_label_pairs": int(a.size),
    }
