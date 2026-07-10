"""Partition synthetic null data into exactly two clusters.

The synthetic null constructed by `null_model.construct_null` has no real
cluster labels (it was simulated without conditioning on any cluster
covariate), so to compute a "null" version of the target DE p-values we first
need some data-driven 2-way partition of the null cells. This module runs a
standard scanpy preprocessing + Leiden pipeline, with a short bisection
search over the `resolution` parameter to land on exactly two communities --
the scanpy analogue of the resolution-bisection search R's
`calcNullPval()` (ClusterDE/R/calc_null_pval.R) runs with Seurat::FindClusters.

Depends only on scanpy/anndata/pandas -- no scdesigner import.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


def null_two_clusters(
    null_adata: AnnData,
    n_pcs: int = 20,
    seed: int = 0,
    max_iter: int = 15,
    key_added: str = "null_cluster",
) -> pd.Categorical:
    """Cluster synthetic null cells into exactly two groups via Leiden.

    Pipeline: normalize_total -> log1p -> pca(n_pcs) -> neighbors -> leiden,
    with `resolution` bisected until exactly 2 communities are found.

    Parameters
    ----------
    null_adata : AnnData
        Synthetic null data (modified in place: normalized/logged counts
        overwrite `.X`, and PCA/neighbors/Leiden results are added).
    n_pcs : int
        Number of principal components used for the neighbor graph.
    seed : int
        Random seed for PCA, neighbors, and Leiden (deterministic given a
        fixed seed and a fixed `resolution`).
    max_iter : int
        Maximum number of resolution values to try before giving up.
    key_added : str
        Column name to write the final two-level cluster assignment into,
        both in `null_adata.obs` and in the returned Categorical.

    Returns
    -------
    pd.Categorical
        Length-n_cells categorical with exactly two levels ("0", "1").

    Raises
    ------
    RuntimeError
        If no resolution within `max_iter` tries yields exactly 2 clusters.
    """
    sc.pp.normalize_total(null_adata)
    sc.pp.log1p(null_adata)
    sc.pp.pca(null_adata, n_comps=min(n_pcs, min(null_adata.shape) - 1), random_state=seed)
    sc.pp.neighbors(null_adata, random_state=seed)

    def n_clusters_at(resolution: float) -> int:
        sc.tl.leiden(
            null_adata,
            resolution=resolution,
            random_state=seed,
            flavor="igraph",
            n_iterations=2,
            key_added=key_added,
        )
        return null_adata.obs[key_added].nunique()

    # Phase 1: double `right` until it yields >= 2 clusters (mirrors R
    # calc_null_pval.R's initial doubling loop before its bisection search).
    right = 0.3
    n = n_clusters_at(right)
    for _ in range(max_iter):
        if n >= 2:
            break
        right *= 2
        n = n_clusters_at(right)
    else:
        raise RuntimeError(
            f"Could not reach >= 2 Leiden clusters within {max_iter} resolution doublings."
        )

    # Phase 2: bisect resolution in [left, right] until exactly 2 clusters.
    left = 0.0
    for _ in range(max_iter):
        if n == 2:
            break
        mid = (left + right) / 2
        n = n_clusters_at(mid)
        if n < 2:
            left = mid
        else:
            right = mid
    else:
        raise RuntimeError(
            f"Could not reach exactly 2 Leiden clusters within {max_iter} bisection steps "
            f"(last resolution={right}, n_clusters={n})."
        )

    labels = null_adata.obs[key_added].astype("category")
    return labels.values
