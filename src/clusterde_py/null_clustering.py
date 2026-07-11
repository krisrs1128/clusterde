"""Partition synthetic null data into two clusters.

Given a synthetic null dataset (no true clusters), we create two artificial
clsuters to compute null DE p-values. This step implements scanpy preprocessing
+ Leiden clustering with a bisection search over the resolution parameter to
ensure two communities are created. The approach mirrors R's `calcNullPval()`
but uses scanpy instead of Seurat.
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
    """Bipartition null cells via Leiden clustering.

    Pipeline: normalize → log1p → PCA → neighbors → Leiden, with resolution
    tuned via bisection to yield exactly two clusters.

    Parameters
    ----------
    null_adata : AnnData
        Synthetic null data (modified in place: normalized/logged counts
        overwrite `.X`; PCA/neighbors/Leiden results added).
    n_pcs : int
        Principal components for neighbor graph.
    seed : int
        Random seed (deterministic for fixed seed and resolution).
    max_iter : int
        Maximum resolution trials before failure.
    key_added : str
        Column name for cluster assignments in `null_adata.obs` and return
        value.

    Returns
    -------
    pd.Categorical
        Two-level categorical ("0", "1") of length n_cells.

    Raises
    ------
    RuntimeError
        If no resolution yields exactly two clusters within `max_iter` trials.
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

    # Phase 1: Double resolution until ≥2 clusters.
    right = 0.3
    n = n_clusters_at(right)
    for _ in range(max_iter):
        if n >= 2:
            break
        right *= 2
        n = n_clusters_at(right)
    else:
        raise RuntimeError(f"Failed to reach ≥2 clusters within {max_iter} doublings.")

    # Phase 2: Bisection to exactly 2 clusters.
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
        raise RuntimeError(f"Failed to reach exactly 2 clusters within {max_iter} steps (resolution={right}, n_clusters={n}).")

    labels = null_adata.obs[key_added].astype("category")
    return labels.values
