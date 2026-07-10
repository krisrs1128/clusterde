"""Top-level ClusterDE orchestration.

The main steps in this pipeline are,

  - Construct synthetic null data where there are no clusters.
  - Cluster these null data and compute p-values for the null "marker genes"
  - Apply Clipper to score real gene p-values against the synthetic null contrasts
"""

from typing import Optional

import pandas as pd
from anndata import AnnData

from .clipper import clipper_q, contrast_scores
from .de_pvalues import rank_pvalues
from .null_clustering import null_two_clusters
from .null_model import construct_null


def find_markers(
    adata: AnnData,
    cluster_key: str,
    group1: str,
    group2: str,
    fdr: float = 0.05,
    null_kwargs: Optional[dict] = None,
    cluster_kwargs: Optional[dict] = None,
    contrast_method: str = "diff",
    threshold: str = "DS",
) -> pd.DataFrame:
    """Run the full ClusterDE pipeline comparing `group1` vs. `group2`.

    Steps
    -----
    1. Compute target p-values from the real two-cluster comparison,
       filtered to genes up-regulated in `group1` (one-sided, matching R
       `findMarkers()`'s `avg_log2FC > 0` filter).
    2. Construct a synthetic null dataset via `null_model.construct_null`.
    3. Partition the null data into two clusters via
       `null_clustering.null_two_clusters`.
    4. Compute null p-values from that null two-cluster comparison
       (unfiltered, two-sided).
    5. Compute contrast scores and Clipper-style q-values.

    Parameters
    ----------
    adata : AnnData
        Real data containing both clusters in `adata.obs[cluster_key]`.
    cluster_key : str
        Column of `adata.obs` holding the two real cluster labels.
    group1, group2 : str
        The two cluster labels to compare (group1 vs. reference group2).
    fdr : float
        Target false discovery rate.
    null_kwargs : dict or None
        Extra keyword arguments forwarded to `null_model.construct_null`.
    cluster_kwargs : dict or None
        Extra keyword arguments forwarded to `null_clustering.null_two_clusters`.
    contrast_method : {"diff", "max"}
        Forwarded to `clipper.contrast_scores`.
    threshold : {"DS", "BC"}
        Forwarded to `clipper.clipper_q`. DS means data splitting and BC means
        the Barber-Candes threshold.

    Returns
    -------
    pd.DataFrame
        Gene-indexed table with columns ["target_pval", "null_pval",
        "contrast_score", "q_value", "is_DE"], sorted by contrast_score.
    """
    target_pvals = rank_pvalues(
        adata, cluster_key, group1, group2, filter_upregulated=True
    )

    # generate the null p-values
    null_adata = construct_null(adata, **(null_kwargs or {}))
    null_labels = null_two_clusters(null_adata, **(cluster_kwargs or {}))
    null_adata.obs["null_cluster"] = null_labels
    null_pvals = rank_pvalues(null_adata, "null_cluster", "0", "1")

    # compute contrast statistics and q-values
    cs = contrast_scores(target_pvals, null_pvals, method=contrast_method)
    q = clipper_q(cs, threshold=threshold)

    # merge results
    result = pd.DataFrame(
        {
            "target_pval": target_pvals.reindex(cs.index),
            "null_pval": null_pvals.reindex(cs.index),
            "contrast_score": cs,
            "q_value": q,
        }
    )
    result["is_DE"] = result["q_value"] <= fdr
    return result.sort_values("contrast_score", ascending=False)
