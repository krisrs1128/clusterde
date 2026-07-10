"""Differential-expression analysis

This wraps `scanpy.tl.rank_genes_groups` to return the p-values for use in
Clipper. It is called on both the real and the synthetic null data.
"""

import pandas as pd
import scanpy as sc
from anndata import AnnData


def rank_pvalues(
    adata: AnnData,
    groupby: str,
    group1: str,
    group2: str,
    method: str = "wilcoxon",
    filter_upregulated: bool = False,
) -> pd.Series:
    """Genewise p-values comparing `group1` vs. `group2`.

    This first log-normalizes the input `adata` (`sc.pp.normalize_total` +
    `sc.pp.log1p`, similar to Seurat::NormalizeData, used by ClusterDE).  Then
    run `sc.tl.rank_genes_groups(..., groups=[group1], reference=group2,
    method=method)` to obtain genewise p-values.

    Parameters
    ----------
    adata : AnnData
        Dataset containing both groups in `adata.obs[groupby]`, with raw
        counts in `.X`.
    groupby : str
        Column name of `adata.obs` where the group1 and group2 labels live.
    group1, group2 : str
        The two group labels to compare (group1 vs. reference group2).
    method : str
        Hypothesis test used in `sc.tl.rank_genes_groups` (default "wilcoxon",
        mirroring Seurat::FindMarkers, which is used by ClusterDE in R.
    filter_upregulated : bool
        If True, genes with `logfoldchanges <= 0` (i.e. not higher in group1)
        are removed, mirroring the `findMarkers()` filter for `avg_log2FC > 0`
        in the reference ClusterDE implementation. We only run this on real
        data, not the synthetic null.

    Returns
    -------
    pd.Series
        Unadjusted genewise p-values
    """
    normalized = adata.copy()
    sc.pp.normalize_total(normalized)
    sc.pp.log1p(normalized)

    sc.tl.rank_genes_groups(
        normalized, groupby, groups=[group1], reference=group2, method=method
    )
    df = sc.get.rank_genes_groups_df(normalized, group1).set_index("names")

    if filter_upregulated:
        df = df[df["logfoldchanges"] > 0]

    return df["pvals"].rename("pval")
