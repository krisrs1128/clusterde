"""Contrast-score FDR selection.

This implements the main ClusterDE algorithm. It takes target and null p-values
based on real and synthetic data, then applies either data splitting or
Barber-Candes thresholds for FDR-guaranteed marker gene selection. It doesn't
run any null simulation, since the null p-values are already computed by this
point.

It is based off the callDE() and and cs2q() functions from the ClusterDE R
implementation.
"""

import numpy as np
import pandas as pd


def contrast_scores(
    target_pvals: pd.Series,
    null_pvals: pd.Series,
    nlog_transform: bool = True,
    method: str = "diff",
) -> pd.Series:
    """Genewise contrast scores from target and null p-values.

    Gene names must be shared in both the target and null p-value sets (those
    not in the intersection are dropped).

    Parameters
    ----------
    target_pvals : pd.Series
        Genewise p-values from the real data's cluster comparison.
    null_pvals : pd.Series
        Genewise p-values from the synthetic-null comparison.
    nlog_transform : bool
        If True (default), -log10 transform the p-values before computing a
        contrast score. This means that larger scores correspond to higher
        significance.
    method : {"diff", "max"}
        "diff": cs = target - null.
        "max": cs = max(target, null) * sign(target - null).

    Returns
    -------
    pd.Series
        Genewise contrast scores.
    """
    common = target_pvals.index.intersection(null_pvals.index)
    target = target_pvals.loc[common].astype(float)
    null = null_pvals.loc[common].astype(float)

    if nlog_transform:
        target = -np.log10(target)
        null = -np.log10(null)

    if method == "diff":
        cs = target - null
    elif method == "max":
        cs = np.maximum(target, null) * np.sign(target - null)
    else:
        raise ValueError(f"method must be 'diff' or 'max', got {method!r}")

    return cs


def clipper_q(contrast_scores: pd.Series, threshold: str = "DS") -> pd.Series:
    """Convert contrast scores to q-values using a Clipper estimator.

    This is based off the `cs2q()` function in the ClusterDE R package. For each
    threshold t, we estimate,

        DS: fdp(t) = min(sum(cs <= -t) / sum(cs >= t), 1)
        BC: fdp(t) = min((1 + sum(cs <= -t)) / sum(cs >= t), 1)

    fdp(t) is then made monotone in t and each gene's q-value is the fdp at the
    threshold equal to its own |cs|. Genes with cs <= 0 are assigned q = 1,
    since only positive contrast scores are evidence for true DE effects.

    Parameters
    ----------
    contrast_scores : pd.Series
        Genewise contrast scores from `contrast_scores`.
    threshold : {"DS", "BC"}
        Which threshold estimator to use (data splitting or Barber-Candes).

    Returns
    -------
    pd.Series
        Genewise q-values with the same gene names as `contrast_scores`.
    """
    if threshold not in ("DS", "BC"):
        raise ValueError(f"threshold must be 'DS' or 'BC', got {threshold!r}")

    values = contrast_scores.to_numpy(dtype=float)
    c_abs = np.unique(np.abs(values[values != 0]))
    c_abs.sort()

    emp_fdp = np.empty(len(c_abs))
    for i, t in enumerate(c_abs):
        num_neg = np.sum(values <= -t)
        num_pos = np.sum(values >= t)
        numerator = (1 + num_neg) if threshold == "BC" else num_neg
        fdp = min(numerator / num_pos, 1.0) if num_pos > 0 else 1.0
        if i >= 1:
            fdp = min(fdp, emp_fdp[i - 1])
        emp_fdp[i] = fdp

    lookup = dict(zip(c_abs, emp_fdp))
    q = np.array([lookup.get(v, 1.0) if v > 0 else 1.0 for v in values])
    return pd.Series(q, index=contrast_scores.index, name="q_value")


def select_genes(q_values: pd.Series, fdr: float = 0.05) -> list:
    """Genes with q-value equal to or less than `fdr`."""
    return list(q_values[q_values <= fdr].index)
