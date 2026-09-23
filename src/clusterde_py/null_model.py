"""Synthetic null construction via scdesigner.

This function applies scDesigner to fit a model where there is no true clsuter
difference. We are only supporting negative-binomial copulas here but could
switch other models in without much difficulty. Since the formula for this model
doesn't include cluster labels, the resulting synthetic data has no
cluster-driven differential expression, and the population has homogeneous
marginal and correlation structure. This approach mirrors constructNull() in the
ClusterDE R package.
"""

from typing import Optional

import torch
from anndata import AnnData
from scdesigner.simulators import NegBinCopula


def construct_null(
    adata: AnnData,
    mean_formula: str = "~ 1",
    dispersion_formula: str = "~ 1",
    copula_formula: str = "~ 1",
    seed: Optional[int] = None,
    init_kwargs: Optional[dict] = None,
    fit_kwargs: Optional[dict] = None,
) -> AnnData:
    """Sample synthetic null data from an NB Copula

    Parameters
    ----------
    adata : AnnData
        Real data (both clusters pooled together); only `adata.X` and
        `adata.var` are used for fitting, and `adata.obs` is reused as the
        covariate template for sampling.
    mean_formula, dispersion_formula, copula_formula : str
        Passed to `scdesigner.simulators.NegBinCopula`. Defaults are
        intercept-only ("~ 1"), so there is no cluster effect.
    seed : int or None
        Random number state to use for NB estimating and sampling.
    init_kwargs : dict or None
        Keyword arguments to pass to `NegBinCopula` (e.g. `top_k`, `estimator`),
        which can control the copula modeling (see
        `scdesigner.simulators.NegBinCopula`).
    fit_kwargs : dict or None
        Extra keyword arguments forwarded to `NegBinCopula.fit` (e.g.
        `max_epochs`, `batch_size`).

    Returns
    -------
    AnnData
        Synthetic null data with the same shape, `obs`, and `var` as `adata`,
        but resampled `X`.
    """
    if seed is not None:
        torch.manual_seed(seed)

    sim = NegBinCopula(
        mean_formula=mean_formula,
        dispersion_formula=dispersion_formula,
        copula_formula=copula_formula,
        **(init_kwargs or {}),
    )
    sim.fit(adata, **(fit_kwargs or {}))

    if seed is not None:
        torch.manual_seed(seed)
    return sim.sample(obs=adata.obs)
