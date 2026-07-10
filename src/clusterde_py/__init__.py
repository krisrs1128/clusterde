"""ClusterDE Algorithm

A minimal implementation of the ClusterDE algorithm (Song et al., 2023) that
uses scDesigner instead of scDesign3. Synthetic null simulation is accomplished
with `null_model` and `null_clustering`. The contrast statistic selection logic
is accomplished using `contrast_scores`, `clipper_q` and `select_genes`.
"""

from .clipper import clipper_q, contrast_scores, select_genes
from .de_pvalues import rank_pvalues
from .null_clustering import null_two_clusters
from .null_model import construct_null
from .pipeline import find_markers

__all__ = [
    "find_markers",
    "construct_null",
    "null_two_clusters",
    "rank_pvalues",
    "contrast_scores",
    "clipper_q",
    "select_genes",
]
