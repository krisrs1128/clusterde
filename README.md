# clusterde_py

A Python implementation of the ClusterDE algorithm for differential expression
analysis with synthetic null comparison. You can install it with:

```bash
pip install clusterde_py
```

## Example

This example is further developed and compared with a naive baseline in examples/pancreas.ipynb.
A second example, replicating the [ClusterDE PBMC vignette](https://songdongyuan1994.github.io/ClusterDE/articles/ClusterDE-PBMC.html)
by comparing CD14+ vs. FCGR3A+ (CD16+) monocyte clusters, is in examples/pbmc.ipynb.

```python
from clusterde_py import find_markers
from scdesigner.datasets import pancreas

# Load example data
adata = pancreas()

# Subset to two groups for comparison
sub = adata[adata.obs["cell_type"].isin(["Ngn3 low EP", "Ngn3 high EP"])].copy()

# Find cluster DE supported markers
result = find_markers(
    sub,
    cluster_key="cell_type",
    group1="Ngn3 low EP",
    group2="Ngn3 high EP",
    fdr=0.05
)
```
