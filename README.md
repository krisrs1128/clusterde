# clusterde_py

A Python implementation of the ClusterDE algorithm for differential expression
analysis with synthetic null comparison, based on
[ClusterDE](https://songdongyuan1994.github.io/ClusterDE). You can install it
with:

```bash
pip install clusterde_py
```

## Example

This example is further developed in `examples/pancreas.ipynb`.

```python
from clusterde_py import find_markers
from scdesigner.datasets import pancreas

# Load example data
adata = pancreas()

# Subset to two groups for comparison
adata_filtered = adata[adata.obs["cell_type"].isin(["Ngn3 low EP", "Ngn3 high EP"])].copy()

# Find cluster DE supported markers
result = find_markers(
    adata_filtered,
    cluster_key="cell_type",
    group1="Ngn3 low EP",
    group2="Ngn3 high EP",
    fdr=0.05
)
```
