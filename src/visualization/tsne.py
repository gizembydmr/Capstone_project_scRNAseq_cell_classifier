import scanpy as sc
import matplotlib.pyplot as plt
import os


adata = sc.read_h5ad(
    "pbmc68k_preprocessed_for_training.h5ad"
)

sc.pp.pca(adata, n_comps=30)

sc.tl.tsne(
    adata,
    use_rep="X_pca"
)

os.makedirs("figures", exist_ok=True)

sc.pl.tsne(
    adata,
    color="cell_type_level_1",
    show=False
)

plt.savefig(
    "figures/tsne_cell_type_level_1.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("t-SNE saved.")
