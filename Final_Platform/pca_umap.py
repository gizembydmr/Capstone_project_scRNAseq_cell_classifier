import scanpy as sc
import matplotlib.pyplot as plt
import os


def load_data(file_path):
    print("Loading data...")
    adata = sc.read_h5ad(file_path)
    print("Data loaded!")
    return adata


def run_pca_umap(adata, n_neighbors=15, n_pcs=30, min_dist=0.3):
    print("Running PCA...")
    sc.tl.pca(adata, n_comps=50, svd_solver='arpack')

    print("Computing neighbors...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    print("Running UMAP...")
    sc.tl.umap(adata, min_dist=min_dist, random_state=42)

    print("PCA + UMAP completed!")
    return adata


def plot_umap(adata, color_key):
    print(f"Plotting UMAP: {color_key}")

    os.makedirs("figures", exist_ok=True)

    fig = sc.pl.umap(adata, color=color_key, show=False, return_fig=True)

    fig.savefig(f"figures/umap_{color_key}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_pca(adata, color_key):
    print(f"Plotting PCA: {color_key}")

    os.makedirs("figures", exist_ok=True)

    fig = sc.pl.pca(adata, color=color_key, show=False, return_fig=True)

    fig.savefig(f"figures/pca_{color_key}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    print("STARTED")

    data_path = "pbmc68k_preprocessed_for_training.h5ad"

    adata = load_data(data_path)
    adata = run_pca_umap(adata)

    # UMAP
    plot_umap(adata, "cell_type")
    plot_umap(adata, "cell_type_level_1")

    # PCA
    plot_pca(adata, "cell_type")
    plot_pca(adata, "cell_type_level_1")

    print("DONE")


if __name__ == "__main__":
    main()