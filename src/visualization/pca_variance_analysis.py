import scanpy as sc
import matplotlib.pyplot as plt
import os


def load_data(file_path):
    print("Loading data...")
    adata = sc.read_h5ad(file_path)
    print("Data loaded!")
    return adata


def run_pca(adata, n_comps=50):
    print("Running PCA...")

    sc.tl.pca(
        adata,
        n_comps=n_comps,
        svd_solver="arpack"
    )

    print("PCA completed!")
    return adata


def plot_pca_variance(adata):

    os.makedirs("figures", exist_ok=True)

    variance_ratio = adata.uns["pca"]["variance_ratio"]

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(variance_ratio) + 1),
        variance_ratio,
        marker="o"
    )

    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Explained Variance")

    plt.savefig(
        "figures/pca_variance_ratio.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("PCA variance plot saved.")


def plot_cumulative_variance(adata):

    os.makedirs("figures", exist_ok=True)

    variance_ratio = adata.uns["pca"]["variance_ratio"]

    cumulative_variance = variance_ratio.cumsum()

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker="o"
    )

    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative PCA Explained Variance")

    plt.savefig(
        "figures/pca_cumulative_variance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Cumulative variance plot saved.")


def main():

    data_path = "pbmc68k_preprocessed_for_training.h5ad"

    adata = load_data(data_path)

    adata = run_pca(
        adata,
        n_comps=50
    )

    plot_pca_variance(adata)

    plot_cumulative_variance(adata)

    print("DONE")


if __name__ == "__main__":
    main()