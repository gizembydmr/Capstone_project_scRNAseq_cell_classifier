import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def load_data(path):
    # Load dataset
    adata = sc.read_h5ad(path)
    return adata


def show_available_groups(adata, groupby):
    # Show available group names
    groups = adata.obs[groupby].unique().tolist()

    print(f"\nAvailable groups in '{groupby}':")
    print(", ".join(groups[:10]))

    return groups


def run_pairwise_dge(adata, groupby, group1, group2):
    # Run DGE for selected pair
    print(f"\nRunning DGE: {group1} vs {group2}")

    adata_copy = adata.copy()

    sc.tl.rank_genes_groups(
        adata_copy,
        groupby=groupby,
        groups=[group1],
        reference=group2,
        method="wilcoxon"
    )

    result = adata_copy.uns["rank_genes_groups"]

    df = pd.DataFrame({
        "gene": result["names"][group1],
        "logfoldchange": result["logfoldchanges"][group1].round(4),
        "pval_adj": result["pvals_adj"][group1],
        "score": result["scores"][group1].round(4),
    })

    # Clean (numeric işlemler string dönüşümünden önce)
    df = df.dropna()
    df = df[df["pval_adj"] > 0]

    # Sort by significance
    df = df.sort_values(by="pval_adj")


    return df


def save_results(df, group1, group2, prefix):
    # Save full results
    os.makedirs("results", exist_ok=True)

    safe_name = f"{group1}_vs_{group2}".replace("/", "_").replace(" ", "_")

    filepath = f"results/dge_{prefix}_{safe_name}.csv"
    df.to_csv(filepath, index=False)

    print(f"\nResults saved to: {filepath}")


def volcano_plot(df, group1, group2, prefix):
    # Create volcano plot with p=0.05 threshold

    os.makedirs("figures", exist_ok=True)

    safe_name = f"{group1}_vs_{group2}".replace("/", "_").replace(" ", "_")

    df = df.dropna()
    df = df[df["pval_adj"] > 0]

    x = df["logfoldchange"]
    y = -np.log10(df["pval_adj"])

    plt.figure(figsize=(6, 5))

    # Color by direction
    colors = ["red" if val < 0 else "blue" for val in x]

    plt.scatter(
        x,
        y,
        c=colors,
        s=10,
        alpha=0.6
    )

    # Threshold lines
    threshold = -np.log10(0.05)
    plt.axhline(threshold, linestyle='--')
    plt.axvline(0, linestyle='--')

    # Label for p-value threshold
    plt.text(
        x.max(),
        threshold,
        " p = 0.05",
        fontsize=9,
        verticalalignment='bottom'
    )

    plt.xlabel("Log Fold Change")
    plt.ylabel("-log10(p-value)")
    plt.title(f"{group1} vs {group2}")

    plt.savefig(f"figures/{prefix}_{safe_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Volcano plot saved.")


def main():
    print("=== Pairwise DGE Tool ===")

    path = "pbmc68k_preprocessed_for_training.h5ad"
    adata = load_data(path)

    # --- SELECT LEVEL ---
    print("\nSelect comparison level:")
    print("1 → Detailed cell types (cell_type)")
    print("2 → General cell groups (cell_type_level_1)")

    choice = input("Enter 1 or 2: ")

    if choice == "1":
        groupby = "cell_type"
    elif choice == "2":
        groupby = "cell_type_level_1"
    else:
        print("Invalid choice!")
        return

    # --- SHOW GROUPS ---
    groups = show_available_groups(adata, groupby)

    # --- EXAMPLES ---
    print("\nExample input:")
    if groupby == "cell_type":
        print("Example: CD4+/CD45RO+ Memory  vs  CD8+ Cytotoxic T")
    else:
        print("Example: T cell  vs  B cell")

    # --- USER INPUT ---
    group1 = input("\nEnter first group EXACTLY as shown above: ")
    group2 = input("Enter second group EXACTLY as shown above: ")

    if group1 not in groups or group2 not in groups:
        print("Invalid group names! Please copy exactly.")
        return

    # --- RUN DGE ---
    df = run_pairwise_dge(adata, groupby, group1, group2)

    # --- SAVE ---
    save_results(df, group1, group2, groupby)

    # --- PLOT ---
    volcano_plot(df, group1, group2, groupby)

    print("\nDONE")


if __name__ == "__main__":
    main()
