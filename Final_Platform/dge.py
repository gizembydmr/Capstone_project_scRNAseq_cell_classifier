import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from adjustText import adjust_text

def load_data(path):
    adata = sc.read_h5ad(path)
    return adata


def show_available_groups(adata, groupby):
    groups = sorted(adata.obs[groupby].unique().tolist())

    print(f"\nAvailable groups in '{groupby}':")

    for g in groups:
        print("-", g)

    return groups


def run_pairwise_dge(adata, groupby, group1, group2):

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

    gene_ids = [str(x) for x in result["names"][group1]]

    gene_map = dict(
        zip(
            adata.var.index.astype(str),
            adata.var["gene_symbol"].astype(str)
        )
    )

    gene_symbols = [
        gene_map.get(gene_id, gene_id)
        for gene_id in gene_ids
    ]

    df = pd.DataFrame({
        "gene_id": gene_ids,
        "gene": gene_symbols,
        "logfoldchange": result["logfoldchanges"][group1].round(4),
        "pval_adj": result["pvals_adj"][group1],
        "score": result["scores"][group1].round(4)
    })

    df = df.dropna()
    df = df[df["pval_adj"] > 0]
    df = df.sort_values("pval_adj")

    print("\nTop mapped genes:")
    print(df[["gene_id", "gene"]].head(10))

    return df


def save_results(df, group1, group2, prefix):

    os.makedirs("results", exist_ok=True)

    safe_name = f"{group1}_vs_{group2}"
    safe_name = safe_name.replace("/", "_").replace(" ", "_")

    filepath = f"results/dge_{prefix}_{safe_name}.csv"

    df.to_csv(filepath, index=False)

    print(f"\nResults saved to: {filepath}")


def volcano_plot(df, group1, group2, prefix):

    os.makedirs("figures", exist_ok=True)

    safe_name = f"{group1}_vs_{group2}"
    safe_name = safe_name.replace("/", "_").replace(" ", "_")

    df = df.dropna()
    df = df[df["pval_adj"] > 0]

    x = df["logfoldchange"]
    y = -np.log10(df["pval_adj"])

    plt.figure(figsize=(18, 12))

    colors = ["red" if val < 0 else "blue" for val in x]

    plt.scatter(
        x,
        y,
        c=colors,
        s=20,
        alpha=0.6
    )

    threshold = -np.log10(0.05)

    plt.axhline(
        threshold,
        linestyle="--"
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.text(
        x.max(),
        threshold,
        " p = 0.05",
        fontsize=10,
        va="bottom"
    )

    label_df = df[df["pval_adj"] < 0.05]

    label_df = label_df[
        ~label_df["gene"].str.startswith("ENSG", na=False)
    ]
    label_df = label_df[
        ~label_df["gene"].str.startswith("RPL", na=False)
    ]

    label_df = label_df[
        ~label_df["gene"].str.startswith("RPS", na=False)
    ]

    label_df = label_df.nsmallest(20, "pval_adj")

    texts = []

    for _, row in label_df.iterrows():
        texts.append(
            plt.text(
                row["logfoldchange"],
                -np.log10(row["pval_adj"]),
                row["gene"],
                fontsize=8
            )
        )

    adjust_text(
        texts,
        arrowprops=dict(
            arrowstyle="-",
            color="gray",
            lw=0.5
        )
    )

    plt.xlabel("Log Fold Change")
    plt.ylabel("-log10(adjusted p-value)")
    plt.title(f"{group1} vs {group2}")

    filepath = f"figures/{prefix}_{safe_name}.png"

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return filepath


def main():

    print("=== Pairwise DGE Tool ===")

    path = "pbmc68k_preprocessed_for_training.h5ad"

    adata = load_data(path)

    print("\nSelect comparison level:")
    print("1 -> Detailed cell types")
    print("2 -> General cell groups")

    choice = input("Enter 1 or 2: ")

    if choice == "1":
        groupby = "cell_type"
    elif choice == "2":
        groupby = "cell_type_level_1"
    else:
        print("Invalid choice!")
        return

    groups = show_available_groups(adata, groupby)

    print("\nExample:")
    print(groups[0])
    print(groups[1])

    group1 = input("\nEnter first group: ")
    group2 = input("Enter second group: ")

    if group1 not in groups:
        print("First group not found!")
        return

    if group2 not in groups:
        print("Second group not found!")
        return

    df = run_pairwise_dge(
        adata,
        groupby,
        group1,
        group2
    )

    save_results(
        df,
        group1,
        group2,
        groupby
    )

    volcano_plot(
        df,
        group1,
        group2,
        groupby
    )

    print("\nDONE")


if __name__ == "__main__":
    main()