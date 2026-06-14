# check_pancreas_dataset_structure.py

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from pathlib import Path


# Paths
DATA_PATH = Path(
    r"C:\Users\ferid\Downloads\capstone_demo\pancreas_model\Tabula_Sapiens_Pancreas.h5ad"
)

OUTPUT_DIR = DATA_PATH.parent / "pancreas_structure_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def matrix_basic_stats(X, name="X"):
    print_section(f"MATRIX STATISTICS: {name}")

    print(f"Shape: {X.shape}")
    print(f"Is sparse: {sparse.issparse(X)}")
    print(f"Data type: {X.dtype}")

    if sparse.issparse(X):
        values = X.data
        cell_sums = np.asarray(X.sum(axis=1)).flatten()
        gene_sums = np.asarray(X.sum(axis=0)).flatten()
    else:
        values = X.flatten()
        cell_sums = np.asarray(X.sum(axis=1)).flatten()
        gene_sums = np.asarray(X.sum(axis=0)).flatten()

    print(f"Stored/nonzero values: {len(values)}")

    if len(values) > 0:
        print(f"Minimum stored value: {values.min()}")
        print(f"Maximum stored value: {values.max()}")
        print(f"Mean stored value: {values.mean()}")

        rounded_values = np.round(values)
        integer_like_fraction = np.mean(np.isclose(values, rounded_values))
        print(f"Fraction of integer-like stored values: {integer_like_fraction:.4f}")

    print("\nPer-cell total statistics:")
    print(pd.Series(cell_sums).describe())

    print("\nPer-gene total statistics:")
    print(pd.Series(gene_sums).describe())

    return cell_sums, gene_sums


def find_possible_label_columns(obs):
    keywords = [
        "cell",
        "type",
        "annotation",
        "label",
        "class",
        "ontology",
        "compartment",
        "tissue",
        "organ"
    ]

    possible_cols = []

    for col in obs.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in keywords):
            n_unique = obs[col].nunique(dropna=False)
            if 1 < n_unique <= 100:
                possible_cols.append(col)

    return possible_cols


def summarize_categorical_columns(obs, output_dir):
    rows = []

    for col in obs.columns:
        n_unique = obs[col].nunique(dropna=False)
        n_missing = obs[col].isna().sum()
        dtype = str(obs[col].dtype)

        example_values = obs[col].dropna().astype(str).unique()[:5]
        example_values = ", ".join(example_values)

        rows.append({
            "column": col,
            "dtype": dtype,
            "n_unique": n_unique,
            "n_missing": n_missing,
            "example_values": example_values
        })

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["n_unique", "column"])

    output_path = output_dir / "obs_column_summary.csv"
    summary.to_csv(output_path, index=False)

    print(f"Saved obs column summary to: {output_path}")
    return summary


def summarize_gene_metadata(var, var_names, output_dir):
    rows = []

    rows.append({
        "field": "var_names",
        "n_unique": var_names.nunique(),
        "n_missing": 0,
        "example_values": ", ".join(list(map(str, var_names[:5])))
    })

    for col in var.columns:
        n_unique = var[col].nunique(dropna=False)
        n_missing = var[col].isna().sum()
        example_values = var[col].dropna().astype(str).unique()[:5]
        example_values = ", ".join(example_values)

        rows.append({
            "field": col,
            "n_unique": n_unique,
            "n_missing": n_missing,
            "example_values": example_values
        })

    summary = pd.DataFrame(rows)

    output_path = output_dir / "var_column_summary.csv"
    summary.to_csv(output_path, index=False)

    print(f"Saved var column summary to: {output_path}")
    return summary


def plot_distribution(adata, column, output_dir):
    counts = adata.obs[column].value_counts(dropna=False)

    csv_path = output_dir / f"{column}_distribution.csv"
    counts.to_csv(csv_path, header=["n_cells"])

    plt.figure(figsize=(12, 6))
    counts.plot(kind="bar")
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Number of cells")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plot_path = output_dir / f"{column}_distribution.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved distribution CSV: {csv_path}")
    print(f"Saved distribution plot: {plot_path}")


def inspect_possible_label_columns(adata, possible_cols, output_dir):
    if len(possible_cols) == 0:
        print("No obvious label-like columns found.")
        return None

    print("Possible label-like columns:")
    for col in possible_cols:
        print(f"- {col}")

    best_col = None
    best_score = -1

    for col in possible_cols:
        n_unique = adata.obs[col].nunique(dropna=False)
        min_count = adata.obs[col].value_counts(dropna=False).min()

        score = 0

        if 3 <= n_unique <= 40:
            score += 2
        elif 40 < n_unique <= 80:
            score += 1

        if min_count >= 20:
            score += 1

        if "cell_type" in col.lower() or "celltype" in col.lower():
            score += 3
        elif "ontology" in col.lower() or "annotation" in col.lower():
            score += 2
        elif "cell" in col.lower() and "type" in col.lower():
            score += 2

        if score > best_score:
            best_score = score
            best_col = col

    for col in possible_cols:
        print_section(f"DISTRIBUTION: {col}")

        counts = adata.obs[col].value_counts(dropna=False)
        print(counts)
        print(f"\nUnique values: {adata.obs[col].nunique(dropna=False)}")

        if adata.obs[col].nunique(dropna=False) <= 80:
            plot_distribution(adata, col, output_dir)

    return best_col


def check_gene_identifier_fields(adata):
    print_section("GENE IDENTIFIER INSPECTION")

    print(f"var_names unique: {adata.var_names.nunique() == adata.n_vars}")
    print(f"Number of unique var_names: {adata.var_names.nunique()}")
    print(f"Number of genes: {adata.n_vars}")
    print("\nFirst 10 var_names:")
    print(list(adata.var_names[:10]))

    gene_like_cols = []

    for col in adata.var.columns:
        col_lower = col.lower()
        if "gene" in col_lower or "symbol" in col_lower or "ensembl" in col_lower or "feature" in col_lower:
            gene_like_cols.append(col)

    if len(gene_like_cols) == 0:
        print("\nNo obvious gene identifier columns found in adata.var.")
    else:
        print("\nPossible gene identifier columns:")
        for col in gene_like_cols:
            print(f"- {col}")
            print(list(adata.var[col].head(10)))


def check_count_source(adata):
    print_section("POSSIBLE COUNT DATA SOURCE")

    matrix_sources = {"X": adata.X}

    for layer_name in adata.layers.keys():
        matrix_sources[f"layers/{layer_name}"] = adata.layers[layer_name]

    summaries = []

    for name, X in matrix_sources.items():
        if sparse.issparse(X):
            values = X.data
            cell_sums = np.asarray(X.sum(axis=1)).flatten()
        else:
            values = X.flatten()
            cell_sums = np.asarray(X.sum(axis=1)).flatten()

        if len(values) == 0:
            integer_like_fraction = np.nan
            max_value = np.nan
            mean_cell_sum = np.nan
        else:
            integer_like_fraction = np.mean(np.isclose(values, np.round(values)))
            max_value = values.max()
            mean_cell_sum = cell_sums.mean()

        summaries.append({
            "source": name,
            "integer_like_fraction": integer_like_fraction,
            "max_value": max_value,
            "mean_cell_sum": mean_cell_sum
        })

    summary = pd.DataFrame(summaries)
    print(summary)

    likely_counts = summary.sort_values(
        ["integer_like_fraction", "max_value"],
        ascending=False
    ).iloc[0]["source"]

    print(f"\nMost count-like source: {likely_counts}")

    return summary, likely_counts


# Load dataset
print_section("LOADING DATASET")

adata = sc.read_h5ad(DATA_PATH)

print("Dataset loaded successfully.")
print(adata)


# General structure
print_section("GENERAL STRUCTURE")

print(f"Number of cells: {adata.n_obs}")
print(f"Number of genes: {adata.n_vars}")
print(f"Expression matrix shape: {adata.X.shape}")

print("\nAnnData slots:")
print(f"obs columns: {len(adata.obs.columns)}")
print(f"var columns: {len(adata.var.columns)}")
print(f"layers: {list(adata.layers.keys())}")
print(f"obsm keys: {list(adata.obsm.keys())}")
print(f"varm keys: {list(adata.varm.keys())}")
print(f"uns keys: {list(adata.uns.keys())}")


# Cell metadata
print_section("CELL METADATA")

print("First 5 rows of adata.obs:")
print(adata.obs.head())

print("\nobs columns:")
for col in adata.obs.columns:
    print(f"- {col}")

obs_summary = summarize_categorical_columns(adata.obs, OUTPUT_DIR)

print("\nMost useful obs columns by number of unique values:")
print(obs_summary.head(30))


# Gene metadata
print_section("GENE METADATA")

print("First 5 rows of adata.var:")
print(adata.var.head())

print("\nvar columns:")
for col in adata.var.columns:
    print(f"- {col}")

var_summary = summarize_gene_metadata(adata.var, adata.var_names, OUTPUT_DIR)
print("\nGene metadata summary:")
print(var_summary)

check_gene_identifier_fields(adata)


# Matrix data
matrix_basic_stats(adata.X, name="adata.X")

for layer_name in adata.layers.keys():
    matrix_basic_stats(adata.layers[layer_name], name=f"adata.layers['{layer_name}']")


# Raw
print_section("RAW")

if adata.raw is not None:
    print("adata.raw exists.")
    print(f"raw shape: {adata.raw.shape}")
    print(f"raw var columns: {list(adata.raw.var.columns)}")
else:
    print("adata.raw does not exist.")


# Count source
count_summary, likely_counts = check_count_source(adata)

count_summary_path = OUTPUT_DIR / "matrix_source_summary.csv"
count_summary.to_csv(count_summary_path, index=False)
print(f"Saved matrix source summary to: {count_summary_path}")


# Labels
print_section("LABEL INSPECTION")

possible_label_cols = find_possible_label_columns(adata.obs)
recommended_label_col = inspect_possible_label_columns(
    adata,
    possible_label_cols,
    OUTPUT_DIR
)

if recommended_label_col is not None:
    print_section("RECOMMENDED LABEL COLUMN")
    print(recommended_label_col)
    print("\nLabel counts:")
    print(adata.obs[recommended_label_col].value_counts(dropna=False))
else:
    print_section("RECOMMENDED LABEL COLUMN")
    print("No recommended label column found.")


# Dataset suitability
print_section("DATASET SUITABILITY SUMMARY")

has_labels = recommended_label_col is not None
has_unique_genes = adata.var_names.nunique() == adata.n_vars
has_cells = adata.n_obs > 0
has_genes = adata.n_vars > 0

print(f"Has cells: {has_cells}")
print(f"Has genes: {has_genes}")
print(f"Has unique var_names: {has_unique_genes}")
print(f"Has a likely label column: {has_labels}")
print(f"Recommended label column: {recommended_label_col}")
print(f"Most count-like source: {likely_counts}")

if has_labels:
    n_classes = adata.obs[recommended_label_col].nunique(dropna=False)
    min_class_size = adata.obs[recommended_label_col].value_counts(dropna=False).min()

    print(f"Number of label classes: {n_classes}")
    print(f"Smallest class size: {min_class_size}")

    if min_class_size < 20:
        print("Some classes are very small and may need filtering or merging before training.")

summary = {
    "dataset_path": str(DATA_PATH),
    "n_cells": adata.n_obs,
    "n_genes": adata.n_vars,
    "X_shape": str(adata.X.shape),
    "X_is_sparse": sparse.issparse(adata.X),
    "layers": ", ".join(list(adata.layers.keys())),
    "obsm_keys": ", ".join(list(adata.obsm.keys())),
    "uns_keys": ", ".join(list(adata.uns.keys())),
    "recommended_label_column": recommended_label_col,
    "most_count_like_source": likely_counts,
    "var_names_unique": has_unique_genes,
    "has_raw": adata.raw is not None,
}

summary_path = OUTPUT_DIR / "pancreas_dataset_summary.csv"
pd.DataFrame([summary]).to_csv(summary_path, index=False)

print(f"\nSaved summary report to: {summary_path}")


print_section("DONE")
print(f"All outputs saved in: {OUTPUT_DIR}")