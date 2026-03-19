import scanpy as sc
from pathlib import Path

# -----------------------------
# 1) Load your annotated dataset
# -----------------------------
input_path = Path(r"C:\Users\ferid\Downloads\capstone_demo\pbmc68k_annotated.h5ad")
output_path = Path(r"C:\Users\ferid\Downloads\capstone_demo\pbmc68k_annotated_with_levels.h5ad")

adata = sc.read_h5ad(input_path)

# --------------------------------------------------------
# 2) Keep the original dataset label as cell_type_original
# --------------------------------------------------------
adata.obs["cell_type_original"] = adata.obs["cell_type"]

# --------------------------------------------------------
# 3) Define mapping from original labels to new label levels
# --------------------------------------------------------
label_map = {
    "CD8+/CD45RA+ Naive Cytotoxic": {
        "cell_type_standardized": "CD8 naive T",
        "cell_type_level_1": "T cell",
        "cell_type_level_2": "CD8 T",
        "cell_type_level_3": "CD8 naive T",
    },
    "CD8+ Cytotoxic T": {
        "cell_type_standardized": "CD8 cytotoxic T",
        "cell_type_level_1": "T cell",
        "cell_type_level_2": "CD8 T",
        "cell_type_level_3": "CD8 cytotoxic T",
    },
    "CD4+/CD25 T Reg": {
        "cell_type_standardized": "Regulatory T",
        "cell_type_level_1": "T cell",
        "cell_type_level_2": "CD4 T",
        "cell_type_level_3": "Regulatory T",
    },
    "CD4+/CD45RO+ Memory": {
        "cell_type_standardized": "CD4 memory T",
        "cell_type_level_1": "T cell",
        "cell_type_level_2": "CD4 T",
        "cell_type_level_3": "CD4 memory T",
    },
    "CD4+/CD45RA+/CD25- Naive T": {
        "cell_type_standardized": "CD4 naive T",
        "cell_type_level_1": "T cell",
        "cell_type_level_2": "CD4 T",
        "cell_type_level_3": "CD4 naive T",
    },
    "CD4+ T Helper2": {
        "cell_type_standardized": "CD4 helper T",
        "cell_type_level_1": "T cell",
        "cell_type_level_2": "CD4 T",
        "cell_type_level_3": "CD4 helper T",
    },
    "CD56+ NK": {
        "cell_type_standardized": "NK cell",
        "cell_type_level_1": "NK cell",
        "cell_type_level_2": "NK",
        "cell_type_level_3": "NK cell",
    },
    "CD19+ B": {
        "cell_type_standardized": "B cell",
        "cell_type_level_1": "B cell",
        "cell_type_level_2": "B",
        "cell_type_level_3": "B cell",
    },
    "CD14+ Monocyte": {
        "cell_type_standardized": "Monocyte",
        "cell_type_level_1": "Monocyte",
        "cell_type_level_2": "Classical mono",
        "cell_type_level_3": "Monocyte",
    },
    "Dendritic": {
        "cell_type_standardized": "Dendritic cell",
        "cell_type_level_1": "Dendritic",
        "cell_type_level_2": "DC",
        "cell_type_level_3": "Dendritic cell",
    },
    "CD34+": {
        "cell_type_standardized": "Progenitor",
        "cell_type_level_1": "Progenitor",
        "cell_type_level_2": "Stem",
        "cell_type_level_3": "Progenitor",
    },
}

# ------------------------------------------------
# 4) Create the new metadata columns from the map
# ------------------------------------------------
for new_col in [
    "cell_type_standardized",
    "cell_type_level_1",
    "cell_type_level_2",
    "cell_type_level_3",
]:
    adata.obs[new_col] = adata.obs["cell_type_original"].map(
        lambda x: label_map[x][new_col]
    )

# ----------------------------------------
# 5) Optional: make them categorical columns
# ----------------------------------------
for col in [
    "cell_type_original",
    "cell_type_standardized",
    "cell_type_level_1",
    "cell_type_level_2",
    "cell_type_level_3",
]:
    adata.obs[col] = adata.obs[col].astype("category")

# -------------------------
# 6) Quick sanity check
# -------------------------
print(adata.obs[[
    "cell_type_original",
    "cell_type_standardized",
    "cell_type_level_1",
    "cell_type_level_2",
    "cell_type_level_3"
]].head())

print("\nLevel 1 counts:")
print(adata.obs["cell_type_level_1"].value_counts())

# -------------------------
# 7) Save updated AnnData
# -------------------------
adata.write(output_path)

print(adata.obs.isna().sum())

print(f"\nSaved updated file to:\n{output_path}")