import scanpy as sc
from pathlib import Path


# Load annotated dataset
input_path = Path(r"C:\Users\ferid\Downloads\capstone_demo\pbmc68k_annotated.h5ad")
output_path = Path(r"C:\Users\ferid\Downloads\capstone_demo\pbmc68k_annotated_with_levels.h5ad")

adata = sc.read_h5ad(input_path)


# Keep the original dataset label as cell_type_original
adata.obs["cell_type_original"] = adata.obs["cell_type"]


# Define mapping from original labels to new label levels
label_map = {
    "CD8+/CD45RA+ Naive Cytotoxic": {
        "cell_type_standardized": "CD8 naive T cell",
        "cell_type_level_1": "T cell",
        "cell_type_level_2": "CD8 T cell",
        "cell_type_level_3": "CD8 naive T cell",
    },
    "CD8+ Cytotoxic T": {
        "cell_type_standardized": "CD8 cytotoxic T cell",
        "cell_type_level_1": "T cell",
        "cell_type_level_2": "CD8 T cell",
        "cell_type_level_3": "CD8 cytotoxic T cell",
    },
    "CD4+/CD25 T Reg": {
        "cell_type_standardized": "Regulatory T cell",
        "cell_type_level_1": "T cell",
        "cell_type_level_2": "CD4 T cell",
        "cell_type_level_3": "Regulatory T cell",
    },
    "CD4+/CD45RO+ Memory": {
        "cell_type_standardized": "CD4 memory T cell",
        "cell_type_level_1": "T cell",
        "cell_type_level_2": "CD4 T cell",
        "cell_type_level_3": "CD4 memory T cell",
    },
    "CD4+/CD45RA+/CD25- Naive T": {
        "cell_type_standardized": "CD4 naive T cell",
        "cell_type_level_1": "T cell",
        "cell_type_level_2": "CD4 T cell",
        "cell_type_level_3": "CD4 naive T cell",
    },
    "CD4+ T Helper2": {
        "cell_type_standardized": "CD4 helper T cell",
        "cell_type_level_1": "T cell",
        "cell_type_level_2": "CD4 T cell",
        "cell_type_level_3": "CD4 helper T cell",
    },
    "CD56+ NK": {
        "cell_type_standardized": "NK cell",
        "cell_type_level_1": "NK cell",
        "cell_type_level_2": "NK cell",
        "cell_type_level_3": "NK cell",
    },
    "CD19+ B": {
        "cell_type_standardized": "B cell",
        "cell_type_level_1": "B cell",
        "cell_type_level_2": "B cell",
        "cell_type_level_3": "B cell",
    },
    "CD14+ Monocyte": {
        "cell_type_standardized": "Monocyte",
        "cell_type_level_1": "Monocyte",
        "cell_type_level_2": "Monocyte",
        "cell_type_level_3": "Monocyte",
    },
    "Dendritic": {
        "cell_type_standardized": "Dendritic cell",
        "cell_type_level_1": "Dendritic cell",
        "cell_type_level_2": "Dendritic cell",
        "cell_type_level_3": "Dendritic cell",
    },
    "CD34+": {
        "cell_type_standardized": "CD34+ cell",
        "cell_type_level_1": "CD34+ cell",
        "cell_type_level_2": "CD34+ cell",
        "cell_type_level_3": "CD34+ cell",
    },
}


# Create the new metadata columns from the map
for new_col in [
    "cell_type_standardized",
    "cell_type_level_1",
    "cell_type_level_2",
    "cell_type_level_3",
]:
    adata.obs[new_col] = adata.obs["cell_type_original"].map(
        lambda x: label_map[x][new_col]
    )


# make columns categorical 
for col in [
    "cell_type_original",
    "cell_type_standardized",
    "cell_type_level_1",
    "cell_type_level_2",
    "cell_type_level_3",
]:
    adata.obs[col] = adata.obs[col].astype("category")


# check
print(adata.obs[[
    "cell_type_original",
    "cell_type_standardized",
    "cell_type_level_1",
    "cell_type_level_2",
    "cell_type_level_3"
]].head())

print("\nLevel 1 counts:")
print(adata.obs["cell_type_level_1"].value_counts())


# Save updated AnnData

adata.write(output_path)

print(adata.obs.isna().sum())

print(f"\nSaved updated file to:\n{output_path}")
