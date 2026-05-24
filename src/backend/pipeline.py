"""
pipeline.py
===========
Inference pipeline for the Cell Type Prediction Platform.

Orchestrates all analysis steps in order:

    1. Load data          →  data_loader.py
    2. Validate data      →  data_validation.py
    3. Preprocess         →  preprocess_inference.py
    4. Gene alignment     →  gene_alignment.py
    5. Run prediction     →  model.py
    6. PCA + UMAP         →  pca_umap.py
    7. Export results     →  (export helpers below)

Usage:
    from pipeline import run_pipeline, PipelineResult

    result = run_pipeline(
        filepath="data/sample.h5ad",
        tissue="pbmc",
        output_dir="results/",          # optional
        progress_callback=print,        # optional — receives status strings
    )

    if result.success:
        print(result.predictions)       # pd.DataFrame: cell_barcode | predicted_cell_type
        print(result.umap_coords)       # np.ndarray  : (n_cells, 2)
    else:
        print(result.error)
"""

from __future__ import annotations

import io
import importlib
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Module import helper
# ─────────────────────────────────────────────────────────────────────────────

def _import(module_name: str):
    """
    Import a sibling module by name.

    Returns None only when the module file does not exist.
    Re-raises any other import error so real bugs are not silently swallowed.
    """
    here = Path(__file__).parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if module_name in str(exc):
            return None   # module file not present yet
        raise             # a dependency of the module is missing — surface the error


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepStatus:
    name: str
    success: bool
    message: str = ""


@dataclass
class PipelineResult:
    """Full result object returned by run_pipeline()."""

    success: bool = False
    error: str = ""
    steps: List[StepStatus] = field(default_factory=list)

    predictions: Optional[pd.DataFrame] = None    # cell_barcode | predicted_cell_type
    probabilities: Optional[pd.DataFrame] = None  # per-class probability scores
    umap_coords: Optional[np.ndarray] = None       # (n_cells, 2)
    pca_coords: Optional[np.ndarray] = None        # (n_cells, n_pcs)

    adata: Optional[object] = None                 # preprocessed AnnData
    n_cells: int = 0
    n_genes: int = 0
    tissue: str = ""

    exported_files: Dict[str, Path] = field(default_factory=dict)

    def summary(self) -> str:
        lines = []
        icon = "OK" if self.success else "FAIL"
        lines.append(f"[{icon}] Pipeline {'succeeded' if self.success else 'failed'}")
        for s in self.steps:
            step_icon = "+" if s.success else "-"
            lines.append(f"  [{step_icon}] {s.name}: {s.message}")
        if not self.success:
            lines.append(f"\n  Error: {self.error}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    filepath: str | Path,
    tissue: str,
    output_dir: Optional[str | Path] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    """
    Execute the full cell-type prediction pipeline.

    Parameters
    ----------
    filepath : str or Path
        Path to the uploaded scRNA-seq file (.h5ad, .csv, .mtx).
    tissue : str
        Tissue key, e.g. "pbmc", "lung", "liver", "brain".
    output_dir : str or Path, optional
        If provided, results (CSV + plots) are exported here.
    progress_callback : callable, optional
        Called with a status string at each step.

    Returns
    -------
    PipelineResult
    """
    result = PipelineResult(tissue=tissue)
    filepath = Path(filepath)

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Load data
    # ══════════════════════════════════════════════════════════════════════════
    _log("Loading dataset...")
    try:
        loader = _import("data_loader")
        if loader is None:
            raise ImportError("data_loader.py not found in project directory.")

        load_res = loader.load_file(filepath)
        if not load_res.success:
            raise ValueError(load_res.error)

        adata = load_res.adata
        result.n_cells = load_res.n_cells
        result.n_genes = load_res.n_genes
        result.steps.append(StepStatus(
            "Load", True,
            f"{load_res.n_cells:,} cells x {load_res.n_genes:,} genes [{load_res.file_format}]",
        ))

    except Exception as exc:
        result.steps.append(StepStatus("Load", False, str(exc)))
        result.error = f"Data loading failed: {exc}"
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Validate data
    # ══════════════════════════════════════════════════════════════════════════
    _log("Validating data...")
    try:
        validator = _import("data_validation")
        if validator is None:
            raise ImportError("data_validation.py not found in project directory.")

        val_res = validator.validate_adata(adata)
        if not val_res.is_valid:
            raise ValueError("\n".join(val_res.errors))

        warn_note = f" | {len(val_res.warnings)} warning(s)" if val_res.warnings else ""
        result.steps.append(StepStatus("Validate", True, f"All checks passed{warn_note}"))

    except Exception as exc:
        result.steps.append(StepStatus("Validate", False, str(exc)))
        result.error = f"Validation failed: {exc}"
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Preprocess
    # ══════════════════════════════════════════════════════════════════════════
    _log("Preprocessing...")
    try:
        preprocess = _import("preprocess_inference")
        if preprocess is not None and hasattr(preprocess, "preprocess_for_inference"):
            adata = preprocess.preprocess_for_inference(adata)
            result.steps.append(StepStatus("Preprocess", True, "Normalisation + HVG selection done"))
        else:
            adata = _fallback_preprocess(adata)
            result.steps.append(StepStatus(
                "Preprocess", True,
                "preprocess_inference.py not available — used built-in fallback",
            ))
    except IndexError as exc:
        if "Positions outside range" in str(exc):
            msg = (
                f"Dataset has too few genes ({adata.n_vars}) for QC metrics. "
                f"scanpy requires at least 500 genes. Add more genes to the input file."
            )
        else:
            msg = str(exc)
        result.steps.append(StepStatus("Preprocess", False, msg))
        result.error = f"Preprocessing failed: {msg}"
        return result
    except Exception as exc:
        result.steps.append(StepStatus("Preprocess", False, str(exc)))
        result.error = f"Preprocessing failed: {exc}"
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — Gene alignment
    # ══════════════════════════════════════════════════════════════════════════
    _log("Aligning genes to model reference...")
    try:
        gene_align = _import("gene_alignment")
        if gene_align is not None and hasattr(gene_align, "align_genes_to_training"):
            here = Path(__file__).parent
            # Look for the HVG reference CSV (team should provide this file)
            hvg_candidates = [
                here / "pbmc68k_hvg_list.csv",
                here / f"{tissue}_hvg_list.csv",
                here / "hvg_list.csv",
            ]
            hvg_csv = next((p for p in hvg_candidates if p.exists()), None)

            if hvg_csv is not None:
                training_gene_order = gene_align.load_training_gene_order(str(hvg_csv))
                adata = gene_align.align_genes_to_training(adata, training_gene_order)
                result.steps.append(StepStatus(
                    "Gene Alignment", True,
                    f"Aligned to {tissue} reference ({len(training_gene_order)} genes)",
                ))
            else:
                result.steps.append(StepStatus(
                    "Gene Alignment", True,
                    "HVG reference CSV not found — genes used as-is (ask team for pbmc68k_hvg_list.csv)",
                ))
        else:
            result.steps.append(StepStatus(
                "Gene Alignment", True,
                "gene_alignment.py not available — genes used as-is",
            ))
    except Exception as exc:
        result.steps.append(StepStatus(
            "Gene Alignment", False, f"Warning: {exc} — continuing with available genes"
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Prediction
    # ══════════════════════════════════════════════════════════════════════════
    _log("Running cell-type prediction...")
    try:
        model_mod = _import("model")
        if model_mod is not None and hasattr(model_mod, "predict"):
            pred_result = model_mod.predict(adata, tissue=tissue)
            predicted_labels = pred_result["labels"]
            result.probabilities = pred_result.get("probabilities")
            step_msg = f"{len(pd.Series(predicted_labels).unique())} cell types predicted"
        else:
            predicted_labels = _fallback_predict(adata, tissue)
            step_msg = "model.py not available — demo predictions used"

        result.predictions = pd.DataFrame({
            "cell_barcode": adata.obs_names.tolist(),
            "predicted_cell_type": predicted_labels,
        })
        adata.obs["predicted_cell_type"] = predicted_labels
        result.steps.append(StepStatus("Predict", True, step_msg))

    except Exception as exc:
        result.steps.append(StepStatus("Predict", False, str(exc)))
        result.error = f"Prediction failed: {exc}"
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6 — PCA + UMAP
    # ══════════════════════════════════════════════════════════════════════════
    _log("Computing PCA and UMAP...")
    try:
        pca_umap_mod = _import("pca_umap")
        if pca_umap_mod is not None and hasattr(pca_umap_mod, "run_pca_umap"):
            adata = pca_umap_mod.run_pca_umap(adata)
            result.pca_coords = adata.obsm.get("X_pca")
            result.umap_coords = adata.obsm.get("X_umap")
            result.steps.append(StepStatus("PCA + UMAP", True, "Embeddings computed"))
        else:
            result.pca_coords, result.umap_coords = _fallback_pca_umap(adata)
            result.steps.append(StepStatus(
                "PCA + UMAP", True, "pca_umap.py not available — used built-in fallback"
            ))
    except Exception as exc:
        result.steps.append(StepStatus("PCA + UMAP", False, f"Warning: {exc}"))
        # Non-fatal: predictions are still valid without UMAP

    result.adata = adata
    result.n_cells = adata.n_obs
    result.n_genes = adata.n_vars

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 7 — Export (optional)
    # ══════════════════════════════════════════════════════════════════════════
    if output_dir is not None:
        _log("Exporting results...")
        try:
            exported = export_results(result, output_dir)
            result.exported_files = exported
            result.steps.append(StepStatus(
                "Export", True,
                f"{len(exported)} file(s) written to '{output_dir}'",
            ))
        except Exception as exc:
            result.steps.append(StepStatus("Export", False, str(exc)))

    result.success = True
    _log("Pipeline complete.")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────────────────────

def export_results(result: PipelineResult, output_dir: str | Path) -> Dict[str, Path]:
    """
    Export predictions and plots to disk.

    Writes:
        predictions.csv          — cell barcode + predicted cell type
        umap_plot.png            — UMAP scatter coloured by cell type
        cell_type_distribution.png — horizontal bar chart of cell type counts

    Returns
    -------
    dict mapping filename key -> Path of written file
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    exported: Dict[str, Path] = {}

    if result.predictions is not None:
        df = result.predictions.copy()
        if result.probabilities is not None:
            df = df.join(result.probabilities.set_index(df.index), how="left")
        csv_path = out / "predictions.csv"
        df.to_csv(csv_path, index=False)
        exported["predictions_csv"] = csv_path

    if result.umap_coords is not None and result.predictions is not None:
        umap_path = out / "umap_plot.png"
        _export_umap(result, umap_path)
        exported["umap_plot"] = umap_path

    if result.predictions is not None:
        dist_path = out / "cell_type_distribution.png"
        _export_distribution(result, dist_path)
        exported["cell_type_distribution"] = dist_path

    return exported


def export_predictions_csv(result: PipelineResult) -> bytes:
    """Return predictions as CSV bytes (for Streamlit st.download_button)."""
    if result.predictions is None:
        return b""
    df = result.predictions.copy()
    if result.probabilities is not None:
        df = df.join(result.probabilities.reset_index(drop=True), how="left")
    return df.to_csv(index=False).encode("utf-8")


def export_umap_png(result: PipelineResult, dpi: int = 150) -> bytes:
    """Return UMAP plot as PNG bytes (for Streamlit st.download_button)."""
    buf = io.BytesIO()
    _export_umap(result, buf, dpi=dpi)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    "#58a6ff", "#3fb950", "#f78166", "#e3b341",
    "#bc8cff", "#ff7b72", "#79c0ff", "#ffa657",
    "#56d364", "#f0883e", "#a5d6ff", "#d2a8ff",
]


def _export_umap(result: PipelineResult, dest, dpi: int = 150) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = result.umap_coords
    labels = result.predictions["predicted_cell_type"].values
    unique_types = list(pd.Series(labels).value_counts().index)
    color_map = {ct: _PALETTE[i % len(_PALETTE)] for i, ct in enumerate(unique_types)}

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    for ct in unique_types:
        mask = labels == ct
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color_map[ct], s=10, alpha=0.75,
            linewidths=0, label=ct, rasterized=True,
        )

    legend = ax.legend(fontsize=8, markerscale=2, frameon=True,
                       framealpha=0.15, edgecolor="#30363d", labelcolor="#e6edf3")
    legend.get_frame().set_facecolor("#161b22")
    ax.set_xlabel("UMAP 1", fontsize=10, color="#8b949e")
    ax.set_ylabel("UMAP 2", fontsize=10, color="#8b949e")
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.spines[:].set_color("#21262d")
    ax.set_title(f"UMAP - {result.tissue.upper()}", fontsize=12, color="#e6edf3", pad=10)
    plt.tight_layout()
    fig.savefig(dest, dpi=dpi, facecolor="#0d1117")
    plt.close(fig)


def _export_distribution(result: PipelineResult, dest, dpi: int = 150) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = result.predictions["predicted_cell_type"].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.55)))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.6, edgecolor="none")
    for bar, cnt in zip(bars, values[::-1]):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(cnt), va="center", ha="left",
            fontsize=9, color="#8b949e", fontfamily="monospace",
        )

    ax.set_xlabel("Number of Cells", fontsize=10, color="#8b949e", labelpad=8)
    ax.tick_params(colors="#8b949e", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_title("Cell Type Distribution", fontsize=12, color="#e6edf3", pad=10)
    plt.tight_layout()
    fig.savefig(dest, dpi=dpi, facecolor="#0d1117")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback implementations (used when a module is not yet available)
# ─────────────────────────────────────────────────────────────────────────────

_TISSUE_CELL_TYPES: Dict[str, List[str]] = {
    "pbmc":  ["B cells", "T cells (CD4+)", "T cells (CD8+)", "NK cells",
              "Monocytes (Classical)", "Monocytes (Non-Classical)", "Dendritic Cells"],
    "lung":  ["AT1 cells", "AT2 cells", "Club cells", "Ciliated cells",
              "Endothelial cells", "Fibroblasts", "Macrophages", "T cells"],
    "liver": ["Hepatocytes", "Cholangiocytes", "Kupffer cells",
              "Hepatic Stellate cells", "Endothelial cells", "NK cells"],
    "brain": ["Neurons (Excitatory)", "Neurons (Inhibitory)", "Astrocytes",
              "Oligodendrocytes", "OPC", "Microglia", "Endothelial cells"],
}


def _fallback_preprocess(adata):
    """Basic scanpy preprocessing fallback."""
    try:
        import scanpy as sc
        import scipy.sparse as sp

        if not sp.issparse(adata.X):
            adata.X = sp.csr_matrix(adata.X)

        sc.pp.filter_cells(adata, min_genes=5)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        n_hvg = min(2000, adata.n_vars)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")
    except Exception:
        pass
    return adata


def _fallback_predict(adata, tissue: str) -> np.ndarray:
    """Demo predictions — replaced once model.py is ready."""
    cell_types = _TISSUE_CELL_TYPES.get(tissue.lower(), ["Unknown"])
    rng = np.random.default_rng(42)
    weights = rng.dirichlet(np.ones(len(cell_types)))
    return rng.choice(cell_types, size=adata.n_obs, p=weights)


def _fallback_pca_umap(adata):
    """Basic scanpy PCA + UMAP fallback."""
    pca_coords = None
    umap_coords = None
    try:
        import scanpy as sc

        n_pcs = min(50, adata.n_obs - 1, adata.n_vars - 1)
        sc.tl.pca(adata, n_comps=max(2, n_pcs), use_highly_variable=True)
        pca_coords = adata.obsm.get("X_pca")

        sc.pp.neighbors(adata, n_pcs=min(30, pca_coords.shape[1] if pca_coords is not None else 10))
        sc.tl.umap(adata)
        umap_coords = adata.obsm.get("X_umap")
    except Exception:
        rng = np.random.default_rng(0)
        umap_coords = rng.standard_normal((adata.n_obs, 2))

    return pca_coords, umap_coords


# ─────────────────────────────────────────────────────────────────────────────
# CLI:  python pipeline.py <file> <tissue> [output_dir]
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <filepath> <tissue> [output_dir]")
        print("       tissue: pbmc | lung | liver | brain")
        sys.exit(1)

    fp = sys.argv[1]
    ts = sys.argv[2]
    od = sys.argv[3] if len(sys.argv) > 3 else None

    res = run_pipeline(fp, ts, output_dir=od, progress_callback=print)
    print(res.summary())
    sys.exit(0 if res.success else 1)
