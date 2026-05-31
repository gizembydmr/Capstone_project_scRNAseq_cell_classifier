"""
backend.py
==========
Integration layer between the GUI and all backend modules.

The GUI imports exclusively from this module so that changes to individual
backend files do not require touching gui.py.

Public API
----------
run_analysis(filepath, tissue, progress_callback) -> AnalysisResult
    Full pipeline: load → validate → preprocess → predict → PCA/UMAP.

run_dge(adata, group1, group2) -> DGEResult
    Pairwise differential gene expression between two predicted cell types.

get_pca_plot_bytes(pca_coords, labels, tissue) -> bytes
    PCA scatter plot as PNG bytes (for st.image / st.download_button).

get_volcano_plot_bytes(dge_result) -> bytes
    Volcano plot for a completed DGEResult as PNG bytes.

Usage in gui.py
---------------
    from backend import run_analysis, run_dge, get_pca_plot_bytes, AnalysisResult

    # Replace inline pipeline block with:
    result = run_analysis(tmp_path, tissue_key, progress_callback=status_text.markdown)
    if result.success:
        st.session_state.predictions  = result.predictions
        st.session_state.umap_coords  = result.umap_coords
        st.session_state.pca_coords   = result.pca_coords
        st.session_state.adata        = result.adata
        st.session_state.run_complete = True

    # DGE on demand:
    dge = run_dge(result.adata, group1, group2)
    if dge.success:
        st.dataframe(dge.table)
        st.download_button("Download volcano", dge.volcano_png, "volcano.png")
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from pipeline import run_pipeline, PipelineResult, StepStatus

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DGEResult:
    """Result of a pairwise DGE analysis between two predicted cell types."""

    group1: str
    group2: str
    table: Optional[pd.DataFrame] = None    # gene | logfoldchange | pval_adj | score
    volcano_png: Optional[bytes] = None     # PNG bytes for st.download_button
    error: str = ""

    @property
    def success(self) -> bool:
        return self.table is not None and not self.error


@dataclass
class AnalysisResult:
    """
    Complete result returned to the GUI by run_analysis().

    Fields mirror PipelineResult so the GUI can access everything it needs
    from a single object without importing pipeline.py directly.
    """

    success: bool = False
    error: str = ""
    steps: List[StepStatus] = field(default_factory=list)

    predictions: Optional[pd.DataFrame] = None      # cell_barcode | predicted_cell_type
    probabilities: Optional[pd.DataFrame] = None    # per-class probability scores
    umap_coords: Optional[np.ndarray] = None        # (n_cells, 2)
    pca_coords: Optional[np.ndarray] = None         # (n_cells, n_pcs)
    adata: Optional[object] = None                  # preprocessed AnnData
    n_cells: int = 0
    n_genes: int = 0
    tissue: str = ""

    @classmethod
    def from_pipeline(cls, pr: PipelineResult) -> "AnalysisResult":
        return cls(
            success=pr.success,
            error=pr.error,
            steps=pr.steps,
            predictions=pr.predictions,
            probabilities=pr.probabilities,
            umap_coords=pr.umap_coords,
            pca_coords=pr.pca_coords,
            adata=pr.adata,
            n_cells=pr.n_cells,
            n_genes=pr.n_genes,
            tissue=pr.tissue,
        )

    def cell_types(self) -> List[str]:
        """Sorted list of unique predicted cell types."""
        if self.predictions is None:
            return []
        return sorted(self.predictions["predicted_cell_type"].unique().tolist())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_analysis(
    filepath: str | Path,
    tissue: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    min_counts: int = 500,
    min_genes: int = 200,
) -> AnalysisResult:
    """
    Run the full cell-type prediction pipeline.

    This replaces the inline preprocessing/prediction block in gui.py.
    All backend steps (load, validate, preprocess, align, predict, PCA/UMAP)
    are handled by pipeline.py; this function wraps the result for the GUI.

    Parameters
    ----------
    filepath : str or Path
        Path to the uploaded scRNA-seq file (.h5ad, .csv, .mtx).
    tissue : str
        Tissue key: "pbmc" | "lung" | "liver" | "brain".
    progress_callback : callable, optional
        Receives status strings — pass ``st.empty().markdown`` for live updates.

    Returns
    -------
    AnalysisResult
    """
    pr = run_pipeline(
        filepath=filepath,
        tissue=tissue,
        progress_callback=progress_callback,
        min_counts=min_counts,
        min_genes=min_genes,
    )
    return AnalysisResult.from_pipeline(pr)


def run_dge(adata, group1: str, group2: str) -> DGEResult:
    """
    Run pairwise differential gene expression between two predicted cell types.

    Calls dge.run_pairwise_dge() and returns results + a volcano plot as bytes.
    Requires 'predicted_cell_type' to be present in adata.obs (set by the pipeline).

    Parameters
    ----------
    adata : anndata.AnnData
        Preprocessed AnnData with 'predicted_cell_type' in .obs.
    group1, group2 : str
        Cell type labels to compare (must be present in predictions).

    Returns
    -------
    DGEResult
        .success      True if analysis completed without error
        .table        pd.DataFrame with gene, logfoldchange, pval_adj, score
        .volcano_png  PNG bytes of volcano plot
        .error        Human-readable error message on failure
    """
    result = DGEResult(group1=group1, group2=group2)

    if adata is None:
        result.error = "No AnnData available. Run analysis first."
        return result

    if "predicted_cell_type" not in adata.obs.columns:
        result.error = (
            "Column 'predicted_cell_type' not found in adata.obs. "
            "Run the prediction pipeline before DGE."
        )
        return result

    available = adata.obs["predicted_cell_type"].unique().tolist()
    for label in (group1, group2):
        if label not in available:
            result.error = f"'{label}' not found in predicted cell types: {available[:8]}"
            return result

    try:
        from dge import run_pairwise_dge
    except ImportError:
        result.error = "dge.py not found in project directory."
        return result

    try:
        df = run_pairwise_dge(adata, "predicted_cell_type", group1, group2)
        result.table = df
        result.volcano_png = _volcano_to_bytes(df, group1, group2)
    except Exception as exc:
        result.error = str(exc)

    return result


def get_pca_plot_bytes(
    pca_coords: np.ndarray,
    labels: np.ndarray,
    tissue: str,
    dpi: int = 150,
) -> bytes:
    """
    Render a PCA scatter plot (PC1 vs PC2) and return PNG bytes.

    Intended for st.image() display and st.download_button() export in gui.py.

    Parameters
    ----------
    pca_coords : np.ndarray  shape (n_cells, n_pcs)
    labels : np.ndarray      predicted cell type label per cell
    tissue : str             used in plot title
    dpi : int                image resolution

    Returns
    -------
    bytes  PNG image
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique_types = list(pd.Series(labels).value_counts().index)
    color_map = {ct: _PALETTE[i % len(_PALETTE)] for i, ct in enumerate(unique_types)}

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    for ct in unique_types:
        mask = labels == ct
        ax.scatter(
            pca_coords[mask, 0],
            pca_coords[mask, 1],
            c=color_map[ct],
            s=8 if len(labels) > 2000 else 18,
            alpha=0.75,
            linewidths=0,
            label=ct,
            rasterized=True,
        )

    legend = ax.legend(fontsize=8, markerscale=2, frameon=True,
                       framealpha=0.15, edgecolor="#30363d", labelcolor="#e6edf3")
    legend.get_frame().set_facecolor("#161b22")
    ax.set_xlabel("PC 1", fontsize=10, color="#8b949e")
    ax.set_ylabel("PC 2", fontsize=10, color="#8b949e")
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.spines[:].set_color("#21262d")
    ax.set_title(f"PCA · {tissue.upper()}", fontsize=11, color="#e6edf3", pad=10)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def get_volcano_plot_bytes(dge_result: DGEResult, dpi: int = 150) -> Optional[bytes]:
    """Return the cached volcano plot bytes from a DGEResult, or None."""
    return dge_result.volcano_png if dge_result.success else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PALETTE = [
    "#58a6ff", "#3fb950", "#f78166", "#e3b341",
    "#bc8cff", "#ff7b72", "#79c0ff", "#ffa657",
    "#56d364", "#f0883e", "#a5d6ff", "#d2a8ff",
]


def _volcano_to_bytes(df: pd.DataFrame, group1: str, group2: str, dpi: int = 150) -> bytes:
    """Render a volcano plot to PNG bytes (matching the GUI dark theme)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = df.dropna()
    df = df[df["pval_adj"] > 0]

    x = df["logfoldchange"]
    y = -np.log10(df["pval_adj"])
    colors = ["#f78166" if v < 0 else "#58a6ff" for v in x]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    ax.scatter(x, y, c=colors, s=10, alpha=0.6)

    threshold = -np.log10(0.05)
    ax.axhline(threshold, linestyle="--", color="#8b949e", linewidth=0.8)
    ax.axvline(0, linestyle="--", color="#8b949e", linewidth=0.8)
    ax.text(x.max(), threshold, "  p = 0.05", fontsize=8, color="#8b949e", va="bottom")

    ax.set_xlabel("Log Fold Change", fontsize=10, color="#8b949e")
    ax.set_ylabel("-log10(p-value)", fontsize=10, color="#8b949e")
    ax.set_title(f"{group1}  vs  {group2}", fontsize=11, color="#e6edf3", pad=8)
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.spines[:].set_color("#21262d")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
