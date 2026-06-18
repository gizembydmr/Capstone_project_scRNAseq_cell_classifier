"""
gui.py
======
Streamlit graphical user interface for the
Tissue-Specific Cell Type Prediction Platform.

Run with:
    streamlit run gui.py

Requires:
    pip install streamlit anndata scanpy matplotlib seaborn umap-learn scikit-learn
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless backend – must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import streamlit as st

import tempfile
from data_validation import validate_file, ValidationResult

# ── Optional heavy imports (graceful fallback) ───────────────────────────────
try:
    import scanpy as sc
    SCANPY_OK = True
except ImportError:
    SCANPY_OK = False

try:
    import umap
    UMAP_OK = True
except ImportError:
    UMAP_OK = False

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CellPredict · Cell Type Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS  –  clean scientific look
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Typography ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,600;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Global background ──────────────────────────── */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* ── Header banner ──────────────────────────────── */
.header-banner {
    background: linear-gradient(135deg, #161b22 0%, #1a2332 100%);
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.header-title {
    font-size: 28px;
    font-weight: 600;
    letter-spacing: -0.5px;
    color: #e6edf3;
    margin: 0;
}
.header-sub {
    font-size: 13px;
    color: #8b949e;
    margin: 4px 0 0 0;
    font-family: 'IBM Plex Mono', monospace;
}
.accent { color: #58a6ff; }

/* ── Section cards ──────────────────────────────── */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 20px;
}
.card-title {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #58a6ff;
    margin: 0 0 16px 0;
}

/* ── File requirements list ─────────────────────── */
.req-list {
    font-size: 13px;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1.9;
    margin: 0;
    padding-left: 16px;
}

/* ── Stat boxes ─────────────────────────────────── */
.stat-grid {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 8px;
}
.stat-box {
    flex: 1;
    min-width: 140px;
    min-height: 100px;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.stat-value {
    font-size: 28px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    color: #58a6ff;
}
.stat-label {
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* ── Slider labels & values ─────────────────────── */
[data-testid="stSlider"] label,
[data-testid="stSlider"] > label,
[data-testid="stSlider"] p,
.stSlider label {
    color: #e6edf3 !important;
    font-size: 13px !important;
    opacity: 1 !important;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] output {
    color: #58a6ff !important;
    opacity: 1 !important;
}

/* ── Buttons ────────────────────────────────────── */
.stButton > button {
    background: #1f6feb;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
    padding: 10px 16px;
    width: 100%;
    min-width: 80px;
    white-space: nowrap;
    cursor: pointer;
    transition: background 0.2s;
}
.stButton > button:hover { background: #388bfd; }

/* ── File uploader ──────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0d1117;
    border: 1.5px dashed #30363d;
    border-radius: 10px;
    padding: 20px;
}
[data-testid="stFileUploader"]:hover {
    border-color: #58a6ff;
}

/* ── Progress bar ───────────────────────────────── */
.stProgress > div > div > div {
    background: #1f6feb;
    border-radius: 4px;
}

/* ── Alert overrides ────────────────────────────── */
.stAlert { border-radius: 8px; }

/* ── Download button ────────────────────────────── */
.stDownloadButton > button {
    background: #238636;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
    padding: 10px 24px;
    width: 100%;
}
.stDownloadButton > button:hover { background: #2ea043; }

/* ── Divider ────────────────────────────────────── */
.section-divider {
    height: 1px;
    background: #21262d;
    margin: 28px 0;
}

/* ── Validation messages ────────────────────────── */
.val-error {
    background: #1a0a0a;
    border-left: 3px solid #f85149;
    border-radius: 0 6px 6px 0;
    padding: 10px 16px;
    margin: 6px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #f85149;
}
.val-warning {
    background: #1a140a;
    border-left: 3px solid #e3b341;
    border-radius: 0 6px 6px 0;
    padding: 10px 16px;
    margin: 6px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #e3b341;
}
.val-ok {
    background: #0a1a0d;
    border-left: 3px solid #3fb950;
    border-radius: 0 6px 6px 0;
    padding: 10px 16px;
    margin: 6px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #3fb950;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: tissue model registry
# ─────────────────────────────────────────────────────────────────────────────

TISSUE_MODELS: dict[str, dict] = {
    "PBMC (Peripheral Blood Mononuclear Cells)": {
        "key": "pbmc",
        "cell_types": [
        "B cell",
        "CD34+ cell",
        "Dendritic cell",
        "Monocyte",
        "NK cell",
        "Regulatory T cell",
        "CD4 memory T cell",
        "CD4 naive T cell",
        "CD4 helper T cell",
        "CD8 naive T cell",
        "CD8 cytotoxic T cell",
    ],
        "model_file": "models/pbmc/LR_level3_no_weight_final_model_bundle_with_unassigned_threshold_050.joblib",
        "description": "Pre-trained on 10x Genomics PBMC 3k & 68k datasets.",
        "available": True,
    },
    "Pancreas": {
        "key": "pancreas",
        "cell_types": ["B cell", "CD4+ T cell", "CD8+ T cell",
                        "Classical Monocytes", "Endothelial cells", "Fibroblasts",
                        "Intermediate Monocytes", "Macrophages", "NK cells",
                        "Pancreatic A cells (Alpha)", "Pancreatic Acinar cells",
                        "Pancreatic Ductal cells", "Pancreatic Stellate cells",
                        "Type B Pancreatic cells (Beta)"],
        "model_file": "models/pancreas/pancreas_LR_balanced_level3_final_model_bundle_with_unassigned_threshold_085.joblib",
        "description": "Pre-trained on human pancreas scRNA-seq reference data.",
        "available": True,
    },
    "Lung (Coming Soon)": {
        "key": "lung",
        "cell_types": ["AT1 cells", "AT2 cells", "Club cells", "Ciliated cells",
                        "Endothelial cells", "Fibroblasts", "Macrophages", "T cells"],
        "model_file": "models/lung_model.pkl",
        "description": "Coming soon — model not yet available.",
        "available": False,
    },
    "Liver (Coming Soon)": {
        "key": "liver",
        "cell_types": ["Hepatocytes", "Cholangiocytes", "Kupffer cells",
                        "Hepatic Stellate cells", "Endothelial cells",
                        "NK cells", "B cells", "T cells"],
        "model_file": "models/liver_model.pkl",
        "description": "Coming soon — model not yet available.",
        "available": False,
    },
    "Brain (Coming Soon)": {
        "key": "brain",
        "cell_types": ["Neurons (Excitatory)", "Neurons (Inhibitory)",
                        "Astrocytes", "Oligodendrocytes", "OPC",
                        "Microglia", "Endothelial cells", "Pericytes"],
        "model_file": "models/brain_model.pkl",
        "description": "Coming soon — model not yet available.",
        "available": False,
    },
}

CELL_PALETTE = [
    "#58a6ff", "#3fb950", "#f78166", "#e3b341",
    "#bc8cff", "#ff7b72", "#79c0ff", "#ffa657",
    "#56d364", "#f0883e", "#a5d6ff", "#d2a8ff",
]

def get_unassigned_threshold(model_info: dict) -> float | None:
    model_path = Path(__file__).parent / model_info["model_file"]

    if not JOBLIB_OK or not model_path.exists():
        return None

    try:
        model_bundle = joblib.load(model_path)
    except Exception:
        return None

    possible_keys = [
        "unassigned_threshold",
        "confidence_threshold",
        "threshold",
    ]

    for key in possible_keys:
        if key in model_bundle:
            try:
                return float(model_bundle[key])
            except (TypeError, ValueError):
                pass

    preprocessing = model_bundle.get("preprocessing", {})
    if isinstance(preprocessing, dict):
        for key in possible_keys:
            if key in preprocessing:
                try:
                    return float(preprocessing[key])
                except (TypeError, ValueError):
                    pass

    metadata = model_bundle.get("metadata", {})
    if isinstance(metadata, dict):
        for key in possible_keys:
            if key in metadata:
                try:
                    return float(metadata[key])
                except (TypeError, ValueError):
                    pass

    return None

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "validated": False,
        "adata": None,
        "validation_result": None,
        "predictions": None,
        "probabilities": None,
        "umap_coords": None,
        "run_complete": False,
        "n_cells": 0,
        "n_genes": 0,
        "selected_tissue": list(TISSUE_MODELS.keys())[0],
        "predicted_tissue": None,
        "pca_coords": None,
        "pca_variance_ratio": None,
        "dge_result": None,
        "dge_groups_done": ("", ""),
        "shap_global_df": None,
        "shap_group_df": None,
        "shap_class_df": None,
        "min_counts_val": 500,
        "min_genes_val": 200,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-banner">
  <div style="font-size:42px; line-height:1">🧬</div>
  <div>
    <p class="header-title">Cell<span class="accent">Predict</span></p>
    <p class="header-sub">Tissue-Specific Cell Type Prediction · scRNA-seq Analysis Platform</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_predict, tab_model_info = st.tabs(["🔬  Prediction", "📊  Model Info"])


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL INFO TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_model_info:
    import json

    # Each tissue's model bundle has a matching metadata JSON sitting next to
    # it. Add new tissues here as their metadata files become available.
    TISSUE_METADATA_PATHS: dict[str, str] = {
    "pbmc": "models/pbmc/LR_level3_no_weight_final_model_metadata_with_unassigned_threshold_050.json",
    "pancreas": "models/pancreas/pancreas_LR_balanced_level3_final_model_metadata_with_unassigned_threshold_085.json",
}

    # Only offer tissues that actually have a trained model — "Coming Soon"
    # tissues have no metadata to show.
    _info_tissue_options = [
        name for name, info in TISSUE_MODELS.items() if info.get("available", False)
    ]

    st.markdown('<div class="card-title">Select Model to Inspect</div>', unsafe_allow_html=True)

    _info_tissue_choice = st.selectbox(
        "Model",
        options=_info_tissue_options,
        index=0,
        label_visibility="collapsed",
        key="model_info_tissue_choice",
        help="Independent of the tissue selected on the Prediction tab — "
             "browsing here won't change what model is used to run predictions.",
    )

    _info_tissue_key = TISSUE_MODELS[_info_tissue_choice]["key"]
    _meta_rel_path = TISSUE_METADATA_PATHS.get(_info_tissue_key)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    meta_path = (
        Path(__file__).parent / _meta_rel_path
        if _meta_rel_path is not None
        else None
    )

    if meta_path is not None and meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)

        metrics = (
            meta.get("baseline_test_metrics_without_unassigned")
            or meta.get("test_metrics")
            or {}
        )

        unassigned_metrics = meta.get("test_metrics_with_unassigned", {})

        preprocessing = meta.get("preprocessing", {})

        st.markdown('<div class="card-title">Model Overview</div>', unsafe_allow_html=True)

        m_cols = st.columns(4)
        overview = [
            ("Logistic Regression", "Algorithm"),
            (str(meta.get("n_features", "—")), "HVG Features"),
            (str(meta.get("n_classes", "—")), "Cell Type Classes"),
            (f"{meta.get('n_cells_total', 0):,}", "Training Cells"),
        ]
        for col, (val, label) in zip(m_cols, overview):
            with col:
                st.markdown(f"""
<div class="stat-box">
  <div class="stat-value" style="font-size:20px">{val}</div>
  <div class="stat-label">{label}</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="card-title">Test Set Performance</div>', unsafe_allow_html=True)

        perf_cols = st.columns(4)
        perf_stats = [
            (f"{metrics.get('accuracy', 0) * 100:.1f}%", "Accuracy"),
            (f"{metrics.get('f1_macro', 0) * 100:.1f}%", "F1 Macro"),
            (f"{metrics.get('balanced_accuracy', 0) * 100:.1f}%", "Balanced Accuracy"),
            (f"{metrics.get('roc_auc_macro_ovr', 0) * 100:.1f}%", "ROC-AUC"),
        ]
        for col, (val, label) in zip(perf_cols, perf_stats):
            with col:
                st.markdown(f"""
<div class="stat-box">
  <div class="stat-value" style="font-size:22px; color:#3fb950">{val}</div>
  <div class="stat-label">{label}</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.markdown('<div class="card-title">Predicted Cell Types</div>', unsafe_allow_html=True)
            st.markdown("""
            <style>
            .cell-type-list li {
                white-space: nowrap;
                font-family: 'IBM Plex Mono', monospace;
                color: #58f29b;
                margin-bottom: 10px;
            }
            </style>
            """, unsafe_allow_html=True)

            cell_types_html = "<ul class='cell-type-list'>"
            for ct in meta.get("class_names", []):
                cell_types_html += f"<li>{ct}</li>"
            cell_types_html += "</ul>"

            st.markdown(cell_types_html, unsafe_allow_html=True)

        with detail_col2:
            st.markdown('<div class="card-title">Preprocessing Parameters</div>', unsafe_allow_html=True)
            st.markdown(f"- **Target sum:** `{preprocessing.get('target_sum', '—')}`")
            st.markdown(f"- **Min counts:** `{preprocessing.get('min_counts', '—')}`")
            st.markdown(f"- **Min genes:** `{preprocessing.get('min_genes', '—')}`")
            st.markdown(f"- **Normalization:** `{preprocessing.get('normalization', '—')}`")
            st.markdown(f"- **Log transform:** `{preprocessing.get('log_transform', '—')}`")
            st.markdown(f"- **Feature selection:** `{preprocessing.get('feature_selection', '—')}`")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="card-title">All Metrics</div>', unsafe_allow_html=True)
        metrics_df = pd.DataFrame([
            {"Metric": k.replace("_", " ").title(), "Score": f"{v:.4f}"}
            for k, v in metrics.items()
        ])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    else:
        if _meta_rel_path is None:
            st.warning(
                f"No metadata file is configured for '{_info_tissue_choice}' yet."
            )
        else:
            st.warning(
                f"Model metadata file not found for '{_info_tissue_choice}' "
                f"at '{meta_path}'."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Prediction Tab — full layout
# ─────────────────────────────────────────────────────────────────────────────

with tab_predict:

    col_left, col_right = st.columns([1, 1.6], gap="large")

    # ═══════════════════════════════════════════════════════════════════════════
    # LEFT COLUMN  –  Upload & Configuration
    # ═══════════════════════════════════════════════════════════════════════════

    with col_left:

        # ── Upload ────────────────────────────────────────────────────────────
        st.markdown('<div class="card-title">01 · Upload Dataset</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            label="Drop your scRNA-seq file here",
            type=["h5ad", "csv", "mtx"],
            help="Accepted formats: .h5ad (AnnData), .csv (cell × gene matrix), .mtx (Matrix Market)",
            label_visibility="collapsed",
        )

        # ── File Requirements ─────────────────────────────────────────────────
        with st.expander("📋  File Requirements", expanded=False):
            st.markdown("""
<ul class="req-list">
  <li><b>.h5ad</b> – AnnData HDF5 file (Scanpy output).<br>
      adata.X must contain raw UMI counts (cells × genes).</li>
  <li><b>.csv</b> – Plain comma-separated matrix.<br>
      First column = cell barcodes, header row = gene IDs.</li>
  <li><b>.mtx</b> – Matrix Market sparse format (10x Genomics).<br>
      Place <code>barcodes.tsv</code> and <code>features.tsv</code> in the same folder.</li>
  <li>Minimum: <b>10 cells</b> · <b>50 genes</b>.</li>
  <li>Values must be <b>non-negative integers</b> (raw counts, not normalised).</li>
  <li>No duplicate cell barcodes or gene IDs.</li>
  <li>No NaN / infinite values.</li>
</ul>
""", unsafe_allow_html=True)

        # ── Validation feedback ───────────────────────────────────────────────
        if uploaded_file is not None:
            new_file = uploaded_file.name != st.session_state.get("last_uploaded_filename", "")

            if new_file:
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as _f:
                    _f.write(uploaded_file.getvalue())
                    tmp_path = Path(_f.name)
                st.session_state["tmp_path"] = tmp_path

                with st.spinner("Validating file…"):
                    vr: ValidationResult = validate_file(tmp_path)

                st.session_state.validation_result = vr
                st.session_state.validated = vr.is_valid
                st.session_state.last_uploaded_filename = uploaded_file.name

                if vr.is_valid:
                    st.session_state.adata = vr.data
                    st.session_state.n_cells = vr.n_cells
                    st.session_state.n_genes = vr.n_genes
                    st.session_state.run_complete = False
                    st.session_state.predictions = None
                    st.session_state.umap_coords = None
                    st.session_state.predicted_tissue = None

            vr = st.session_state.get("validation_result")
            if vr and vr.is_valid:
                st.markdown(
                    f'<div class="val-ok">✅ &nbsp;File accepted &nbsp;|&nbsp; '
                    f'{vr.n_cells:,} cells &nbsp;×&nbsp; {vr.n_genes:,} genes</div>',
                    unsafe_allow_html=True,
                )
                for w in vr.warnings:
                    st.markdown(f'<div class="val-warning">⚠ &nbsp;{w}</div>', unsafe_allow_html=True)
            elif vr:
                for e in vr.errors:
                    st.markdown(f'<div class="val-error">✖ &nbsp;{e}</div>', unsafe_allow_html=True)
                for w in vr.warnings:
                    st.markdown(f'<div class="val-warning">⚠ &nbsp;{w}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Model / Tissue Selection ──────────────────────────────────────────
        st.markdown('<div class="card-title">02 · Select Tissue Model</div>', unsafe_allow_html=True)

        tissue_choice = st.selectbox(
            "Tissue",
            options=list(TISSUE_MODELS.keys()),
            index=list(TISSUE_MODELS.keys()).index(
                st.session_state.get("selected_tissue", list(TISSUE_MODELS.keys())[0])
            ),
            label_visibility="collapsed",
            key="selected_tissue",
        )
        model_info = TISSUE_MODELS[tissue_choice]

        st.caption(f"ℹ️  {model_info['description']}")
        st.caption(
            "Predicted cell types: " + " · ".join(
                f"`{ct}`" for ct in model_info["cell_types"]
            )
        )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Quality Filter Sliders ────────────────────────────────────────────
        st.markdown('<div class="card-title">03 · Quality Filters</div>', unsafe_allow_html=True)
        st.caption("Cells below these thresholds will be removed before prediction.")

        filter_col, reset_col = st.columns([3, 1])

        with reset_col:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("↺  Default", use_container_width=True):
                st.session_state["min_counts_val"] = 500
                st.session_state["min_genes_val"] = 200

        with filter_col:
            min_counts_val = st.slider(
                "Min. UMI counts per cell",
                min_value=0,
                max_value=2000,
                value=st.session_state.get("min_counts_val", 500),
                step=50,
                key="min_counts_val",
                help="Cells with fewer total UMI counts than this will be filtered out.",
            )

            min_genes_val = st.slider(
                "Min. genes per cell",
                min_value=0,
                max_value=1000,
                value=st.session_state.get("min_genes_val", 200),
                step=25,
                key="min_genes_val",
                help="Cells expressing fewer genes than this will be filtered out.",
            )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Run button ────────────────────────────────────────────────────────
        st.markdown('<div class="card-title">04 · Run Prediction</div>', unsafe_allow_html=True)

        model_available = model_info.get("available", False)
        run_disabled = not st.session_state.validated or not model_available
        run_clicked = st.button(
            "▶  RUN PREDICTION",
            disabled=run_disabled,
            help="Upload and validate a dataset first." if not st.session_state.validated
            else "Model not available for this tissue yet." if not model_available
            else "Start cell-type prediction.",
        )

        if not model_available:
            st.warning("This tissue model is not yet available. Please select PBMC or Pancreas.")
        elif run_disabled and uploaded_file is None:
            st.caption("Upload a dataset to enable prediction.")
        elif run_disabled:
            st.caption("Fix validation errors before running.")

        # ── Progress & Pipeline ───────────────────────────────────────────────
        if run_clicked and st.session_state.validated:
            st.session_state.dge_result = None
            st.session_state.dge_groups_done = ("", "")

            progress_bar = st.progress(0)
            status_text = st.empty()

            tmp_path = st.session_state.get("tmp_path") or Path(f"/tmp/{uploaded_file.name}")
            tissue_key = model_info["key"]

            _step_counter = {"n": 0}
            _TOTAL_STEPS = 6

            def _progress_cb(msg: str) -> None:
                _step_counter["n"] = min(_step_counter["n"] + 1, _TOTAL_STEPS)
                progress_bar.progress(_step_counter["n"] / _TOTAL_STEPS)
                status_text.markdown(f"⏳ &nbsp;`{msg}`")

            from backend import run_analysis
            result = run_analysis(
                tmp_path,
                tissue_key,
                progress_callback=_progress_cb,
                min_counts=min_counts_val,
                min_genes=min_genes_val,
            )

            progress_bar.progress(1.0)
            progress_bar.empty()
            status_text.empty()

            if result.success:
                st.session_state.predictions        = result.predictions
                st.session_state.probabilities      = result.probabilities
                st.session_state.umap_coords        = result.umap_coords
                st.session_state.pca_coords         = result.pca_coords
                st.session_state.pca_variance_ratio = result.pca_variance_ratio
                st.session_state.shap_global_df     = result.shap_global_df
                st.session_state.shap_group_df      = result.shap_group_df
                st.session_state.shap_class_df      = result.shap_class_df
                st.session_state.adata              = result.adata
                st.session_state.predicted_tissue   = tissue_choice
                st.session_state.run_complete       = True
                st.session_state.n_cells            = result.n_cells
                st.session_state.n_genes            = result.n_genes
                st.rerun()
            else:
                st.error(f"**Pipeline failed:** {result.error}")
                for step in result.steps:
                    icon = "✅" if step.success else "❌"
                    st.caption(f"{icon} {step.name}: {step.message}")

    # ═══════════════════════════════════════════════════════════════════════════
    # RIGHT COLUMN  –  Results
    # ═══════════════════════════════════════════════════════════════════════════

    with col_right:

        if not st.session_state.run_complete:
            st.markdown("""
<div style="height:340px; display:flex; flex-direction:column;
            align-items:center; justify-content:center;
            background:#161b22; border:1px dashed #21262d;
            border-radius:12px; color:#30363d; text-align:center; padding:32px;">
  <div style="font-size:56px; margin-bottom:16px;">🔬</div>
  <div style="font-size:15px; font-weight:600; color:#484f58;">
      Results will appear here
  </div>
  <div style="font-size:13px; margin-top:8px; color:#30363d; font-family:'IBM Plex Mono',monospace;">
      Upload a dataset → select tissue → run prediction
  </div>
</div>
""", unsafe_allow_html=True)

        else:
            predictions: pd.DataFrame = st.session_state.predictions
            umap_coords: np.ndarray = st.session_state.umap_coords
            n_cells = st.session_state.n_cells
            n_genes = st.session_state.n_genes

            cell_type_counts = predictions["predicted_cell_type"].value_counts()
            n_cell_types = len(cell_type_counts)
            result_tissue = st.session_state.get("predicted_tissue") or st.session_state.selected_tissue
            result_tissue_short = result_tissue.split("(")[0].strip()

            # ── Quick Stats ───────────────────────────────────────────────────
            st.markdown('<div class="card-title">Results · Quick Stats</div>', unsafe_allow_html=True)

            stat_cols = st.columns(4)
            stats = [
                (f"{n_cells:,}", "Cells"),
                (f"{n_genes:,}", "Genes"),
                (f"{n_cell_types}", "Cell Types"),
                (result_tissue_short, "Tissue"),
            ]
            for col, (val, label) in zip(stat_cols, stats):
                with col:
                    # Tissue label can be long (e.g. "Pancreas") — slightly smaller font so it fits on one line
                    val_style = 'font-size:22px;' if label == "Tissue" else ''
                    st.markdown(f"""
<div class="stat-box">
  <div class="stat-value" style="{val_style}">{val}</div>
  <div class="stat-label">{label}</div>
</div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ── Cell Type Distribution ─────────────────────────────────────────
            st.markdown('<div class="card-title">Cell Type Distribution</div>', unsafe_allow_html=True)

            fig_hist, ax = plt.subplots(figsize=(7, 3.2))
            fig_hist.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#161b22")

            labels = cell_type_counts.index.tolist()
            counts = cell_type_counts.values.tolist()
            colors = [CELL_PALETTE[i % len(CELL_PALETTE)] for i in range(len(labels))]

            bars = ax.barh(labels[::-1], counts[::-1], color=colors[::-1],
                           height=0.6, edgecolor="none")

            for bar, cnt in zip(bars, counts[::-1]):
                ax.text(
                    bar.get_width() + max(counts) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    str(cnt),
                    va="center", ha="left",
                    fontsize=9, color="#8b949e",
                    fontfamily="monospace",
                )

            ax.set_xlabel("Number of Cells", fontsize=10, color="#8b949e", labelpad=8)
            ax.tick_params(colors="#8b949e", labelsize=9)
            ax.spines[:].set_visible(False)
            ax.xaxis.set_tick_params(length=0)
            ax.yaxis.set_tick_params(length=0)
            ax.set_xlim(0, max(counts) * 1.15)
            plt.tight_layout(pad=0.5)

            st.pyplot(fig_hist, use_container_width=True)
            plt.close(fig_hist)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ── Predictions Table ─────────────────────────────────────────────
            st.markdown('<div class="card-title">Predictions Table</div>', unsafe_allow_html=True)

            probabilities = st.session_state.get("probabilities")

            result_model_info = TISSUE_MODELS.get(result_tissue, {})
            unassigned_threshold = get_unassigned_threshold(result_model_info)

            if unassigned_threshold is not None:
                threshold_text = f"{unassigned_threshold:.2f}"
            else:
                threshold_text = "the selected model threshold"

            display_predictions = predictions.copy().reset_index(drop=True)

            # Keep one scientific confidence column only.
            # If confidence_score is missing, derive it from the probability table.
            if "confidence_score" not in display_predictions.columns and probabilities is not None:
                prob_reset = probabilities.reset_index(drop=True)
                display_predictions["confidence_score"] = prob_reset.max(axis=1)

            # Remove backend/helper columns that may confuse users
            display_predictions = display_predictions.drop(
                columns=[
                    col for col in ["is_unassigned", "confidence"]
                    if col in display_predictions.columns
                ],
                errors="ignore",
            )

            # Rename columns for clearer GUI/download output
            display_predictions = display_predictions.rename(
                columns={
                    "cell_barcode": "cell_barcode",
                    "predicted_cell_type": "predicted_cell_type",
                    "predicted_label_before_threshold": "prediction_before_unassigned_threshold",
                    "confidence_score": "confidence_score",
                }
            )

            # Keep only the user-facing columns in a consistent order
            prediction_table_columns = [
                "cell_barcode",
                "predicted_cell_type",
                "prediction_before_unassigned_threshold",
                "confidence_score",
            ]

            display_predictions = display_predictions[
                [col for col in prediction_table_columns if col in display_predictions.columns]
            ]

            st.dataframe(
                display_predictions,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "cell_barcode": st.column_config.TextColumn(
                        "cell_barcode",
                        help="Unique barcode or identifier of each cell in the uploaded dataset.",
                    ),
                    "predicted_cell_type": st.column_config.TextColumn(
                       "predicted_cell_type",
                        help=(
                              f"Final cell type shown after applying the unassigned threshold "
                              f"for this model: confidence score ≥ {threshold_text}. "
                              "Cells below this threshold are shown as Unassigned."
                        ),
                    ),
                    "prediction_before_unassigned_threshold": st.column_config.TextColumn(
                        "prediction_before_unassigned_threshold",
                        help=(
                            f"The model's best predicted cell type before applying the unassigned threshold "
                            f"for this model: confidence score ≥ {threshold_text}. "
                            "This helps users inspect what the model would have predicted for low-confidence cells."
                        ),
                    ),
                    "confidence_score": st.column_config.NumberColumn(
                        "confidence_score",
                        help=(
                            "Model confidence score for the prediction, shown as a value between 0 and 1. "
                            "Higher values indicate stronger model confidence."
                        ),
                        format="%.4f",
                    ),
                },
            )

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ── Download ──────────────────────────────────────────────────────
            st.markdown('<div class="card-title">Download Predictions</div>', unsafe_allow_html=True)

            csv_bytes = display_predictions.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇  Download Predictions (CSV)",
                data=csv_bytes,
                file_name="cell_type_predictions.csv",
                mime="text/csv",
            )

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ── Embeddings: UMAP + PCA tabs ───────────────────────────────────
            st.markdown('<div class="card-title">Embeddings</div>', unsafe_allow_html=True)

            tab_umap, tab_pca = st.tabs(["UMAP", "PCA"])

            pred_labels = predictions["predicted_cell_type"].values
            unique_types = list(cell_type_counts.index)
            color_map = {ct: CELL_PALETTE[i % len(CELL_PALETTE)] for i, ct in enumerate(unique_types)}
            tissue_short = result_tissue_short

            def _scatter(ax, coords, title, xlabel, ylabel):
                for ct in unique_types:
                    mask = pred_labels == ct
                    ax.scatter(
                        coords[mask, 0], coords[mask, 1],
                        c=color_map[ct],
                        s=8 if n_cells > 2000 else 18,
                        alpha=0.75, linewidths=0, label=ct, rasterized=True,
                    )
                legend = ax.legend(fontsize=8, markerscale=2, frameon=True,
                                   framealpha=0.15, edgecolor="#30363d", labelcolor="#e6edf3")
                legend.get_frame().set_facecolor("#161b22")
                ax.set_xlabel(xlabel, fontsize=10, color="#8b949e")
                ax.set_ylabel(ylabel, fontsize=10, color="#8b949e")
                ax.tick_params(colors="#8b949e", labelsize=8)
                ax.spines[:].set_color("#21262d")
                ax.set_title(title, fontsize=11, color="#e6edf3", pad=10)

            with tab_umap:
                if umap_coords is not None:
                    fig_umap, ax_u = plt.subplots(figsize=(7, 5.5))
                    fig_umap.patch.set_facecolor("#0d1117")
                    ax_u.set_facecolor("#0d1117")
                    _scatter(ax_u, umap_coords, f"UMAP · {tissue_short}", "UMAP 1", "UMAP 2")
                    plt.tight_layout(pad=0.5)

                    buf = io.BytesIO()
                    fig_umap.savefig(buf, format="png", dpi=150, facecolor="#0d1117")
                    buf.seek(0)

                    st.pyplot(fig_umap, use_container_width=True)
                    plt.close(fig_umap)

                    st.download_button(
                        label="⬇  Export UMAP Plot (PNG)",
                        data=buf, file_name="umap_plot.png", mime="image/png",
                    )
                else:
                    st.info("UMAP could not be computed. Install `umap-learn` and `scanpy`.")

            with tab_pca:
                pca_coords = st.session_state.get("pca_coords")
                if pca_coords is not None:
                    fig_pca, ax_p = plt.subplots(figsize=(7, 5.5))
                    fig_pca.patch.set_facecolor("#0d1117")
                    ax_p.set_facecolor("#0d1117")
                    _scatter(ax_p, pca_coords, f"PCA · {tissue_short}", "PC 1", "PC 2")
                    plt.tight_layout(pad=0.5)

                    buf_pca = io.BytesIO()
                    fig_pca.savefig(buf_pca, format="png", dpi=150, facecolor="#0d1117")
                    buf_pca.seek(0)

                    st.pyplot(fig_pca, use_container_width=True)
                    plt.close(fig_pca)

                    st.download_button(
                        label="⬇  Export PCA Plot (PNG)",
                        data=buf_pca, file_name="pca_plot.png", mime="image/png",
                    )
                else:
                    st.info("PCA coordinates not available.")

    # ── 05 · SHAP Gene Importance ─────────────────────────────────────────────
    if st.session_state.run_complete:
        shap_global_df = st.session_state.get("shap_global_df")
        shap_group_df  = st.session_state.get("shap_group_df")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">05 · SHAP Gene Importance</div>', unsafe_allow_html=True)
        st.caption("Which genes drove each cell type prediction? (SHAP LinearExplainer)")

        if shap_global_df is not None:
            from backend import get_shap_bar_plot_bytes

            shap_tab_global, shap_tab_group = st.tabs(["Global Top Genes", "Per Cell Type"])

            with shap_tab_global:
                shap_plot_col, shap_table_col = st.columns([2, 1], gap="large")

                with shap_plot_col:
                    st.markdown('<div class="card-title">Top 20 Genes — All Classes</div>', unsafe_allow_html=True)
                    img_global = get_shap_bar_plot_bytes(
                        shap_global_df, title="Global SHAP Gene Importance", top_n=20
                    )
                    st.image(img_global, use_container_width=True)
                    st.download_button(
                        label="⬇  Download Plot (PNG)",
                        data=img_global,
                        file_name="shap_global_top_genes.png",
                        mime="image/png",
                        key="dl_shap_global_png",
                    )

                with shap_table_col:
                    st.markdown('<div class="card-title">Top Genes</div>', unsafe_allow_html=True)
                    st.dataframe(
                        shap_global_df[["gene_label", "mean_abs_shap"]].head(20).reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        label="⬇  Download CSV",
                        data=shap_global_df.to_csv(index=False).encode("utf-8"),
                        file_name="shap_global_gene_importance.csv",
                        mime="text/csv",
                        key="dl_shap_global_csv",
                    )

            with shap_tab_group:
                if shap_group_df is not None:
                    shap_sel_col, shap_info_col = st.columns([2, 1], gap="large")

                    with shap_sel_col:
                        available_groups = sorted(shap_group_df["predicted_group"].unique().tolist())
                        selected_group = st.selectbox(
                            "Select cell type", options=available_groups, key="shap_group_sel"
                        )

                    group_df = shap_group_df[shap_group_df["predicted_group"] == selected_group]
                    n_cells_group = int(group_df["n_cells"].iloc[0]) if "n_cells" in group_df.columns else "?"

                    shap_g_plot, shap_g_table = st.columns([2, 1], gap="large")

                    with shap_g_plot:
                        st.markdown(f'<div class="card-title">Top Genes · {selected_group}</div>', unsafe_allow_html=True)
                        st.caption(f"{n_cells_group} cells predicted as {selected_group}")
                        img_group = get_shap_bar_plot_bytes(
                            group_df, title=f"Top Genes · {selected_group}", top_n=15,
                        )
                        st.image(img_group, use_container_width=True)
                        st.download_button(
                            label="⬇  Download Plot (PNG)",
                            data=img_group,
                            file_name=f"shap_{selected_group.replace(' ','_')}.png",
                            mime="image/png",
                            key="dl_shap_group_png",
                        )

                    with shap_g_table:
                        st.markdown('<div class="card-title">Top Genes</div>', unsafe_allow_html=True)
                        st.dataframe(
                            group_df[["gene_label", "mean_abs_shap"]].head(15).reset_index(drop=True),
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.download_button(
                            label="⬇  Download CSV",
                            data=shap_group_df.to_csv(index=False).encode("utf-8"),
                            file_name="shap_group_gene_importance.csv",
                            mime="text/csv",
                            key="dl_shap_group_csv",
                        )
        else:
            st.info("SHAP analysis not available. Install `shap` package: pip install shap")

    # ── 04 · Differential Gene Expression ────────────────────────────────────
    if st.session_state.run_complete and st.session_state.predictions is not None:

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">04 · Differential Gene Expression</div>', unsafe_allow_html=True)
        st.caption("Compare two predicted cell types to find differentially expressed genes.")

        available_types = sorted(
            [
                ct for ct in st.session_state.predictions["predicted_cell_type"].unique().tolist()
                if str(ct).lower() != "unassigned"
            ]
        )

        if len(available_types) < 2:
            st.info(
                "DGE requires at least two predicted biological cell types. "
                "Unassigned cells are excluded from DGE because they do not represent a biological cell type."
            )
            run_dge_clicked = False
            dge_group1 = None
            dge_group2 = None

        else:
            dge_sel1, dge_sel2, dge_btn_col = st.columns([2, 2, 1])

            with dge_sel1:
                dge_group1 = st.selectbox(
                    "Group 1",
                    options=available_types,
                    key="dge_sel_group1",
                )

            with dge_sel2:
                group2_opts = [ct for ct in available_types if ct != dge_group1]
                dge_group2 = st.selectbox(
                    "Group 2",
                    options=group2_opts,
                    key="dge_sel_group2",
                )

            with dge_btn_col:
                st.markdown("<br>", unsafe_allow_html=True)
                run_dge_clicked = st.button("▶  Run DGE", use_container_width=True)
        if run_dge_clicked:
            with st.spinner(f"Comparing {dge_group1} vs {dge_group2}…"):
                from backend import run_dge
                dge_res = run_dge(st.session_state.adata, dge_group1, dge_group2)
                st.session_state.dge_result = dge_res
                st.session_state.dge_groups_done = (dge_group1, dge_group2)

        dge_res = st.session_state.get("dge_result")
        if dge_res is not None:
            g1, g2 = st.session_state.get("dge_groups_done", ("", ""))

            if dge_res.success:
                dge_plot_col, dge_table_col = st.columns([2, 1], gap="large")

                with dge_plot_col:
                    st.markdown(
                        f'<div class="card-title">Volcano Plot · {g1} vs {g2}</div>',
                        unsafe_allow_html=True,
                    )
                    if dge_res.volcano_png:
                        st.image(dge_res.volcano_png, use_container_width=True)

                        if isinstance(dge_res.volcano_png, (bytes, bytearray)):
                            _volcano_bytes = bytes(dge_res.volcano_png)
                        else:
                            with open(dge_res.volcano_png, "rb") as _f:
                                _volcano_bytes = _f.read()
                        safe_name = f"{g1}_vs_{g2}".replace("/", "_").replace(" ", "_")
                        st.download_button(
                            label="⬇  Download Volcano Plot (PNG)",
                            data=_volcano_bytes,
                            file_name=f"volcano_{safe_name}.png",
                            mime="image/png",
                        )

                with dge_table_col:
                    st.markdown(
                        '<div class="card-title">Top Differentially Expressed Genes</div>',
                        unsafe_allow_html=True,
                    )
                    display_dge = dge_res.table.head(20).reset_index(drop=True).copy()

                    display_dge = display_dge[
                        ["gene", "gene_id", "logfoldchange", "pval_adj", "score"]
                    ]

                    display_dge["pval_adj"] = display_dge["pval_adj"].apply(
                        lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}"
                    )

                    st.dataframe(
                        display_dge,
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        label="⬇  Download DGE Results (CSV)",
                        data=dge_res.table.to_csv(index=False).encode("utf-8"),
                        file_name=f"dge_{g1}_vs_{g2}.csv".replace(" ", "_"),
                        mime="text/csv",
                    )
            else:
                st.error(f"DGE analysis failed: {dge_res.error}")

    # ── Reset ─────────────────────────────────────────────────────────────────
    if st.session_state.run_complete:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        if st.button("↺  Upload New Dataset"):
            for key in ["validated", "adata", "validation_result", "predictions",
                        "probabilities", "umap_coords", "pca_coords", "pca_variance_ratio",
                        "shap_global_df", "shap_group_df", "shap_class_df",
                        "run_complete", "n_cells", "n_genes", "dge_result", "dge_groups_done",
                        "predicted_tissue",
                        "last_uploaded_filename", "tmp_path"]:
                st.session_state.pop(key, None)
            st.rerun()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
<div style="margin-top:40px; text-align:center; font-size:11px;
            font-family:'IBM Plex Mono',monospace; color:#30363d;">
  CellTypePrediction · Capstone Project 2025–2026 ·
  Bahçeşehir University · Computer Engineering &amp; Software Engineering<br>
  ⚠ Predictions are computational estimates for research use only — not for clinical applications.
</div>
""", unsafe_allow_html=True)
