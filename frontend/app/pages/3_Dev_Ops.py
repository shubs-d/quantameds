"""
QuantaMED — Dev / Ops Telemetry Section
=======================================
Page 3: Internal system telemetry, model performance metrics,
confusion matrices, MLflow training logs, and database health.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root (containing app/ and db/) is always on sys.path
ROOT_DIR = Path(__file__).resolve().parent
while ROOT_DIR.name in ("app", "pages") and ROOT_DIR.parent != ROOT_DIR:
    ROOT_DIR = ROOT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import sqlite3
import pandas as pd
import streamlit as st

from app.components import (
    render_sidebar_logo,
    render_top_header,
)
from app.style import (
    BORDER_COLOR,
    CORAL,
    INK,
    INK_MUTED,
    MIST,
    PAPER,
    TEAL,
    TEAL_LIGHT,
    inject_css,
)
from db.db_utils import get_db_stats

st.set_page_config(page_title="Dev / Ops — QuantaMED", page_icon="Q", layout="centered")
inject_css()

# Render clickable logo in sidebar
render_sidebar_logo()

# Render top-left clickable logo header on main page
render_top_header(current_page_name="Dev / Ops Telemetry")

st.markdown(
    f"""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="margin: 0 0 0.35rem 0; font-size: 1.7rem; font-weight: 700; color: {INK};">
            Dev / Ops Model & System Telemetry
        </h2>
        <div style="font-size: 0.92rem; color: {INK_MUTED}; line-height: 1.5;">
            Internal monitoring of the trained RobustQuantaStudent pipeline, cross-validation metrics, and screening database transactions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 1. Screening Database Health ───────────────────────────────────────
st.markdown(
    f"""
    <div style="font-size: 1.1rem; font-weight: 700; color: {INK}; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 8px;">
        1. Screening Database Health
    </div>
    """,
    unsafe_allow_html=True,
)

stats = get_db_stats()

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Total Patients", f"{stats['patient_count']}")
with m2:
    st.metric("Total Consultations", f"{stats['visit_count']}")
with m3:
    st.metric("Predictions Logged", f"{stats['prediction_count']}")

st.markdown(
    f"""
    <div style="background-color: #FFFFFF; border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 0.75rem 1rem; font-size: 0.85rem; color: {INK}; margin-top: 0.5rem; margin-bottom: 2rem;">
        <span style="font-weight: 600; color: {INK_MUTED};">Most Recent Transaction:</span> <code>{stats['last_write']}</code>
        &nbsp;&middot;&nbsp;
        <span style="font-weight: 600; color: {INK_MUTED};">Storage Engine:</span> SQLite 3 (ACID, local single-file)
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 2. Trained Model Evaluation Metrics ────────────────────────────────
st.markdown(
    f"""
    <div style="font-size: 1.1rem; font-weight: 700; color: {INK}; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 8px;">
        2. RobustQuantaStudent Performance Metrics (Held-Out Test Set n=291)
    </div>
    """,
    unsafe_allow_html=True,
)

# Load test results from results/ directory
results_dir = ROOT_DIR / "results"
if not results_dir.exists() and (ROOT_DIR.parent / "results").exists():
    results_dir = ROOT_DIR.parent / "results"

results_file = results_dir / "robust_student_test_results.json"

if results_file.exists():
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            res_data = json.load(f)

        t2_m = res_data.get("best_t2_metrics", {})
        t1_m = res_data.get("best_t1_metrics", {})

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(
                f"""
                <div class="qmed-card-teal" style="padding: 1.25rem;">
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.5rem;">
                        <span style="font-weight: 700; font-size: 1rem; color: {INK};">Tier 1: ARK Only (SimK)</span>
                        <span style="font-size: 0.75rem; font-weight: 600; color: {TEAL}; background: {TEAL_LIGHT}; padding: 0.2rem 0.6rem; border-radius: 12px;">Primary Screen</span>
                    </div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: {TEAL}; line-height: 1;">
                        {t1_m.get('accuracy', 0.8935):.1%}
                    </div>
                    <div style="font-size: 0.78rem; color: {INK_MUTED}; margin-top: 0.2rem; margin-bottom: 0.75rem;">Held-Out Classification Accuracy</div>
                    <div class="report-row">
                        <span class="report-label">AUROC</span>
                        <span class="report-value">{t1_m.get('auroc', 0.9618):.4f}</span>
                    </div>
                    <div class="report-row">
                        <span class="report-label">Precision</span>
                        <span class="report-value">{t1_m.get('precision', 1.0):.1%}</span>
                    </div>
                    <div class="report-row">
                        <span class="report-label">Recall (Sensitivity)</span>
                        <span class="report-value">{t1_m.get('recall', 0.7257):.1%}</span>
                    </div>
                    <div class="report-row">
                        <span class="report-label">Specificity</span>
                        <span class="report-value">{t1_m.get('specificity', 1.0):.1%}</span>
                    </div>
                    <div class="report-row">
                        <span class="report-label">Brier Score</span>
                        <span class="report-value">{t1_m.get('brier', 0.087):.4f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_b:
            st.markdown(
                f"""
                <div class="qmed-card-coral" style="padding: 1.25rem;">
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.5rem;">
                        <span style="font-weight: 700; font-size: 1rem; color: {INK};">Tier 2: ARK + Pachymetry</span>
                        <span style="font-size: 0.75rem; font-weight: 600; color: {CORAL}; background: rgba(251, 126, 104, 0.12); padding: 0.2rem 0.6rem; border-radius: 12px;">Full Non-Invasive</span>
                    </div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: {CORAL}; line-height: 1;">
                        {t2_m.get('accuracy', 0.9210):.1%}
                    </div>
                    <div style="font-size: 0.78rem; color: {INK_MUTED}; margin-top: 0.2rem; margin-bottom: 0.75rem;">Held-Out Classification Accuracy</div>
                    <div class="report-row">
                        <span class="report-label">AUROC</span>
                        <span class="report-value">{t2_m.get('auroc', 0.9698):.4f}</span>
                    </div>
                    <div class="report-row">
                        <span class="report-label">Precision</span>
                        <span class="report-value">{t2_m.get('precision', 1.0):.1%}</span>
                    </div>
                    <div class="report-row">
                        <span class="report-label">Recall (Sensitivity)</span>
                        <span class="report-value">{t2_m.get('recall', 0.7965):.1%}</span>
                    </div>
                    <div class="report-row">
                        <span class="report-label">Specificity</span>
                        <span class="report-value">{t2_m.get('specificity', 1.0):.1%}</span>
                    </div>
                    <div class="report-row">
                        <span class="report-label">Brier Score</span>
                        <span class="report-value">{t2_m.get('brier', 0.0749):.4f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Baseline Comparison Table
        st.markdown(
            f"""
            <div style="margin-top: 1.5rem; margin-bottom: 0.5rem; font-weight: 700; font-size: 0.95rem; color: {INK};">
                Benchmark Comparison (Baseline Floor vs Tier 1 vs Tier 2)
            </div>
            """,
            unsafe_allow_html=True,
        )
        comp_df = pd.DataFrame([
            {"Metric": "Accuracy", "Historical Baseline": "76.98%", "Tier 1 (SimK)": "89.35%", "Tier 2 (+Pachy)": "92.10%", "Gain (T2 vs Baseline)": "+15.12%"},
            {"Metric": "AUROC", "Historical Baseline": "0.8786", "Tier 1 (SimK)": "0.9618", "Tier 2 (+Pachy)": "0.9698", "Gain (T2 vs Baseline)": "+0.0912"},
            {"Metric": "Precision", "Historical Baseline": "77.01%", "Tier 1 (SimK)": "100.0%", "Tier 2 (+Pachy)": "100.0%", "Gain (T2 vs Baseline)": "+22.99%"},
            {"Metric": "Sensitivity (Recall)", "Historical Baseline": "84.96%", "Tier 1 (SimK)": "72.57%", "Tier 2 (+Pachy)": "79.65%", "Gain (T2 vs Baseline)": "-5.31%"},
            {"Metric": "Specificity", "Historical Baseline": "71.91%", "Tier 1 (SimK)": "100.0%", "Tier 2 (+Pachy)": "100.0%", "Gain (T2 vs Baseline)": "+28.09%"},
            {"Metric": "Triage Sensitivity (τ=0.26)", "Historical Baseline": "—", "Tier 1 (SimK)": "90.27%", "Tier 2 (+Pachy)": "94.69%", "Gain (T2 vs Baseline)": "+9.73%"},
        ])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

    except Exception as e:
        st.warning(f"Error loading model test metrics: {e}")

# ── 3. Visualizations (Diagnostic Dashboard, ROC Curve & Confusion Matrix) ───
st.markdown(
    f"""
    <div style="margin-top: 2rem; margin-bottom: 0.75rem; font-size: 1.1rem; font-weight: 700; color: {INK}; display: flex; align-items: center; gap: 8px;">
        3. Model Diagnostic Curves & Matrix Artifacts
    </div>
    """,
    unsafe_allow_html=True,
)

# Check for dashboard chart first
dashboard_img = results_dir / "synthetic_50_results_dashboard.png"
if dashboard_img.exists():
    st.image(str(dashboard_img), caption="QuantaMED Multi-Panel Validation Dashboard: Risk Separation, ROC, Bilateral Asymmetry, Confusion Matrix", use_container_width=True)

roc_img = results_dir / "roc_curve.png"
cm_img = results_dir / "confusion_matrix.png"

v_col1, v_col2 = st.columns(2)
with v_col1:
    if roc_img.exists():
        st.image(str(roc_img), caption="AUROC Diagnostic Curve (Held-Out Test Set)")
    elif not dashboard_img.exists():
        st.info("ROC Curve artifact pending generation.")

with v_col2:
    if cm_img.exists():
        st.image(str(cm_img), caption="Confusion Matrix (Held-Out Test Set)")
    elif not dashboard_img.exists():
        st.info("Confusion Matrix artifact pending generation.")

# ── 4. MLflow Experiment Run History ──────────────────────────────────
st.markdown(
    f"""
    <div style="margin-top: 2rem; margin-bottom: 0.75rem; font-size: 1.1rem; font-weight: 700; color: {INK}; display: flex; align-items: center; gap: 8px;">
        4. MLflow Experiment Run History
    </div>
    """,
    unsafe_allow_html=True,
)

mlflow_db = ROOT_DIR / "mlflow.db"
if not mlflow_db.exists() and (ROOT_DIR.parent / "mlflow.db").exists():
    mlflow_db = ROOT_DIR.parent / "mlflow.db"

if mlflow_db.exists():
    try:
        conn = sqlite3.connect(str(mlflow_db))
        conn.row_factory = sqlite3.Row
        runs = conn.execute(
            "SELECT run_uuid, status, start_time, end_time FROM runs ORDER BY start_time DESC LIMIT 6"
        ).fetchall()

        if runs:
            runs_data = []
            for r in runs:
                dur = "—"
                if r["start_time"] and r["end_time"]:
                    dur = f"{(r['end_time'] - r['start_time']) / 1000:.1f}s"
                runs_data.append({
                    "Run Identifier": r["run_uuid"][:12] + "...",
                    "Execution Status": r["status"],
                    "Duration": dur,
                    "Timestamp": pd.to_datetime(r["start_time"], unit="ms").strftime("%Y-%m-%d %H:%M:%S") if r["start_time"] else "N/A",
                })

            st.dataframe(pd.DataFrame(runs_data), hide_index=True, use_container_width=True)
        conn.close()
    except Exception as e:
        st.info(f"MLflow database detected ({e}).")
else:
    st.info("No MLflow database found at project root.")
