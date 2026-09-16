"""
QuantaMED — Detection Section
============================
Page 1: Clinician entry form and instant risk assessment with calculation breakdown.
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

import streamlit as st

from app.components import (
    calculation_breakdown,
    graph_placeholder,
    heuristic_disclaimer,
    patient_badge,
    render_sidebar_logo,
    render_top_header,
    report_panel,
    risk_display,
)
from app.model_interface import predict_risk
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
from db.db_utils import (
    generate_patient_id,
    insert_prediction,
    insert_visit,
)

st.set_page_config(page_title="Detection — QuantaMED", page_icon="Q", layout="centered")
inject_css()

# Render clickable logo in sidebar
render_sidebar_logo()

# Render top-left clickable logo header on main page
render_top_header(current_page_name="Detection Entry")

st.markdown(
    f"""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="margin: 0 0 0.35rem 0; font-size: 1.7rem; font-weight: 700; color: {INK};">
            Keratoconus Detection Entry
        </h2>
        <div style="font-size: 0.92rem; color: {INK_MUTED}; line-height: 1.5;">
            Input five standard autorefractor-keratometer (ARK) measurements to compute instant corneal risk probability.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Mode Selection Segmented Control ───────────────────────────────────
st.markdown(
    f"""
    <div style="font-size: 0.82rem; font-weight: 600; color: {INK_MUTED}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem;">
        Clinical Acquisition Mode
    </div>
    """,
    unsafe_allow_html=True,
)
mode_label = st.radio(
    "Clinical Acquisition Mode",
    options=["ARK only (Tier 1)", "ARK + Pachymetry (Tier 2)"],
    horizontal=True,
    label_visibility="collapsed",
    help="Select whether optical or ultrasonic corneal pachymetry is available alongside standard autokeratometry.",
)
mode = "ark_pachy" if "Pachymetry" in mode_label else "ark_only"

st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

# ── Entry Form Card ────────────────────────────────────────────────────
with st.form(key="detection_form"):
    st.markdown(
        f"""
        <div style="font-size: 0.85rem; font-weight: 700; color: {INK}; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 6px;">
            Patient Information
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Optional Patient ID input
    patient_id_input = st.text_input(
        "Patient ID (Optional — leave blank to auto-generate)",
        value="",
        placeholder="e.g. QMED-000123 (Blank = New Patient ID auto-assigned)",
        help="If left blank, a sequential ID will be created and displayed on result generation.",
    )

    st.markdown(
        f"""
        <div style="margin-top: 1.25rem; margin-bottom: 0.6rem; font-size: 0.85rem; font-weight: 700; color: {INK}; display: flex; align-items: center; gap: 6px;">
            Autokeratometry & Astigmatism (ARK)
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5 standard ARK inputs in a clean 2-column layout
    col1, col2 = st.columns(2)

    with col1:
        flat_k = st.number_input(
            "K1 / Flat K (D)",
            min_value=30.0,
            max_value=70.0,
            value=43.50,
            step=0.25,
            format="%.2f",
            help="Flat keratometry meridian in Diopters.",
        )
        flat_axis = st.number_input(
            "Flat K Axis (°)",
            min_value=0.0,
            max_value=180.0,
            value=180.0,
            step=1.0,
            format="%.0f",
            help="Axis angle for K1 (0–180 degrees).",
        )
        cylinder = st.number_input(
            "Measured Cylinder (D)",
            min_value=-15.0,
            max_value=15.0,
            value=1.00,
            step=0.25,
            format="%.2f",
            help="Manifest / refractive cylinder magnitude from autorefractor.",
        )

    with col2:
        steep_k = st.number_input(
            "K2 / Steep K (D)",
            min_value=30.0,
            max_value=75.0,
            value=44.75,
            step=0.25,
            format="%.2f",
            help="Steep keratometry meridian in Diopters.",
        )
        steep_axis = st.number_input(
            "Steep K Axis (°)",
            min_value=0.0,
            max_value=180.0,
            value=90.0,
            step=1.0,
            format="%.0f",
            help="Axis angle for K2 (0–180 degrees).",
        )

    # Conditional Pachymetry Input (Tier 2)
    pachy_val = None
    if mode == "ark_pachy":
        st.markdown(
            f"""
            <div style="margin-top: 1.25rem; margin-bottom: 0.5rem; font-size: 0.85rem; font-weight: 700; color: {INK}; display: flex; align-items: center; gap: 6px;">
                Central Corneal Pachymetry (Tier 2)
            </div>
            """,
            unsafe_allow_html=True,
        )
        pachy_val = st.number_input(
            "Central Corneal Thickness (µm)",
            min_value=200.0,
            max_value=800.0,
            value=535.0,
            step=5.0,
            format="%.0f",
            help="Ultrasonic or optical central corneal thickness in microns.",
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    submit_btn = st.form_submit_button("Run Screening Detection", use_container_width=True)

# ── Submission & Processing ────────────────────────────────────────────
if submit_btn:
    # 1. Resolve Patient ID
    resolved_id = patient_id_input.strip() if patient_id_input.strip() else generate_patient_id()

    # 2. Prepare reading payload
    readings = {
        "flat_k_D": flat_k,
        "flat_k_axis_deg": flat_axis,
        "steep_k_D": steep_k,
        "steep_k_axis_deg": steep_axis,
        "cylinder_D": cylinder,
        "pachy_central_um": pachy_val if mode == "ark_pachy" else None,
    }

    # 3. Call Detection Contract
    result = predict_risk(readings, mode)
    risk_score = result["risk_score"]
    model_version = result["model_version"]
    calculation_trail = result["calculation_trail"]

    # 4. Persist Visit & Prediction to SQLite
    visit_id = insert_visit(resolved_id, mode, readings)
    insert_prediction(visit_id, model_version, risk_score, calculation_trail)

    # Save to session_state so view remains stable
    st.session_state["latest_detection"] = {
        "patient_id": resolved_id,
        "visit_id": visit_id,
        "mode": mode,
        "readings": readings,
        "risk_score": risk_score,
        "model_version": model_version,
        "calculation_trail": calculation_trail,
    }

# ── Render Result Section ──────────────────────────────────────────────
if "latest_detection" in st.session_state:
    det = st.session_state["latest_detection"]

    st.markdown("<hr style='margin: 2.25rem 0 1.5rem 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.5rem;">
            <div style="font-size: 1.25rem; font-weight: 700; color: {INK};">
                Screening Evaluation Result
            </div>
            <span style="font-size: 0.8rem; font-weight: 600; color: {INK_MUTED}; background: {MIST}; padding: 0.25rem 0.65rem; border-radius: 6px; border: 1px solid {BORDER_COLOR};">
                Visit Record #{det['visit_id']}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Patient ID Badge
    patient_badge(det["patient_id"])

    # 2. Risk Display (Large percentage) + Supporting Clinical Gauge
    r_col1, r_col2 = st.columns([1.15, 0.85])

    with r_col1:
        risk_display(det["risk_score"], model_version=det["model_version"])
        heuristic_disclaimer(model_version=det["model_version"])

    with r_col2:
        graph_placeholder(height_px=155, risk_score=det["risk_score"])

    # 3. Clinical Triage Recommendation Callout
    triage_text = det.get("triage_recommendation", "")
    pred_label = det.get("prediction_label", "")
    if triage_text:
        is_referral = det["risk_score"] >= 26.2
        border_col = CORAL if det["risk_score"] >= 50 else ("#E67E22" if is_referral else TEAL)
        bg_col = "rgba(251, 126, 104, 0.08)" if det["risk_score"] >= 50 else ("rgba(243, 156, 18, 0.08)" if is_referral else "rgba(12, 191, 222, 0.08)")
        st.markdown(
            f"""
            <div style="background: {bg_col}; border-left: 4px solid {border_col}; border-radius: 6px; padding: 0.85rem 1rem; margin-top: 1rem; margin-bottom: 0.5rem;">
                <div style="font-weight: 700; font-size: 0.92rem; color: {INK}; margin-bottom: 0.25rem;">
                    Clinical Triage Recommendation: {pred_label}
                </div>
                <div style="font-size: 0.84rem; color: {INK_MUTED}; line-height: 1.5;">
                    {triage_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Clinical Report Panel (ARK + Pachymetry mode only)
    if det["mode"] == "ark_pachy":
        report_panel(det["readings"], det["mode"])

    # 5. Calculation Breakdown Toggle View (§5b)
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    with st.expander("How was this calculated? (Factor Contribution Breakdown)", expanded=False):
        calculation_breakdown(det["calculation_trail"], det["risk_score"])

    # 6. Biomechanical Latent Vector (LUPI)
    if det.get("latent_vector"):
        from app.components import render_latent_representation
        with st.expander("Privileged Biomechanical Latents (z ∈ ℝ⁸)", expanded=False):
            render_latent_representation(det["latent_vector"])
