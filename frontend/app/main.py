"""
QuantaMED — Clinical Dashboard
==============================
Main entry point for the Streamlit multi-page application.
Initializes the database, configures page settings, and renders navigation overview.
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

from app.components import render_sidebar_logo, render_top_header
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
from db.db_utils import get_db_stats, init_db

# Configure page
st.set_page_config(
    page_title="QuantaMED Screening Suite",
    page_icon="Q",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Initialize database schema
init_db()

# Apply enhanced QuantaMED design system styles
inject_css()

# Render clickable logo in sidebar
render_sidebar_logo()

# Render top-left clickable logo header on main page
render_top_header(current_page_name="Home Suite")

# Hero Section
st.markdown(
    f"""
    <div style="margin-bottom: 2rem; padding: 1.5rem 0 0.5rem 0;">
        <div style="display: inline-flex; align-items: center; gap: 8px; font-size: 0.82rem; font-weight: 600; color: {TEAL}; letter-spacing: 0.08em; text-transform: uppercase; background: {TEAL_LIGHT}; padding: 0.35rem 0.85rem; border-radius: 20px; border: 1px solid rgba(12, 191, 222, 0.25); margin-bottom: 0.8rem;">
            QuantaMED Clinical Suite &middot; v2.0
        </div>
        <h1 style="margin: 0.2rem 0 0.6rem 0; font-size: 2.3rem; font-weight: 700; letter-spacing: -0.03em; color: {INK};">
            Super-Early Keratoconus Screening
        </h1>
        <p style="font-size: 1.05rem; color: {INK_MUTED}; margin: 0; line-height: 1.6; max-width: 820px;">
            Screen patients using standard Autorefractor-Keratometer (ARK) readings alone or with optional pachymetry.
            Backed by a trained two-tier quantum-classical student network achieving <strong>92.1% accuracy (0.970 AUROC)</strong>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Section Cards Overview
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div class="qmed-card-teal" style="height: 100%;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                <span style="font-size: 0.82rem; font-weight: 700; color: {TEAL}; text-transform: uppercase; letter-spacing: 0.05em;">Section 1</span>
                <span style="font-size: 0.75rem; font-weight: 600; color: {TEAL}; background: {TEAL_LIGHT}; padding: 0.2rem 0.6rem; border-radius: 12px;">
                    Clinician Workflow
                </span>
            </div>
            <div style="font-weight: 700; font-size: 1.15rem; color: {INK}; margin-bottom: 0.4rem;">
                1. Detection View
            </div>
            <div style="font-size: 0.88rem; color: {INK_MUTED}; line-height: 1.55; margin-bottom: 1.25rem;">
                Enter 5 standard ARK parameters (K1, K2, axes, measured cylinder) with optional central pachymetry. Instant risk assessment with factor-by-factor calculation breakdown.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open Detection Page →", key="goto_detect", use_container_width=True):
        st.switch_page("pages/1_Detection.py")

with col2:
    st.markdown(
        f"""
        <div class="qmed-card-coral" style="height: 100%;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                <span style="font-size: 0.82rem; font-weight: 700; color: {CORAL}; text-transform: uppercase; letter-spacing: 0.05em;">Section 2</span>
                <span style="font-size: 0.75rem; font-weight: 600; color: {CORAL}; background: rgba(251, 126, 104, 0.12); padding: 0.2rem 0.6rem; border-radius: 12px;">
                    History & Longitudinal
                </span>
            </div>
            <div style="font-weight: 700; font-size: 1.15rem; color: {INK}; margin-bottom: 0.4rem;">
                2. Patient Profile
            </div>
            <div style="font-size: 0.88rem; color: {INK_MUTED}; line-height: 1.55; margin-bottom: 1.25rem;">
                Search patient by ID (<code>QMED-XXXXXX</code>). Inspect chronological visit logs, review permanently stored calculation trails, track multi-visit risk trends, and save clinical comments.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open Patient Profile →", key="goto_profile", use_container_width=True):
        st.switch_page("pages/2_Patient_Profile.py")

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# Telemetry Overview Card
st.markdown(
    f"""
    <div class="qmed-card" style="margin-top: 0.5rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
            <div style="font-weight: 700; font-size: 1.05rem; color: {INK}; display: flex; align-items: center; gap: 8px;">
                3. Dev / Ops Telemetry & Performance
            </div>
            <span style="font-size: 0.75rem; font-weight: 600; color: #16A34A; background: #DCFCE7; padding: 0.2rem 0.6rem; border-radius: 12px;">
                Verified Pipeline
            </span>
        </div>
        <div style="font-size: 0.86rem; color: {INK_MUTED}; line-height: 1.5; margin-bottom: 1rem;">
            Review trained model performance (Tier 1: 89.35% Acc / 0.962 AUC &middot; Tier 2: 92.10% Acc / 0.970 AUC), confusion matrices, MLflow tracking runs, and screening database health.
        </div>
    """,
    unsafe_allow_html=True,
)

db_stats = get_db_stats()
t_col1, t_col2, t_col3, t_col4 = st.columns(4)
with t_col1:
    st.metric("Total Patients", f"{db_stats['patient_count']}")
with t_col2:
    st.metric("Total Visits", f"{db_stats['visit_count']}")
with t_col3:
    st.metric("Tier 2 Accuracy", "92.1%")
with t_col4:
    st.metric("Tier 2 AUROC", "0.970")

st.markdown("</div>", unsafe_allow_html=True)

if st.button("View Dev / Ops Telemetry →", key="goto_devops", use_container_width=True):
    st.switch_page("pages/3_Dev_Ops.py")

st.markdown(
    f"""
    <div style="margin-top: 3rem; padding-top: 1.25rem; border-top: 1px solid {BORDER_COLOR}; font-size: 0.82rem; color: {INK_MUTED}; text-align: center;">
        QuantaMED Diagnostic Technologies &middot; Keratoconus Screening Protocol &middot; Clinician Demo
    </div>
    """,
    unsafe_allow_html=True,
)
