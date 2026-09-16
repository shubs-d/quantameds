"""
QuantaMED — Patient Profile Section
===================================
Page 2: Longitudinal patient lookup, historical visit reviews,
persisted calculation breakdowns, cross-visit trend comparisons, and clinical notes.
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

import pandas as pd
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
    get_patient,
    get_visits,
    update_visit_notes,
)

st.set_page_config(page_title="Patient Profile — QuantaMED", page_icon="Q", layout="centered")
inject_css()

# Render clickable logo in sidebar
render_sidebar_logo()

# Render top-left clickable logo header on main page
render_top_header(current_page_name="Patient Profile")

st.markdown(
    f"""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="margin: 0 0 0.35rem 0; font-size: 1.7rem; font-weight: 700; color: {INK};">
            Longitudinal Patient Profile
        </h2>
        <div style="font-size: 0.92rem; color: {INK_MUTED}; line-height: 1.5;">
            Look up patient records, inspect historical factor breakdowns, track corneal progression, and record preventive clinical measures.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Lookup Bar Card ────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="background: #FFFFFF; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 1.25rem; box-shadow: 0 4px 18px -2px rgba(15, 23, 42, 0.04); margin-bottom: 1.5rem;">
        <div style="font-size: 0.85rem; font-weight: 700; color: {INK}; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 6px;">
            Find Patient Record
        </div>
    """,
    unsafe_allow_html=True,
)

search_col1, search_col2 = st.columns([3.2, 1.2])

with search_col1:
    search_id = st.text_input(
        "Enter Patient ID",
        value=st.session_state.get("latest_detection", {}).get("patient_id", ""),
        placeholder="Enter ID, e.g. QMED-000001",
        label_visibility="collapsed",
    )

with search_col2:
    search_clicked = st.button("Search Patient", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

if search_id.strip():
    clean_id = search_id.strip()
    patient = get_patient(clean_id)
    visits = get_visits(clean_id)

    if not patient and not visits:
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px dashed {CORAL}88; padding: 2rem; border-radius: 12px; text-align: center; margin-top: 1rem;">
                <div style="font-weight: 700; color: {INK}; font-size: 1.05rem;">No Records Found</div>
                <div style="font-size: 0.88rem; color: {INK_MUTED}; margin-top: 0.3rem;">
                    Patient identifier <code>{clean_id}</code> has no registered visits in the screening database.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        num_visits = len(visits)
        st.markdown(
            f"""
            <div style="margin-top: 1rem; padding: 1rem 1.25rem; background: #FFFFFF; border: 1px solid {BORDER_COLOR}; border-left: 4px solid {TEAL}; border-radius: 10px; display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div>
                        <div style="font-weight: 700; font-size: 1.1rem; color: {INK};">{clean_id}</div>
                        <div style="font-size: 0.78rem; color: {INK_MUTED};">Registered in screening database</div>
                    </div>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.85rem; font-weight: 700; color: {TEAL}; background: {TEAL_LIGHT}; padding: 0.35rem 0.8rem; border-radius: 20px;">
                        {num_visits} recorded visit{'s' if num_visits != 1 else ''}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Multi-Visit Comparison Trend Chart (if 2+ visits exist) ─────
        if num_visits >= 2:
            st.markdown(
                f"""
                <div style="margin-top: 1.75rem; margin-bottom: 0.5rem;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: {INK}; display: flex; align-items: center; gap: 6px;">
                        Longitudinal Progression Trend
                    </div>
                    <div style="font-size: 0.82rem; color: {INK_MUTED};">
                        Chronological tracking of Keratoconus risk percentage and steep keratometry across consultation dates.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            chronological = sorted(visits, key=lambda v: str(v["visit_date"]))
            chart_data = pd.DataFrame([
                {
                    "Visit Date": str(v["visit_date"])[:16],
                    "KC Risk (%)": float(v["risk_score"]) if v.get("risk_score") is not None else 0.0,
                    "K2 / Steep K (D)": float(v["steep_k_D"]) if v.get("steep_k_D") is not None else 0.0,
                }
                for v in chronological
            ]).set_index("Visit Date")

            st.line_chart(chart_data[["KC Risk (%)", "K2 / Steep K (D)"]], color=[TEAL, CORAL])

        # ── Chronological Visit Log (Newest First) ─────────────────────
        st.markdown(
            f"""
            <div style="margin-top: 2rem; margin-bottom: 0.8rem;">
                <div style="font-size: 1.1rem; font-weight: 700; color: {INK}; display: flex; align-items: center; gap: 6px;">
                    Chronological Consultation Logs
                </div>
                <div style="font-size: 0.82rem; color: {INK_MUTED};">
                    Newest visits listed first. Click any visit to view full autokeratometry readings, calculation trail, and clinical notes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for idx, visit in enumerate(visits):
            v_date = str(visit.get("visit_date", "Unknown"))[:16]
            v_score = visit.get("risk_score")
            score_text = f"{v_score:.0f}% Risk" if v_score is not None else "No score"
            is_newest = (idx == 0)

            expander_title = f"{'▸ LATEST VISIT: ' if is_newest else '▸ '} {v_date}  |  {score_text}"

            with st.expander(expander_title, expanded=is_newest):
                v_col1, v_col2 = st.columns([1.1, 0.9])

                with v_col1:
                    if v_score is not None:
                        risk_display(v_score, model_version=visit.get("model_version") or "RobustQuantaStudent-v1")
                    heuristic_disclaimer(model_version=visit.get("model_version") or "RobustQuantaStudent-v1")

                with v_col2:
                    graph_placeholder(height_px=140)

                # Parameter Summary Metrics
                st.markdown(
                    f"""
                    <div style="margin-top: 1rem; margin-bottom: 0.5rem; font-size: 0.88rem; font-weight: 700; color: {INK};">
                        Observed Meridian Readings (Mode: {'ARK + Pachymetry' if visit['mode'] == 'ark_pachy' else 'ARK only'})
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("K1 (Flat K)", f"{visit['flat_k_D']:.2f} D")
                    st.metric("Flat Axis", f"{visit['flat_k_axis_deg']:.0f}°")
                with c2:
                    st.metric("K2 (Steep K)", f"{visit['steep_k_D']:.2f} D")
                    st.metric("Steep Axis", f"{visit['steep_k_axis_deg']:.0f}°")
                with c3:
                    st.metric("Cylinder", f"{visit['cylinder_D']:.2f} D")
                    if visit.get("pachy_central_um") is not None:
                        st.metric("Pachymetry", f"{visit['pachy_central_um']:.0f} µm")

                # Report panel for pachy mode
                if visit["mode"] == "ark_pachy":
                    readings_dict = {
                        "flat_k_D": visit["flat_k_D"],
                        "steep_k_D": visit["steep_k_D"],
                        "pachy_central_um": visit["pachy_central_um"],
                    }
                    report_panel(readings_dict, visit["mode"])

                # Persisted Calculation Breakdown
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                with st.expander("Stored Factor Calculation Breakdown (Historical)", expanded=False):
                    calculation_breakdown(visit.get("calculation_trail", []), v_score or 0.0)

                # Free-text clinical notes per visit
                st.markdown(
                    f"""
                    <div style="margin-top: 1.25rem; margin-bottom: 0.4rem; font-size: 0.88rem; font-weight: 700; color: {INK};">
                        Doctor's Clinical Observations & Preventive Measures
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                notes_key = f"notes_{visit['visit_id']}"
                current_notes = visit.get("notes") or ""
                new_notes = st.text_area(
                    "Doctor's notes",
                    value=current_notes,
                    key=notes_key,
                    placeholder="Enter clinical observations, follow-up timeline, or corneal cross-linking recommendations...",
                    label_visibility="collapsed",
                )

                if st.button("Save Visit Notes", key=f"save_btn_{visit['visit_id']}"):
                    update_visit_notes(visit["visit_id"], new_notes)
                    st.success("Clinical notes updated successfully.")
                    st.rerun()
else:
    st.info("Enter a Patient ID above (e.g. QMED-000001) to view longitudinal history and consultation records.")
