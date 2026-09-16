"""
QuantaMED — Reusable UI Components
==================================
Modular, brand-aligned visual components including clickable logo headers,
risk cards, factor breakdown lists, and laboratory report rows.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st

from app.style import (
    BORDER_COLOR,
    CORAL,
    CORAL_HOVER,
    CORAL_LIGHT,
    INK,
    INK_MUTED,
    MIST,
    PAPER,
    TEAL,
    TEAL_HOVER,
    TEAL_LIGHT,
)

# Cache logo base64 string
_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.png"


@st.cache_data
def get_logo_base64() -> str:
    """Read and return base64 encoded logo image for reliable inline embedding."""
    if _LOGO_PATH.exists():
        return base64.b64encode(_LOGO_PATH.read_bytes()).decode("utf-8")
    return ""


def render_top_header(current_page_name: str = "Detection") -> None:
    """Render the top-left clickable logo navigation header across all dashboard pages.

    Clicking the logo or brand wordmark routes the clinician back to the homepage.
    """
    b64_logo = get_logo_base64()
    logo_img_tag = (
        f'<img src="data:image/png;base64,{b64_logo}" class="top-logo-img" alt="QuantaMED Logo" />'
        if b64_logo
        else '<span style="font-size: 1.2rem; font-weight: 800; color: {TEAL};">QM</span>'
    )

    st.markdown(
        f"""
        <div class="top-nav-header">
            <a href="/" target="_self" class="top-logo-link" title="Return to QuantaMED Home">
                {logo_img_tag}
                <div>
                    <div class="top-logo-title">Quanta<span>MED</span></div>
                    <div class="top-logo-subtitle">Super-Early KC Screening</div>
                </div>
            </a>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 0.8rem; font-weight: 600; color: {INK_MUTED}; background: {MIST}; padding: 0.35rem 0.8rem; border-radius: 20px; border: 1px solid {BORDER_COLOR};">
                    Current: <strong style="color: {TEAL_HOVER};">{current_page_name}</strong>
                </span>
                <a href="/" target="_self" style="font-size: 0.82rem; font-weight: 600; color: {TEAL_HOVER}; text-decoration: none; padding: 0.35rem 0.75rem; border-radius: 6px; background: {TEAL_LIGHT};">
                    &larr; Home
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_logo() -> None:
    """Render the clickable logo in the sidebar top area."""
    b64_logo = get_logo_base64()
    logo_img_tag = (
        f'<img src="data:image/png;base64,{b64_logo}" style="width: 44px; height: 44px; object-fit: contain; filter: drop-shadow(0 2px 8px rgba(12, 191, 222, 0.35));" alt="QuantaMED Logo" />'
        if b64_logo
        else '<span style="font-size: 1.2rem; font-weight: 800; color: {TEAL};">QM</span>'
    )

    st.sidebar.markdown(
        f"""
        <div style="padding: 0.25rem 0 1.25rem 0; border-bottom: 1px solid {BORDER_COLOR}; margin-bottom: 1rem;">
            <a href="/" target="_self" style="text-decoration: none; display: flex; align-items: center; gap: 12px;">
                {logo_img_tag}
                <div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: {INK}; letter-spacing: -0.02em;">
                        Quanta<span style="color: {TEAL};">MED</span>
                    </div>
                    <div style="font-size: 0.68rem; color: {INK_MUTED}; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">
                        Home Navigation
                    </div>
                </div>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def patient_badge(patient_id: str) -> None:
    """Render an elevated patient identifier pill badge."""
    st.markdown(
        f"""
        <div style="margin: 0.25rem 0 1rem 0;">
            <span class="patient-pill">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: {TEAL};"></span>
                Patient Ref: <strong>{patient_id}</strong>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_display(score: float, model_version: str = "RobustQuantaStudent-v1") -> None:
    """Render the elevated risk assessment card with visual progress bar and status indicator."""
    is_elevated = score >= 50.0
    hero_class = "risk-hero-elevated" if is_elevated else "risk-hero-low"
    num_color_class = "risk-color-elevated" if is_elevated else "risk-color-low"
    status_pill_class = "status-pill-elevated" if is_elevated else "status-pill-low"
    status_text = "Elevated Risk — Clinical Follow-up Advised" if is_elevated else "Low Risk — Normal Corneal Profile"
    meter_color = CORAL if is_elevated else TEAL

    st.markdown(
        f"""
        <div class="risk-hero-card {hero_class}">
            <div>
                <div style="font-size: 0.8rem; font-weight: 600; color: {INK_MUTED}; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.25rem;">
                    Keratoconus Risk Assessment
                </div>
                <div class="risk-big-number {num_color_class}">
                    {score:.0f}<span style="font-size: 2.2rem; font-weight: 500;">%</span>
                </div>
                <div style="margin-top: 0.4rem;">
                    <span class="status-pill {status_pill_class}">
                        {status_text}
                    </span>
                </div>
            </div>
            <div style="text-align: right; min-width: 140px;">
                <div style="font-size: 0.75rem; color: {INK_MUTED}; margin-bottom: 0.35rem;">
                    Confidence Meter
                </div>
                <div style="width: 130px; height: 10px; background: {MIST}; border-radius: 6px; overflow: hidden; border: 1px solid {BORDER_COLOR};">
                    <div style="width: {score}%; height: 100%; background: {meter_color}; border-radius: 6px;"></div>
                </div>
                <div style="font-size: 0.72rem; color: {INK_MUTED}; margin-top: 0.4rem;">
                    Model: <strong>{model_version}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def graph_placeholder(height_px: int = 175, risk_score: Optional[float] = None) -> None:
    """Render an interactive clinical risk gauge & triage operating threshold meter."""
    if risk_score is None:
        if "latest_detection" in st.session_state:
            risk_score = st.session_state["latest_detection"].get("risk_score", 15.0)
        else:
            risk_score = 15.0

    pct = max(0.0, min(float(risk_score), 100.0))

    if pct >= 50.0:
        band_title = "⚠️ High Ectatic Risk Band"
        band_color = CORAL
        band_desc = "Above confirmatory referral threshold (τ ≥ 50%)"
    elif pct >= 26.2:
        band_title = "❓ Borderline / High Triage Risk"
        band_color = "#E67E22"
        band_desc = "Flagged under high-sensitivity screening (τ ≥ 26.2%)"
    else:
        band_title = "✅ Normal Physiological Band"
        band_color = TEAL
        band_desc = "Below triage threshold (τ < 26.2%)"

    st.markdown(
        f"""
        <div style="
            background: #FFFFFF;
            border: 1px solid {BORDER_COLOR};
            border-radius: 12px;
            padding: 1.15rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.03);
            margin: 0.8rem 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.5rem;">
                <span style="font-size: 0.85rem; font-weight: 700; color: {INK};">
                    Clinical Triage Gauge
                </span>
                <span style="font-size: 0.78rem; font-weight: 700; color: {band_color};">
                    {band_title}
                </span>
            </div>

            <!-- Visual 3-zone gradient track -->
            <div style="position: relative; width: 100%; height: 16px; background: #E2E8F0; border-radius: 8px; overflow: hidden; margin-top: 0.5rem; margin-bottom: 0.3rem;">
                <div style="display: flex; width: 100%; height: 100%;">
                    <div style="width: 26.2%; background: #27AE60; opacity: 0.85;" title="Low Risk (0-26%)"></div>
                    <div style="width: 23.8%; background: #F39C12; opacity: 0.85;" title="Borderline (26-50%)"></div>
                    <div style="width: 50.0%; background: #E74C3C; opacity: 0.85;" title="Keratoconus (50-100%)"></div>
                </div>
            </div>

            <!-- Pointer marker -->
            <div style="position: relative; width: 100%; height: 16px; margin-bottom: 0.4rem;">
                <div style="position: absolute; left: calc({pct}% - 6px); top: 0; font-size: 0.82rem; font-weight: 800; color: {band_color};">
                    ▲
                </div>
            </div>

            <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: {INK_MUTED};">
                <span>0%</span>
                <span style="color: #E67E22; font-weight: 600;">τ=26.2% (Triage)</span>
                <span style="color: #E74C3C; font-weight: 600;">τ=50% (Confirmatory)</span>
                <span>100%</span>
            </div>

            <div style="margin-top: 0.6rem; padding-top: 0.5rem; border-top: 1px solid {BORDER_COLOR}; font-size: 0.75rem; color: {INK_MUTED};">
                {band_desc}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_latent_representation(latent_vector: List[float]) -> None:
    """Render the model's 8D continuous biomechanical latent vector."""
    if not latent_vector or len(latent_vector) != 8:
        return

    st.markdown(
        f"""
        <div style="margin-top: 1rem; margin-bottom: 0.5rem; font-size: 0.85rem; font-weight: 700; color: {INK};">
            Privileged Biomechanical Latent Vector (z ∈ ℝ⁸)
        </div>
        <div style="font-size: 0.78rem; color: {INK_MUTED}; margin-bottom: 0.6rem;">
            8 continuous latent features distilled from the frozen corneal topography Teacher CNN:
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(8)
    for i, (val, col) in enumerate(zip(latent_vector, cols)):
        with col:
            intensity = min(1.0, max(0.0, float(val) / 3.1415))
            bg = f"rgba(12, 191, 222, {0.15 + intensity * 0.75})"
            st.markdown(
                f"""
                <div style="background: {bg}; border: 1px solid {BORDER_COLOR}; border-radius: 6px; padding: 0.35rem 0.2rem; text-align: center;">
                    <div style="font-size: 0.65rem; color: {INK_MUTED};">z_{i+1}</div>
                    <div style="font-size: 0.78rem; font-weight: 700; color: {INK};">{val:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def report_panel(readings: Dict[str, Any], mode: str) -> None:
    """Render auto-derived clinical laboratory rows with modern parameter chips."""
    if mode != "ark_pachy":
        return

    pachy = readings.get("pachy_central_um")
    steep_k = float(readings.get("steep_k_D", 0))
    flat_k = float(readings.get("flat_k_D", 0))
    corneal_toricity = round(steep_k - flat_k, 2)

    st.markdown(
        f"""
        <div class="qmed-card" style="padding: 1.2rem; margin-top: 1.25rem;">
            <div style="font-size: 0.9rem; font-weight: 700; color: {INK}; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 8px;">
                Derived Clinical Laboratory Parameters
            </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Central Corneal Thickness
    if pachy is not None and float(pachy) > 0:
        pachy_val = float(pachy)
        if pachy_val < 500.0:
            badge_html = f'<span class="factor-chip-raise">Thin (&lt; 500 µm) &middot; Flagged</span>'
        elif pachy_val > 565.0:
            badge_html = f'<span class="factor-chip-ok">Thick (&gt; 565 µm) &middot; Normal</span>'
        else:
            badge_html = f'<span class="factor-chip-ok">Normal (500–565 µm)</span>'

        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid {BORDER_COLOR};">
                <div>
                    <div style="font-weight: 600; font-size: 0.9rem; color: {INK};">Central Corneal Thickness</div>
                    <div style="font-size: 0.78rem; color: {INK_MUTED};">Normative reference: 520 – 570 µm</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-weight: 700; font-size: 0.95rem; color: {INK};">{pachy_val:.0f} µm</div>
                    <div style="margin-top: 3px;">{badge_html}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Corneal Toricity (ΔK = K2 - K1)
    is_toricity_high = corneal_toricity > 2.0
    toricity_badge = (
        f'<span class="factor-chip-raise">Elevated (&gt; 2.0 D)</span>'
        if is_toricity_high
        else f'<span class="factor-chip-ok">Within Tolerance</span>'
    )

    st.markdown(
        f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0;">
                <div>
                    <div style="font-weight: 600; font-size: 0.9rem; color: {INK};">Corneal Toricity (ΔK = K2 - K1)</div>
                    <div style="font-size: 0.78rem; color: {INK_MUTED};">Meridional power differential</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-weight: 700; font-size: 0.95rem; color: {INK};">{corneal_toricity:+.2f} D</div>
                    <div style="margin-top: 3px;">{toricity_badge}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def calculation_breakdown(trail: List[Dict[str, Any]], risk_score: float) -> None:
    """Render the factor-by-factor transparency view (§5b) with modern factor cards."""
    if not trail:
        st.info("No calculation trail recorded for this prediction.")
        return

    st.markdown(
        f"""
        <div style="margin-top: 0.25rem; margin-bottom: 0.8rem;">
            <div style="font-size: 0.95rem; font-weight: 700; color: {INK};">
                Factor Contribution Breakdown (§5b Transparency View)
            </div>
            <div style="font-size: 0.8rem; color: {INK_MUTED};">
                Detailed clinical weight and threshold judgment for each input meridian.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for item in trail:
        factor_name = item.get("factor", "Factor")
        val = item.get("value", 0.0)
        unit = item.get("unit", "")
        ref = item.get("ref_range", "")
        contrib = item.get("contribution", 0.0)
        direction = item.get("direction", "")

        if contrib > 0:
            chip_class = "factor-chip-raise"
            chip_text = f"+{contrib:+.1f} pts ({direction})"
        elif contrib < 0:
            chip_class = "factor-chip-ok"
            chip_text = f"{contrib:+.1f} pts ({direction})"
        else:
            chip_class = "factor-chip-ok"
            chip_text = "+0.0 pts (normal reference)"

        st.markdown(
            f"""
            <div class="factor-card">
                <div>
                    <div class="factor-title">{factor_name}</div>
                    <div class="factor-ref">Observed: <strong>{val} {unit}</strong> &nbsp;&middot;&nbsp; Reference: {ref}</div>
                </div>
                <div>
                    <span class="{chip_class}">{chip_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="padding-top: 0.75rem; font-size: 0.92rem; font-weight: 700; color: {INK}; text-align: right;">
            Combined Clinical Risk Score: 
            <span style="color: {CORAL if risk_score >= 50 else TEAL}; font-size: 1.15rem;">
                {risk_score:.0f}%
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def heuristic_disclaimer(model_version: str = "RobustQuantaStudent-v1") -> None:
    """Render a clean disclaimer tag."""
    st.markdown(
        f"""
        <div class="disclaimer-pill">
            Verified model pipeline &middot; {model_version} &middot; Super-early screening protocol
        </div>
        """,
        unsafe_allow_html=True,
    )
