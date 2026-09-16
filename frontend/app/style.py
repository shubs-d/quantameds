"""
QuantaMED — Premium Design System
==================================
Elevated medical-tech aesthetic built around QuantaMED's signature brand tokens:
- Primary Brand Teal: #0CBFDE
- Secondary Brand Coral: #FB7E68
- Deep Slate Ink: #0F172A
- Clinical Paper: #FFFFFF
- Subtle Atmospheric Mist: #F1F6F9
"""

from __future__ import annotations

import streamlit as st

# ── Color tokens (Exact QuantaMED brand palette) ──────────────────────
TEAL = "#0CBFDE"           # Primary brand teal
TEAL_HOVER = "#0897B4"     # Deeper teal for button hover and active states
TEAL_LIGHT = "rgba(12, 191, 222, 0.09)"
TEAL_GLOW = "0 8px 24px -4px rgba(12, 191, 222, 0.35)"

CORAL = "#FB7E68"          # Secondary brand coral
CORAL_HOVER = "#E05A44"    # Deeper coral for elevated emphasis
CORAL_LIGHT = "rgba(251, 126, 104, 0.10)"
CORAL_GLOW = "0 8px 24px -4px rgba(251, 126, 104, 0.38)"

INK = "#0F172A"            # Deep slate for primary typography
INK_MUTED = "#64748B"      # Slate for subtext and metadata
INK_LIGHT = "#94A3B8"      # Light slate for borders and placeholders
PAPER = "#FFFFFF"          # Background pure white
MIST = "#F8FAFC"           # Subtle cool card background
BORDER_COLOR = "#E2E8F0"   # Refined card border


def inject_css() -> None:
    """Inject polished stylesheet with refined typography, soft elevation, and gradients."""
    st.markdown(
        f"""
        <style>
        /* ── Typography: Inter font ───────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: {INK};
        }}

        /* ── Background & Ambient Atmosphere ─────────────────── */
        .stApp {{
            background: linear-gradient(180deg, #F8FCFD 0%, #F1F6F9 50%, #FFFFFF 100%);
            background-attachment: fixed;
        }}

        /* Main Container layout */
        .main .block-container {{
            max-width: 960px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }}

        /* ── Sidebar Styling ─────────────────────────────────── */
        [data-testid="stSidebar"] {{
            background-color: #FFFFFF;
            border-right: 1px solid {BORDER_COLOR};
            box-shadow: 4px 0 20px rgba(15, 23, 42, 0.03);
        }}
        [data-testid="stSidebarNav"] {{
            padding-top: 1rem;
        }}
        [data-testid="stSidebarNav"] a {{
            border-radius: 8px;
            margin: 3px 8px;
            padding: 8px 12px;
            font-weight: 500;
            color: {INK};
            transition: all 0.15s ease-in-out;
        }}
        [data-testid="stSidebarNav"] a:hover {{
            background-color: {TEAL_LIGHT};
            color: {TEAL_HOVER};
        }}
        [data-testid="stSidebarNav"] a[aria-current="page"] {{
            background: linear-gradient(90deg, rgba(12, 191, 222, 0.15) 0%, rgba(12, 191, 222, 0.05) 100%);
            color: {TEAL_HOVER} !important;
            font-weight: 600;
            border-left: 3px solid {TEAL};
        }}

        /* ── Headers ─────────────────────────────────────────── */
        h1, h2, h3, h4 {{
            font-family: 'Inter', sans-serif !important;
            color: {INK} !important;
            font-weight: 700 !important;
            letter-spacing: -0.025em;
        }}
        h1 {{
            font-size: 2.1rem !important;
            line-height: 1.25 !important;
        }}
        h2 {{
            font-size: 1.5rem !important;
            margin-top: 0.5rem;
        }}
        h3 {{
            font-size: 1.15rem !important;
        }}

        /* ── Buttons: Vibrant Teal Gradient ─────────────────── */
        .stButton > button[kind="primary"],
        .stButton > button {{
            background: linear-gradient(135deg, {TEAL} 0%, {TEAL_HOVER} 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.4rem !important;
            box-shadow: {TEAL_GLOW} !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
            letter-spacing: 0.01em;
        }}
        .stButton > button:hover {{
            opacity: 0.94;
            transform: translateY(-1px);
            box-shadow: 0 10px 28px -4px rgba(12, 191, 222, 0.45) !important;
        }}
        .stButton > button:active {{
            transform: translateY(0);
        }}

        /* Secondary Button (for save/search) */
        .stButton.secondary > button {{
            background: #FFFFFF !important;
            color: {INK} !important;
            border: 1px solid {BORDER_COLOR} !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
        }}

        /* ── Modern Form Inputs ──────────────────────────────── */
        .stNumberInput input, .stTextInput input, .stTextArea textarea {{
            font-family: 'Inter', sans-serif;
            background-color: #FFFFFF !important;
            border: 1px solid {BORDER_COLOR} !important;
            border-radius: 8px !important;
            color: {INK} !important;
            font-size: 0.92rem !important;
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.03);
            transition: all 0.15s ease;
        }}
        .stNumberInput input:focus, .stTextInput input:focus, .stTextArea textarea:focus {{
            border-color: {TEAL} !important;
            box-shadow: 0 0 0 3px rgba(12, 191, 222, 0.22) !important;
        }}

        /* Radio Mode Toggle — Segmented style */
        [data-testid="stRadio"] {{
            background: #FFFFFF;
            padding: 8px 12px;
            border-radius: 10px;
            border: 1px solid {BORDER_COLOR};
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
            display: inline-block;
        }}

        /* ── Expanders ───────────────────────────────────────── */
        .streamlit-expanderHeader {{
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            color: {INK} !important;
            background-color: #FFFFFF;
            border: 1px solid {BORDER_COLOR};
            border-radius: 8px !important;
            padding: 0.75rem 1rem !important;
            transition: all 0.15s ease;
        }}
        .streamlit-expanderHeader:hover {{
            border-color: {TEAL};
            background-color: {TEAL_LIGHT};
        }}
        .streamlit-expanderContent {{
            background-color: #FFFFFF;
            border: 1px solid {BORDER_COLOR};
            border-top: none;
            border-bottom-left-radius: 8px;
            border-bottom-right-radius: 8px;
            padding: 1.25rem !important;
        }}

        /* ── Card Containers ─────────────────────────────────── */
        .qmed-card {{
            background: #FFFFFF;
            border: 1px solid {BORDER_COLOR};
            border-radius: 14px;
            padding: 1.5rem;
            box-shadow: 0 8px 30px -6px rgba(15, 23, 42, 0.05), 0 2px 6px -1px rgba(15, 23, 42, 0.03);
            margin-bottom: 1.25rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .qmed-card:hover {{
            box-shadow: 0 12px 35px -6px rgba(15, 23, 42, 0.08), 0 4px 10px -2px rgba(15, 23, 42, 0.04);
        }}

        .qmed-card-teal {{
            background: #FFFFFF;
            border-top: 4px solid {TEAL};
            border-left: 1px solid {BORDER_COLOR};
            border-right: 1px solid {BORDER_COLOR};
            border-bottom: 1px solid {BORDER_COLOR};
            border-radius: 14px;
            padding: 1.4rem;
            box-shadow: 0 8px 24px -4px rgba(12, 191, 222, 0.08);
        }}

        .qmed-card-coral {{
            background: #FFFFFF;
            border-top: 4px solid {CORAL};
            border-left: 1px solid {BORDER_COLOR};
            border-right: 1px solid {BORDER_COLOR};
            border-bottom: 1px solid {BORDER_COLOR};
            border-radius: 14px;
            padding: 1.4rem;
            box-shadow: 0 8px 24px -4px rgba(251, 126, 104, 0.09);
        }}

        /* ── Risk Metric Badge & Visuals ─────────────────────── */
        .risk-hero-card {{
            background: #FFFFFF;
            border-radius: 16px;
            border: 1px solid {BORDER_COLOR};
            padding: 1.75rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 10px 30px -4px rgba(15, 23, 42, 0.06);
            margin: 1.25rem 0;
            position: relative;
            overflow: hidden;
        }}
        .risk-hero-card::before {{
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 6px;
        }}
        .risk-hero-low::before {{
            background: linear-gradient(180deg, {TEAL} 0%, {TEAL_HOVER} 100%);
        }}
        .risk-hero-elevated::before {{
            background: linear-gradient(180deg, {CORAL} 0%, {CORAL_HOVER} 100%);
        }}

        .risk-big-number {{
            font-size: 4.2rem;
            font-weight: 700;
            line-height: 1;
            letter-spacing: -0.04em;
        }}
        .risk-color-low {{
            color: {TEAL};
        }}
        .risk-color-elevated {{
            color: {CORAL};
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85rem;
            font-weight: 600;
            padding: 0.35rem 0.9rem;
            border-radius: 20px;
            letter-spacing: 0.02em;
        }}
        .status-pill-low {{
            background: {TEAL_LIGHT};
            color: {TEAL_HOVER};
            border: 1px solid rgba(12, 191, 222, 0.3);
        }}
        .status-pill-elevated {{
            background: {CORAL_LIGHT};
            color: {CORAL_HOVER};
            border: 1px solid rgba(251, 126, 104, 0.35);
        }}

        /* Patient Badge */
        .patient-pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: #FFFFFF;
            color: {INK};
            font-weight: 600;
            font-size: 0.9rem;
            border: 1.5px solid {TEAL};
            border-radius: 24px;
            padding: 0.3rem 0.9rem;
            box-shadow: 0 2px 8px rgba(12, 191, 222, 0.12);
        }}

        /* Breakdown Factor Row */
        .factor-card {{
            background: #FFFFFF;
            border: 1px solid {BORDER_COLOR};
            border-radius: 10px;
            padding: 0.85rem 1.1rem;
            margin-bottom: 0.6rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.15s ease;
        }}
        .factor-card:hover {{
            border-color: {TEAL};
            transform: translateX(3px);
        }}
        .factor-title {{
            font-weight: 600;
            font-size: 0.92rem;
            color: {INK};
        }}
        .factor-ref {{
            font-size: 0.8rem;
            color: {INK_MUTED};
            margin-top: 2px;
        }}
        .factor-chip-raise {{
            background: {CORAL_LIGHT};
            color: {CORAL_HOVER};
            font-weight: 600;
            font-size: 0.82rem;
            padding: 0.25rem 0.65rem;
            border-radius: 12px;
            border: 1px solid rgba(251, 126, 104, 0.3);
        }}
        .factor-chip-ok {{
            background: {TEAL_LIGHT};
            color: {TEAL_HOVER};
            font-weight: 600;
            font-size: 0.82rem;
            padding: 0.25rem 0.65rem;
            border-radius: 12px;
            border: 1px solid rgba(12, 191, 222, 0.25);
        }}

        /* Disclaimer tag */
        .disclaimer-pill {{
            font-size: 0.78rem;
            color: {INK_MUTED};
            background: rgba(15, 23, 42, 0.04);
            border: 1px solid {BORDER_COLOR};
            border-radius: 6px;
            padding: 0.3rem 0.7rem;
            display: inline-block;
        }}

        /* ── Top Nav Header with Clickable Logo ──────────────── */
        .top-nav-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #FFFFFF;
            border: 1px solid {BORDER_COLOR};
            border-radius: 12px;
            padding: 0.75rem 1.25rem;
            margin-bottom: 1.75rem;
            box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.04);
        }}
        .top-logo-link {{
            display: flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
            color: {INK};
            transition: opacity 0.15s ease;
        }}
        .top-logo-link:hover {{
            opacity: 0.85;
        }}
        .top-logo-img {{
            width: 44px;
            height: 44px;
            object-fit: contain;
            border-radius: 8px;
            filter: drop-shadow(0 2px 6px rgba(12, 191, 222, 0.25));
        }}
        .top-logo-title {{
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: {INK};
        }}
        .top-logo-title span {{
            color: {TEAL};
        }}
        .top-logo-subtitle {{
            font-size: 0.72rem;
            color: {INK_MUTED};
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
