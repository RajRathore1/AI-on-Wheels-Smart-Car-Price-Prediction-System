# =============================================================
# 🚗 CAR PRICE PREDICTION — AI VALUATION APP
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image

from condition_assessment import assess_condition_multi, condition_label
from car_recognition import recognize_from_images, match_to_price_dataset
from valuation import (
    build_report,
    damage_impact,
    depreciation_curve,
    market_position,
    predict_with_range,
    similar_listings,
)

# =============================================================
# PAGE CONFIGURATION
# =============================================================
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="auto",  # collapsed on phones, expanded on desktop
)

# =============================================================
# LOAD MODEL & DATA
# =============================================================
@st.cache_resource
def load_model():
    return pickle.load(open("PriceModel.pkl", "rb"))

@st.cache_data
def load_data():
    return pd.read_csv("Cleaned_Car_data.csv")

pipe = load_model()
df = load_data()

# =============================================================
# DESIGN SYSTEM
# One stylesheet for the whole app: neutral dark surfaces, a single
# accent colour, and restrained motion. Everything else inherits.
# =============================================================
ACCENT = "#79aaff"
SURFACE = "#131518"
BG = "#090a0c"
TEXT_MUTED = "#a8adb7"

st.markdown(f"""
    <style>
    :root {{
        --accent: {ACCENT};
        --accent-soft: rgba(121, 170, 255, 0.13);
        --bg: {BG};
        --surface: {SURFACE};
        --surface-2: #1a1d22;
        --border: rgba(255, 255, 255, 0.10);
        --border-strong: rgba(222, 229, 241, 0.24);
        --text: #f7f5f0;
        --text-muted: {TEXT_MUTED};
        --radius: 12px;
    }}

    /* ---------- Base ---------- */
    [data-testid="stAppViewContainer"] {{
        background:
            radial-gradient(circle at 3% -12%, rgba(75, 107, 164, 0.22), transparent 32rem),
            radial-gradient(circle at 95% 0%, rgba(255, 255, 255, 0.055), transparent 26rem),
            var(--bg);
    }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"] {{
        background: #0d0f12;
        border-right: 1px solid var(--border);
    }}
    .block-container {{ padding: 2.75rem 1.5rem 4rem; max-width: 1220px; }}

    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
        color: var(--text);
    }}
    h1, h2, h3, h4 {{ color: var(--text); font-weight: 650; letter-spacing: -0.01em; }}
    p, span, label, li {{ color: var(--text); }}
    a {{ color: var(--accent); }}

    /* ---------- Page header ---------- */
    .page-kicker {{
        color: #b9c9ec;
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 0 0 10px;
    }}
    .hero-shell {{
        position: relative;
        isolation: isolate;
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(250px, 0.65fr);
        gap: 1.5rem;
        overflow: hidden;
        min-height: 265px;
        padding: 2.25rem 2.3rem;
        margin: 0 0 1.55rem;
        border: 1px solid rgba(210, 224, 255, 0.17);
        border-radius: 22px;
        background:
            linear-gradient(108deg, rgba(31, 42, 65, 0.92), rgba(15, 17, 21, 0.96) 58%),
            #101216;
        box-shadow: 0 24px 56px rgba(0, 0, 0, 0.26);
    }}
    .hero-shell::before {{
        content: "";
        position: absolute;
        z-index: -1;
        width: 34rem;
        height: 34rem;
        right: -11rem;
        top: -14rem;
        border: 1px solid rgba(154, 189, 255, 0.15);
        border-radius: 50%;
        box-shadow: 0 0 0 3rem rgba(121, 170, 255, 0.035), 0 0 0 7rem rgba(121, 170, 255, 0.025);
    }}
    .hero-copy {{ display: flex; flex-direction: column; justify-content: center; }}
    .page-title {{
        font-size: 2.5rem;
        font-weight: 760;
        line-height: 1.08;
        letter-spacing: -0.045em;
        margin: 0 0 10px 0;
        max-width: 760px;
    }}
    .page-sub {{
        color: var(--text-muted);
        font-size: 1rem;
        margin: 0 0 32px 0;
        max-width: 680px;
        line-height: 1.65;
    }}
    .hero-pills {{ display: flex; flex-wrap: wrap; gap: 0.55rem; margin-top: 0.15rem; }}
    .hero-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.42rem 0.7rem;
        color: #dce7ff;
        font-size: 0.77rem;
        font-weight: 650;
        border: 1px solid rgba(211, 225, 255, 0.17);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.045);
    }}
    .hero-pill-dot {{
        width: 0.42rem; height: 0.42rem; border-radius: 50%;
        background: #78a8ff; box-shadow: 0 0 11px #78a8ff;
    }}
    .hero-art {{
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 190px;
        padding-left: 0.4rem;
    }}
    .car-silhouette {{ width: min(100%, 330px); filter: drop-shadow(0 14px 16px rgba(0,0,0,0.45)); }}

    /* ---------- Cards ---------- */
    .card {{
        background: linear-gradient(145deg, rgba(255,255,255,0.055), rgba(255,255,255,0.018)), var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 18px 44px rgba(0, 0, 0, 0.14);
    }}
    .stat-card {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(145deg, rgba(255,255,255,0.055), rgba(255,255,255,0.018)), var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px 22px;
        height: 100%;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.12);
        transition: border-color 0.18s ease, transform 0.18s ease, background 0.18s ease;
    }}
    .stat-card:hover {{
        background: linear-gradient(145deg, rgba(121, 170, 255, 0.14), rgba(255,255,255,0.025)), var(--surface);
        border-color: rgba(185, 207, 251, 0.40);
        transform: translateY(-3px);
    }}
    .stat-card::after {{
        content: "";
        position: absolute;
        top: 0; right: 0; width: 5.5rem; height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent));
        opacity: 0.85;
    }}
    .stat-value {{ font-size: 1.9rem; font-weight: 750; letter-spacing: -0.03em; }}
    .stat-label {{ color: var(--text-muted); font-size: 0.85rem; margin-top: 4px; }}

    /* ---------- Home market intelligence cards ---------- */
    .market-card {{
        position: relative;
        overflow: hidden;
        min-height: 188px;
        padding: 1.35rem 1.45rem 1.2rem;
        border: 1px solid rgba(215, 228, 255, 0.15);
        border-radius: 18px;
        background:
            linear-gradient(135deg, rgba(120, 165, 255, 0.14), transparent 48%),
            linear-gradient(145deg, rgba(255,255,255,0.055), rgba(255,255,255,0.015)),
            #14171c;
        box-shadow: 0 16px 34px rgba(0,0,0,0.20);
        transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
    }}
    .market-card::before {{
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        opacity: 0.26;
        background-image: linear-gradient(90deg, rgba(255,255,255,0.045) 1px, transparent 1px),
                          linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px);
        background-size: 22px 22px;
        mask-image: linear-gradient(to bottom, black, transparent 72%);
    }}
    .market-card:hover {{
        transform: translateY(-5px);
        border-color: rgba(174, 202, 255, 0.46);
        box-shadow: 0 22px 42px rgba(0,0,0,0.30), 0 0 0 1px rgba(121,170,255,0.08) inset;
    }}
    .market-card--brands {{
        background:
            linear-gradient(135deg, rgba(159, 124, 255, 0.16), transparent 48%),
            linear-gradient(145deg, rgba(255,255,255,0.055), rgba(255,255,255,0.015)),
            #14171c;
    }}
    .market-card--models {{
        background:
            linear-gradient(135deg, rgba(75, 201, 179, 0.14), transparent 48%),
            linear-gradient(145deg, rgba(255,255,255,0.055), rgba(255,255,255,0.015)),
            #14171c;
    }}
    .market-card-top {{ display: flex; align-items: center; justify-content: space-between; }}
    .market-card-icon {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2.6rem; height: 2.6rem;
        color: #dbe8ff;
        border: 1px solid rgba(190, 214, 255, 0.26);
        border-radius: 12px;
        background: rgba(113, 158, 248, 0.13);
    }}
    .market-card--brands .market-card-icon {{ color: #e4d8ff; background: rgba(159, 124, 255, 0.13); }}
    .market-card--models .market-card-icon {{ color: #d6fff6; background: rgba(75, 201, 179, 0.12); }}
    .market-card-icon svg {{ width: 1.32rem; height: 1.32rem; stroke: currentColor; }}
    .market-card-tag {{
        color: #bbc9e4;
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }}
    .market-card-value {{
        position: relative;
        margin-top: 1rem;
        color: #f7f5f0;
        font-size: clamp(2rem, 1.5rem + 1.5vw, 2.55rem);
        font-weight: 780;
        letter-spacing: -0.05em;
        line-height: 1;
    }}
    .market-card-headline {{
        position: relative;
        margin-top: 1rem;
        color: #f7f5f0;
        font-size: clamp(1.25rem, 1.05rem + 0.55vw, 1.55rem);
        font-weight: 740;
        letter-spacing: -0.035em;
        line-height: 1.14;
    }}
    .market-card-label {{ margin-top: 0.42rem; color: #b5bac4; font-size: 0.91rem; font-weight: 540; }}
    .market-card-footer {{
        position: relative;
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 0.8rem;
        margin-top: 1rem;
        padding-top: 0.78rem;
        border-top: 1px solid rgba(255,255,255,0.09);
        color: #8e97a5;
        font-size: 0.73rem;
    }}
    .mini-bars {{ display: flex; align-items: end; gap: 3px; height: 1.05rem; }}
    .mini-bars span {{ width: 4px; height: var(--bar); border-radius: 4px; background: #7aaaff; opacity: 0.9; }}
    .market-card--brands .mini-bars span {{ background: #ac8dff; }}
    .market-card--models .mini-bars span {{ background: #5fd4c0; }}

    /* ---------- Home decision tools ---------- */
    .section-eyebrow {{
        margin: 0 0 0.45rem;
        color: #94b8ff;
        font-size: 0.69rem;
        font-weight: 760;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }}
    .home-section-title {{
        margin: 0;
        color: #f7f5f0;
        font-size: clamp(1.45rem, 1.2rem + 0.65vw, 1.9rem);
        font-weight: 735;
        letter-spacing: -0.04em;
    }}
    .home-section-sub {{ margin: 0.56rem 0 1.2rem; color: #9ea6b3; font-size: 0.93rem; line-height: 1.55; }}
    .brand-insight-card {{
        position: relative;
        overflow: hidden;
        padding: 1.45rem;
        border: 1px solid rgba(209, 224, 255, 0.17);
        border-radius: 18px;
        background: linear-gradient(130deg, rgba(121,170,255,0.16), rgba(255,255,255,0.035) 48%, rgba(255,255,255,0.02));
        box-shadow: 0 18px 36px rgba(0,0,0,0.18);
    }}
    .brand-insight-card::after {{
        content: "";
        position: absolute;
        width: 13rem; height: 13rem;
        right: -5.8rem; bottom: -8rem;
        border: 1px solid rgba(157,192,255,0.22);
        border-radius: 50%;
        box-shadow: 0 0 0 1.6rem rgba(121,170,255,0.045);
    }}
    .brand-card-label {{ color: #c7d1e3; font-size: 0.78rem; font-weight: 650; }}
    .brand-card-value {{
        margin-top: 0.5rem;
        color: #f7f5f0;
        font-size: clamp(2rem, 1.6rem + 1.2vw, 2.6rem);
        font-weight: 780;
        letter-spacing: -0.055em;
        line-height: 1;
    }}
    .brand-card-note {{ margin-top: 0.55rem; color: #aab3c2; font-size: 0.82rem; }}
    .brand-card-stats {{
        position: relative;
        z-index: 1;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin-top: 1.3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.12);
    }}
    .brand-card-stat {{ padding: 0.1rem 0.65rem 0.1rem 0; }}
    .brand-card-stat + .brand-card-stat {{ border-left: 1px solid rgba(255,255,255,0.12); padding-left: 0.85rem; }}
    .brand-card-stat-label {{ color: #98a1af; font-size: 0.7rem; }}
    .brand-card-stat-value {{ margin-top: 0.2rem; color: #e9edf6; font-size: 0.94rem; font-weight: 690; }}
    .use-case-list {{ display: grid; gap: 0.72rem; margin-top: 0.2rem; }}
    .use-case {{
        display: grid;
        grid-template-columns: 2.45rem 1fr;
        gap: 0.85rem;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 14px;
        background: rgba(255,255,255,0.025);
        transition: transform 180ms ease, border-color 180ms ease, background 180ms ease;
    }}
    .use-case:hover {{
        transform: translateX(4px);
        border-color: rgba(161,194,255,0.38);
        background: rgba(121,170,255,0.07);
    }}
    .use-case-icon {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2.45rem; height: 2.45rem;
        color: #dbe8ff;
        border: 1px solid rgba(181,207,255,0.25);
        border-radius: 10px;
        background: rgba(121,170,255,0.13);
    }}
    .use-case-icon svg {{ width: 1.18rem; height: 1.18rem; stroke: currentColor; }}
    .use-case-title {{ color: #f2f5fb; font-size: 0.97rem; font-weight: 720; }}
    .use-case-desc {{ margin-top: 0.22rem; color: #a8b0bd; font-size: 0.82rem; line-height: 1.48; }}

    /* ---------- Analytics workspace ---------- */
    .filter-toolbar {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.18rem 0 1.1rem;
        margin-bottom: 1.15rem;
        border-bottom: 1px solid rgba(255,255,255,0.09);
    }}
    .filter-toolbar-title {{ color: #eef3ff; font-size: 1rem; font-weight: 730; letter-spacing: -0.015em; }}
    .filter-toolbar-note {{ margin-top: 0.2rem; color: #9ca6b7; font-size: 0.8rem; line-height: 1.45; }}
    .filter-toolbar-status {{
        flex: 0 0 auto;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.38rem 0.6rem;
        color: #bbcdf1;
        border: 1px solid rgba(160,192,248,0.21);
        border-radius: 999px;
        background: rgba(121,170,255,0.08);
        font-size: 0.69rem;
        font-weight: 720;
    }}
    .filter-toolbar-status::before {{
        content: "";
        width: 0.38rem; height: 0.38rem;
        border-radius: 50%;
        background: #78a8ff;
        box-shadow: 0 0 10px rgba(120,168,255,0.8);
    }}
    [data-testid="stVerticalBlockBorderWrapper"]:has(.filter-toolbar) {{
        margin-top: 0.4rem;
        padding: 1.25rem 1.35rem 1.35rem;
        border: 1px solid rgba(216,228,255,0.14);
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(121,170,255,0.07), rgba(255,255,255,0.018));
        box-shadow: 0 16px 32px rgba(0,0,0,0.13);
    }}
    .analytics-metric {{
        position: relative;
        overflow: hidden;
        min-height: 166px;
        padding: 1.15rem 1.2rem;
        border: 1px solid rgba(255,255,255,0.11);
        border-radius: 16px;
        background: linear-gradient(145deg, rgba(255,255,255,0.055), rgba(255,255,255,0.018));
        box-shadow: 0 12px 28px rgba(0,0,0,0.15);
    }}
    .analytics-metric::before {{
        content: "";
        position: absolute;
        top: 0; left: 1.2rem; right: 1.2rem; height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent), transparent);
    }}
    .analytics-metric-head {{ display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; }}
    .analytics-metric-icon {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 2.1rem; height: 2.1rem;
        color: #d9e7ff;
        border: 1px solid rgba(184,210,255,0.20);
        border-radius: 9px;
        background: rgba(121,170,255,0.11);
    }}
    .analytics-metric-icon svg {{ width: 1.02rem; height: 1.02rem; stroke: currentColor; }}
    .analytics-metric-kicker {{ color: #8f9caf; font-size: 0.66rem; font-weight: 740; letter-spacing: 0.1em; text-transform: uppercase; }}
    .analytics-metric-value {{ margin-top: 1.05rem; color: #f7f5f0; font-size: clamp(1.7rem, 1.35rem + 0.8vw, 2.2rem); font-weight: 770; letter-spacing: -0.05em; line-height: 1; }}
    .analytics-metric-label {{ margin-top: 0.45rem; color: #b4bcc9; font-size: 0.82rem; }}
    .analytics-metric-foot {{ margin-top: 0.85rem; padding-top: 0.65rem; border-top: 1px solid rgba(255,255,255,0.08); color: #8792a3; font-size: 0.69rem; }}

    /* ---------- Numbered steps ---------- */
    .step {{
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 1.15rem;
        font-weight: 650;
        margin: 38px 0 8px 0;
    }}
    .step-num {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px; height: 26px;
        border-radius: 50%;
        background: var(--accent-soft);
        border: 1px solid rgba(121,170,255,0.42);
        color: var(--accent);
        font-size: 0.85rem;
        font-weight: 700;
    }}
    .chip {{
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--text-muted);
        border: 1px solid var(--border-strong);
        border-radius: 999px;
        padding: 2px 9px;
    }}

    /* ---------- Result panel ---------- */
    .result {{
        background: linear-gradient(135deg, rgba(121,170,255,0.14), rgba(255,255,255,0.025) 48%, var(--surface));
        border: 1px solid rgba(185,207,251,0.34);
        border-radius: 16px;
        padding: 34px 24px;
        text-align: center;
        box-shadow: 0 22px 55px rgba(0, 0, 0, 0.20);
    }}
    .result-label {{
        color: var(--text-muted);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .result-value {{
        font-size: 2.9rem;
        font-weight: 750;
        letter-spacing: -0.03em;
        margin: 6px 0 2px 0;
    }}
    .result-meta {{ color: var(--text-muted); font-size: 0.9rem; }}
    .result-meta b {{ color: var(--text); font-weight: 600; }}

    /* ---------- Feature rows ---------- */
    .feature {{
        border-left: 2px solid var(--accent);
        padding: 5px 0 5px 14px;
        margin-bottom: 20px;
    }}
    .feature-title {{ font-weight: 650; margin-bottom: 2px; }}
    .feature-desc {{ color: var(--text-muted); font-size: 0.9rem; line-height: 1.5; }}

    /* ---------- Streamlit widget polish ---------- */
    .stButton > button {{
        background: linear-gradient(135deg, #6f9eff, #4e7df2);
        color: #fff !important;
        border: 1px solid rgba(206, 220, 246, 0.38);
        border-radius: 11px;
        min-height: 2.7rem;
        padding: 0.65rem 1rem;
        font-weight: 700;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .stButton > button:hover {{
        background: linear-gradient(135deg, #8ab5ff, #5a84e6);
        color: #fff;
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(79, 127, 241, 0.28);
    }}
    .stButton > button:focus {{ box-shadow: 0 0 0 3px var(--accent-soft); color: #fff; }}

    div[data-baseweb="select"] > div, .stNumberInput input {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 9px !important;
        min-height: 2.75rem;
    }}
    div[data-baseweb="select"] > div:hover {{ border-color: var(--border-strong) !important; }}

    [data-testid="stFileUploaderDropzone"] {{
        background: linear-gradient(135deg, rgba(121, 170, 255, 0.10), rgba(255,255,255,0.025));
        border: 1px dashed rgba(188, 208, 248, 0.48);
        border-radius: 16px;
        padding: 1.5rem 1rem;
    }}
    [data-testid="stMetricValue"] {{ font-size: 2rem; font-weight: 700; }}
    [data-testid="stMetricLabel"] {{ color: var(--text-muted); }}
    .stProgress > div > div > div {{ background: var(--accent); }}

    div[role="radiogroup"] label {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 9px;
        padding: 9px 13px;
        margin-bottom: 7px;
        width: 100%;
        transition: border-color 0.15s ease;
    }}
    div[role="radiogroup"] label:hover {{ border-color: var(--accent); }}

    details {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
    }}
    hr {{ border: none; border-top: 1px solid var(--border); margin: 30px 0; }}

    /* ---------- Sidebar ---------- */
    .brand {{
        display: flex; align-items: center; gap: 9px;
        color: #f7f5f0;
        font-size: 1.02rem; font-weight: 740;
        letter-spacing: -0.025em;
        padding: 8px 0 18px; margin-bottom: 12px;
        border-bottom: 1px solid var(--border);
    }}
    .brand-mark {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 28px; height: 28px; border-radius: 8px;
        background: linear-gradient(135deg, rgba(121,170,255,0.28), rgba(121,170,255,0.06));
        border: 1px solid rgba(121,170,255,0.42);
        box-shadow: 0 6px 14px rgba(0,0,0,0.20);
    }}
    .brand-text {{ display: flex; flex-direction: column; gap: 1px; }}
    .brand-overline {{
        color: #8e9bb0;
        font-size: 0.61rem;
        font-weight: 740;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }}
    .side-note {{ color: var(--text-muted); font-size: 0.8rem; line-height: 1.55; }}

    /* ---------- Sidebar nav: icon-labelled radio row instead of a stacked
       radio group, and full-width tap targets on touch screens ---------- */
    [data-testid="stSidebar"] div[role="radiogroup"] label {{
        display: flex; align-items: center; gap: 10px;
        padding: 12px 14px;
        border-radius: 12px;
        margin: 0 0 0.45rem;
        background: transparent;
        border-color: transparent;
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{ display: none; }}
    [data-testid="stSidebar"] div[role="radiogroup"] label p {{ font-size: 0.91rem; font-weight: 620; }}
    [data-testid="stSidebar"] .stRadio label[data-checked="true"] {{
        background: rgba(121, 170, 255, 0.13);
        border-color: rgba(185, 207, 251, 0.40);
    }}
    [data-testid="stSidebar"] .stRadio label:has(input:checked) {{
        background: linear-gradient(90deg, rgba(121,170,255,0.16), rgba(121,170,255,0.04));
        border-color: rgba(185,207,251,0.35);
        box-shadow: inset 3px 0 0 var(--accent);
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {{
        color: #a5abb6 !important;
        font-size: 0.7rem;
        font-weight: 750;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }}

    /* ---------- Fluid type + spacing ---------- */
    .page-title {{ font-size: clamp(2rem, 1.4rem + 2.2vw, 2.8rem); }}
    .page-sub {{ font-size: clamp(0.9rem, 0.85rem + 0.3vw, 1rem); }}
    .result-value {{ font-size: clamp(2rem, 1.5rem + 2vw, 3rem); word-break: break-word; }}
    .step {{ font-size: clamp(1rem, 0.9rem + 0.4vw, 1.15rem); flex-wrap: wrap; }}

    /* ---------- Responsive breakpoints ---------- */
    @media (max-width: 900px) {{
        .block-container {{ padding-left: 1rem; padding-right: 1rem; padding-top: 1.5rem; }}
        .stat-card {{ padding: 16px 18px; }}
        .result {{ padding: 22px 16px; }}

        /* Streamlit's sidebar shares flex space with the content even when the
           user opens it on a phone, squeezing the page to a sliver. Taking it out
           of flow makes it overlay on top instead, so the content underneath
           keeps its full width regardless of whether the drawer is open. */
        [data-testid="stSidebar"] {{
            position: fixed !important;
            top: 0; left: 0; height: 100vh !important;
            z-index: 999;
            box-shadow: 4px 0 28px rgba(0,0,0,0.55);
        }}
    }}
    @media (max-width: 640px) {{
        .block-container {{ padding: 1.25rem 0.85rem 2.5rem; }}
        .page-kicker {{ margin-bottom: 7px; }}
        .page-title {{ line-height: 1.12; }}
        .page-sub {{ margin-bottom: 24px; line-height: 1.55; }}
        .hero-shell {{
            display: block;
            min-height: auto;
            padding: 1.55rem 1.25rem 0;
            border-radius: 18px;
        }}
        .hero-art {{ min-height: 150px; margin-top: 0.15rem; }}
        .car-silhouette {{ max-width: 270px; }}
        .filter-toolbar {{ flex-direction: column; gap: 0.7rem; }}
        [data-testid="stVerticalBlockBorderWrapper"]:has(.filter-toolbar) {{ padding: 1rem; border-radius: 14px; }}
        .analytics-metric {{ min-height: 150px; }}
        .step-num {{ width: 22px; height: 22px; font-size: 0.75rem; }}
        .result-meta {{ font-size: 0.82rem; line-height: 1.6; }}
        [data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
        .card, .stat-card {{ border-radius: 14px; }}
        [data-testid="stFileUploaderDropzone"] {{ padding: 1rem 0.5rem; }}
        [data-testid="stDownloadButton"] button {{ width: 100%; min-height: 2.9rem; }}
    }}

    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{ transition: none !important; scroll-behavior: auto !important; }}
    }}

    /* ---------- Tables & dataframes ---------- */
    [data-testid="stDataFrame"] {{
        border: 1px solid var(--border);
        border-radius: var(--radius);
        overflow: hidden;
    }}

    /* ---------- Tabs ---------- */
    div[data-baseweb="tab-list"] {{
        overflow-x: auto; overflow-y: hidden; flex-wrap: nowrap;
        scrollbar-width: thin;
    }}
    button[data-baseweb="tab"] {{
        color: var(--text-muted) !important;
        font-weight: 600;
        white-space: nowrap;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{ color: var(--text) !important; }}
    div[data-baseweb="tab-highlight"] {{ background-color: var(--accent) !important; }}
    div[data-baseweb="tab-border"] {{ background-color: var(--border) !important; }}

    /* ---------- Alerts ---------- */
    div[data-testid="stAlert"] {{ border-radius: var(--radius); border: 1px solid var(--border); }}

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: var(--surface-2); border-radius: 8px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--border-strong); }}

    footer, #MainMenu {{ visibility: hidden; }}
    </style>
""", unsafe_allow_html=True)

# Charts inherit the same palette so they don't look bolted on.
pio.templates["carapp"] = pio.templates["plotly_dark"]
pio.templates["carapp"].layout.update(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system, Segoe UI, Roboto, sans-serif", color="#e8ecf3", size=13),
    colorway=[ACCENT, "#22c55e", "#f59e0b", "#a78bfa", "#ec4899"],
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
    margin=dict(t=54, b=40, l=40, r=20),
)
pio.templates.default = "carapp"

# =============================================================
# SIDEBAR
# =============================================================
st.sidebar.markdown(
    """
    <div class='brand'>
        <span class='brand-mark'>
            <svg width='17' height='17' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'>
                <path d='m4 15 1.8-5.2c.3-1 1.2-1.7 2.3-1.7h7.8c1.1 0 2 .7 2.3 1.7L20 15'/>
                <path d='M3 15h18v3c0 .8-.7 1.5-1.5 1.5h-1c-.8 0-1.5-.7-1.5-1.5V17H6v1c0 .8-.7 1.5-1.5 1.5h-1C3.7 19.5 3 18.8 3 18v-3Z'/>
                <circle cx='7.5' cy='15.5' r='1'/><circle cx='16.5' cy='15.5' r='1'/>
            </svg>
        </span>
        <span class='brand-text'><span>Car Price AI</span><span class='brand-overline'>AI on Wheels</span></span>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Workspace",
    ["Overview", "Market analytics", "Value your car"],
    index=0,
)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown(
    f"""
    <div class='card' style='padding:16px 18px;'>
        <div style='font-weight:650; margin-bottom:6px;'>About this tool</div>
        <div class='side-note'>
            Valuations from {df.shape[0]:,} real listings across
            {df['company'].nunique()} brands, with optional photo-based
            damage and model detection.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<div class='side-note' style='text-align:center; margin-top:18px; opacity:0.7;'>"
    "Market intelligence · condition aware</div>",
    unsafe_allow_html=True,
)
def page_header(
    title: str,
    subtitle: str,
    kicker: str = "AI-powered car intelligence",
    primary_pill: str = "Live market signals",
    secondary_pill: str = "Photo-aware valuation",
):
    st.markdown(
        f"""
        <section class='hero-shell'>
            <div class='hero-copy'>
                <div class='page-kicker'>{kicker}</div>
                <div class='page-title'>{title}</div>
                <div class='page-sub'>{subtitle}</div>
                <div class='hero-pills'>
                    <span class='hero-pill'><span class='hero-pill-dot'></span>{primary_pill}</span>
                    <span class='hero-pill'>{secondary_pill}</span>
                </div>
            </div>
            <div class='hero-art' aria-hidden='true'>
                <svg class='car-silhouette' viewBox='0 0 540 260' fill='none' xmlns='http://www.w3.org/2000/svg'>
                    <path d='M72 177h31l20-54c6-16 18-28 35-34l87-31c20-7 42-9 63-5l68 12c18 3 35 12 47 27l44 53h25c20 0 36 16 36 36v16H72v-20Z' fill='url(#body)' stroke='#BDD2FF' stroke-width='3'/>
                    <path d='m160 100 49-20c15-6 32-8 48-5l35 7-8 42H145l15-24Z' fill='#0B101A' stroke='#91B7FF' stroke-width='3'/>
                    <path d='m305 84 39 7c14 3 26 10 35 22l11 14h-85V84Z' fill='#0B101A' stroke='#91B7FF' stroke-width='3'/>
                    <path d='M77 177h358' stroke='#F7F5F0' stroke-opacity='.45' stroke-width='3'/>
                    <path d='M108 156h27' stroke='#E7F0FF' stroke-width='7' stroke-linecap='round'/>
                    <path d='M412 154h20' stroke='#90B9FF' stroke-width='7' stroke-linecap='round'/>
                    <circle cx='165' cy='197' r='35' fill='#090A0C' stroke='#BDD2FF' stroke-width='5'/>
                    <circle cx='165' cy='197' r='14' fill='#526B9E'/>
                    <circle cx='389' cy='197' r='35' fill='#090A0C' stroke='#BDD2FF' stroke-width='5'/>
                    <circle cx='389' cy='197' r='14' fill='#526B9E'/>
                    <path d='M18 228h504' stroke='url(#road)' stroke-width='2' stroke-dasharray='12 15'/>
                    <defs>
                        <linearGradient id='body' x1='90' y1='53' x2='440' y2='220' gradientUnits='userSpaceOnUse'><stop stop-color='#668CD7'/><stop offset='.48' stop-color='#263B63'/><stop offset='1' stop-color='#17243B'/></linearGradient>
                        <linearGradient id='road' x1='18' y1='228' x2='522' y2='228' gradientUnits='userSpaceOnUse'><stop stop-color='#79AAFF' stop-opacity='0'/><stop offset='.5' stop-color='#DCE8FF'/><stop offset='1' stop-color='#79AAFF' stop-opacity='0'/></linearGradient>
                    </defs>
                </svg>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


# =============================================================
# PAGE 1: HOME
# =============================================================
if page == "Overview":
    page_header(
        "Know what your car is really worth",
        "An AI valuation tool trained on real Indian resale listings. "
        "Add photos and it will also check the car's condition and identify the model.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
            <div class='market-card'>
                <div class='market-card-top'>
                    <span class='market-card-icon'>
                        <svg viewBox='0 0 24 24' fill='none' stroke-width='1.8'><path d='M3 18.5V9.8c0-.7.4-1.3 1-1.6l7-3.2c.6-.3 1.3-.3 1.9 0l7 3.2c.6.3 1 1 1 1.6v8.7'/><path d='M3 14h18M7 18v-3.3m5 3.3v-3.3m5 3.3v-3.3'/><path d='M2 19.5h20'/></svg>
                    </span>
                    <span class='market-card-tag'>Market data</span>
                </div>
                <div class='market-card-value'>{df.shape[0]:,}<span style='color:#8fb5ff'>+</span></div>
                <div class='market-card-label'>Listings analysed</div>
                <div class='market-card-footer'>
                    <span>Live resale benchmark</span>
                    <span class='mini-bars'><span style='--bar:35%'></span><span style='--bar:55%'></span><span style='--bar:74%'></span><span style='--bar:100%'></span></span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class='market-card market-card--brands'>
                <div class='market-card-top'>
                    <span class='market-card-icon'>
                        <svg viewBox='0 0 24 24' fill='none' stroke-width='1.8'><circle cx='12' cy='12' r='8.5'/><path d='M3.8 12h16.4M12 3.5c2.1 2.3 3.1 5.1 3.1 8.5S14.1 18.2 12 20.5C9.9 18.2 8.9 15.4 8.9 12S9.9 5.8 12 3.5'/></svg>
                    </span>
                    <span class='market-card-tag'>Coverage</span>
                </div>
                <div class='market-card-value'>{df['company'].nunique()}<span style='color:#c3abff'>+</span></div>
                <div class='market-card-label'>Brands covered</div>
                <div class='market-card-footer'>
                    <span>India's popular makes</span>
                    <span class='mini-bars'><span style='--bar:42%'></span><span style='--bar:66%'></span><span style='--bar:82%'></span><span style='--bar:100%'></span></span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class='market-card market-card--models'>
                <div class='market-card-top'>
                    <span class='market-card-icon'>
                        <svg viewBox='0 0 24 24' fill='none' stroke-width='1.8'><path d='m5 15 1.8-5.2c.3-1 1.2-1.7 2.3-1.7h5.8c1.1 0 2 .7 2.3 1.7L19 15'/><path d='M4 15h16v3c0 .8-.7 1.5-1.5 1.5h-1c-.8 0-1.5-.7-1.5-1.5V17H8v1c0 .8-.7 1.5-1.5 1.5h-1C4.7 19.5 4 18.8 4 18v-3Z'/><circle cx='7.5' cy='15.5' r='.8' fill='currentColor'/><circle cx='16.5' cy='15.5' r='.8' fill='currentColor'/></svg>
                    </span>
                    <span class='market-card-tag'>Precision</span>
                </div>
                <div class='market-card-value'>{df['name'].nunique():,}<span style='color:#7ae2d0'>+</span></div>
                <div class='market-card-label'>Model variants</div>
                <div class='market-card-footer'>
                    <span>Specific model matching</span>
                    <span class='mini-bars'><span style='--bar:30%'></span><span style='--bar:60%'></span><span style='--bar:76%'></span><span style='--bar:100%'></span></span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("<div class='section-eyebrow'>Start with the market</div>", unsafe_allow_html=True)
        st.markdown("<div class='home-section-title'>Price your car with context</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='home-section-sub'>Select a brand to see the resale signal before you set an asking price or start negotiating.</div>",
            unsafe_allow_html=True,
        )

        brands = sorted(df['company'].unique())
        selected_brand = st.selectbox(
            "Brand", brands,
            index=brands.index("Hyundai") if "Hyundai" in brands else 0,
        )

        brand_data = df[df['company'] == selected_brand]
        price_low, price_high = brand_data['Price'].quantile([0.25, 0.75])
        st.markdown(f"""
            <div class='brand-insight-card'>
                <div class='brand-card-label'>Estimated average asking price · {selected_brand}</div>
                <div class='brand-card-value'>₹ {brand_data['Price'].mean():,.0f}</div>
                <div class='brand-card-note'>A useful starting point before you list or negotiate.</div>
                <div class='brand-card-stats'>
                    <div class='brand-card-stat'>
                        <div class='brand-card-stat-label'>Typical price band</div>
                        <div class='brand-card-stat-value'>₹ {price_low:,.0f} – ₹ {price_high:,.0f}</div>
                    </div>
                    <div class='brand-card-stat'>
                        <div class='brand-card-stat-label'>Comparable listings</div>
                        <div class='brand-card-stat-value'>{len(brand_data):,}+ cars</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("<div class='section-eyebrow'>Everyday decisions</div>", unsafe_allow_html=True)
        st.markdown("<div class='home-section-title'>Built for the real journey</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='home-section-sub'>Use the same market signal whether you are selling, buying, or assessing a car after an incident.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("""
            <div class='use-case-list'>
                <div class='use-case'>
                    <span class='use-case-icon'>
                        <svg viewBox='0 0 24 24' fill='none' stroke-width='1.9'><path d='M4 20V7.5c0-.8.7-1.5 1.5-1.5h9L20 11.5V20c0 .6-.4 1-1 1H5c-.6 0-1-.4-1-1Z'/><path d='M14 6v6h6M8 16h8M8 12h3'/></svg>
                    </span>
                    <div><div class='use-case-title'>Selling your car</div><div class='use-case-desc'>Set a realistic asking price and avoid leaving money on the table.</div></div>
                </div>
                <div class='use-case'>
                    <span class='use-case-icon'>
                        <svg viewBox='0 0 24 24' fill='none' stroke-width='1.9'><path d='M3 10.5h18M5 6h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2Z'/><path d='M7 15h3m4 0h3'/></svg>
                    </span>
                    <div><div class='use-case-title'>Buying used</div><div class='use-case-desc'>Compare a seller's expectation with the prevailing resale market.</div></div>
                </div>
                <div class='use-case'>
                    <span class='use-case-icon'>
                        <svg viewBox='0 0 24 24' fill='none' stroke-width='1.9'><path d='M5 19 19 5M6.5 5.5l12 12M4 8.5l3-3 4.2 4.2-3 3L4 8.5Zm8.8 8.8 3-3 4.2 4.2-3 3-4.2-4.2Z'/></svg>
                    </span>
                    <div><div class='use-case-title'>After a repair or bump</div><div class='use-case-desc'>Upload photos to understand how visible condition can affect value.</div></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# =============================================================
# PAGE 2: EDA DASHBOARD
# =============================================================
elif page == "Market analytics":
    page_header(
        "See the market before you make a move",
        "Narrow the view to the cars that matter, then compare demand, pricing and age trends with confidence.",
        kicker="Market intelligence",
        primary_pill="Flexible market filters",
        secondary_pill="Decision-ready trends",
    )

    with st.container(border=True):
        st.markdown(
            """
            <div class='filter-toolbar'>
                <div>
                    <div class='filter-toolbar-title'>Build your market view</div>
                    <div class='filter-toolbar-note'>Refine by brand, model year and fuel type. Your insights update instantly.</div>
                </div>
                <span class='filter-toolbar-status'>Filters ready</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        f1, f2, f3 = st.columns(3)
        with f1:
            all_brands = sorted(df['company'].unique())
            selected_brands_filter = st.multiselect(
                "Brands", all_brands, default=[], placeholder="All brands"
            )
        with f2:
            min_yr, max_yr = int(df['year'].min()), int(df['year'].max())
            selected_years_filter = st.slider(
                "Year range", min_value=min_yr, max_value=max_yr, value=(min_yr, max_yr)
            )
        with f3:
            all_fuels = sorted(df['fuel_type'].unique())
            selected_fuels_filter = st.multiselect(
                "Fuel types", all_fuels, default=[], placeholder="All fuel types"
            )

        filtered_df = df.copy()
        if selected_brands_filter:
            filtered_df = filtered_df[filtered_df['company'].isin(selected_brands_filter)]
        filtered_df = filtered_df[
            (filtered_df['year'] >= selected_years_filter[0])
            & (filtered_df['year'] <= selected_years_filter[1])
        ]
        if selected_fuels_filter:
            filtered_df = filtered_df[filtered_df['fuel_type'].isin(selected_fuels_filter)]

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    avg_price = filtered_df['Price'].mean() if not filtered_df.empty else 0
    for col, value, kicker, label, foot, icon in [
        (k1, f"{filtered_df.shape[0]:,}", "Filtered market", "matching listings", "Updates with your view",
         "<svg viewBox='0 0 24 24' fill='none' stroke-width='1.9'><path d='M4 4h6v6H4V4Zm10 0h6v6h-6V4ZM4 14h6v6H4v-6Zm10 0h6v6h-6v-6Z'/></svg>"),
        (k2, f"{filtered_df['company'].nunique()}", "Brand mix", "makes in this view", "Compare brand by brand",
         "<svg viewBox='0 0 24 24' fill='none' stroke-width='1.9'><circle cx='12' cy='12' r='8.5'/><path d='M3.8 12h16.4M12 3.5c2.1 2.3 3.1 5.1 3.1 8.5S14.1 18.2 12 20.5C9.9 18.2 8.9 15.4 8.9 12S9.9 5.8 12 3.5'/></svg>"),
        (k3, f"{filtered_df['fuel_type'].nunique()}", "Powertrain mix", "fuel types present", "See demand by fuel type",
         "<svg viewBox='0 0 24 24' fill='none' stroke-width='1.9'><path d='M7 20h10M8 16h8M9 3h6l2 6.5c.4 1.5-.7 3-2.2 3H9.2c-1.5 0-2.6-1.5-2.2-3L9 3Z'/><path d='M12 12.5V16'/></svg>"),
        (k4, f"₹ {avg_price/100000:.1f}L", "Market pricing", "average asking price", "Across this filtered view",
         "<svg viewBox='0 0 24 24' fill='none' stroke-width='1.9'><path d='M4 19V5m0 14h16'/><path d='m7 15 3.5-4 3 2 4.5-6'/><path d='M15 7h3v3'/></svg>"),
    ]:
        with col:
            st.markdown(
                f"""
                <div class='analytics-metric'>
                    <div class='analytics-metric-head'>
                        <span class='analytics-metric-icon'>{icon}</span>
                        <span class='analytics-metric-kicker'>{kicker}</span>
                    </div>
                    <div class='analytics-metric-value'>{value}</div>
                    <div class='analytics-metric-label'>{label}</div>
                    <div class='analytics-metric-foot'>{foot}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<hr>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Price distribution", "Trend by year", "Top models", "Fuel mix"]
    )

    with tab1:
        if not filtered_df.empty:
            fig = px.histogram(filtered_df, x="Price", nbins=40, title="How prices are distributed")
            st.plotly_chart(fig, width="stretch")
            st.caption("Where most resale values cluster.")
        else:
            st.info("No listings match the current filters.")

    with tab2:
        if not filtered_df.empty:
            year_trend = filtered_df.groupby("year")["Price"].mean().reset_index()
            fig2 = px.line(year_trend, x="year", y="Price", markers=True,
                           title="Average price by manufacturing year")
            fig2.update_traces(line=dict(width=2.5))
            st.plotly_chart(fig2, width="stretch")
            st.caption("How value falls away with age.")
        else:
            st.info("No listings match the current filters.")

    with tab3:
        if not filtered_df.empty and "name" in filtered_df.columns:
            top_models = filtered_df.groupby("name")["Price"].mean().nlargest(5).reset_index()
            fig3 = px.bar(top_models, x="name", y="Price", text="Price",
                          title="Highest average resale value")
            fig3.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside",
                               marker_color=ACCENT)
            fig3.update_layout(xaxis_title="", yaxis_title="Average price")
            st.plotly_chart(fig3, width="stretch")
        else:
            st.info("Not enough data for this chart.")

    with tab4:
        if not filtered_df.empty:
            fig4 = px.pie(filtered_df, names="fuel_type", hole=0.55, title="Share of listings by fuel type")
            fig4.update_traces(textinfo="percent+label")
            st.plotly_chart(fig4, width="stretch")
        else:
            st.info("No listings match the current filters.")

# =============================================================
# PAGE 3: PRICE PREDICTION
# =============================================================
elif page == "Value your car":
    page_header(
        "Value your car",
        "Add photos for a condition-adjusted estimate, or just fill in the details below.",
    )

    # ── STEP 1 ── Photos: condition assessment + auto brand/model detection
    st.markdown(
        "<div class='step'><span class='step-num'>1</span>Add photos"
        "<span class='chip'>optional</span></div>",
        unsafe_allow_html=True,
    )
    st.caption("Several angles give a more complete damage check and a more reliable model match.")
    uploaded_images = st.file_uploader(
        "Car photos",
        type=["jpg", "jpeg", "jfif", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    condition_result = None
    auto_company = None
    auto_name = None
    auto_fuel = None

    is_new_upload = False
    upload_identity = None
    if uploaded_images:
        upload_identity = "|".join(sorted(f"{f.name}-{f.size}" for f in uploaded_images))
        # "Have we seen these exact photos before?" -- kept separate from which
        # candidate has been applied, so re-running for a candidate change doesn't
        # look like a fresh upload and reset the user's choice.
        is_new_upload = st.session_state.get("seen_upload") != upload_identity

        images = [Image.open(f) for f in uploaded_images]
        with st.spinner(f"Analysing {len(images)} photo{'s' if len(images) > 1 else ''}..."):
            condition_result = assess_condition_multi(images)
            recognized, cars_seen = recognize_from_images(images)

        score = condition_result["condition_score"]

        # ── STEP 2 ── What the photos told us
        st.markdown(
            "<div class='step'><span class='step-num'>2</span>What we found</div>",
            unsafe_allow_html=True,
        )

        col_score, col_issues = st.columns([1, 1.4])
        with col_score:
            st.metric("Condition score", f"{score}/100", condition_label(score))
            st.progress(score / 100)
            st.caption(
                f"From {len(images)} photo{'s' if len(images) > 1 else ''}"
                + (f" · {cars_seen} recognised as a car" if cars_seen != len(images) else "")
            )
        with col_issues:
            if condition_result["summary"]:
                st.markdown("**Damage detected**")
                for item in condition_result["summary"]:
                    times = f" ×{item['count']}" if item["count"] > 1 else ""
                    st.markdown(f"- {item['class']}{times} · up to {item['confidence']:.0%} confidence")
            else:
                st.markdown("**No visible damage detected**")
                st.caption("Fine scratches, paint chips and corrosion are still hard for the model to spot reliably.")

        with st.expander(f"View analysed photos ({len(images)})", expanded=False):
            cols = st.columns(min(3, len(images)))
            for i, res in enumerate(condition_result["per_image"]):
                with cols[i % len(cols)]:
                    n_found = len(res["detections"])
                    st.image(res["annotated_image"], width="stretch",
                             caption=f"Photo {i+1} — {n_found} issue{'s' if n_found != 1 else ''}")

        if recognized:
            # Only offer candidates whose brand actually exists in our price data.
            usable = []
            for cand in recognized:
                m = match_to_price_dataset(cand["brand"], cand["model"], df)
                if m:
                    usable.append((cand, m))

            if usable:
                st.markdown("**Which of these is your car?**")
                st.caption(
                    "Our best guess is first. Picking the right one fills in the details below — "
                    "if none look right, choose the last option and select your car manually."
                )

                def candidate_label(cand):
                    # Reference labels already carry the brand ("Tata Indica",
                    # "HYUNDAI Atos"), so don't prefix it again -- just normalise the
                    # brand's casing to match our own naming.
                    model_name = cand["model"]
                    if model_name.lower().startswith(cand["brand"].lower()):
                        model_name = cand["brand"] + model_name[len(cand["brand"]):]
                    else:
                        model_name = f"{cand['brand']} {model_name}"
                    return f"{model_name}  ·  {cand['similarity']:.0%} match"

                options = [candidate_label(c) for c, _ in usable]
                options.append("None of these — I'll pick manually")

                # Reset the choice back to the top guess whenever new photos arrive
                # (and only then -- otherwise picking a candidate would be undone).
                if is_new_upload:
                    st.session_state["candidate_pick"] = options[0]
                    st.session_state["seen_upload"] = upload_identity
                if st.session_state.get("candidate_pick") not in options:
                    st.session_state.pop("candidate_pick", None)

                picked = st.radio(
                    "Candidates", options, key="candidate_pick", label_visibility="collapsed"
                )

                picked_idx = options.index(picked)
                if picked_idx < len(usable):
                    match = usable[picked_idx][1]
                    auto_company = match["company"]
                    auto_name = match["name"]
                    if auto_name is not None:
                        auto_fuel = df[(df['company'] == auto_company) & (df['name'] == auto_name)]['fuel_type'].iloc[0]
                    else:
                        # Brand matched but no confident specific model -- still pick a
                        # fuel type that actually exists for this brand, otherwise the
                        # fuel filter below can exclude it and silently drop the brand too.
                        auto_fuel = df[df['company'] == auto_company]['fuel_type'].mode().iloc[0]

                    # Re-apply whenever the photos OR the chosen candidate change. Selectboxes
                    # ignore a changed `index=` once they hold a value, so write session_state
                    # directly, before those widgets are created below.
                    applied_key = f"{upload_identity}#{picked_idx}"
                    if st.session_state.get("autofill_applied_for") != applied_key:
                        st.session_state["fuel_type_select"] = auto_fuel
                        st.session_state["brand_select"] = auto_company
                        if auto_name is not None:
                            st.session_state["model_select"] = auto_name
                        else:
                            # Clear any leftover model from a previous photo so it can't
                            # linger under the new brand.
                            st.session_state.pop("model_select", None)
                        st.session_state["autofill_applied_for"] = applied_key
            else:
                top = recognized[0]
                st.warning(
                    f"Looks closest to a **{top['brand']} {top['model']}**, but that brand isn't in our "
                    "price data — please pick your car below manually."
                )
        else:
            st.warning("These don't look like photos of a car, so we skipped model detection. Please pick your car below.")

    # ── STEP 3 ── Car details
    step_no = 3 if uploaded_images else 2
    st.markdown(
        f"<div class='step'><span class='step-num'>{step_no}</span>Confirm the details</div>",
        unsafe_allow_html=True,
    )

    fuel_types = sorted(df['fuel_type'].unique())

    col1, col2 = st.columns(2)
    with col2:
        kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=100)
        fuel_type = st.selectbox("Fuel Type", fuel_types, key="fuel_type_select")

    # Only offer brands/models that actually exist with the selected fuel type
    fuel_filtered_df = df[df['fuel_type'] == fuel_type]

    with col1:
        companies = sorted(fuel_filtered_df['company'].unique())
        if st.session_state.get("brand_select") not in companies:
            st.session_state.pop("brand_select", None)
        company = st.selectbox("Car Brand", companies, key="brand_select")
        # Dynamically filter car models based on the selected brand + fuel type
        valid_car_names = sorted(fuel_filtered_df[fuel_filtered_df['company'] == company]['name'].unique())
        if st.session_state.get("model_select") not in valid_car_names:
            st.session_state.pop("model_select", None)
        name = st.selectbox("Car Model Name", valid_car_names, key="model_select")
        year = st.number_input("Year of Purchase", min_value=1995, max_value=2025, value=2019)

    # ── Valuation ── recalculated live as the inputs change, so there's no
    # "press the button again" step after tweaking a field.
    st.markdown(
        f"<div class='step'><span class='step-num'>{step_no + 1}</span>Your valuation</div>",
        unsafe_allow_html=True,
    )

    try:
        car = {
            'name': name,
            'company': company,
            'year': year,
            'kms_driven': kms_driven,
            'fuel_type': fuel_type,
        }
        input_df = pd.DataFrame([car])

        est = predict_with_range(pipe, input_df)
        multiplier = condition_result["price_multiplier"] if condition_result else 1.0
        final_price = est["price"] * multiplier
        low, high = est["low"] * multiplier, est["high"] * multiplier

        conf_tone = {"High": "#22c55e", "Moderate": "#f59e0b", "Low": "#ef4444"}[est["confidence"]]
        cond_line = ""
        if condition_result:
            cs = condition_result["condition_score"]
            cond_line = (
                f"&nbsp;·&nbsp; Condition <b>{cs}/100 ({condition_label(cs)})</b>"
                f"&nbsp;·&nbsp; Before condition <b>₹ {est['price']:,.0f}</b>"
            )

        st.markdown(f"""
            <div class='result'>
                <div class='result-label'>
                    {'Condition-adjusted value' if condition_result else 'Estimated market value'}
                </div>
                <div class='result-value'>₹ {final_price:,.0f}</div>
                <div class='result-meta'>
                    8 in 10 similar cars sell within
                    <b>₹ {low:,.0f} – ₹ {high:,.0f}</b>
                    &nbsp;·&nbsp; Confidence for this car
                    <b style='color:{conf_tone};'>{est['confidence']}</b>
                    {cond_line}
                </div>
            </div>
        """, unsafe_allow_html=True)

        impacts = damage_impact(condition_result, est["price"])

        t_market, t_depr, t_similar, t_cond = st.tabs(
            ["Market position", "How it ages", "Similar cars", "Condition detail"]
        )

        # --- Where this car sits against comparable listings ---
        with t_market:
            pos = market_position(df, company, final_price)
            fig = px.histogram(pos["pool"], nbins=45, title=f"{company} listings by price")
            fig.add_vline(x=final_price, line_width=2.5, line_color=ACCENT,
                          annotation_text="Your car", annotation_position="top")
            fig.update_layout(showlegend=False, xaxis_title="Price (₹)", yaxis_title="Listings")
            st.plotly_chart(fig, width="stretch")
            st.caption(
                f"Priced above about **{pos['percentile']:.0f}%** of {company} listings. "
                f"Typical {company} listing sits at ₹ {pos['median']:,.0f}."
            )

        # --- Value against manufacturing year ---
        with t_depr:
            curve = depreciation_curve(pipe, car)
            fig2 = px.line(curve, x="year", y="price", markers=True,
                           title="What the same car is worth by model year")
            fig2.add_vline(x=year, line_width=2, line_dash="dot", line_color=ACCENT,
                           annotation_text="Yours", annotation_position="top")
            fig2.update_layout(xaxis_title="Manufacturing year", yaxis_title="Estimated price (₹)")
            st.plotly_chart(fig2, width="stretch")
            newer = curve[curve["year"] > year]["price"]
            if len(newer):
                st.caption(
                    f"A model year newer is worth about ₹ {newer.iloc[0] - est['price']:,.0f} more, "
                    "based on how the model values age."
                )

        # --- Real comparable listings ---
        with t_similar:
            sims = similar_listings(df, company, name, year, kms_driven)
            if sims.empty:
                st.info("No comparable listings for this car in the dataset.")
            else:
                show = sims.rename(columns={
                    "name": "Model", "year": "Year", "kms_driven": "Kilometers",
                    "fuel_type": "Fuel", "Price": "Listed price",
                })
                st.dataframe(
                    show.style.format({"Listed price": "₹ {:,.0f}", "Kilometers": "{:,.0f}"}),
                    width="stretch", hide_index=True,
                )
                st.caption("Actual listings closest to your car on age and distance driven.")

        # --- Condition breakdown ---
        with t_cond:
            if not condition_result:
                st.info("Add photos in step 1 to see a condition assessment and its price impact.")
            else:
                g1, g2 = st.columns([1, 1.3])
                with g1:
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=condition_result["condition_score"],
                        number={"suffix": "/100"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": ACCENT},
                            "steps": [
                                {"range": [0, 55], "color": "rgba(239,68,68,0.25)"},
                                {"range": [55, 75], "color": "rgba(245,158,11,0.25)"},
                                {"range": [75, 100], "color": "rgba(34,197,94,0.25)"},
                            ],
                        },
                    ))
                    gauge.update_layout(height=240, margin=dict(t=10, b=10, l=20, r=20))
                    st.plotly_chart(gauge, width="stretch")
                with g2:
                    if impacts:
                        st.markdown("**Estimated impact on value**")
                        for imp in impacts:
                            times = f" ×{imp['count']}" if imp["count"] > 1 else ""
                            st.markdown(f"- {imp['class']}{times} — about **−₹ {imp['cost']:,.0f}**")
                        st.caption(
                            f"Total condition adjustment: −₹ {est['price'] - final_price:,.0f}"
                        )
                    else:
                        st.markdown("**No visible damage detected**")
                        st.caption("No condition deduction was applied to the estimate.")

        st.download_button(
            "Download valuation report",
            data=build_report(car, {**est, "price": final_price, "low": low, "high": high},
                              condition_result, impacts),
            file_name=f"valuation-{name.replace(' ', '-').lower()}.txt",
            mime="text/plain",
        )

    except Exception as e:
        st.error(f"Could not calculate a valuation: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption(
        "Estimates come from an Extra Trees model trained on Indian resale listings. "
        "Checked against held-out cars it wasn't trained on, half of estimates land "
        "within 19% of the real price and the quoted range contains it 8 times out of 10. "
        "Treat it as a guide: condition, service history and location all move the "
        "final number."
    )
