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
ACCENT = "#4f8df7"
SURFACE = "#141922"
BG = "#0b0e14"
TEXT_MUTED = "#8b97a8"

st.markdown(f"""
    <style>
    :root {{
        --accent: {ACCENT};
        --accent-soft: rgba(79, 141, 247, 0.12);
        --bg: {BG};
        --surface: {SURFACE};
        --surface-2: #1b2130;
        --border: rgba(255, 255, 255, 0.08);
        --border-strong: rgba(255, 255, 255, 0.14);
        --text: #e8ecf3;
        --text-muted: {TEXT_MUTED};
        --radius: 12px;
    }}

    /* ---------- Base ---------- */
    [data-testid="stAppViewContainer"] {{ background: var(--bg); }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"] {{
        background: #0e1218;
        border-right: 1px solid var(--border);
    }}
    .block-container {{ padding-top: 2.5rem; max-width: 1180px; }}

    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
        color: var(--text);
    }}
    h1, h2, h3, h4 {{ color: var(--text); font-weight: 650; letter-spacing: -0.01em; }}
    p, span, label, li {{ color: var(--text); }}
    a {{ color: var(--accent); }}

    /* ---------- Page header ---------- */
    .page-title {{
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0 0 6px 0;
    }}
    .page-sub {{
        color: var(--text-muted);
        font-size: 1rem;
        margin: 0 0 28px 0;
        max-width: 640px;
    }}

    /* ---------- Cards ---------- */
    .card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 22px 24px;
    }}
    .stat-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 20px 22px;
        height: 100%;
        transition: border-color 0.18s ease, transform 0.18s ease;
    }}
    .stat-card:hover {{ border-color: var(--border-strong); transform: translateY(-2px); }}
    .stat-value {{ font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em; }}
    .stat-label {{ color: var(--text-muted); font-size: 0.85rem; margin-top: 4px; }}

    /* ---------- Numbered steps ---------- */
    .step {{
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 1.15rem;
        font-weight: 650;
        margin: 34px 0 4px 0;
    }}
    .step-num {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px; height: 26px;
        border-radius: 7px;
        background: var(--accent-soft);
        border: 1px solid rgba(79,141,247,0.35);
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
        background: linear-gradient(180deg, var(--surface-2), var(--surface));
        border: 1px solid rgba(79,141,247,0.35);
        border-radius: 16px;
        padding: 30px 24px;
        text-align: center;
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
        padding: 2px 0 2px 14px;
        margin-bottom: 18px;
    }}
    .feature-title {{ font-weight: 650; margin-bottom: 2px; }}
    .feature-desc {{ color: var(--text-muted); font-size: 0.9rem; line-height: 1.5; }}

    /* ---------- Streamlit widget polish ---------- */
    .stButton > button {{
        background: var(--accent);
        color: #fff;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-weight: 600;
        transition: background 0.15s ease;
    }}
    .stButton > button:hover {{ background: #3d7ae8; color: #fff; }}
    .stButton > button:focus {{ box-shadow: 0 0 0 3px var(--accent-soft); color: #fff; }}

    div[data-baseweb="select"] > div, .stNumberInput input {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 9px !important;
    }}
    div[data-baseweb="select"] > div:hover {{ border-color: var(--border-strong) !important; }}

    [data-testid="stFileUploaderDropzone"] {{
        background: var(--surface);
        border: 1px dashed var(--border-strong);
        border-radius: var(--radius);
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
        font-size: 1.05rem; font-weight: 700;
        padding-bottom: 14px; margin-bottom: 6px;
        border-bottom: 1px solid var(--border);
    }}
    .brand-mark {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 28px; height: 28px; border-radius: 8px;
        background: var(--accent-soft); border: 1px solid rgba(79,141,247,0.35);
    }}
    .side-note {{ color: var(--text-muted); font-size: 0.8rem; line-height: 1.55; }}

    /* ---------- Sidebar nav: icon-labelled radio row instead of a stacked
       radio group, and full-width tap targets on touch screens ---------- */
    [data-testid="stSidebar"] div[role="radiogroup"] label {{
        display: flex; align-items: center; gap: 10px;
        padding: 11px 14px;
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] label p {{ font-size: 0.95rem; }}

    /* ---------- Fluid type + spacing ---------- */
    .page-title {{ font-size: clamp(1.5rem, 1.1rem + 1.6vw, 2.1rem); }}
    .page-sub {{ font-size: clamp(0.9rem, 0.85rem + 0.3vw, 1rem); }}
    .result-value {{ font-size: clamp(2rem, 1.5rem + 2vw, 3rem); word-break: break-word; }}
    .step {{ font-size: clamp(1rem, 0.9rem + 0.4vw, 1.15rem); flex-wrap: wrap; }}

    /* ---------- Responsive breakpoints ---------- */
    @media (max-width: 900px) {{
        .block-container {{ padding-left: 1rem; padding-right: 1rem; padding-top: 1.5rem; }}
        .stat-card {{ padding: 16px 18px; }}
        .result {{ padding: 22px 16px; }}
        div[data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
        div[data-testid="column"] {{ min-width: 100% !important; }}

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
        .step-num {{ width: 22px; height: 22px; font-size: 0.75rem; }}
        .result-meta {{ font-size: 0.82rem; line-height: 1.6; }}
        [data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
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
    "<div class='brand'><span class='brand-mark'>🚗</span> Car Price AI</div>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 EDA Dashboard", "💰 Price Prediction"],
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
    "Built with Streamlit · scikit-learn · CLIP</div>",
    unsafe_allow_html=True,
)


def page_header(title: str, subtitle: str):
    st.markdown(f"<div class='page-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-sub'>{subtitle}</div>", unsafe_allow_html=True)


# =============================================================
# PAGE 1: HOME
# =============================================================
if page == "🏠 Home":
    page_header(
        "Know what your car is really worth",
        "An AI valuation tool trained on real Indian resale listings. "
        "Add photos and it will also check the car's condition and identify the model.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
            <div class='stat-card'>
                <div class='stat-value'>{df.shape[0]:,}</div>
                <div class='stat-label'>Listings analysed</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class='stat-card'>
                <div class='stat-value'>{df['company'].nunique()}</div>
                <div class='stat-label'>Brands covered</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class='stat-card'>
                <div class='stat-value'>{df['name'].nunique():,}</div>
                <div class='stat-label'>Model variants</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("#### Quick brand insight")
        st.caption("See where a brand typically sits in the resale market.")

        brands = sorted(df['company'].unique())
        selected_brand = st.selectbox(
            "Brand", brands,
            index=brands.index("Hyundai") if "Hyundai" in brands else 0,
        )

        brand_data = df[df['company'] == selected_brand]
        st.markdown(f"""
            <div class='card' style='margin-top:12px;'>
                <div class='stat-label'>Average resale value · {selected_brand}</div>
                <div class='stat-value' style='margin-top:6px;'>₹ {brand_data['Price'].mean():,.0f}</div>
                <div class='result-meta' style='margin-top:14px;'>
                    Highest listed <b>₹ {brand_data['Price'].max():,.0f}</b>
                    &nbsp;·&nbsp; From <b>{len(brand_data):,}</b> listings
                </div>
            </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("#### What's inside")
        st.markdown("""
            <div style='margin-top:14px;'>
                <div class='feature'>
                    <div class='feature-title'>Photo-based condition check</div>
                    <div class='feature-desc'>Upload photos and a detection model looks for
                    dents, broken and missing parts, then adjusts the price for condition.</div>
                </div>
                <div class='feature'>
                    <div class='feature-title'>Automatic model detection</div>
                    <div class='feature-desc'>Recognises the car from your photos and suggests
                    the closest matches, so you don't have to hunt through dropdowns.</div>
                </div>
                <div class='feature'>
                    <div class='feature-title'>Market analytics</div>
                    <div class='feature-desc'>Explore price distributions, depreciation by year
                    and fuel-type trends across the dataset.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# =============================================================
# PAGE 2: EDA DASHBOARD
# =============================================================
elif page == "📊 EDA Dashboard":
    page_header(
        "Market analytics",
        "Filter the dataset and explore how prices move with brand, age and fuel type.",
    )

    with st.container():
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
    for col, value, label in [
        (k1, f"{filtered_df.shape[0]:,}", "Listings"),
        (k2, f"{filtered_df['company'].nunique()}", "Brands"),
        (k3, f"{filtered_df['fuel_type'].nunique()}", "Fuel types"),
        (k4, f"₹ {avg_price/100000:.1f}L", "Average price"),
    ]:
        with col:
            st.markdown(
                f"<div class='stat-card'><div class='stat-value'>{value}</div>"
                f"<div class='stat-label'>{label}</div></div>",
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
elif page == "💰 Price Prediction":
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
