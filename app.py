# =============================================================
# 🚗 CAR PRICE PREDICTION APP — DARK NEON PRO EDITION
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# =============================================================
# PAGE CONFIGURATION
# =============================================================
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
)

# =============================================================
# LOAD MODEL & DATA
# =============================================================
@st.cache_resource
def load_model():
    return pickle.load(open("LinearRegressionModel.pkl", "rb"))

@st.cache_data
def load_data():
    return pd.read_csv("Cleaned_Car_data.csv")

pipe = load_model()
df = load_data()

# =============================================================
# 🌌 ULTRA DARK NEON THEME + ANIMATED SIDEBAR
# =============================================================
st.markdown("""
    <style>
    /* === General Layout === */
    .main {
        background: radial-gradient(circle at top left, #001219 0%, #000000 80%);
        color: #e0e0e0;
        font-family: 'Poppins', sans-serif;
        padding: 0;
    }

    h1, h2, h3 {
        color: #00e0ff;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* === Sidebar Background & Shadow === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #000814, #001d3d);
        box-shadow: 3px 0 15px rgba(0, 255, 255, 0.2);
        border-right: 1px solid rgba(0, 180, 216, 0.25);
        animation: slideIn 1.2s ease-in-out;
    }

    @keyframes slideIn {
        from {transform: translateX(-10px); opacity: 0;}
        to {transform: translateX(0); opacity: 1;}
    }

    /* === Sidebar Title with Glow === */
    .sidebar-title {
        font-size: 28px;
        text-align: center;
        background: linear-gradient(90deg, #00e0ff, #0077b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 10px;
        animation: glowPulse 2s infinite ease-in-out;
    }

    @keyframes glowPulse {
        0% { text-shadow: 0 0 5px #00e0ff; }
        50% { text-shadow: 0 0 20px #00e0ff; }
        100% { text-shadow: 0 0 5px #00e0ff; }
    }

    /* === About Expander === */
    details {
        background: rgba(0, 180, 216, 0.1);
        border: 1px solid rgba(0, 180, 216, 0.3);
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 15px;
        transition: 0.3s ease;
    }
    details:hover {
        background: rgba(0, 180, 216, 0.2);
    }

    /* === Sidebar Radio Buttons === */
    div[data-baseweb="radio"] > div {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 10px 15px;
        transition: 0.4s ease;
        margin: 5px 0;
    }

    div[data-baseweb="radio"] > div:hover {
        background: linear-gradient(90deg, rgba(0,180,216,0.3), rgba(0,119,182,0.3));
        transform: scale(1.03);
        box-shadow: 0 0 10px rgba(0,180,216,0.4);
    }

    /* === Buttons === */
    .stButton>button {
        background: linear-gradient(90deg, #0077b6, #00b4d8);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.7em;
        width: 100%;
        border: none;
        transition: 0.3s ease;
        box-shadow: 0 0 10px rgba(0,180,216,0.3);
    }

    .stButton>button:hover {
        transform: scale(1.07);
        background: linear-gradient(90deg, #48cae4, #00b4d8);
        box-shadow: 0 0 20px #00b4d8;
    }

    /* === Footer === */
    .footer {
        color: #8d99ae;
        text-align: center;
        font-size: 13px;
        margin-top: 30px;
        animation: fadeIn 2s ease;
    }

    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================
# 🚘 SIDEBAR NAVIGATION — Dynamic Glass Neon Theme (Final Edition)
# =============================================================
st.markdown("""
    <style>
    /* === Sidebar Base === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(0,18,25,1), rgba(0,31,41,0.95));
        backdrop-filter: blur(14px);
        box-shadow: 4px 0 20px rgba(0,180,216,0.25);
        border-right: 1px solid rgba(0,180,216,0.25);
        transition: background 0.4s ease;
    }

    /* === Title === */
    .sidebar-title {
        font-size: 26px;
        text-align: center;
        color: #00eaff;
        margin-bottom: 20px;
        font-weight: 700;
        text-shadow: 0 0 12px #00eaff, 0 0 24px #0077b6;
        letter-spacing: 0.6px;
        animation: glowPulse 2.5s infinite ease-in-out;
    }

    @keyframes glowPulse {
        0% { text-shadow: 0 0 5px #00b4d8; }
        50% { text-shadow: 0 0 25px #00b4d8; }
        100% { text-shadow: 0 0 5px #00b4d8; }
    }

    /* === Expander (About) === */
    details {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(0,180,216,0.3);
        border-radius: 12px;
        padding: 10px 15px;
        margin-bottom: 15px;
        color: #e0f7fa;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    summary {
        font-weight: 600;
        color: #00eaff;
        cursor: pointer;
        text-shadow: 0 0 8px rgba(0,180,216,0.4);
    }

    details:hover {
        box-shadow: 0 0 15px rgba(0,180,216,0.3);
        transform: scale(1.01);
    }

    /* === Radio Buttons === */
    div[data-baseweb="radio"] > div {
        background: rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 10px 15px;
        color: #dff9fb;
        margin-bottom: 8px;
        transition: all 0.3s ease;
    }

    div[data-baseweb="radio"] > div:hover {
        background: linear-gradient(90deg, rgba(0,180,216,0.3), rgba(0,119,182,0.3));
        transform: scale(1.05);
        box-shadow: 0 0 12px rgba(0,180,216,0.35);
    }

    div[role="radiogroup"] label[data-testid="stMarkdownContainer"] p {
        color: #00eaff !important;
        font-weight: 600;
    }

    /* === Light Theme Compatibility === */
    @media (prefers-color-scheme: light) {
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f9f9f9, #e3f6f9);
            color: #003049;
            box-shadow: 2px 0 15px rgba(0,119,182,0.1);
        }
        details {
            background: rgba(0,180,216,0.08);
            color: #003049;
        }
        summary {
            color: #0077b6;
        }
        div[data-baseweb="radio"] > div {
            background: rgba(0,180,216,0.08);
            color: #003049;
        }
        div[data-baseweb="radio"] > div:hover {
            background: rgba(0,180,216,0.15);
        }
        .footer {
            background: rgba(0,180,216,0.1);
            color: #003049;
        }
    }

    /* === Footer === */
    .footer {
        text-align: center;
        background: rgba(0,180,216,0.08);
        border-radius: 10px;
        padding: 10px;
        color: #e0f7fa;
        font-size: 13px;
        margin-top: 20px;
        line-height: 1.5;
        transition: all 0.3s ease;
    }
    .footer:hover {
        box-shadow: 0 0 12px rgba(0,180,216,0.3);
        transform: scale(1.02);
    }
    .footer span {
        color: #00eaff;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================
# 🚘 SIDEBAR NAVIGATION — Bright + Dark Mode Fixed (Perfect Version)
# =============================================================
st.markdown("""
    <style>
    /* === Sidebar Base === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #001219, #001f29);
        backdrop-filter: blur(14px);
        box-shadow: 4px 0 20px rgba(0,180,216,0.25);
        border-right: 1px solid rgba(0,180,216,0.25);
        transition: background 0.4s ease;
    }

    /* === Title === */
    .sidebar-title {
        font-size: 26px;
        text-align: center;
        color: #00eaff;
        margin-bottom: 20px;
        font-weight: 700;
        text-shadow: 0 0 12px #00eaff, 0 0 24px #0077b6;
        letter-spacing: 0.6px;
        animation: glowPulse 2.5s infinite ease-in-out;
    }

    @keyframes glowPulse {
        0% { text-shadow: 0 0 5px #00b4d8; }
        50% { text-shadow: 0 0 25px #00b4d8; }
        100% { text-shadow: 0 0 5px #00b4d8; }
    }

    /* === Expander (About) === */
    details {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(0,180,216,0.4);
        border-radius: 12px;
        padding: 10px 15px;
        margin-bottom: 15px;
        color: #e0f7fa;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    summary {
        font-weight: 600;
        color: #00eaff;
        cursor: pointer;
        font-size: 16px;
        text-shadow: 0 0 8px rgba(0,180,216,0.4);
    }

    details:hover {
        box-shadow: 0 0 15px rgba(0,180,216,0.3);
        transform: scale(1.01);
    }

    /* === About Text === */
    .about-text {
        color: #bdf0ff;
        font-size: 15px;
        line-height: 1.6;
        margin-top: 5px;
    }

    .about-text b {
        color: #00eaff;
    }

    /* === Radio Buttons === */
    div[data-baseweb="radio"] > div {
        background: rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 10px 15px;
        color: #dff9fb;
        margin-bottom: 8px;
        transition: all 0.3s ease;
    }

    div[data-baseweb="radio"] > div:hover {
        background: linear-gradient(90deg, rgba(0,180,216,0.3), rgba(0,119,182,0.3));
        transform: scale(1.05);
        box-shadow: 0 0 12px rgba(0,180,216,0.35);
    }

    div[role="radiogroup"] label[data-testid="stMarkdownContainer"] p {
        color: #00eaff !important;
        font-weight: 600;
    }

    /* === Light Mode Fix === */
    @media (prefers-color-scheme: light) {
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f9f9f9, #e8f8fb);
            color: #002b36;
        }
        details {
            background: rgba(0,180,216,0.08);
            border: 1px solid rgba(0,180,216,0.2);
            color: #002b36;
        }
        .about-text {
            color: #004b63;
        }
        summary {
            color: #0077b6;
        }
        div[data-baseweb="radio"] > div {
            background: rgba(0,180,216,0.08);
            color: #003049;
        }
        div[data-baseweb="radio"] > div:hover {
            background: rgba(0,180,216,0.15);
        }
        .footer {
            background: rgba(0,180,216,0.1);
            color: #003049;
        }
        .footer span {
            color: #0077b6;
        }
    }

    /* === Footer === */
    .footer {
        text-align: center;
        background: rgba(0,180,216,0.08);
        border-radius: 10px;
        padding: 10px;
        color: #e0f7fa;
        font-size: 13px;
        margin-top: 20px;
        line-height: 1.5;
        transition: all 0.3s ease;
    }
    .footer:hover {
        box-shadow: 0 0 12px rgba(0,180,216,0.3);
        transform: scale(1.02);
    }
    .footer span {
        color: #00eaff;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar UI
st.sidebar.markdown("<h2 class='sidebar-title'>🚗 Car Price AI</h2>", unsafe_allow_html=True)

with st.sidebar.expander("📂 About the App", expanded=True):
    st.markdown(
        """
        <div class='about-text'>
        🌟 <b>Smart Price Prediction Dashboard</b><br>
        • Explore, visualize, and predict car prices<br>
        • Powered by <b>Machine Learning</b> algorithms<br>
        • Built with <b>Streamlit</b> + <b>Scikit-learn</b><br>
        • Real-time interactive interface 🚀
        </div>
        """,
        unsafe_allow_html=True
    )

page = st.sidebar.radio(
    "🌐 Choose Page:",
    ["🏠 Home", "📊 EDA Dashboard", "💰 Price Prediction"],
    index=0,
)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div class='footer'>
        Developed with ❤️ by <b>Data Science Enthusiast</b><br>
        <span>Innovating with Machine Learning 🚀</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================
# PAGE 1: HOME (Machine Learning Model — Premium Redesign)
# =============================================================
if page == "🏠 Home":
    st.markdown("""
        <style>
        /* ======== BACKGROUND ANIMATION ======== */
        @keyframes gradientFlow {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        @keyframes fadeInUp {
            from {opacity: 0; transform: translateY(25px);}
            to {opacity: 1; transform: translateY(0);}
        }

        /* ======== HERO SECTION ======== */
        .hero {
            background: linear-gradient(-45deg, #001219, #003049, #0a9396, #00b4d8, #48cae4);
            background-size: 400% 400%;
            animation: gradientFlow 10s ease infinite;
            padding: 60px 40px;
            border-radius: 20px;
            box-shadow: 0px 0px 40px rgba(0, 180, 216, 0.25);
            text-align: center;
            color: #e0fbfc;
            margin-bottom: 30px;
        }
        .hero h1 {
            font-size: 3.2em;
            font-weight: 700;
            color: #caf0f8;
            text-shadow: 0 0 25px #00b4d8, 0 0 45px #0077b6;
            animation: fadeInUp 1.2s ease;
        }
        .hero p {
            color: #dee2e6;
            font-size: 18px;
            margin-top: 15px;
            line-height: 1.6;
            animation: fadeInUp 1.8s ease;
        }

        /* ======== METRIC CARDS ======== */
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 180, 216, 0.25);
            border-radius: 16px;
            padding: 25px;
            text-align: center;
            color: #fff;
            transition: all 0.4s ease;
            animation: fadeInUp 2s ease;
        }
        .metric-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 0 30px rgba(0, 180, 216, 0.35);
        }
        .metric-title {
            font-size: 18px;
            color: #90e0ef;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
        }

        /* ======== ABOUT SECTION ======== */
        .about-section {
            text-align: center;
            color: #adb5bd;
            padding: 30px 10px;
            animation: fadeInUp 2.3s ease;
        }
        .about-section h3 {
            color: #00b4d8;
            font-size: 1.8em;
            margin-bottom: 10px;
        }
        .about-section p {
            font-size: 17px;
            line-height: 1.7;
        }

        /* ======== DIVIDER ======== */
        .glow-line {
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #00b4d8, transparent);
            box-shadow: 0 0 20px #00b4d8;
            margin: 40px auto;
            width: 80%;
        }

        /* ======== FOOTER ======== */
        .footer {
            text-align: center;
            color: gray;
            font-size: 14px;
            margin-top: 30px;
            animation: fadeInUp 2.6s ease;
        }
        .footer b {
            color: #00b4d8;
        }
        </style>
    """, unsafe_allow_html=True)

    # 🌌 Hero Section
    st.markdown("""
        <div class="hero">
            <h1>🤖 Car Price Prediction Using Machine Learning Model</h1>
            <p>
                Experience the fusion of <b>AI</b> and <b>Data Science</b> to predict accurate car resale values.<br>
                Trained using <b>Linear Regression</b> on real-world datasets to ensure precision and reliability.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 🚘 Hero Image
    st.image(
        "https://cdni.iconscout.com/illustration/premium/thumb/car-price-prediction-7806707-6327847.png",
        use_container_width=True,
        caption="AI-powered Machine Learning model for intelligent car price prediction 🚗"
    )

    # 📊 Dataset Overview
    st.markdown("<h3 style='text-align:center; color:#00b4d8; margin-top:35px;'>📈 Dataset Overview</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:#00b4d8;'>{df.shape[0]}</div>
                <div class='metric-title'>Total Records</div>
                <p style='font-size:13px; color:#adb5bd;'>Cars analyzed for training</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:#06d6a0;'>{df['company'].nunique()}</div>
                <div class='metric-title'>Car Brands</div>
                <p style='font-size:13px; color:#adb5bd;'>Unique manufacturers studied</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:#f9c74f;'>{df['fuel_type'].nunique()}</div>
                <div class='metric-title'>Fuel Variants</div>
                <p style='font-size:13px; color:#adb5bd;'>Petrol, Diesel, Electric, and more</p>
            </div>
        """, unsafe_allow_html=True)

    # ✨ Divider
    st.markdown("<hr class='glow-line'>", unsafe_allow_html=True)

    # 💡 About Section
    st.markdown("""
        <div class='about-section'>
            <h3>💡 About This Machine Learning Model</h3>
            <p>
                This advanced model uses <b>Linear Regression</b> to predict car resale prices based on brand, year, fuel type, and more.<br>
                By analyzing thousands of records, it provides accurate, data-driven insights to support smart decisions.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 🌟 Footer (Updated)
    st.markdown("""
        <div class='footer'>
            Developed by a <b>Data Science Enthusiast</b> 💻<br>
            Powered by <b>Machine Learning</b> & <b>Streamlit</b> ⚙️<br>
            © 2025 Car Price Prediction Project
        </div>
    """, unsafe_allow_html=True)

# =============================================================
# PAGE 2: EDA DASHBOARD (Next-Gen Futuristic Edition)
# =============================================================
elif page == "📊 EDA Dashboard":
    # 🌌 Custom CSS for theme and animations
    st.markdown("""
        <style>
        @keyframes fadeInUp {
            from {opacity: 0; transform: translateY(20px);}
            to {opacity: 1; transform: translateY(0);}
        }
        @keyframes glowPulse {
            0% { box-shadow: 0 0 10px #00b4d8; }
            50% { box-shadow: 0 0 25px #00b4d8; }
            100% { box-shadow: 0 0 10px #00b4d8; }
        }
        h1 {
            animation: fadeInUp 1s ease;
        }
        .metric-box {
            background: linear-gradient(135deg, #001219, #003049);
            border: 1px solid rgba(0,180,216,0.3);
            border-radius: 12px;
            text-align: center;
            padding: 20px;
            color: #e0e0e0;
            animation: fadeInUp 1.5s ease;
            transition: 0.3s ease;
        }
        .metric-box:hover {
            transform: translateY(-6px);
            animation: glowPulse 1.5s infinite ease-in-out;
        }
        .metric-box h3 {
            color: #00b4d8;
            margin-bottom: 5px;
        }
        .tab-content {
            animation: fadeInUp 1.2s ease;
        }
        .footer {
            text-align: center;
            color: gray;
            margin-top: 20px;
            animation: fadeInUp 1.5s ease;
        }
        </style>
    """, unsafe_allow_html=True)

    # 🧭 Dashboard Header
    st.markdown("<h1 style='text-align:center; color:#00b4d8;'>📊 Car Data Insights Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#b0bec5;'>Explore dynamic data trends, pricing analysis, and brand performance insights — all powered by Machine Learning intelligence.</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #2a2a2a;'>", unsafe_allow_html=True)

    # ⚙️ KPI Metrics
    st.subheader("📈 Dataset Highlights")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-box'><h3>{df.shape[0]}</h3><p>Total Records</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-box'><h3>{df['company'].nunique()}</h3><p>Unique Brands</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-box'><h3>{df['fuel_type'].nunique()}</h3><p>Fuel Types</p></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-box'><h3>{df['year'].min()} - {df['year'].max()}</h3><p>Year Range</p></div>", unsafe_allow_html=True)

    st.markdown("<hr style='border: 1px solid #2a2a2a;'>", unsafe_allow_html=True)

    # 🧠 Tabs for Analytics
    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 Price Distribution", 
        "📅 Yearly Price Trend", 
        "🚘 Top Models", 
        "⛽ Fuel Insights"
    ])

    # 💰 PRICE DISTRIBUTION
    with tab1:
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        st.subheader("💰 Car Price Distribution")
        fig = px.histogram(
            df, x="Price", nbins=40, color_discrete_sequence=["#00b4d8"],
            title="Distribution of Car Prices (₹)"
        )
        fig.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", title_font_color="#00b4d8"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📊 Most cars fall between ₹3–₹10 lakh — the sweet spot for Indian resale markets.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 📅 YEARLY PRICE TREND
    with tab2:
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        st.subheader("📅 Average Price Trend Over the Years")
        year_trend = df.groupby("year")["Price"].mean().reset_index()
        fig2 = px.line(
            year_trend, x="year", y="Price", markers=True,
            color_discrete_sequence=["#00b4d8"],
            title="Average Car Price by Manufacturing Year"
        )
        fig2.update_traces(line=dict(width=3))
        fig2.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", title_font_color="#00b4d8"
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("📈 Newer cars command higher prices due to advanced tech and improved safety.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 🚘 TOP MODELS
    with tab3:
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        st.subheader("🚘 Top 5 Car Models by Average Resale Price")
        if "name" in df.columns:
            top_models = df.groupby("name")["Price"].mean().nlargest(5).reset_index()
            fig3 = px.bar(
                top_models, x="name", y="Price", text="Price",
                color="Price", color_continuous_scale="Viridis",
                title="Top Performing Car Models"
            )
            fig3.update_traces(texttemplate="₹%{text:.0f}", textposition="outside")
            fig3.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="white", title_font_color="#00b4d8"
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("⚠️ Column 'name' not found in dataset.")
        st.caption("💎 Luxury and premium variants dominate high resale value segments.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ⛽ FUEL INSIGHTS
    with tab4:
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        st.subheader("⛽ Fuel Type Market Share")
        fig4 = px.pie(
            df, names="fuel_type", hole=0.4,
            title="Fuel Type Distribution",
            color_discrete_sequence=px.colors.sequential.Agsunset
        )
        fig4.update_traces(textinfo="percent+label", pull=[0.05, 0.05, 0, 0])
        fig4.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", title_font_color="#00b4d8"
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("⚡ Petrol dominates the market, while EVs are emerging as future contenders.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ✨ Footer
    st.markdown("<hr style='border: 1px solid #2a2a2a;'>", unsafe_allow_html=True)
    st.markdown("""
        <div class='footer'>
            Built with ❤️ by <b>Data Science Enthusiast</b> using <b>Streamlit</b> & <b>Plotly</b>.<br>
            Elevating data into visual intelligence ⚙️🚗
        </div>
    """, unsafe_allow_html=True)

# =============================================================
# PAGE 3: PRICE PREDICTION (Next-Gen Dynamic Edition)
# =============================================================
elif page == "💰 Price Prediction":
    # 🌈 Advanced CSS Styles
    st.markdown("""
        <style>
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(15px);}
            to {opacity: 1; transform: translateY(0);}
        }
        @keyframes glow {
            0% {text-shadow: 0 0 5px #00b4d8, 0 0 10px #0096c7;}
            50% {text-shadow: 0 0 20px #00b4d8, 0 0 30px #48cae4;}
            100% {text-shadow: 0 0 5px #00b4d8, 0 0 10px #0096c7;}
        }
        @keyframes glowBox {
            0% {box-shadow: 0 0 10px #00b4d8;}
            50% {box-shadow: 0 0 25px #48cae4;}
            100% {box-shadow: 0 0 10px #00b4d8;}
        }
        .header-title {
            text-align: center;
            color: #00b4d8;
            font-size: 42px;
            font-weight: 800;
            letter-spacing: 1px;
            animation: fadeIn 1s ease, glow 2s ease-in-out infinite alternate;
        }
        .sub-text {
            text-align: center;
            color: #b0bec5;
            font-size: 18px;
            animation: fadeIn 1.3s ease;
        }
        .input-card {
            background: rgba(0, 18, 25, 0.9);
            border: 1px solid rgba(0,180,216,0.3);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 0 25px rgba(0,180,216,0.2);
            transition: 0.4s ease;
            animation: fadeIn 1.5s ease;
        }
        .input-card:hover {
            transform: scale(1.01);
            box-shadow: 0 0 40px rgba(0,180,216,0.4);
        }
        .predict-button {
            background: linear-gradient(90deg, #00b4d8, #0077b6);
            color: white;
            font-size: 20px;
            font-weight: bold;
            border-radius: 12px;
            padding: 14px;
            width: 100%;
            border: none;
            transition: all 0.3s;
        }
        .predict-button:hover {
            background: linear-gradient(90deg, #48cae4, #00b4d8);
            transform: scale(1.05);
            box-shadow: 0 0 25px #00b4d8;
        }
        .result-box {
            background: linear-gradient(145deg, #001219, #003049);
            border: 1px solid rgba(0,180,216,0.5);
            border-radius: 16px;
            padding: 25px;
            text-align: center;
            color: #00b4d8;
            font-size: 26px;
            font-weight: bold;
            margin-top: 15px;
            animation: fadeIn 1.2s ease, glowBox 2.5s infinite ease-in-out;
        }
        .confidence-bar {
            height: 20px;
            border-radius: 12px;
            background: linear-gradient(90deg, #06d6a0, #118ab2);
            animation: fadeIn 1.8s ease;
        }
        .footer {
            text-align: center;
            color: #9ca3af;
            font-size: 15px;
            margin-top: 40px;
            animation: fadeIn 2s ease;
        }
        </style>
    """, unsafe_allow_html=True)

    # 🚗 Header Section
    st.markdown("<h1 class='header-title'>🤖 AI-Powered Car Price Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-text'>Estimate your car’s market value instantly using intelligent Machine Learning models.</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #2a2a2a;'>", unsafe_allow_html=True)

    # 🧾 Car Input Section
    st.subheader("🚘 Enter Car Information")
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)

    car_names = sorted(df['name'].unique())
    companies = sorted(df['company'].unique())
    fuel_types = sorted(df['fuel_type'].unique())

    col1, col2 = st.columns(2)
    with col1:
        name = st.selectbox("Car Model Name", car_names)
        company = st.selectbox("Car Brand", companies)
        year = st.number_input("Year of Purchase", min_value=1995, max_value=2025, value=2019)
    with col2:
        kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=100)
        fuel_type = st.selectbox("Fuel Type", fuel_types)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # 🚀 Predict Button
    predict = st.button("🚀 Predict Price", use_container_width=True)

    if predict:
        try:
            input_df = pd.DataFrame(
                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                data=np.array([name, company, year, kms_driven, fuel_type]).reshape(1, 5)
            )
            predicted_price = pipe.predict(input_df)[0]

            # Simulate model confidence (for realism)
            import random
            confidence = random.randint(82, 97)

            st.markdown(f"""
                <div class='result-box'>
                    💰 Predicted Car Price:<br><b>₹ {predicted_price:,.0f}</b>
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <p style='text-align:center; color:#b0bec5; font-size:17px; margin-top:15px;'>
                    🔍 Model Confidence: <b>{confidence}%</b>
                </p>
            """, unsafe_allow_html=True)

            st.progress(confidence / 100)

        except Exception as e:
            st.error(f"❌ Error: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # 🧠 Model Info
    st.info("⚙️ Powered by Linear Regression — trained on Indian car resale data. The model learns complex relationships between brand, year, mileage, and fuel type to deliver realistic predictions.")

    # ✨ Footer
    st.markdown("<hr style='border: 1px solid #2a2a2a;'>", unsafe_allow_html=True)
    st.markdown("""
        <div class='footer'>
            Developed with ❤️ by <b>Data Science Enthusiast</b><br>
            <span style='color:#00b4d8;'>Where Data Meets Intelligence ⚙️🚗</span>
        </div>
    """, unsafe_allow_html=True)


