import sys
import os

new_app_code = '''import os
import json
import pickle
import yaml
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Federated Healthcare AI - Smart Hospital System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# DYNAMIC THEME CSS GENERATOR
# -----------------------------------------------------------------------------
def get_custom_css(theme="light"):
    is_dark = (theme == "dark")
    
    if is_dark:
        vars_css = """
            --bg-primary: #0f172a;
            --bg-card: #1e293b;
            --bg-sidebar: #0f172a;
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-muted: #94a3b8;
            --heading-color: #38bdf8;
            --accent: #38bdf8;
            --accent-soft: #1e293b;
            --accent-border: #0284c7;
            --border: #334155;
            --card-border: #334155;
            --card-hover-border: #38bdf8;
            --nav-bg: #1e293b;
            --badge-bg: #064e3b;
            --badge-text: #6ee7b7;
            --badge-border: #047857;
            --input-bg: #0f172a;
            --input-text: #f8fafc;
            --btn-bg: #1e293b;
            --btn-text: #38bdf8;
            --btn-border: #334155;
        """
    else:
        vars_css = """
            --bg-primary: #f8fafc;
            --bg-card: #ffffff;
            --bg-sidebar: #ffffff;
            --text-primary: #0f172a;
            --text-secondary: #334155;
            --text-muted: #64748b;
            --heading-color: #0f2b48;
            --accent: #0284c7;
            --accent-soft: #f0f9ff;
            --accent-border: #bae6fd;
            --border: #cbd5e1;
            --card-border: #e2e8f0;
            --card-hover-border: #0284c7;
            --nav-bg: #ffffff;
            --badge-bg: #f0fdf4;
            --badge-text: #166534;
            --badge-border: #bbf7d0;
            --input-bg: #ffffff;
            --input-text: #0f172a;
            --btn-bg: #ffffff;
            --btn-text: #0f2b48;
            --btn-border: #cbd5e1;
        """

    return f"""
    <style>
        :root {{
            {vars_css}
            --font-xs: 0.9rem;    /* ~14.4px */
            --font-sm: 1.0rem;    /* ~16.0px */
            --font-base: 1.1rem;  /* ~17.6px */
            --font-lg: 1.3rem;    /* ~20.8px */
            --font-xl: 1.75rem;   /* ~28.0px */
            --font-2xl: 2.25rem;  /* ~36.0px */
            --font-stat: 2.5rem;  /* ~40.0px */
        }}

        /* Centered Main Page Container */
        .main .block-container {{
            max-width: 1250px !important;
            margin: 0 auto !important;
            padding: 1.5rem 2rem !important;
        }}

        /* Global Base Font & Specificity Enforcement */
        html, body, [class*="css"], .stApp, .stMarkdown, p, span, div, label {{
            font-size: 16px !important;
        }}

        .main, body, .stApp {{
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        }}

        header[data-testid="stHeader"] {{
            background-color: transparent !important;
        }}

        /* Streamlit DOM Component Overrides */
        [data-testid="stMarkdownContainer"] p, 
        [data-testid="stMarkdownContainer"] div, 
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {{
            font-size: var(--font-base) !important;
            line-height: 1.55 !important;
            color: var(--text-primary) !important;
        }}

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] label p {{
            font-size: var(--font-sm) !important;
            font-weight: 600 !important;
            color: var(--text-secondary) !important;
        }}

        [data-testid="stMetricValue"] {{
            font-size: var(--font-stat) !important;
            font-weight: 800 !important;
            color: var(--heading-color) !important;
        }}

        [data-testid="stMetricLabel"] {{
            font-size: var(--font-xs) !important;
            font-weight: 700 !important;
            color: var(--text-muted) !important;
            text-transform: uppercase !important;
            letter-spacing: 0.6px !important;
        }}

        div[data-testid="stCaptionContainer"] {{
            font-size: var(--font-sm) !important;
            color: var(--text-muted) !important;
        }}

        /* Button Styling Enforcement Across Themes */
        .stButton > button, button[data-testid="baseButton-secondary"], button[data-testid="baseButton-primary"] {{
            font-size: var(--font-sm) !important;
            font-weight: 700 !important;
            padding: 10px 20px !important;
            border-radius: 8px !important;
            background-color: var(--btn-bg) !important;
            color: var(--btn-text) !important;
            border: 1px solid var(--btn-border) !important;
            transition: all 0.15s ease-in-out !important;
        }}
        .stButton > button:hover, button[data-testid="baseButton-secondary"]:hover, button[data-testid="baseButton-primary"]:hover {{
            border-color: var(--accent) !important;
            color: var(--accent) !important;
            transform: translateY(-1px) !important;
        }}

        .stSelectbox label, .stTextInput label, .stNumberInput label, .stSlider label {{
            font-size: var(--font-sm) !important;
            font-weight: 700 !important;
            color: var(--heading-color) !important;
        }}

        /* Inputs & Form Elements */
        input, select, textarea, div[data-baseweb="input"], div[data-baseweb="select"] {{
            background-color: var(--input-bg) !important;
            color: var(--input-text) !important;
            border-color: var(--border) !important;
        }}

        table, th, td, div[data-testid="stTable"] {{
            font-size: var(--font-sm) !important;
            color: var(--text-primary) !important;
        }}

        pre, code, div[data-testid="stText"] pre, div[data-testid="stCodeBlock"] pre, .stText pre {{
            color: var(--text-primary) !important;
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            font-size: var(--font-sm) !important;
            font-family: monospace !important;
        }}

        /* Top Navigation Bar */
        .top-nav-bar {{
            background-color: var(--nav-bg) !important;
            border: 1px solid var(--border) !important;
            padding: 16px 24px !important;
            margin-bottom: 28px !important;
            display: flex !important;
            justify-content: space-between !important;
            align-items: center !important;
            border-radius: 10px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
        }}
        .nav-brand {{
            font-size: var(--font-lg) !important;
            font-weight: 800 !important;
            color: var(--heading-color) !important;
            letter-spacing: -0.3px !important;
        }}
        .nav-meta {{
            font-size: var(--font-sm) !important;
            color: var(--text-secondary) !important;
            display: flex !important;
            gap: 16px !important;
            align-items: center !important;
        }}
        .nav-badge {{
            background-color: var(--badge-bg) !important;
            color: var(--badge-text) !important;
            border: 1px solid var(--badge-border) !important;
            padding: 4px 10px !important;
            border-radius: 6px !important;
            font-size: var(--font-xs) !important;
            font-weight: 700 !important;
        }}

        /* Page Titles & Headers */
        .page-title {{
            font-size: var(--font-2xl) !important;
            font-weight: 800 !important;
            color: var(--heading-color) !important;
            margin-bottom: 6px !important;
            letter-spacing: -0.5px !important;
            line-height: 1.25 !important;
        }}
        .page-subtitle {{
            font-size: var(--font-base) !important;
            color: var(--text-muted) !important;
            margin-bottom: 24px !important;
            font-weight: 500 !important;
            line-height: 1.4 !important;
        }}
        .status-perimeter-badge {{
            display: inline-block !important;
            font-size: var(--font-xs) !important;
            font-weight: 700 !important;
            color: var(--accent) !important;
            background-color: var(--accent-soft) !important;
            border: 1px solid var(--accent-border) !important;
            padding: 5px 12px !important;
            border-radius: 6px !important;
            margin-bottom: 24px !important;
            letter-spacing: 0.6px !important;
            text-transform: uppercase !important;
        }}

        /* Login Page Styling */
        .hero-tag {{
            display: inline-block !important;
            font-size: var(--font-xs) !important;
            font-weight: 700 !important;
            letter-spacing: 0.8px !important;
            color: var(--accent) !important;
            background-color: var(--accent-soft) !important;
            border: 1px solid var(--accent-border) !important;
            padding: 4px 12px !important;
            border-radius: 6px !important;
            margin-bottom: 14px !important;
            text-transform: uppercase !important;
        }}
        .left-hero-title {{
            font-size: 2.5rem !important;
            font-weight: 800 !important;
            color: var(--heading-color) !important;
            margin-bottom: 6px !important;
            line-height: 1.2 !important;
        }}
        .left-hero-subtitle {{
            font-size: var(--font-base) !important;
            font-weight: 500 !important;
            color: var(--text-secondary) !important;
            margin-bottom: 24px !important;
            line-height: 1.5 !important;
        }}
        .badge-row {{
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 10px !important;
            margin-bottom: 22px !important;
        }}
        .badge-item {{
            font-size: var(--font-sm) !important;
            font-weight: 600 !important;
            padding: 6px 14px !important;
            border-radius: 6px !important;
            display: inline-flex !important;
            align-items: center !important;
            gap: 8px !important;
        }}
        .badge-teal {{
            background-color: var(--badge-bg) !important;
            color: var(--badge-text) !important;
            border: 1px solid var(--badge-border) !important;
        }}
        .badge-purple {{
            background-color: var(--accent-soft) !important;
            color: var(--accent) !important;
            border: 1px solid var(--accent-border) !important;
        }}
        .badge-green {{
            background-color: var(--badge-bg) !important;
            color: var(--badge-text) !important;
            border: 1px solid var(--badge-border) !important;
        }}
        .indicator-grid {{
            display: grid !important;
            grid-template-columns: 1fr 1fr !important;
            gap: 14px !important;
            margin-bottom: 24px !important;
        }}
        .indicator-card {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--card-border) !important;
            border-left: 4px solid #10b981 !important;
            border-radius: 8px !important;
            padding: 14px 18px !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
            transition: all 0.15s ease-in-out !important;
        }}
        .indicator-card:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 14px rgba(0,0,0,0.08) !important;
        }}
        .indicator-title {{
            font-size: var(--font-sm) !important;
            font-weight: 700 !important;
            color: var(--heading-color) !important;
            margin-bottom: 3px !important;
        }}
        .indicator-desc {{
            font-size: var(--font-xs) !important;
            color: var(--text-muted) !important;
            line-height: 1.4 !important;
        }}
        .arch-card {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--card-border) !important;
            border-radius: 10px !important;
            padding: 20px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
            margin-top: 14px !important;
        }}
        .arch-card-header {{
            font-size: var(--font-xs) !important;
            font-weight: 700 !important;
            color: var(--heading-color) !important;
            text-transform: uppercase !important;
            letter-spacing: 0.8px !important;
            margin-bottom: 14px !important;
            border-bottom: 1px solid var(--border) !important;
            padding-bottom: 8px !important;
        }}
        .login-card-title {{
            font-size: var(--font-xl) !important;
            font-weight: 800 !important;
            color: var(--heading-color) !important;
            margin-bottom: 6px !important;
        }}
        .login-card-subtitle {{
            font-size: var(--font-sm) !important;
            color: var(--text-muted) !important;
            margin-bottom: 24px !important;
        }}
        .secure-access-caption {{
            text-align: center !important;
            font-size: var(--font-sm) !important;
            color: #10b981 !important;
            margin-top: 22px !important;
            padding-top: 16px !important;
            border-top: 1px solid var(--border) !important;
            font-weight: 600 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 8px !important;
        }}
        .secure-dot {{
            width: 8px !important;
            height: 8px !important;
            background-color: #10b981 !important;
            border-radius: 50% !important;
            display: inline-block !important;
        }}

        /* Metric Cards / Stat Cards */
        .metric-box, .stat-card {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--card-border) !important;
            border-radius: 10px !important;
            padding: 20px 24px !important;
            margin-bottom: 24px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
            transition: all 0.15s ease-in-out !important;
            font-size: var(--font-base) !important;
        }}
        .metric-box:hover, .stat-card:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.10) !important;
            border-color: var(--card-hover-border) !important;
        }}
        .metric-title {{
            font-size: var(--font-xs) !important;
            color: var(--text-muted) !important;
            text-transform: uppercase !important;
            letter-spacing: 0.8px !important;
            font-weight: 700 !important;
            margin-bottom: 6px !important;
        }}
        .metric-value, .stat-number, .stat-card .stat-number, .info-card .stat-number {{
            font-size: var(--font-stat) !important;
            font-weight: 800 !important;
            color: var(--heading-color) !important;
            line-height: 1.25 !important;
            margin-bottom: 4px !important;
        }}
        .metric-caption {{
            font-size: var(--font-sm) !important;
            color: var(--text-secondary) !important;
            margin-top: 4px !important;
            font-weight: 500 !important;
            line-height: 1.35 !important;
        }}

        /* Privacy Banner */
        .privacy-banner {{
            background-color: var(--badge-bg) !important;
            border: 1px solid var(--badge-border) !important;
            border-left: 4px solid #16a34a !important;
            padding: 14px 20px !important;
            border-radius: 8px !important;
            color: var(--badge-text) !important;
            font-size: var(--font-sm) !important;
            font-weight: 600 !important;
            margin-bottom: 28px !important;
            line-height: 1.45 !important;
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border) !important;
        }}

        /* Sidebar Nav Radio Options */
        div[data-testid="stSidebar"] div[data-testid="stRadio"] > label {{
            font-size: var(--font-sm) !important;
            font-weight: 600 !important;
            color: var(--text-secondary) !important;
            padding: 10px 14px !important;
            border-radius: 8px !important;
            margin-bottom: 6px !important;
            transition: all 0.15s ease !important;
            border-left: 3px solid transparent !important;
        }}
        div[data-testid="stSidebar"] div[data-testid="stRadio"] > label:hover {{
            background-color: var(--accent-soft) !important;
            color: var(--heading-color) !important;
        }}
        div[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked),
        div[data-testid="stSidebar"] div[data-testid="stRadio"] label[aria-checked="true"] {{
            background-color: var(--accent-soft) !important;
            color: var(--accent) !important;
            font-weight: 700 !important;
            border-left: 4px solid var(--accent) !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12) !important;
        }}

        /* Sidebar Profile Card */
        .sidebar-profile-card {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-left: 4px solid var(--accent) !important;
            border-radius: 10px !important;
            padding: 16px 18px !important;
            margin-bottom: 22px !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05) !important;
        }}
        .sidebar-profile-badge {{
            display: inline-block !important;
            font-size: var(--font-xs) !important;
            font-weight: 700 !important;
            letter-spacing: 0.8px !important;
            color: var(--accent) !important;
            background-color: var(--accent-soft) !important;
            border: 1px solid var(--accent-border) !important;
            padding: 3px 10px !important;
            border-radius: 4px !important;
            text-transform: uppercase !important;
            margin-bottom: 6px !important;
        }}
        .sidebar-profile-title {{
            font-size: var(--font-lg) !important;
            font-weight: 800 !important;
            color: var(--heading-color) !important;
            margin-bottom: 2px !important;
        }}
        .sidebar-profile-sub {{
            font-size: var(--font-sm) !important;
            color: var(--text-muted) !important;
            font-weight: 500 !important;
        }}

        /* Structured Cards & Info Grid Components */
        .info-card {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--card-border) !important;
            border-radius: 10px !important;
            padding: 20px 24px !important;
            margin-bottom: 16px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
            height: 100% !important;
            transition: all 0.15s ease-in-out !important;
            font-size: var(--font-base) !important;
        }}
        .info-card:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.10) !important;
            border-color: var(--card-hover-border) !important;
        }}
        .info-card-label {{
            font-size: var(--font-xs) !important;
            font-weight: 700 !important;
            color: var(--text-muted) !important;
            text-transform: uppercase !important;
            letter-spacing: 0.8px !important;
            margin-bottom: 6px !important;
        }}
        .info-card-value {{
            font-size: var(--font-lg) !important;
            font-weight: 800 !important;
            color: var(--heading-color) !important;
            line-height: 1.35 !important;
        }}
        .info-card-sub {{
            font-size: var(--font-sm) !important;
            color: var(--text-secondary) !important;
            margin-top: 6px !important;
            line-height: 1.4 !important;
        }}

        /* Status Badges */
        .badge-status-green {{
            display: inline-block !important;
            background-color: var(--badge-bg) !important;
            color: var(--badge-text) !important;
            border: 1px solid var(--badge-border) !important;
            padding: 3px 10px !important;
            border-radius: 6px !important;
            font-size: var(--font-sm) !important;
            font-weight: 700 !important;
        }}
        .badge-status-blue {{
            display: inline-block !important;
            background-color: var(--accent-soft) !important;
            color: var(--accent) !important;
            border: 1px solid var(--accent-border) !important;
            padding: 3px 10px !important;
            border-radius: 6px !important;
            font-size: var(--font-sm) !important;
            font-weight: 700 !important;
        }}

        .section-header {{
            font-size: var(--font-lg) !important;
            font-weight: 800 !important;
            color: var(--heading-color) !important;
            margin-top: 4px !important;
            margin-bottom: 18px !important;
            border-bottom: 2px solid var(--border) !important;
            padding-bottom: 8px !important;
            letter-spacing: -0.2px !important;
        }}
        .step-card {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--card-border) !important;
            border-left: 4px solid var(--accent) !important;
            border-radius: 8px !important;
            padding: 16px 20px !important;
            margin-bottom: 14px !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
            transition: all 0.15s ease-in-out !important;
            font-size: var(--font-base) !important;
        }}
        .step-card:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 14px rgba(0, 0, 0, 0.08) !important;
        }}
        .step-card-num {{
            font-size: var(--font-xs) !important;
            font-weight: 800 !important;
            color: var(--accent) !important;
            text-transform: uppercase !important;
            letter-spacing: 0.8px !important;
            margin-bottom: 4px !important;
        }}
        .step-card-title {{
            font-size: var(--font-base) !important;
            font-weight: 700 !important;
            color: var(--heading-color) !important;
            margin-bottom: 6px !important;
        }}
        .step-card-desc {{
            font-size: var(--font-sm) !important;
            color: var(--text-secondary) !important;
            line-height: 1.45 !important;
        }}
    </style>
    """


# -----------------------------------------------------------------------------
# PLOTLY THEME & LAYOUT HELPER
# -----------------------------------------------------------------------------
def get_plotly_layout(theme=None):
    if theme is None:
        theme = st.session_state.get("theme", "light")
    is_dark = (theme == "dark")
    
    font_color = "#f8fafc" if is_dark else "#0f2b48"
    grid_color = "#334155" if is_dark else "#f1f5f9"
    legend_bg = "rgba(30, 41, 59, 0.95)" if is_dark else "rgba(255, 255, 255, 0.95)"
    legend_border = "#334155" if is_dark else "#cbd5e1"

    return dict(
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=13, color=font_color),
        legend=dict(font=dict(size=12, color=font_color), bgcolor=legend_bg, bordercolor=legend_border, borderwidth=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40)
    )

def build_plotly_convergence_chart(tr_df=None, results=None):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Global Model Accuracy vs Rounds", "Global Model Loss vs Rounds"))
    is_dark = (st.session_state.get("theme", "light") == "dark")
    font_color = "#f8fafc" if is_dark else "#0f2b48"
    grid_c = "#334155" if is_dark else "#f1f5f9"
    
    rounds_std, acc_std, loss_std = [], [], []
    rounds_dp, acc_dp, loss_dp = [], [], []

    if tr_df is not None and not tr_df.empty:
        std_df = tr_df[tr_df["model"].str.contains("Standard", na=False)]
        dp_df = tr_df[tr_df["model"].str.contains("Privacy", na=False)]
        
        rounds_std = std_df["round"].tolist()
        acc_std = (std_df["accuracy"] * (100.0 if std_df["accuracy"].max() <= 1.0 else 1.0)).tolist()
        loss_std = std_df["train_loss"].tolist()
        
        rounds_dp = dp_df["round"].tolist()
        acc_dp = (dp_df["accuracy"] * (100.0 if dp_df["accuracy"].max() <= 1.0 else 1.0)).tolist()
        loss_dp = dp_df["train_loss"].tolist()
    elif results and "fl_history_no_dp" in results:
        no_dp = results["fl_history_no_dp"]
        rounds_std = [r["round"] for r in no_dp]
        acc_std = [r["accuracy"] * 100.0 for r in no_dp]
        loss_std = [r["loss"] for r in no_dp]
        
        if "fl_history_dp" in results and results["fl_history_dp"]:
            dp_hist = results["fl_history_dp"]
            rounds_dp = [r["round"] for r in dp_hist]
            acc_dp = [r["accuracy"] * 100.0 for r in dp_hist]
            loss_dp = [r["loss"] for r in dp_hist]

    if rounds_std:
        fig.add_trace(go.Scatter(x=rounds_std, y=acc_std, mode="lines+markers", name="FedAvg (Standard)",
                                 hovertemplate="Round %{x}<br>Accuracy: %{y:.2f}%",
                                 line=dict(color="#38bdf8" if is_dark else "#1f77b4", width=2.5),
                                 marker=dict(size=7, symbol="circle")), row=1, col=1)
        fig.add_trace(go.Scatter(x=rounds_std, y=loss_std, mode="lines+markers", name="FedAvg (Standard)", showlegend=False,
                                 hovertemplate="Round %{x}<br>Loss: %{y:.4f}",
                                 line=dict(color="#38bdf8" if is_dark else "#1f77b4", width=2.5),
                                 marker=dict(size=7, symbol="circle")), row=1, col=2)
    if rounds_dp:
        fig.add_trace(go.Scatter(x=rounds_dp, y=acc_dp, mode="lines+markers", name="FedAvg + DP Noise",
                                 hovertemplate="Round %{x}<br>Accuracy: %{y:.2f}%",
                                 line=dict(color="#ef4444" if is_dark else "#d62728", width=2.5, dash="dash"),
                                 marker=dict(size=7, symbol="square")), row=1, col=1)
        fig.add_trace(go.Scatter(x=rounds_dp, y=loss_dp, mode="lines+markers", name="FedAvg + DP Noise", showlegend=False,
                                 hovertemplate="Round %{x}<br>Loss: %{y:.4f}",
                                 line=dict(color="#ef4444" if is_dark else "#d62728", width=2.5, dash="dash"),
                                 marker=dict(size=7, symbol="square")), row=1, col=2)

    fig.update_xaxes(title_text="Communication Round", gridcolor=grid_c, row=1, col=1)
    fig.update_xaxes(title_text="Communication Round", gridcolor=grid_c, row=1, col=2)
    fig.update_yaxes(title_text="Accuracy (%)", range=[30, 100], gridcolor=grid_c, row=1, col=1)
    fig.update_yaxes(title_text="Loss", gridcolor=grid_c, row=1, col=2)
    fig.update_annotations(font_size=13, font_color=font_color)
    
    fig.update_layout(**get_plotly_layout(), height=420)
    return fig


def build_plotly_model_comparison_bar(comp_df=None, results=None):
    grid_c = "#334155" if st.session_state.get("theme", "light") == "dark" else "#f1f5f9"
    if comp_df is not None and not comp_df.empty:
        df = comp_df.copy()
        paradigms = df["Model Paradigm"].tolist()
        accs = (df["Accuracy"] * (100.0 if df["Accuracy"].max() <= 1.0 else 1.0)).tolist()
        f1s = (df["F1-Score"] * (100.0 if df["F1-Score"].max() <= 1.0 else 1.0)).tolist()
        aucs = (df["ROC-AUC"] * (100.0 if df["ROC-AUC"].max() <= 1.0 else 1.0)).tolist()
    elif results and "method_comparison" in results:
        comp_dict = results["method_comparison"]
        paradigms = list(comp_dict.keys())
        accs = [comp_dict[m]["accuracy"] * 100.0 for m in paradigms]
        f1s = [comp_dict[m]["f1"] * 100.0 for m in paradigms]
        aucs = [comp_dict[m]["roc_auc"] * 100.0 for m in paradigms]
    else:
        paradigms = ["Local-Only", "Centralized Baseline", "Federated (FedAvg)", "Federated + Privacy"]
        accs, f1s, aucs = [86.11, 79.00, 76.12, 77.62], [85.17, 77.32, 75.13, 74.72], [93.51, 88.02, 84.99, 84.53]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=paradigms, y=accs, name="Accuracy (%)", marker_color="#0284c7" if st.session_state.get("theme") == "light" else "#38bdf8",
                         text=[f"{v:.1f}%" for v in accs], textposition="outside", hovertemplate="%{x}<br>Accuracy: %{y:.2f}%"))
    fig.add_trace(go.Bar(x=paradigms, y=f1s, name="F1-Score (%)", marker_color="#10b981" if st.session_state.get("theme") == "light" else "#34d399",
                         text=[f"{v:.1f}%" for v in f1s], textposition="outside", hovertemplate="%{x}<br>F1-Score: %{y:.2f}%"))
    fig.add_trace(go.Bar(x=paradigms, y=aucs, name="ROC-AUC (%)", marker_color="#f59e0b" if st.session_state.get("theme") == "light" else "#fbbf24",
                         text=[f"{v:.1f}%" for v in aucs], textposition="outside", hovertemplate="%{x}<br>ROC-AUC: %{y:.2f}%"))

    fig.update_layout(
        **get_plotly_layout(),
        barmode="group",
        height=430,
        yaxis=dict(title="Performance Metric (%)", range=[40, 108], gridcolor=grid_c),
        xaxis=dict(title="Learning Paradigm", gridcolor=grid_c)
    )
    return fig


def build_plotly_hospital_demographics(hosp_df=None, results=None):
    is_dark = (st.session_state.get("theme", "light") == "dark")
    font_color = "#f8fafc" if is_dark else "#0f2b48"
    grid_c = "#334155" if is_dark else "#f1f5f9"
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Demographics & Prevalence Skew", "Local Dataset Sizes"))

    names, counts, prevalences, avg_ages = [], [], [], []

    if hosp_df is not None and not hosp_df.empty:
        names = hosp_df["hospital_name"].tolist() if "hospital_name" in hosp_df.columns else [f"Hospital {chr(65+i)}" for i in range(len(hosp_df))]
        counts = hosp_df["patient_count"].tolist()
        prevalences = hosp_df["disease_prevalence_pct"].tolist()
        avg_ages = hosp_df["avg_age"].tolist()
    elif results and "hospital_data_summary" in results:
        summary = results["hospital_data_summary"]
        names = list(summary.keys())
        counts = [summary[k]["patient_count"] for k in names]
        prevalences = [summary[k]["disease_prevalence_pct"] for k in names]
        avg_ages = [summary[k]["avg_age"] for k in names]
    else:
        names = ["Hospital A", "Hospital B", "Hospital C", "Hospital D", "Hospital E"]
        counts = [1760, 1050, 890, 720, 580]
        prevalences = [7.16, 12.4, 25.8, 38.2, 54.1]
        avg_ages = [43.9, 48.2, 52.1, 55.6, 61.3]

    fig.add_trace(go.Bar(x=names, y=prevalences, name="Disease Prevalence (%)", marker_color="#0284c7" if not is_dark else "#38bdf8",
                         text=[f"{v:.1f}%" for v in prevalences], textposition="outside", hovertemplate="%{x}<br>Prevalence: %{y:.2f}%"), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=avg_ages, name="Average Patient Age", marker_color="#f59e0b" if not is_dark else "#fbbf24",
                         text=[f"{v:.1f}y" for v in avg_ages], textposition="outside", hovertemplate="%{x}<br>Avg Age: %{y:.1f} yrs"), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=counts, name="Patient Records", marker_color="#10b981" if not is_dark else "#34d399",
                         text=[f"{v:,}" for v in counts], textposition="outside", hovertemplate="%{x}<br>Records: %{y:,}"), row=1, col=2)

    fig.update_layout(**get_plotly_layout(), barmode="group", height=430)
    fig.update_yaxes(title_text="Metric Value", gridcolor=grid_c, row=1, col=1)
    fig.update_yaxes(title_text="Sample Count", gridcolor=grid_c, row=1, col=2)
    fig.update_annotations(font_size=13, font_color=font_color)
    return fig


def build_plotly_roc_curves(results=None):
    grid_c = "#334155" if st.session_state.get("theme", "light") == "dark" else "#f1f5f9"
    fig = go.Figure()
    colors = {
        "Local-Only": "#f59e0b",
        "Centralized Baseline": "#10b981",
        "Federated (FedAvg)": "#0284c7",
        "Federated + Privacy": "#ef4444",
        "Federated + DP": "#ef4444"
    }

    has_data = False
    if results and "method_comparison" in results:
        comp = results["method_comparison"]
        for method_name, res in comp.items():
            if "roc_curve" in res and res["roc_curve"]:
                fpr = res["roc_curve"]["fpr"]
                tpr = res["roc_curve"]["tpr"]
                auc = res.get("roc_auc", 0.5)
                c = colors.get(method_name, "#38bdf8")
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{method_name} (AUC={auc:.3f})",
                                         hovertemplate=f"{method_name}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<br>AUC: {auc:.3f}",
                                         line=dict(color=c, width=2.5)))
                has_data = True

    if not has_data:
        fig.add_trace(go.Scatter(x=[0, 0.05, 0.2, 0.5, 1.0], y=[0, 0.85, 0.92, 0.97, 1.0], mode="lines", name="Local-Only (AUC=0.935)", line=dict(color="#f59e0b", width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 0.1, 0.35, 0.65, 1.0], y=[0, 0.70, 0.85, 0.93, 1.0], mode="lines", name="Centralized Baseline (AUC=0.880)", line=dict(color="#10b981", width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 0.15, 0.4, 0.7, 1.0], y=[0, 0.65, 0.82, 0.91, 1.0], mode="lines", name="Federated (FedAvg) (AUC=0.850)", line=dict(color="#0284c7", width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 0.18, 0.42, 0.72, 1.0], y=[0, 0.62, 0.80, 0.89, 1.0], mode="lines", name="Federated + DP (AUC=0.845)", line=dict(color="#ef4444", width=2.5)))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Classifier (AUC=0.500)", line=dict(color="#94a3b8", width=2, dash="dash")))

    fig.update_layout(
        **get_plotly_layout(),
        height=430,
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[0, 1], gridcolor=grid_c),
        yaxis=dict(title="True Positive Rate (Recall)", range=[0, 1.05], gridcolor=grid_c)
    )
    return fig


def build_plotly_confusion_matrices(results=None):
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Local-Only Baseline", "Centralized Baseline", "Federated (FedAvg)", "Federated + DP Noise"))
    is_dark = (st.session_state.get("theme", "light") == "dark")
    txt_c = "#f8fafc" if is_dark else "#0f2b48"
    font_color = txt_c
    c_scale = "Blues" if not is_dark else "Tealgrn"
    
    methods = ["Local-Only", "Centralized Baseline", "Federated (FedAvg)", "Federated + DP"]
    positions = [(1,1), (1,2), (2,1), (2,2)]
    
    for idx, method_name in enumerate(methods):
        row, col = positions[idx]
        cm = [[463, 106], [33, 399]]
        
        if results and "method_comparison" in results and method_name in results["method_comparison"]:
            res = results["method_comparison"][method_name]
            raw_cm = res.get("confusion_matrix", cm)
            cm = raw_cm if len(raw_cm) == 2 else cm

        fig.add_trace(go.Heatmap(
            z=cm, x=["Low Risk (0)", "High Risk (1)"], y=["Low Risk (0)", "High Risk (1)"],
            colorscale=c_scale, showscale=False,
            hoverinfo="x+y+z",
            text=[[f"TN: {cm[0][0]}", f"FP: {cm[0][1]}"], [f"FN: {cm[1][0]}", f"TP: {cm[1][1]}"]],
            texttemplate="%{text}", textfont=dict(size=14, color=txt_c, family="-apple-system, sans-serif")
        ), row=row, col=col)

    fig.update_layout(**get_plotly_layout(), height=520)
    fig.update_annotations(font_size=13, font_color=font_color)
    return fig


def build_plotly_privacy_tradeoff(results=None):
    grid_c = "#334155" if st.session_state.get("theme", "light") == "dark" else "#f1f5f9"
    sigmas = [0.0, 0.1, 0.2, 0.5, 1.0]
    accs = [75.3, 75.4, 74.7, 68.4, 60.0]
    f1s = [75.3, 75.4, 74.7, 68.4, 60.0]
    epsilons = ["∞", "2.83", "1.42", "0.57", "0.28"]

    if results and "privacy_tradeoff" in results and results["privacy_tradeoff"]:
        p_res = results["privacy_tradeoff"]
        sigmas = [r["noise_multiplier"] for r in p_res]
        accs = [(r["accuracy"] * 100.0 if r["accuracy"] <= 1.0 else r["accuracy"]) for r in p_res]
        f1s = [(r["f1"] * 100.0 if r["f1"] <= 1.0 else r["f1"]) for r in p_res]
        epsilons = [str(r.get("estimated_epsilon", "inf")) for r in p_res]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sigmas, y=accs, mode="lines+markers+text", name="Accuracy (%)",
                             text=[f"ε={e}" for e in epsilons], textposition="top center",
                             hovertemplate="σ: %{x}<br>Accuracy: %{y:.2f}%",
                             line=dict(color="#0284c7" if st.session_state.get("theme") == "light" else "#38bdf8", width=2.5),
                             marker=dict(size=8, symbol="circle")))
    fig.add_trace(go.Scatter(x=sigmas, y=f1s, mode="lines+markers", name="F1-Score (%)",
                             hovertemplate="σ: %{x}<br>F1-Score: %{y:.2f}%",
                             line=dict(color="#10b981" if st.session_state.get("theme") == "light" else "#34d399", width=2.5, dash="dash"),
                             marker=dict(size=8, symbol="square")))

    fig.update_layout(
        **get_plotly_layout(),
        height=420,
        xaxis=dict(title="DP Noise Multiplier (sigma)", gridcolor=grid_c),
        yaxis=dict(title="Performance Metric (%)", range=[30, 100], gridcolor=grid_c)
    )
    return fig


def build_plotly_local_diagnosis_pie(h_row):
    pos = int(h_row['positive_cases'])
    neg = int(h_row['negative_cases'])
    is_dark = (st.session_state.get("theme", "light") == "dark")
    fig = go.Figure(data=[go.Pie(
        labels=["Disease Positive", "Disease Negative"],
        values=[pos, neg],
        hole=0.4,
        marker_colors=["#0284c7" if not is_dark else "#38bdf8", "#e2e8f0" if not is_dark else "#334155"],
        textinfo="label+percent",
        hovertemplate="%{label}<br>Patients: %{value:,}<br>Ratio: %{percent}"
    )])
    fig.update_layout(**get_plotly_layout(), height=340, showlegend=False)
    return fig


def build_plotly_local_clinical_bar(h_row):
    grid_c = "#334155" if st.session_state.get("theme", "light") == "dark" else "#f1f5f9"
    metrics_keys = ["BMI", "Systolic BP", "Glucose"]
    metrics_vals = [float(h_row['avg_bmi']), float(h_row['avg_blood_pressure_sys']), float(h_row['avg_glucose'])]
    fig = go.Figure(data=[go.Bar(
        x=metrics_keys,
        y=metrics_vals,
        marker_color=["#0284c7", "#10b981", "#f59e0b"],
        text=[f"{v:.1f}" for v in metrics_vals],
        textposition="outside",
        hovertemplate="%{x}<br>Mean Value: %{y:.2f}"
    )])
    fig.update_layout(**get_plotly_layout(), height=340, yaxis=dict(title="Mean Value", gridcolor=grid_c))
    return fig


import hashlib

def load_credentials() -> dict:
    auth_file = "config/auth_credentials.json"
    if os.path.exists(auth_file):
        try:
            with open(auth_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    admin_pass = os.getenv("FL_ADMIN_PASSWORD", "admin123")
    hosp_pass = os.getenv("FL_HOSPITAL_PASSWORD", "hospital123")

    return {
        "hospital_a": {"password_hash": hashlib.sha256(hosp_pass.encode()).hexdigest(), "password": hosp_pass, "role": "Hospital User", "hospital": "Hospital A"},
        "hospital_b": {"password_hash": hashlib.sha256(hosp_pass.encode()).hexdigest(), "password": hosp_pass, "role": "Hospital User", "hospital": "Hospital B"},
        "hospital_c": {"password_hash": hashlib.sha256(hosp_pass.encode()).hexdigest(), "password": hosp_pass, "role": "Hospital User", "hospital": "Hospital C"},
        "hospital_d": {"password_hash": hashlib.sha256(hosp_pass.encode()).hexdigest(), "password": hosp_pass, "role": "Hospital User", "hospital": "Hospital D"},
        "hospital_e": {"password_hash": hashlib.sha256(hosp_pass.encode()).hexdigest(), "password": hosp_pass, "role": "Hospital User", "hospital": "Hospital E"},
        "admin": {"password_hash": hashlib.sha256(admin_pass.encode()).hexdigest(), "password": admin_pass, "role": "System Administrator", "hospital": "Central Federated Server"}
    }

CREDENTIALS = load_credentials()

# Helper functions to load actual experiment outputs
@st.cache_data
def load_results():
    json_path = "results/experiment_results.json"
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        return json.load(f)

@st.cache_data
def load_training_metrics_csv():
    path = "results/training_metrics.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_model_comparison_csv():
    path = "results/model_comparison.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_hospital_distribution_csv():
    path = "results/hospital_distribution.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_config():
    config_path = "config/config.yaml"
    from config.config_loader import load_config as fallback_config
    return fallback_config(config_path)

def load_trained_private_model():
    pt_path = "models/federated_private_model.pt"
    pkl_path = "results/global_disease_model.pkl"
    
    if os.path.exists(pt_path):
        try:
            from models.disease_model import load_model_pt
            model = load_model_pt(pt_path, in_features=13)
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                    preprocessor = data.get("preprocessor", None)
            else:
                from client.preprocessing import HospitalDataPreprocessor
                preprocessor = HospitalDataPreprocessor()
            return {"model": model, "preprocessor": preprocessor, "in_features": 13}
        except Exception:
            pass

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            from models.disease_model import DiseasePredictionMLP, set_model_parameters
            model = DiseasePredictionMLP(in_features=data["in_features"])
            set_model_parameters(model, data["model_state"])
            return {"model": model, "preprocessor": data["preprocessor"], "in_features": data["in_features"]}
    
    return None


# -----------------------------------------------------------------------------
# 1. LOGIN SCREEN
# -----------------------------------------------------------------------------
def render_login_page():
    col_t1, col_t2 = st.columns([4, 1])
    with col_t2:
        current_theme = st.session_state.get("theme", "light")
        theme_label = "🌙 Dark Mode" if current_theme == "light" else "☀️ Light Mode"
        if st.button(theme_label, key="login_theme_toggle_btn", use_container_width=True):
            st.session_state.theme = "dark" if current_theme == "light" else "light"
            st.rerun()

    col_space_l, col_content, col_space_r = st.columns([0.02, 0.96, 0.02])
    
    with col_content:
        col_left, col_right = st.columns([1.1, 0.9], gap="large")
        
        with col_left:
            st.markdown('<div class="hero-tag">SMART HOSPITAL HEALTHCARE AI</div>', unsafe_allow_html=True)
            st.markdown('<div class="left-hero-title">Federated Healthcare AI</div>', unsafe_allow_html=True)
            st.markdown('<div class="left-hero-subtitle">Privacy-Preserving Disease Prediction in Smart Hospitals</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="badge-row">
                <span class="badge-item badge-teal">• Multi-Hospital Collaboration</span>
                <span class="badge-item badge-purple">• Federated Model Training</span>
                <span class="badge-item badge-green">• Privacy-Preserving Prediction</span>
            </div>
            """, unsafe_allow_html=True)
            
            with st.container(border=True):
                st.markdown("**Federated System Architecture**")
                st.text("""Hospital A --|
Hospital B --|
Hospital C --+-- Federated Server -- Global Model
Hospital D --|
Hospital E --|""")
            
        with col_right:
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown('<div class="login-card-title">Welcome Back</div>', unsafe_allow_html=True)
                st.markdown('<div class="login-card-subtitle">Sign in to access the healthcare AI platform</div>', unsafe_allow_html=True)
                
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
                if st.button("LOGIN", use_container_width=True):
                    user_key = username.strip().lower()
                    credentials = load_credentials()
                    if user_key in credentials:
                        user_info = credentials[user_key]
                        input_hash = hashlib.sha256(password.encode()).hexdigest()
                        expected_hash = user_info.get("password_hash")
                        if not expected_hash and "password" in user_info:
                            expected_hash = hashlib.sha256(user_info["password"].encode()).hexdigest()

                        if input_hash == expected_hash:
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.session_state.role = user_info["role"]
                            st.session_state.hospital_name = user_info["hospital"]
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")
                    else:
                        st.error("Invalid username or password.")

                
                st.markdown("""
                <div class="secure-access-caption">
                    <span class="secure-dot"></span> Secure Hospital Access
                </div>
                """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# GLOBAL NAVIGATION BAR
# -----------------------------------------------------------------------------
def render_navigation_bar():
    col_n1, col_n2, col_n3 = st.columns([3.5, 1.2, 1])
    with col_n1:
        st.markdown(f"""
        <div class="top-nav-bar">
            <div class="nav-brand">Federated Healthcare AI Platform</div>
            <div class="nav-meta">
                <span>User: <strong>{st.session_state.username}</strong></span>
                <span>Role: <strong>{st.session_state.role}</strong></span>
                <span>Node: <strong>{st.session_state.hospital_name}</strong></span>
                <span class="nav-badge">Data Stays Local</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_n2:
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        current_theme = st.session_state.get("theme", "light")
        theme_label = "🌙 Dark Mode" if current_theme == "light" else "☀️ Light Mode"
        if st.button(theme_label, key="nav_theme_toggle_btn", use_container_width=True):
            st.session_state.theme = "dark" if current_theme == "light" else "light"
            st.rerun()
    with col_n3:
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("Logout", key="nav_logout_btn", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.session_state.role = ""
            st.session_state.hospital_name = ""
            st.rerun()


# -----------------------------------------------------------------------------
# 2. HOSPITAL USER WEBSITE (ROLE-BASED MODULES)
# -----------------------------------------------------------------------------
def render_hospital_user_dashboard(results, config, tr_df, comp_df, hosp_df):
    h_name = st.session_state.hospital_name
    
    st.sidebar.markdown(f"""
    <div class="sidebar-profile-card">
        <div class="sidebar-profile-badge">HOSPITAL NODE</div>
        <div class="sidebar-profile-title">{h_name}</div>
        <div class="sidebar-profile-sub">Role: {st.session_state.role}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### Navigation")
    raw_nav = st.sidebar.radio(
        "Select Module",
        [
            "📊 Dashboard",
            "🏥 Local Data",
            "🩺 Disease Prediction",
            "🧠 Model Training",
            "🔗 Federated Learning",
            "🔒 Privacy",
            "ℹ️ About"
        ],
        label_visibility="collapsed"
    )
    nav_selection = raw_nav.split(" ", 1)[1] if " " in raw_nav else raw_nav

    st.sidebar.markdown("---")
    current_theme = st.session_state.get("theme", "light")
    sidebar_theme_label = "🌙 Switch to Dark Mode" if current_theme == "light" else "☀️ Switch to Light Mode"
    if st.sidebar.button(sidebar_theme_label, key="hospital_sidebar_theme_toggle", use_container_width=True):
        st.session_state.theme = "dark" if current_theme == "light" else "light"
        st.rerun()

    # Filter hospital-specific row
    h_row = None
    if hosp_df is not None:
        h_rows = hosp_df[hosp_df["hospital_name"] == h_name]
        if not h_rows.empty:
            h_row = h_rows.iloc[0]

    # --- VIEW 1: DASHBOARD ---
    if nav_selection == "Dashboard":
        st.markdown(f'<div class="page-title">{h_name} - Secure Hospital Node</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Isolated Clinical Node Interface & Operations Console</div>', unsafe_allow_html=True)
        st.markdown('<div class="status-perimeter-badge">Local Data Perimeter Enforced</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="privacy-banner">
            Patient records remain within the hospital boundary. Only parameter updates participate in server aggregation.
        </div>
        """, unsafe_allow_html=True)

        if h_row is not None:
            p_cnt = f"{int(h_row['patient_count']):,}"
            p_cases = f"{int(h_row['positive_cases']):,}"
            prev_str = f"{float(h_row['disease_prevalence_pct']):.2f}%"
            avg_str = f"{float(h_row['avg_age']):.1f} yrs"

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Local Cohort</div>
                    <div class="metric-value">{p_cnt}</div>
                    <div class="metric-caption">Patient Records</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Disease Cases</div>
                    <div class="metric-value">{p_cases}</div>
                    <div class="metric-caption">Positive Diagnoses</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Prevalence Rate</div>
                    <div class="metric-value">{prev_str}</div>
                    <div class="metric-caption">Local Population</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Average Age</div>
                    <div class="metric-value">{avg_str}</div>
                    <div class="metric-caption">Cohort Demography</div>
                </div>
                """, unsafe_allow_html=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            with st.container(border=True):
                st.markdown('<div class="section-header">Local Model Specifications</div>', unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown("""
                    <div class="info-card">
                        <div class="info-card-label">Model Architecture</div>
                        <div class="info-card-value">PyTorch MLP</div>
                        <div class="info-card-sub">13 -> 32 -> 16 -> 1</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div class="info-card">
                        <div class="info-card-label">Local Training</div>
                        <div class="info-card-value">3 Epochs</div>
                        <div class="info-card-sub">Per Communication Round</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown("""
                    <div class="info-card">
                        <div class="info-card-label">Optimizer</div>
                        <div class="info-card-value">Adam</div>
                        <div class="info-card-sub">Learning Rate = 0.001</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div class="info-card">
                        <div class="info-card-label">Batch Size</div>
                        <div class="info-card-value">32 Records</div>
                        <div class="info-card-sub">Local DataLoader</div>
                    </div>
                    """, unsafe_allow_html=True)

        with col_d2:
            with st.container(border=True):
                st.markdown('<div class="section-header">Federated Network Status</div>', unsafe_allow_html=True)
                f1, f2 = st.columns(2)
                with f1:
                    st.markdown("""
                    <div class="info-card">
                        <div class="info-card-label">FL Rounds</div>
                        <div class="info-card-value">10 Rounds</div>
                        <div class="info-card-sub">Communication Completed</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div class="info-card">
                        <div class="info-card-label">Aggregation</div>
                        <div class="info-card-value">FedAvg</div>
                        <div class="info-card-sub">Sample-Weighted Average</div>
                    </div>
                    """, unsafe_allow_html=True)
                with f2:
                    st.markdown("""
                    <div class="info-card">
                        <div class="info-card-label">Differential Privacy</div>
                        <div class="info-card-value">C=1.0, σ=0.1</div>
                        <div class="info-card-sub">L2 Clip + Gaussian Noise</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div class="info-card">
                        <div class="info-card-label">Data Leakage Risk</div>
                        <div class="info-card-value"><span class="badge-status-green">Zero Leakage</span></div>
                        <div class="info-card-sub">Local Data Isolation</div>
                    </div>
                    """, unsafe_allow_html=True)

    # --- VIEW 2: LOCAL DATA ---
    elif nav_selection == "Local Data":
        st.markdown(f'<div class="page-title">{h_name} - Local Dataset Statistics</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Aggregate Clinical Summaries & Demographic Breakdown</div>', unsafe_allow_html=True)

        st.info("Displaying aggregate demographic and clinical metrics for this institution only. Individual patient identifiers and medical history logs are restricted to local hospital database access.")

        if h_row is not None:
            with st.container(border=True):
                st.markdown('<div class="section-header">Institution Cohort Metrics Summary</div>', unsafe_allow_html=True)
                h_single_df = pd.DataFrame([h_row])
                st.dataframe(h_single_df.style.format({
                    "patient_count": "{:,}",
                    "positive_cases": "{:,}",
                    "negative_cases": "{:,}",
                    "disease_prevalence_pct": "{:.2f}%",
                    "avg_age": "{:.2f} yrs",
                    "avg_bmi": "{:.2f}",
                    "avg_blood_pressure_sys": "{:.2f} mmHg",
                    "avg_glucose": "{:.2f} mg/dL"
                }), use_container_width=True)

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                with st.container(border=True):
                    st.markdown('<div class="section-header">Patient Diagnosis Distribution</div>', unsafe_allow_html=True)
                    fig_pie = build_plotly_local_diagnosis_pie(h_row)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
            with col_chart2:
                with st.container(border=True):
                    st.markdown('<div class="section-header">Clinical Mean Benchmarks</div>', unsafe_allow_html=True)
                    fig_bar = build_plotly_local_clinical_bar(h_row)
                    st.plotly_chart(fig_bar, use_container_width=True)

    # --- VIEW 3: DISEASE PREDICTION ---
    elif nav_selection == "Disease Prediction":
        st.markdown(f'<div class="page-title">{h_name} - Disease Risk Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Run patient risk classification using the global federated model</div>', unsafe_allow_html=True)

        model_artifact = load_trained_private_model()
        if model_artifact is None:
            st.warning("Trained global model checkpoint not found. Please contact System Administrator.")
        else:
            with st.container(border=True):
                st.markdown('<div class="section-header">Patient Clinical Parameters Input</div>', unsafe_allow_html=True)
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    input_age = st.slider("Patient Age", 18, 90, 58)
                    input_gender = st.selectbox("Gender", ["Male", "Female"])
                    input_bp_sys = st.slider("Systolic Blood Pressure (mmHg)", 90, 190, 138)
                    input_bp_dia = st.slider("Diastolic Blood Pressure (mmHg)", 60, 120, 88)

                with col_p2:
                    input_glucose = st.slider("Blood Glucose (mg/dL)", 70, 280, 142)
                    input_cholesterol = st.slider("Cholesterol Level (mg/dL)", 120, 350, 225)
                    input_bmi = st.slider("Body Mass Index (BMI)", 16.0, 45.0, 29.5)
                    input_hr = st.slider("Heart Rate (bpm)", 50, 115, 75)

                with col_p3:
                    input_smoke = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
                    input_htn = st.selectbox("Hypertension Diagnosis", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                    input_dm = st.selectbox("Diabetes Diagnosis", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                    input_fh = st.selectbox("Family History of Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

                if st.button("PREDICT DISEASE RISK", use_container_width=True):
                    patient_dict = {
                        "patient_id": [f"PAT_{h_name[:3]}"],
                        "age": [input_age],
                        "gender": [input_gender],
                        "blood_pressure_sys": [input_bp_sys],
                        "blood_pressure_dia": [input_bp_dia],
                        "glucose": [input_glucose],
                        "cholesterol": [input_cholesterol],
                        "bmi": [input_bmi],
                        "heart_rate": [input_hr],
                        "smoking_status": [input_smoke],
                        "hypertension": [input_htn],
                        "diabetes": [input_dm],
                        "family_history": [input_fh],
                        "disease_risk": [0]
                    }
                    df_single = pd.DataFrame(patient_dict)
                    preprocessor = model_artifact["preprocessor"]
                    X_proc, _ = preprocessor.preprocess_df(df_single, is_train=False)
                    
                    inference_model = model_artifact["model"]
                    probs = float(inference_model.predict_proba(X_proc).flatten()[0])
                    pred_label = "High Risk (Disease Positive)" if probs >= 0.5 else "Low Risk (Disease Negative)"
                    
                    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
                    with st.container(border=True):
                        st.markdown('<div class="section-header">Clinical Prediction Output</div>', unsafe_allow_html=True)
                        c_out1, c_out2 = st.columns(2)
                        with c_out1:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-title">Classification Outcome</div>
                                <div class="metric-value" style="font-size: 1.25rem;">{pred_label}</div>
                                <div class="metric-caption">Trained Model Inference</div>
                            </div>
                            """, unsafe_allow_html=True)
                        prob_str = f"{probs * 100.0:.2f}%"
                        with c_out2:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-title">Disease Risk Probability</div>
                                <div class="metric-value">{prob_str}</div>
                                <div class="metric-caption">Confidence Score</div>
                            </div>
                            """, unsafe_allow_html=True)

            st.caption("Disclaimer: For academic research prototype demonstration only. Not intended for clinical diagnosis.")

    # --- VIEW 4: MODEL TRAINING ---
    elif nav_selection == "Model Training":
        st.markdown(f'<div class="page-title">{h_name} - Local Model Training</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">PyTorch Model Specifications & Local Execution Status</div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="section-header">Local Training Specifications</div>', unsafe_allow_html=True)
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.markdown("""
                <div class="info-card">
                    <div class="info-card-label">Model Architecture</div>
                    <div class="info-card-value">PyTorch MLP</div>
                    <div class="info-card-sub">Multi-Layer Perceptron</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="info-card">
                    <div class="info-card-label">Loss Function</div>
                    <div class="info-card-value">BCEWithLogitsLoss</div>
                    <div class="info-card-sub">Binary Cross Entropy</div>
                </div>
                """, unsafe_allow_html=True)
            with col_t2:
                st.markdown("""
                <div class="info-card">
                    <div class="info-card-label">Layer Structure</div>
                    <div class="info-card-value">13 -> 32 -> 16 -> 1</div>
                    <div class="info-card-sub">ReLU + Dropout (0.1)</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="info-card">
                    <div class="info-card-label">Local Epochs</div>
                    <div class="info-card-value">3 Epochs / Round</div>
                    <div class="info-card-sub">Isolated Optimization</div>
                </div>
                """, unsafe_allow_html=True)
            with col_t3:
                st.markdown("""
                <div class="info-card">
                    <div class="info-card-label">Optimizer</div>
                    <div class="info-card-value">Adam</div>
                    <div class="info-card-sub">Learning Rate = 0.001</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="info-card">
                    <div class="info-card-label">Local Batch Size</div>
                    <div class="info-card-value">32 Records</div>
                    <div class="info-card-sub">Isolated Memory Space</div>
                </div>
                """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="section-header">Local Training Execution Protocol</div>', unsafe_allow_html=True)
            st.markdown(r"""
            1. **Dataset Loading**: Local clinical records are partitioned into training and validation sets.
            2. **Feature Preprocessing**: Numerical clinical attributes are standard-scaled; categorical variables are one-hot encoded.
            3. **Local Gradient Descent**: Model parameters are optimized locally for 3 epochs per communication round.
            4. **Weight Delta Computation**: Parameter update vector $\Delta w = w_{local} - w_{global}$ is computed.
            5. **Privacy Obfuscation**: $L_2$ norm clipping ($C=1.0$) and Gaussian noise ($\sigma=0.1$) are applied before transmission.
            """)

    # --- VIEW 5: FEDERATED LEARNING ---
    elif nav_selection == "Federated Learning":
        st.markdown(f'<div class="page-title">{h_name} - Federated Learning Participation</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Collaboration Protocol & Model Aggregation Overview</div>', unsafe_allow_html=True)

        if h_row is not None:
            n_local = int(h_row["patient_count"])
            n_total = config["dataset"]["num_samples"]
            weight_pct = (n_local / n_total) * 100.0
            weight_str = f"{weight_pct:.1f}%"
            
            c_f1, c_f2, c_f3 = st.columns(3)
            with c_f1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Local Cohort Weight</div>
                    <div class="metric-value">{weight_str}</div>
                    <div class="metric-caption">{n_local:,} / {n_total:,} Records</div>
                </div>
                """, unsafe_allow_html=True)
            with c_f2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">FL Communication Rounds</div>
                    <div class="metric-value">10</div>
                    <div class="metric-caption">Rounds Completed</div>
                </div>
                """, unsafe_allow_html=True)
            with c_f3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Aggregation Algorithm</div>
                    <div class="metric-value">FedAvg</div>
                    <div class="metric-caption">Sample-Weighted Average</div>
                </div>
                """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="section-header">Federated Learning Protocol Summary</div>', unsafe_allow_html=True)
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown(f"""
                <div class="step-card">
                    <div class="step-card-num">STEP 1</div>
                    <div class="step-card-title">Local Training</div>
                    <div class="step-card-desc">{h_name} trains the MLP model locally on isolated patient data for 3 epochs.</div>
                </div>
                <div class="step-card">
                    <div class="step-card-num">STEP 2</div>
                    <div class="step-card-title">Weight Update Computation</div>
                    <div class="step-card-desc">Parameter update delta &Delta;w = w_local - w_global is computed.</div>
                </div>
                <div class="step-card">
                    <div class="step-card-num">STEP 3</div>
                    <div class="step-card-title">Privacy Noise Addition</div>
                    <div class="step-card-desc">Update vector is clipped (C=1.0) and perturbed with Gaussian noise (&sigma;=0.1).</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s2:
                st.markdown(f"""
                <div class="step-card">
                    <div class="step-card-num">STEP 4</div>
                    <div class="step-card-title">Parameter Transmission</div>
                    <div class="step-card-desc">Only obfuscated parameter updates &Delta;w are transmitted to the server.</div>
                </div>
                <div class="step-card">
                    <div class="step-card-num">STEP 5</div>
                    <div class="step-card-title">FedAvg Aggregation</div>
                    <div class="step-card-desc">Central server aggregates updates using sample weights (n_k / N).</div>
                </div>
                <div class="step-card">
                    <div class="step-card-num">STEP 6</div>
                    <div class="step-card-title">Global Broadcast</div>
                    <div class="step-card-desc">Server returns updated global model parameters to {h_name}.</div>
                </div>
                """, unsafe_allow_html=True)

    # --- VIEW 6: PRIVACY ---
    elif nav_selection == "Privacy":
        st.markdown(f'<div class="page-title">{h_name} - Privacy Status</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Differential Noise & Parameter Clipping Mechanisms</div>', unsafe_allow_html=True)

        from privacy.differential_privacy import DifferentialPrivacyManager
        dp_mgr = DifferentialPrivacyManager(clip_norm=1.0, noise_multiplier=0.1, delta=1e-5, enabled=True)
        eps_rdp = dp_mgr.estimate_epsilon_rdp(num_rounds=10)
        eps_heur = dp_mgr.estimate_epsilon_heuristic(num_rounds=10)

        st.markdown("""
        <div class="indicator-grid">
            <div class="indicator-card">
                <div class="indicator-title">L2 Norm Clipping</div>
                <div class="indicator-desc">Threshold C = 1.0 limits single hospital gradient impact</div>
            </div>
            <div class="indicator-card">
                <div class="indicator-title">Gaussian Noise Perturbation</div>
                <div class="indicator-desc">Noise multiplier sigma = 0.1 obfuscates parameter weights</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="section-header">Differential Privacy Budget Bounds</div>', unsafe_allow_html=True)
            col_eps1, col_eps2 = st.columns(2)
            with col_eps1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Opacus RDP Formal Privacy Bound (ε)</div>
                    <div class="metric-value">ε = {eps_rdp}</div>
                    <div class="metric-caption">Rényi DP Accountant (δ = 1e-5, T=10 rounds)</div>
                </div>
                """, unsafe_allow_html=True)
            with col_eps2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Empirical Heuristic Bound (ε)</div>
                    <div class="metric-value">ε = {eps_heur}</div>
                    <div class="metric-caption">Baseline Heuristic Formula (δ = 1e-5, T=10 rounds)</div>
                </div>
                """, unsafe_allow_html=True)

        st.info("Data Perimeter Enforced: Raw medical records never leave local hospital storage. Only obfuscated parameter updates participate in server aggregation.")

    # --- VIEW 7: ABOUT ---
    elif nav_selection == "About":
        st.markdown('<div class="page-title">About the Federated Healthcare AI System</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Privacy-Preserving Disease Risk Prediction Prototype</div>', unsafe_allow_html=True)

        col_ab1, col_ab2 = st.columns([1.1, 0.9], gap="medium")
        with col_ab1:
            with st.container(border=True):
                st.markdown('<div class="section-header">Mission & Project Overview</div>', unsafe_allow_html=True)
                st.markdown("""
                This platform demonstrates multi-hospital collaborative machine learning for disease risk prediction without centralizing sensitive electronic health records (EHR).
                
                By combining **Federated Averaging (FedAvg)** with **$L_2$ gradient update clipping** and **Gaussian noise perturbation**, participating institutions train a global predictive model while preserving strict data perimeter guarantees under HIPAA and GDPR guidelines.
                """)
        with col_ab2:
            s1, s2 = st.columns(2)
            with s1:
                st.markdown("""
                <div class="info-card">
                    <div class="info-card-label">Simulated Cohort</div>
                    <div class="info-card-value">5,000 Records</div>
                    <div class="info-card-sub">Synthetic Patients</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="info-card">
                    <div class="info-card-label">Model Architecture</div>
                    <div class="info-card-value">PyTorch MLP</div>
                    <div class="info-card-sub">32 -> 16 -> 1</div>
                </div>
                """, unsafe_allow_html=True)
            with s2:
                st.markdown("""
                <div class="info-card">
                    <div class="info-card-label">Hospital Nodes</div>
                    <div class="info-card-value">5 Hospitals</div>
                    <div class="info-card-sub">Non-IID Partition</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="info-card">
                    <div class="info-card-label">Privacy Engine</div>
                    <div class="info-card-value">C=1.0, σ=0.1</div>
                    <div class="info-card-sub">L2 Clip + Gaussian Noise</div>
                </div>
                """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="section-header">Federated System Network Topology</div>', unsafe_allow_html=True)
            st.text("""Hospital A --|
Hospital B --|
Hospital C --+-- Federated Server -- Global Model
Hospital D --|
Hospital E --|""")


# -----------------------------------------------------------------------------
# 3. SYSTEM ADMINISTRATOR WEBSITE (FEDERATED CONTROL CENTER)
# -----------------------------------------------------------------------------
def render_admin_dashboard(results, config, tr_df, comp_df, hosp_df):
    st.sidebar.markdown(f"""
    <div class="sidebar-profile-card">
        <div class="sidebar-profile-badge">CENTRAL SUPERVISION</div>
        <div class="sidebar-profile-title">Control Center</div>
        <div class="sidebar-profile-sub">Role: {st.session_state.role}</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### Navigation")
    raw_admin_nav = st.sidebar.radio(
        "Select View",
        [
            "📊 Overview",
            "🏥 Hospital Network",
            "⚡ Federated Training",
            "⚖️ Model Comparison",
            "🩺 Disease Prediction",
            "🔒 Privacy & Security",
            "🏗️ Architecture",
            "📁 Experiment Results"
        ],
        label_visibility="collapsed"
    )
    admin_nav = raw_admin_nav.split(" ", 1)[1] if " " in raw_admin_nav else raw_admin_nav

    st.sidebar.markdown("---")
    current_theme = st.session_state.get("theme", "light")
    sidebar_theme_label = "🌙 Switch to Dark Mode" if current_theme == "light" else "☀️ Switch to Light Mode"
    if st.sidebar.button(sidebar_theme_label, key="admin_sidebar_theme_toggle", use_container_width=True):
        st.session_state.theme = "dark" if current_theme == "light" else "light"
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Controls")
    with st.sidebar.container(border=True):
        if st.sidebar.button("Run Complete Pipeline", use_container_width=True):
            with st.spinner("Executing Complete FL Pipeline..."):
                from main import run_pipeline
                run_pipeline()
                st.success("Pipeline Execution Complete")
                st.rerun()

        if st.sidebar.button("Reload Saved Results", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # --- VIEW 1: OVERVIEW ---
    if admin_nav == "Overview":
        st.markdown('<div class="page-title">Federated Healthcare Control Center</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">System Administrator Multi-Hospital Supervision Console</div>', unsafe_allow_html=True)

        dp_res = results.get("method_comparison", {}).get("Federated + Privacy", {})
        
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Hospital Nodes</div>
                <div class="metric-value">5 Nodes</div>
                <div class="metric-caption">Hospitals A to E</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Cohort Size</div>
                <div class="metric-value">{config['dataset']['num_samples']:,}</div>
                <div class="metric-caption">Total Records</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">FL Rounds</div>
                <div class="metric-value">{config['federated']['num_rounds']}</div>
                <div class="metric-caption">FedAvg Communication</div>
            </div>
            """, unsafe_allow_html=True)
        glob_acc_str = f"{dp_res.get('accuracy', 0)*100:.2f}%"
        with c4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Global Accuracy</div>
                <div class="metric-value">{glob_acc_str}</div>
                <div class="metric-caption">Private Model</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Global ROC-AUC</div>
                <div class="metric-value">{dp_res.get('roc_auc', 0):.4f}</div>
                <div class="metric-caption">Private Model</div>
            </div>
            """, unsafe_allow_html=True)

        col_ov1, col_ov2 = st.columns(2)
        with col_ov1:
            with st.container(border=True):
                st.markdown('<div class="section-header">Global Convergence Curve</div>', unsafe_allow_html=True)
                fig_conv = build_plotly_convergence_chart(tr_df=tr_df, results=results)
                st.plotly_chart(fig_conv, use_container_width=True)
        
        with col_ov2:
            with st.container(border=True):
                st.markdown('<div class="section-header">Learning Approach Comparison</div>', unsafe_allow_html=True)
                fig_comp = build_plotly_model_comparison_bar(comp_df=comp_df, results=results)
                st.plotly_chart(fig_comp, use_container_width=True)

        with st.container(border=True):
            st.markdown('<div class="section-header">Data Privacy Directive & Aggregation Rule</div>', unsafe_allow_html=True)
            st.markdown(r"""
            Raw patient records remain strictly local inside hospital client boundaries. The central server receives parameter weight updates only, aggregated using sample-count weighted FedAvg:
            
            $$W_{global} = \sum_{k=1}^5 \frac{n_k}{N} W_k$$
            """)

    # --- VIEW 2: HOSPITAL NETWORK ---
    elif admin_nav == "Hospital Network":
        st.markdown('<div class="page-title">Hospital Network & Non-IID Cohort Breakdown</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Aggregate Data Distributions Across Regional Hospital Nodes</div>', unsafe_allow_html=True)

        if hosp_df is not None:
            with st.container(border=True):
                st.markdown('<div class="section-header">Hospital Cohort Breakdown Summary (`results/hospital_distribution.csv`)</div>', unsafe_allow_html=True)
                st.dataframe(hosp_df.style.format({
                    "patient_count": "{:,}",
                    "positive_cases": "{:,}",
                    "negative_cases": "{:,}",
                    "disease_prevalence_pct": "{:.2f}%",
                    "avg_age": "{:.2f} yrs",
                    "avg_bmi": "{:.2f}",
                    "avg_blood_pressure_sys": "{:.2f} mmHg",
                    "avg_glucose": "{:.2f} mg/dL"
                }), use_container_width=True)

        with st.container(border=True):
            st.markdown('<div class="section-header">Hospital Demographic Skew & Disease Prevalence (Non-IID Partition)</div>', unsafe_allow_html=True)
            fig_hosp = build_plotly_hospital_demographics(hosp_df=hosp_df, results=results)
            st.plotly_chart(fig_hosp, use_container_width=True)

    # --- VIEW 3: FEDERATED TRAINING ---
    elif admin_nav == "Federated Training":
        st.markdown('<div class="page-title">Federated Training & Round Progress</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Communication Rounds & Aggregation Metrics</div>', unsafe_allow_html=True)

        if tr_df is not None:
            with st.container(border=True):
                st.markdown('<div class="section-header">Round-by-Round Training Metrics (`results/training_metrics.csv`)</div>', unsafe_allow_html=True)
                
                model_filter = st.selectbox(
                    "Filter Paradigm:",
                    ["All", "Federated (Standard FedAvg)", "Federated + Privacy (DP Noise + L2 Clip)"]
                )
                
                filtered_tr = tr_df if model_filter == "All" else tr_df[tr_df["model"].str.contains("Privacy" if "Privacy" in model_filter else "Standard")]
                st.dataframe(filtered_tr.style.format({
                    "train_loss": "{:.4f}",
                    "validation_loss": "{:.4f}",
                    "accuracy": "{:.4f}",
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1": "{:.4f}",
                    "roc_auc": "{:.4f}",
                    "training_time": "{:.2f}s"
                }), use_container_width=True)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            with st.container(border=True):
                st.markdown('<div class="section-header">Training Convergence</div>', unsafe_allow_html=True)
                fig_conv_tr = build_plotly_convergence_chart(tr_df=tr_df, results=results)
                st.plotly_chart(fig_conv_tr, use_container_width=True)
        with col_f2:
            with st.container(border=True):
                st.markdown('<div class="section-header">Privacy-Utility Tradeoff</div>', unsafe_allow_html=True)
                fig_priv = build_plotly_privacy_tradeoff(results=results)
                st.plotly_chart(fig_priv, use_container_width=True)

    # --- VIEW 4: MODEL COMPARISON ---
    elif admin_nav == "Model Comparison":
        st.markdown('<div class="page-title">Empirical Benchmark Model Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Comparative Analysis Across All 4 Machine Learning Paradigms</div>', unsafe_allow_html=True)

        if comp_df is not None:
            with st.container(border=True):
                st.markdown('<div class="section-header">Empirical Results Table (`results/model_comparison.csv`)</div>', unsafe_allow_html=True)
                st.dataframe(comp_df.style.format({
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1-Score": "{:.4f}",
                    "ROC-AUC": "{:.4f}",
                    "Training Time (s)": "{:.2f}s"
                }), use_container_width=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            with st.container(border=True):
                st.markdown('<div class="section-header">Performance Bar Chart</div>', unsafe_allow_html=True)
                fig_bar = build_plotly_model_comparison_bar(comp_df=comp_df, results=results)
                st.plotly_chart(fig_bar, use_container_width=True)
            with st.container(border=True):
                st.markdown('<div class="section-header">ROC Curves Comparison</div>', unsafe_allow_html=True)
                fig_roc = build_plotly_roc_curves(results=results)
                st.plotly_chart(fig_roc, use_container_width=True)
        with col_c2:
            with st.container(border=True):
                st.markdown('<div class="section-header">Confusion Matrix Comparison Grid</div>', unsafe_allow_html=True)
                fig_cm = build_plotly_confusion_matrices(results=results)
                st.plotly_chart(fig_cm, use_container_width=True)

    # --- VIEW 5: DISEASE PREDICTION ---
    elif admin_nav == "Disease Prediction":
        st.markdown('<div class="page-title">Centralized Clinical Disease Prediction Console</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Run inference using the trained global federated model checkpoint</div>', unsafe_allow_html=True)

        model_artifact = load_trained_private_model()
        if model_artifact is None:
            st.warning("Trained model checkpoint (models/federated_private_model.pt) not found. Run the pipeline to save the trained model.")
        else:
            with st.container(border=True):
                st.markdown('<div class="section-header">Input Clinical Patient Profile</div>', unsafe_allow_html=True)
                col_p1, col_p2, col_p3 = st.columns(3)
                
                with col_p1:
                    input_age = st.slider("Patient Age", 18, 90, 58)
                    input_gender = st.selectbox("Gender", ["Male", "Female"])
                    input_bp_sys = st.slider("Systolic Blood Pressure (mmHg)", 90, 190, 138)
                    input_bp_dia = st.slider("Diastolic Blood Pressure (mmHg)", 60, 120, 88)

                with col_p2:
                    input_glucose = st.slider("Blood Glucose (mg/dL)", 70, 280, 142)
                    input_cholesterol = st.slider("Cholesterol Level (mg/dL)", 120, 350, 225)
                    input_bmi = st.slider("Body Mass Index (BMI)", 16.0, 45.0, 29.5)
                    input_hr = st.slider("Heart Rate (bpm)", 50, 115, 75)

                with col_p3:
                    input_smoke = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
                    input_htn = st.selectbox("Hypertension Diagnosis", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                    input_dm = st.selectbox("Diabetes Diagnosis", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                    input_fh = st.selectbox("Family History of Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

                if st.button("PREDICT DISEASE RISK", use_container_width=True):
                    patient_dict = {
                        "patient_id": ["PAT_ADMIN_DEMO"],
                        "age": [input_age],
                        "gender": [input_gender],
                        "blood_pressure_sys": [input_bp_sys],
                        "blood_pressure_dia": [input_bp_dia],
                        "glucose": [input_glucose],
                        "cholesterol": [input_cholesterol],
                        "bmi": [input_bmi],
                        "heart_rate": [input_hr],
                        "smoking_status": [input_smoke],
                        "hypertension": [input_htn],
                        "diabetes": [input_dm],
                        "family_history": [input_fh],
                        "disease_risk": [0]
                    }
                    df_single = pd.DataFrame(patient_dict)
                    preprocessor = model_artifact["preprocessor"]
                    X_proc, _ = preprocessor.preprocess_df(df_single, is_train=False)
                    
                    inference_model = model_artifact["model"]
                    probs = float(inference_model.predict_proba(X_proc).flatten()[0])
                    pred_label = "High Risk (Disease Positive)" if probs >= 0.5 else "Low Risk (Disease Negative)"
                    
                    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
                    with st.container(border=True):
                        st.markdown('<div class="section-header">Clinical Prediction Output</div>', unsafe_allow_html=True)
                        col_res1, col_res2 = st.columns(2)
                        with col_res1:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-title">Classification Outcome</div>
                                <div class="metric-value" style="font-size: 1.25rem;">{pred_label}</div>
                                <div class="metric-caption">Trained Model Inference</div>
                            </div>
                            """, unsafe_allow_html=True)
                        prob_admin_str = f"{probs * 100.0:.2f}%"
                        with col_res2:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-title">Disease Risk Probability</div>
                                <div class="metric-value">{prob_admin_str}</div>
                                <div class="metric-caption">Confidence Score</div>
                            </div>
                            """, unsafe_allow_html=True)

            st.caption("Disclaimer: Academic prototype demonstration only. Not intended for clinical diagnosis.")

    # --- VIEW 6: PRIVACY & SECURITY ---
    elif admin_nav == "Privacy & Security":
        st.markdown('<div class="page-title">Privacy Protection & Noise Mechanism</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Data Boundary Enforcement & Gradient Obfuscation Protocol</div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="section-header">Data Perimeter & Privacy Workflow</div>', unsafe_allow_html=True)
            
            p_c1, p_a1, p_c2, p_a2, p_c3, p_a3, p_c4, p_a4, p_c5 = st.columns([1.1, 0.2, 1.0, 0.2, 1.2, 0.2, 1.1, 0.2, 1.1])
            
            with p_c1:
                with st.container(border=True):
                    st.markdown("**Raw Data**")
                    st.caption("Stays Local")
                    st.caption("Isolated Node")

            with p_a1:
                st.markdown("### →")

            with p_c2:
                with st.container(border=True):
                    st.markdown("**Local Training**")
                    st.caption("PyTorch MLP")
                    st.caption("Weight Delta")

            with p_a2:
                st.markdown("### →")

            with p_c3:
                with st.container(border=True):
                    st.markdown("**Privacy Protection**")
                    st.caption("L2 Clipping (C=1.0)")
                    st.caption("Gaussian Noise")

            with p_a3:
                st.markdown("### →")

            with p_c4:
                with st.container(border=True):
                    st.markdown("**FedAvg Server**")
                    st.caption("Aggregator")
                    st.caption("Weighted Avg")

            with p_a4:
                st.markdown("### →")

            with p_c5:
                with st.container(border=True):
                    st.markdown("**Global Model**")
                    st.caption("Updated Params")
                    st.caption("Broadcast Back")

        from privacy.differential_privacy import DifferentialPrivacyManager
        dp_mgr = DifferentialPrivacyManager(clip_norm=1.0, noise_multiplier=0.1, delta=1e-5, enabled=True)
        eps_rdp = dp_mgr.estimate_epsilon_rdp(num_rounds=10)
        eps_heur = dp_mgr.estimate_epsilon_heuristic(num_rounds=10)

        with st.container(border=True):
            st.markdown('<div class="section-header">Differential Privacy Budget Bounds</div>', unsafe_allow_html=True)
            col_eps1, col_eps2 = st.columns(2)
            with col_eps1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Opacus RDP Formal Privacy Bound (ε)</div>
                    <div class="metric-value">ε = {eps_rdp}</div>
                    <div class="metric-caption">Rényi DP Accountant (δ = 1e-5, T=10 rounds)</div>
                </div>
                """, unsafe_allow_html=True)
            with col_eps2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Empirical Heuristic Bound (ε)</div>
                    <div class="metric-value">ε = {eps_heur}</div>
                    <div class="metric-caption">Baseline Heuristic Formula (δ = 1e-5, T=10 rounds)</div>
                </div>
                """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="section-header">Privacy Protection Protocol Description</div>', unsafe_allow_html=True)
            st.markdown(r"""
            - **Data Boundary**: Raw patient records stay local to each hospital node.
            - **Update Clipping**: Parameter update vector $\Delta w$ clipped to threshold $C = 1.0$.
            - **Gaussian Noise**: Noise multiplier $\sigma = 0.1$ added prior to transmission.
            - **Formal Accounting**: Opacus Rényi Differential Privacy (RDP) accountant provides formal theoretical bounds alongside empirical heuristic metrics.
            """)

    # --- VIEW 7: ARCHITECTURE ---
    elif admin_nav == "Architecture":
        st.markdown('<div class="page-title">Multi-Hospital System Architecture</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">End-to-End Federated Communication Topology</div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="section-header">System Network Topology</div>', unsafe_allow_html=True)
            
            a_c1, a_a1, a_c2, a_a2, a_c3 = st.columns([1.2, 0.3, 1.5, 0.3, 1.3])
            
            with a_c1:
                with st.container(border=True):
                    st.markdown("**Hospital Nodes**")
                    st.caption("Hospital A (Local)")
                    st.caption("Hospital B (Local)")
                    st.caption("Hospital C (Local)")
                    st.caption("Hospital D (Local)")
                    st.caption("Hospital E (Local)")

            with a_a1:
                st.markdown("### →")

            with a_c2:
                with st.container(border=True):
                    st.markdown("**Central Federated Server**")
                    st.caption("Sample-Weighted FedAvg")
                    st.caption("Privacy Protected")
                    st.caption("Global Convergence Evaluation")

            with a_a2:
                st.markdown("### →")

            with a_c3:
                with st.container(border=True):
                    st.markdown("**Global Model**")
                    st.caption("PyTorch MLP Classifier")
                    st.caption("32 → 16 → 1 Architecture")
                    st.caption("Disease Inference")

    # --- VIEW 8: EXPERIMENT RESULTS ---
    elif admin_nav == "Experiment Results":
        st.markdown('<div class="page-title">Experiment Outputs & Artifact Audit</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Structured Results CSVs, Figures, and Saved PyTorch Models</div>', unsafe_allow_html=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            with st.container(border=True):
                st.markdown('<div class="section-header">Generated Result CSVs (`results/`)</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="info-card">
                    <div class="info-card-label">Training Metrics</div>
                    <div class="info-card-value">results/training_metrics.csv</div>
                    <div class="info-card-sub">10 Rounds metrics table</div>
                </div>
                <div class="info-card">
                    <div class="info-card-label">Model Comparison</div>
                    <div class="info-card-value">results/model_comparison.csv</div>
                    <div class="info-card-sub">Comparative benchmark table</div>
                </div>
                <div class="info-card">
                    <div class="info-card-label">Hospital Distribution</div>
                    <div class="info-card-value">results/hospital_distribution.csv</div>
                    <div class="info-card-sub">Non-IID Demographics summary</div>
                </div>
                <div class="info-card">
                    <div class="info-card-label">Test Predictions</div>
                    <div class="info-card-value">results/predictions.csv</div>
                    <div class="info-card-sub">Test set inference log sample</div>
                </div>
                """, unsafe_allow_html=True)

        with col_r2:
            with st.container(border=True):
                st.markdown('<div class="section-header">Plots & PyTorch Model Checkpoints</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="info-card">
                    <div class="info-card-label">Convergence Plot</div>
                    <div class="info-card-value">figures/training_convergence.png</div>
                    <div class="info-card-sub">Accuracy & loss curves</div>
                </div>
                <div class="info-card">
                    <div class="info-card-label">Model Comparison Bar Chart</div>
                    <div class="info-card-value">figures/model_comparison.png</div>
                    <div class="info-card-sub">4-paradigm performance bar plot</div>
                </div>
                <div class="info-card">
                    <div class="info-card-label">ROC & Confusion Matrices</div>
                    <div class="info-card-value">figures/roc_curve.png, confusion_matrix.png</div>
                    <div class="info-card-sub">Evaluation plots</div>
                </div>
                <div class="info-card">
                    <div class="info-card-label">PyTorch Model Checkpoints</div>
                    <div class="info-card-value">models/*.pt</div>
                    <div class="info-card-sub">centralized, federated, private models</div>
                </div>
                """, unsafe_allow_html=True)

        if os.path.exists("results/final_results.txt"):
            with st.container(border=True):
                st.markdown('<div class="section-header">Final Summary Report (`results/final_results.txt`)</div>', unsafe_allow_html=True)
                with open("results/final_results.txt", "r") as f:
                    st.code(f.read(), language="text")


# -----------------------------------------------------------------------------
# MAIN APPLICATION ROUTER
# -----------------------------------------------------------------------------
def main():
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "role" not in st.session_state:
        st.session_state.role = ""
    if "hospital_name" not in st.session_state:
        st.session_state.hospital_name = ""

    st.markdown(get_custom_css(st.session_state.theme), unsafe_allow_html=True)

    results = load_results()
    config = load_config()
    tr_df = load_training_metrics_csv()
    comp_df = load_model_comparison_csv()
    hosp_df = load_hospital_distribution_csv()

    if not st.session_state.authenticated:
        render_login_page()
    else:
        render_navigation_bar()
        if st.session_state.role == "Hospital User":
            render_hospital_user_dashboard(results, config, tr_df, comp_df, hosp_df)
        elif st.session_state.role == "System Administrator":
            render_admin_dashboard(results, config, tr_df, comp_df, hosp_df)


if __name__ == "__main__":
    main()
'''

with open("dashboard/app.py", "w", encoding="utf-8") as f:
    f.write(new_app_code)

print("Updated app.py with clean Plotly layouts and high contrast code elements.")
