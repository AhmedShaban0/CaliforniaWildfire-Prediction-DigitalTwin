"""
California Wildfire Prediction - California Fire Atlas
Entry point for multi-page geospatial environmental intelligence console.
Matches the reference design layout:
- Left sidebar with clean button-style navigation and prominent bottom branded project card.
- Direct routing to all verified pages with shared design tokens.
"""
import os
import streamlit as st
from PIL import Image

# 1. Page Configuration
logo_path = "wildfire.png"
if not os.path.exists(logo_path) and os.path.exists("assets/wildfire.png"):
    logo_path = "assets/wildfire.png"

page_icon = None
if os.path.exists(logo_path):
    try:
        page_icon = Image.open(logo_path)
    except Exception:
        page_icon = "🔥"
else:
    page_icon = "🔥"

st.set_page_config(
    page_title="California Wildfire Prediction System",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Apply Custom Atlas Design System & CSS
from src.theme import apply_theme, render_sidebar_project_card
apply_theme()

# 3. Sidebar Navigation & Branding
with st.sidebar:
    nav_items = [
        ("🏠 Overview", "Overview"),
        ("🔥 Wildfire Prediction", "Wildfire Prediction"),
        ("🌐 3D Digital Twin", "3D Digital Twin"),
        ("📊 Model Performance", "Model Performance"),
        ("🛰️ Data & Methodology", "Data & Methodology"),
        ("ℹ️ About", "About"),
    ]

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Overview"

    # Handle programmatic nav_target redirects
    if "nav_target" in st.session_state:
        target = st.session_state.pop("nav_target")
        for label, code in nav_items:
            if target.lower() in code.lower() or target.lower() in label.lower():
                st.session_state["current_page"] = code
                break

    # Standard clickable button-style navigation (NO circular radios)
    for label, code in nav_items:
        is_active = (st.session_state["current_page"] == code)
        btn_type = "primary" if is_active else "secondary"
        if st.button(
            label,
            key=f"nav_btn_{code}",
            use_container_width=True,
            type=btn_type,
        ):
            if not is_active:
                st.session_state["current_page"] = code
                st.rerun()

    # Branded project card near bottom of sidebar with enlarged logo
    render_sidebar_project_card()

page = st.session_state["current_page"]

# 4. Page Routing
from pages_impl.overview import render_overview_page
from pages_impl.prediction import render_prediction_page
from pages_impl.digital_twin import render_digital_twin_page
from pages_impl.performance import render_performance_page
from pages_impl.methodology import render_methodology_page
from pages_impl.about import render_about_page

if page == "Overview":
    render_overview_page()
elif page == "Wildfire Prediction":
    render_prediction_page()
elif page == "3D Digital Twin":
    render_digital_twin_page()
elif page == "Model Performance":
    render_performance_page()
elif page == "Data & Methodology":
    render_methodology_page()
elif page == "About":
    render_about_page()
