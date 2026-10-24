"""
Interactive Map Page - California Wildfire Intelligence Platform
Dedicated full-screen geospatial intelligence workstation providing:
- Authentic California Leaflet geographic map with verified keyless, watermark-free basemaps (Esri Topo, Esri Satellite, OpenTopoMap, OpenStreetMap, Esri Dark Gray).
- 0.25° (~25 km) statistical wildfire risk grid with continuous risk colormap.
- Real NASA FIRMS active fire satellite detections with thermal anomaly details.
- Comprehensive scenario switcher across peak wildfire events and winter baseline.
- Real-time cell telemetry inspector and geospatial intelligence metrics.
"""
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

from src.theme import (
    render_main_hero,
    render_section_header,
    render_intelligence_card,
    render_risk_legend,
    render_operational_notice,
    render_html,
    COLOR_BORDER,
    COLOR_DARK_NAVY,
    COLOR_DEEP_NAVY,
    COLOR_ELEVATED,
    COLOR_FIRE_RED,
    COLOR_FIRE_ORANGE,
    COLOR_WARM_ORANGE,
    COLOR_GREEN,
    COLOR_GOLD,
    COLOR_CRIMSON,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    RISK_TIER_COLORS,
)
from src.data_loader import (
    CURATED_SCENARIOS,
    CALIFORNIA_PRESET_LOCATIONS,
    load_spatial_slice_with_predictions,
    open_dataset,
)
from src.leaflet_map import render_california_leaflet_map


def render_interactive_map_page():
    # 1. Header
    html_header = f"""
    <div style="display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 1.25rem; flex-wrap: wrap; gap: 0.75rem;">
        <div>
            <h1 style="font-size: 2.25rem; font-weight: 800; color: #FFFFFF; margin: 0 0 0.25rem 0; letter-spacing: -0.025em; line-height: 1.15;">
                Interactive <span style="background: linear-gradient(90deg, #FF3434 0%, #FF6A24 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Map</span>
            </h1>
            <p style="font-size: 0.92rem; color: #9FB1C5; margin: 0; max-width: 850px; line-height: 1.5;">
                Statewide California geospatial intelligence console monitoring 0.25° risk grid cells, spaceborne active fire anomalies, and meteorological risk factors.
            </p>
        </div>
        <div style="display: inline-flex; align-items: center; gap: 0.55rem; background: #0D2235; border: 1px solid #1C3A50; border-radius: 20px; padding: 0.4rem 0.95rem; font-family: ui-monospace, monospace; font-size: 0.75rem; color: #F4F7FB;">
            <span>🛰️ GEOSPATIAL INTELLIGENCE WORKSTATION</span>
        </div>
    </div>
    """
    render_html(html_header)

    # 2. Control Ribbon: Scenario, Basemap, Target Region, Opacity
    c_scen, c_base, c_loc, c_op = st.columns([1.5, 1.2, 1.1, 0.8], gap="medium")

    with c_scen:
        scenario_key = st.selectbox(
            "Historical Observation Scenario:",
            options=list(CURATED_SCENARIOS.keys()),
            index=0,
            key="imap_scen_select",
        )
        sc = CURATED_SCENARIOS[scenario_key]

    with c_base:
        basemap_choice = st.selectbox(
            "Authentic Basemap Layer:",
            options=[
                "🛰️ Satellite Imagery (Esri World)",
                "🗺️ Terrain & Roads (Esri Topo)",
                "🌑 Dark Analytical (Esri Dark Canvas)",
                "🏔️ Mountain Contours (OpenTopoMap)",
                "🏙️ Streets & Municipal (OpenStreetMap)",
            ],
            index=0,
            key="imap_base_select",
        )

    with c_loc:
        preset_key = st.selectbox(
            "Highlight California Region:",
            options=["None (Statewide View)"] + list(CALIFORNIA_PRESET_LOCATIONS.keys()),
            index=0,
            key="imap_preset_select",
        )

    with c_op:
        opacity = st.slider("Risk Opacity", 0.15, 0.95, 0.55, 0.05, key="imap_op_slider")

    # Load Spatial Data with Model Predictions
    snap = load_spatial_slice_with_predictions("master", sc["timestamp"], land_only=True)
    if snap is None or snap.empty:
        st.error("Spatial slice unavailable for the selected scenario.")
        return

    # Derive Summary Intelligence Metrics
    total_cells = len(snap)
    valid_scores = snap["model_risk_score"].dropna()
    mean_risk = float(valid_scores.mean()) if not valid_scores.empty else 0.0
    alert_cells = int(snap["is_alert"].sum()) if "is_alert" in snap.columns else 0
    firms_count = int((snap["fire_now"] > 0).sum()) if "fire_now" in snap.columns else 0

    # Summary Metrics Row
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1:
        render_intelligence_card("🌐", "MONITORED CELLS", f"{total_cells:,}", "Terrestrial 0.25° Grid")
    with m2:
        render_intelligence_card("📈", "MEAN STATEWIDE RISK", f"{mean_risk:.1%}", f"{sc['region_name'][:25]}")
    with m3:
        border = COLOR_FIRE_RED if alert_cells > 0 else COLOR_BORDER
        render_intelligence_card("🚨", "ALERT CELLS (≥25.1%)", f"{alert_cells}", "Actionable Risk Threshold", border_color=border)
    with m4:
        border = COLOR_FIRE_ORANGE if firms_count > 0 else COLOR_BORDER
        render_intelligence_card("🛰️", "OBSERVED FIRMS", f"{firms_count}", "Thermal Sat Hotspots", border_color=border)

    # Resolve target cell pin
    target_tuple = None
    if preset_key != "None (Statewide View)":
        p_info = CALIFORNIA_PRESET_LOCATIONS[preset_key]
        target_tuple = (p_info["latitude"], p_info["longitude"], preset_key)

    # 3. Main Leaflet Geospatial Map Canvas
    st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
    render_california_leaflet_map(
        df=snap,
        default_basemap=basemap_choice,
        target_cell=target_tuple,
        height=660,
        opacity=opacity,
    )

    # 4. Legend & Operational Guidance
    render_risk_legend(show_note=True)
    render_operational_notice()
