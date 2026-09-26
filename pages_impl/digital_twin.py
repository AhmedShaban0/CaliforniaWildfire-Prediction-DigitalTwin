"""
3D Digital Twin Page - California Wildfire Prediction System
Full-canvas interactive 3D geospatial visualization with real SRTM elevation and aligned environmental layers.
Matches the reference layout:
- Main hero banner with upper-left logo tile + pill.
- Emoji-led section heading: "🌐 California 3D Digital Twin".
- Compact horizontal controls above the full-width WebGL map.
- Dominant interactive 3D map canvas.
- Telemetry status strip and physical boundary notice below.
"""
import streamlit as st
import pandas as pd
import numpy as np
from src.theme import (
    render_main_hero,
    render_section_header,
    render_operational_notice,
    render_html,
    COLOR_BORDER,
    COLOR_DEEP_NAVY,
    COLOR_DARK_NAVY,
    COLOR_ELEVATED,
    COLOR_FIRE_RED,
    COLOR_FIRE_ORANGE,
    COLOR_WARM_ORANGE,
    COLOR_GREEN,
    COLOR_GOLD,
    COLOR_CRIMSON,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
)
from src.data_loader import (
    open_dataset,
    get_dataset_metadata,
    CURATED_SCENARIOS,
    load_spatial_slice_with_predictions,
)
from src.twin_3d import render_pydeck_3d_twin, render_plotly_topographic_twin


def render_digital_twin_page():
    # 1. Main Landing Hero Banner
    render_main_hero(
        title="California 3D Digital Twin",
        subtitle="Interactive 3D volumetric topographic twin fusing Copernicus ERA5 atmospheric reanalysis, SRTM surface elevation, and spaceborne active fire risk across California.",
        eyebrow="🌐 GEOSPATIAL 3D VISUALIZATION",
    )

    ds, ds_err = open_dataset()
    has_fallback = os.path.exists("assets/curated_snapshots.parquet")
    if ds is None and not has_fallback:
        st.error(f"NetCDF dataset `master_dataset_2024_2025.nc` is required. {ds_err}")
        return

    meta = get_dataset_metadata(ds) if ds is not None else {
        "total_cells": 1677,
        "lat_range": (32.5, 42.0),
        "lon_range": (-124.5, -114.0),
        "total_time_steps": 17544,
    }

    # 2. Emoji-led Section Heading
    render_section_header("🌐 California 3D Digital Twin", tag="FULL-VIEWPORT WEBGL CANVASES")

    # 3. Compact Horizontal Control Bar
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([3, 2.2, 2, 1.8], gap="small")

    with ctrl_col1:
        sc_name = st.selectbox(
            "Temporal Snapshot:",
            options=list(CURATED_SCENARIOS.keys()),
            index=0,
        )
        selected_timestamp = CURATED_SCENARIOS[sc_name]["timestamp"]

    with ctrl_col2:
        layer_option = st.selectbox(
            "Active Data Layer:",
            options=[
                "Model Predicted Risk",
                "Elevation (SRTM, m)",
                "Temperature (°C)",
                "Vapor Pressure Deficit (kPa)",
                "Wind Speed (m/s)",
                "FIRMS Active Fire",
            ],
            index=0,
        )

    with ctrl_col3:
        engine_choice = st.selectbox(
            "Rendering Engine:",
            options=["PyDeck 3D Map (WebGL)", "Plotly 3D Orbital Scatter"],
            index=0,
        )

    with ctrl_col4:
        with st.popover("⚙️ Camera & Tilt"):
            pitch_val = st.slider("Pitch (°)", min_value=0, max_value=60, value=48, step=2)
            bearing_val = st.slider("Bearing (°)", min_value=-180, max_value=180, value=-18, step=2)
            elev_scale = st.slider("Elevation Scale", min_value=1.0, max_value=6.0, value=2.5, step=0.5)

    # 4. Load Spatial Data with Model Risk
    with st.spinner(f"Extracting spatial grid for {selected_timestamp} UTC..."):
        slice_df = load_spatial_slice_with_predictions(selected_timestamp, selected_timestamp, land_only=True)

    if slice_df is None or slice_df.empty:
        st.warning("No terrestrial grid cells found for the selected timestamp.")
        return

    # 5. Dominant Interactive 3D Canvas
    if engine_choice.startswith("PyDeck"):
        render_pydeck_3d_twin(
            slice_df,
            color_by=layer_option,
            elevation_scale=elev_scale if "elev_scale" in locals() else 2.5,
            pitch=float(pitch_val) if "pitch_val" in locals() else 48.0,
            bearing=float(bearing_val) if "bearing_val" in locals() else -18.0,
            height=620,
        )
    else:
        render_plotly_topographic_twin(
            slice_df,
            color_by=layer_option,
            height=620,
        )

    # 6. Telemetry Status Strip
    alerts_count = int(slice_df["is_alert"].sum())
    fires_count = int((slice_df["fire_now"] > 0).sum())
    max_risk = slice_df["model_risk_score"].max()
    mean_risk = slice_df["model_risk_score"].mean()

    render_html(f"""
    <div style="background-color: {COLOR_DEEP_NAVY}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 1rem 1.25rem; margin-top: 1rem; display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; text-align: center;">
        <div>
            <div style="font-size: 0.72rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace;">ACTIVE TERRESTRIAL CELLS</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: {COLOR_TEXT_PRIMARY};">{len(slice_df):,}</div>
            <div style="font-size: 0.74rem; color: {COLOR_TEXT_SECONDARY};">0.25° (~25 km) resolution</div>
        </div>
        <div>
            <div style="font-size: 0.72rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace;">HIGH RISK CELLS (≥0.2507)</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: {COLOR_FIRE_ORANGE};">{alerts_count} ({alerts_count / len(slice_df):.1%})</div>
            <div style="font-size: 0.74rem; color: {COLOR_TEXT_SECONDARY};">Operational F₂ alert posture</div>
        </div>
        <div>
            <div style="font-size: 0.72rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace;">PEAK GRID RISK</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: {COLOR_FIRE_RED};">{max_risk:.1%}</div>
            <div style="font-size: 0.74rem; color: {COLOR_TEXT_SECONDARY};">Maximum cell score</div>
        </div>
        <div>
            <div style="font-size: 0.72rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace;">MEAN STATEWIDE RISK</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: {COLOR_TEXT_PRIMARY};">{mean_risk:.1%}</div>
            <div style="font-size: 0.74rem; color: {COLOR_TEXT_SECONDARY};">California terrestrial average</div>
        </div>
        <div>
            <div style="font-size: 0.72rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace;">NASA FIRMS DETECTIONS</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: {COLOR_CRIMSON if fires_count > 0 else COLOR_GREEN};">{fires_count} active</div>
            <div style="font-size: 0.74rem; color: {COLOR_TEXT_SECONDARY};">MODIS / VIIRS spaceborne</div>
        </div>
    </div>
    """)

    # 7. Physical scope limitation disclosure
    render_html(f"""
    <div style="background-color: {COLOR_DARK_NAVY}; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 0.85rem 1.15rem; margin-top: 1rem; font-size: 0.82rem; color: {COLOR_TEXT_SECONDARY}; line-height: 1.5;">
        <b style="color: {COLOR_GOLD};">Physical Scope Notice:</b>
        The 3D Digital Twin renders atmospheric condition grids, terrain elevation, and 24-hour ignition detection probabilities. It does <u>not</u> simulate fluid dynamics (CFD), fire spread fronts, ember casting, or smoke dispersion.
    </div>
    """)

    render_operational_notice()
