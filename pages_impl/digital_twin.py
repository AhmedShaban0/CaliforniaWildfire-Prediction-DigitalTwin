"""
3D Digital Twin Page - California Wildfire Prediction System
Full-canvas interactive geospatial visualization with real California geography:
- Multi-basemap support (Terrain & Geography, Satellite Imagery, Streets, Dark Analytical).
- Transparent 0.25° grid cell polygons showing 24h satellite fire detection risk without obscuring roads, cities, and terrain.
- Interactive toggle between 2D Geographic Flat view and 3D Volumetric Extrusion.
- Direct layer opacity control.
- Separate, distinct NASA FIRMS thermal anomaly detection pins.
- Statewide telemetry status strip and physical boundary notice.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
from src.theme import (
    render_main_hero,
    render_section_header,
    render_operational_notice,
    render_risk_legend,
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
from src.twin_3d import (
    render_pydeck_3d_twin,
    render_plotly_topographic_twin,
    BASEMAP_CATALOG,
)


def render_digital_twin_page():
    # 1. Main Landing Hero Banner
    render_main_hero(
        title="California Wildfire Geospatial Twin",
        subtitle="Interactive geospatial environment fusing Copernicus ERA5 atmospheric reanalysis, SRTM surface elevation, and spaceborne active fire risk across California with recognizable geographic basemaps.",
        eyebrow="🌐 GEOSPATIAL 3D DIGITAL TWIN",
    )

    ds, ds_err = open_dataset()
    has_fallback = os.path.exists("assets/curated_snapshots.parquet")
    if ds is None and not has_fallback:
        st.error(f"NetCDF dataset `master_dataset_2024_2025.nc` is required. {ds_err}")
        return

    # 2. Section Heading
    render_section_header("🌐 California Geospatial Environment", tag="WEBGL INTERACTIVE MAP")

    # 3. Control Bar
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns([2.6, 2.2, 2.0, 1.8, 1.4], gap="small")

    with ctrl_col1:
        sc_name = st.selectbox(
            "Temporal Snapshot:",
            options=list(CURATED_SCENARIOS.keys()),
            index=0,
            help="Select an environmental scenario from 2024-2025.",
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
            help="Variable mapped to cell color.",
        )

    with ctrl_col3:
        basemap_choice = st.selectbox(
            "Basemap Style:",
            options=list(BASEMAP_CATALOG.keys()),
            index=0,
            help="Select a geographic basemap showing California roads, coastline, and topography.",
        )

    with ctrl_col4:
        view_mode = st.selectbox(
            "Perspective Mode:",
            options=["2D Geographic Map", "3D Volumetric (Extruded)"],
            index=0,
            help="Toggle between flat 2D choropleth polygons and 3D extruded terrain blocks.",
        )

    with ctrl_col5:
        with st.popover("⚙️ Controls"):
            engine_choice = st.radio(
                "Engine:",
                options=["PyDeck (WebGL)", "Plotly 3D Scatter"],
                index=0,
            )
            opacity_val = st.slider("Layer Opacity", min_value=0.2, max_value=0.9, value=0.55, step=0.05, help="Adjust transparency so underlying geography is readable.")
            pitch_val = st.slider("Camera Pitch (°)", min_value=0, max_value=60, value=38 if view_mode.startswith("3D") else 0, step=2)
            bearing_val = st.slider("Camera Bearing (°)", min_value=-180, max_value=180, value=-14, step=2)
            elev_scale = st.slider("Elevation Scale", min_value=1.0, max_value=6.0, value=2.5, step=0.5)

    # 4. Load Spatial Data with Model Risk
    with st.spinner(f"Extracting spatial grid for {selected_timestamp} UTC..."):
        slice_df = load_spatial_slice_with_predictions(selected_timestamp, selected_timestamp, land_only=True)

    if slice_df is None or slice_df.empty:
        st.warning("No terrestrial grid cells found for the selected timestamp.")
        return

    # 5. Interactive Map Canvas
    is_pydeck = engine_choice.startswith("PyDeck") if "engine_choice" in locals() else True

    if is_pydeck:
        render_pydeck_3d_twin(
            slice_df,
            color_by=layer_option,
            basemap_style=basemap_choice,
            view_mode=view_mode,
            layer_opacity=opacity_val if "opacity_val" in locals() else 0.55,
            elevation_scale=elev_scale if "elev_scale" in locals() else 2.5,
            pitch=float(pitch_val) if "pitch_val" in locals() else (38.0 if view_mode.startswith("3D") else 0.0),
            bearing=float(bearing_val) if "bearing_val" in locals() else -14.0,
            height=600,
            show_legend=True,
        )
    else:
        render_plotly_topographic_twin(
            slice_df,
            color_by=layer_option,
            height=600,
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
            <div style="font-size: 0.72rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace;">NASA SATELLITE DETECTIONS</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: {COLOR_CRIMSON if fires_count > 0 else COLOR_GREEN};">{fires_count} active</div>
            <div style="font-size: 0.74rem; color: {COLOR_TEXT_SECONDARY};">MODIS / VIIRS spaceborne</div>
        </div>
    </div>
    """)

    # 7. Physical scope limitation disclosure
    render_html(f"""
    <div style="background-color: {COLOR_DARK_NAVY}; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 0.85rem 1.15rem; margin-top: 1rem; font-size: 0.82rem; color: {COLOR_TEXT_SECONDARY}; line-height: 1.5;">
        <b style="color: {COLOR_GOLD};">Physical Scope Notice:</b>
        The Geospatial Digital Twin renders atmospheric condition grids, terrain elevation, and 24-hour ignition detection probabilities for 0.25° (~25 km) cells. It does <u>not</u> simulate fluid dynamics, active flame propagation, combustion fronts, or ember transport.
    </div>
    """)

    render_operational_notice()
