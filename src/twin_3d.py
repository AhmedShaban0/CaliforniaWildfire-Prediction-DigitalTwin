"""
Interactive 3D Digital Twin for California Wildfire Prediction.
Implements GPU-accelerated PyDeck 3D geographic column / terrain visualizations
with true 3D pitch, bearing, real elevation extrusion, and rich tooltips.
Also provides a 3D Plotly topographic view as a resilient alternative.
"""
import pydeck as pdk
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st
from src.theme import (
    COLOR_CRIMSON,
    COLOR_FIRE_RED,
    COLOR_FIRE_ORANGE,
    COLOR_WARM_ORANGE,
    COLOR_GREEN,
    COLOR_GOLD,
    COLOR_BRAND_NAVY,
    COLOR_BORDER,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
)


def get_color_for_metric(val: float, metric: str) -> list:
    """Returns RGBA color quad [R, G, B, A] based on the selected metric value."""
    if pd.isna(val):
        return [40, 60, 80, 100]

    if metric == "Model Predicted Risk":
        # Green (low) -> Yellow (moderate) -> Orange (high) -> Crimson (extreme)
        if val < 0.15:
            return [53, 183, 121, 200]
        elif val < 0.2507:
            return [246, 183, 60, 220]
        elif val < 0.45:
            return [244, 81, 30, 240]
        else:
            return [197, 31, 50, 255]

    elif metric == "Elevation (SRTM, m)":
        # 0m to 4000m
        norm = np.clip(val / 3500.0, 0.0, 1.0)
        r = int(norm * 240 + 20)
        g = int((1.0 - norm) * 160 + 50)
        b = int((1.0 - norm) * 120 + 30)
        return [r, g, b, 220]

    elif metric == "Temperature (°C)":
        # 0C to 45C
        norm = np.clip((val - 5.0) / 38.0, 0.0, 1.0)
        r = int(norm * 220 + 30)
        g = int((1.0 - abs(norm - 0.5) * 2) * 180 + 30)
        b = int((1.0 - norm) * 220 + 30)
        return [r, g, b, 220]

    elif metric == "Vapor Pressure Deficit (kPa)":
        # 0 to 6 kPa
        norm = np.clip(val / 5.0, 0.0, 1.0)
        r = int(norm * 230 + 25)
        g = int((1.0 - norm) * 150 + 40)
        b = int(40)
        return [r, g, b, 220]

    elif metric == "Wind Speed (m/s)":
        # 0 to 18 m/s
        norm = np.clip(val / 15.0, 0.0, 1.0)
        r = int(norm * 200 + 40)
        g = int(norm * 180 + 50)
        b = int(220)
        return [r, g, b, 220]

    elif metric == "FIRMS Active Fire":
        if val > 0:
            return [255, 30, 30, 255]
        return [25, 45, 65, 80]

    return [30, 80, 130, 180]


def render_pydeck_3d_twin(
    df: pd.DataFrame,
    color_by: str = "Model Predicted Risk",
    elevation_scale: float = 3.0,
    pitch: float = 48.0,
    bearing: float = -18.0,
    height: int = 580,
):
    """
    Renders the interactive 3D California Digital Twin using PyDeck.
    Columns are extruded in 3D by real elevation or risk score.
    """
    if df is None or df.empty:
        st.warning("No spatial data available to render the 3D twin.")
        return

    plot_df = df.copy()

    # Determine column height
    if color_by == "Model Predicted Risk":
        plot_df["col_elevation"] = (plot_df["model_risk_score"].fillna(0.0) * 12000.0 * (elevation_scale / 3.0)).clip(lower=100)
    else:
        plot_df["col_elevation"] = (plot_df["elevation"].fillna(0.0) * elevation_scale).clip(lower=50)

    # Compute RGBA colors
    colors = []
    for _, row in plot_df.iterrows():
        if color_by == "Model Predicted Risk":
            c = get_color_for_metric(row.get("model_risk_score"), color_by)
        elif color_by == "Elevation (SRTM, m)":
            c = get_color_for_metric(row.get("elevation"), color_by)
        elif color_by == "Temperature (°C)":
            c = get_color_for_metric(row.get("t2m_celsius"), color_by)
        elif color_by == "Vapor Pressure Deficit (kPa)":
            c = get_color_for_metric(row.get("vpd"), color_by)
        elif color_by == "Wind Speed (m/s)":
            c = get_color_for_metric(row.get("wind_speed"), color_by)
        elif color_by == "FIRMS Active Fire":
            c = get_color_for_metric(row.get("fire_now", 0), color_by)
        else:
            c = [35, 110, 180, 200]
        colors.append(c)

    plot_df["color_rgba"] = colors

    # Formatted display strings for tooltip
    plot_df["risk_display"] = plot_df["model_risk_score"].apply(
        lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A"
    )
    plot_df["temp_display"] = plot_df["t2m_celsius"].apply(
        lambda x: f"{x:.1f}°C" if pd.notnull(x) else "N/A"
    )
    plot_df["wind_display"] = plot_df["wind_speed"].apply(
        lambda x: f"{x:.1f} m/s" if pd.notnull(x) else "N/A"
    )
    plot_df["vpd_display"] = plot_df["vpd"].apply(
        lambda x: f"{x:.2f} kPa" if pd.notnull(x) else "N/A"
    )
    plot_df["elev_display"] = plot_df["elevation"].apply(
        lambda x: f"{x:.0f} m" if pd.notnull(x) else "0 m"
    )
    plot_df["fire_now_display"] = plot_df["fire_now"].apply(
        lambda x: "CONFIRMED (FIRMS Active Fire)" if x > 0 else "None Detected"
    )

    # PyDeck 3D Column Layer (representing 0.25 x 0.25 degree cells)
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=plot_df,
        get_position=["longitude", "latitude"],
        get_elevation="col_elevation",
        elevation_scale=1,
        radius=11500,  # ~11.5 km radius matches 0.25 deg spacing
        get_fill_color="color_rgba",
        pickable=True,
        auto_highlight=True,
    )

    # Initial California ViewState
    view_state = pdk.ViewState(
        latitude=37.1,
        longitude=-119.5,
        zoom=5.7,
        pitch=pitch,
        bearing=bearing,
    )

    tooltip = {
        "html": """
        <div style="background-color: #071A2C; padding: 10px 12px; border: 1px solid #29465E; border-radius: 6px; color: #F8FAFC; font-family: sans-serif; font-size: 12px; line-height: 1.5; min-width: 200px;">
            <div style="font-weight: 700; color: #FF7A24; font-size: 13px; margin-bottom: 4px;">Cell Inspector: {latitude}°N, {longitude}°W</div>
            <div style="border-bottom: 1px solid #29465E; padding-bottom: 4px; margin-bottom: 4px;">
                <b>Terrain Elevation:</b> {elev_display}<br/>
                <b>Land Cover:</b> {land_cover_name}
            </div>
            <div>
                <b>Temperature:</b> {temp_display}<br/>
                <b>Wind Speed:</b> {wind_display}<br/>
                <b>VPD (Dryness):</b> {vpd_display}<br/>
            </div>
            <div style="margin-top: 4px; padding-top: 4px; border-top: 1px solid #29465E;">
                <b>Model Risk Score:</b> <span style="color: #F4511E; font-weight: bold;">{risk_display}</span><br/>
                <b>Alert Status:</b> {risk_level}<br/>
                <b>Observed Fire:</b> {fire_now_display}
            </div>
        </div>
        """,
        "style": {"zIndex": 1000},
    }

    deck = pdk.Deck(
        layers=[column_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    )

    st.pydeck_chart(deck, height=height)


def render_plotly_topographic_twin(
    df: pd.DataFrame,
    color_by: str = "Model Predicted Risk",
    height: int = 580,
):
    """
    Renders a 3D topographic scatter view via Plotly as a resilient fallback.
    Plotly provides full client-side 3D orbital camera controls.
    """
    if df is None or df.empty:
        st.warning("No spatial data available to render the 3D twin.")
        return

    plot_df = df.copy()

    if color_by == "Model Predicted Risk":
        color_col = plot_df["model_risk_score"].fillna(0.0)
        cbar_title = "Risk Estimate"
        colorscale = [[0.0, COLOR_GREEN], [0.25, COLOR_GOLD], [0.45, COLOR_FIRE_ORANGE], [1.0, COLOR_CRIMSON]]
    elif color_by == "Temperature (°C)":
        color_col = plot_df["t2m_celsius"]
        cbar_title = "Temp (°C)"
        colorscale = "Plasma"
    elif color_by == "Vapor Pressure Deficit (kPa)":
        color_col = plot_df["vpd"]
        cbar_title = "VPD (kPa)"
        colorscale = "YlOrRd"
    elif color_by == "Wind Speed (m/s)":
        color_col = plot_df["wind_speed"]
        cbar_title = "Wind (m/s)"
        colorscale = "Viridis"
    else:
        color_col = plot_df["elevation"]
        cbar_title = "Elevation (m)"
        colorscale = "Earth"

    fig = go.Figure(data=[go.Scatter3d(
        x=plot_df["longitude"],
        y=plot_df["latitude"],
        z=plot_df["elevation"],
        mode="markers",
        marker=dict(
            size=6,
            color=color_col,
            colorscale=colorscale,
            opacity=0.9,
            colorbar=dict(
                title=dict(text=cbar_title, font=dict(color=COLOR_TEXT_PRIMARY, size=11)),
                tickfont=dict(color=COLOR_TEXT_SECONDARY),
                thickness=14,
                len=0.75,
            ),
        ),
        text=[
            f"<b>Lat:</b> {row['latitude']:.2f}°N<br>"
            f"<b>Lon:</b> {row['longitude']:.2f}°W<br>"
            f"<b>Elevation:</b> {row['elevation']:.0f} m<br>"
            f"<b>Land Cover:</b> {row.get('land_cover_name', 'N/A')}<br>"
            f"<b>Temperature:</b> {row.get('t2m_celsius', 0):.1f}°C<br>"
            f"<b>VPD:</b> {row.get('vpd', 0):.2f} kPa<br>"
            f"<b>Model Risk:</b> {row.get('model_risk_score', np.nan):.1%}"
            for _, row in plot_df.iterrows()
        ],
        hoverinfo="text",
    )])

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Longitude", backgroundcolor=COLOR_BRAND_NAVY, gridcolor=COLOR_BORDER, color=COLOR_TEXT_SECONDARY),
            yaxis=dict(title="Latitude", backgroundcolor=COLOR_BRAND_NAVY, gridcolor=COLOR_BORDER, color=COLOR_TEXT_SECONDARY),
            zaxis=dict(title="Elevation (m)", backgroundcolor=COLOR_BRAND_NAVY, gridcolor=COLOR_BORDER, color=COLOR_TEXT_SECONDARY),
            camera=dict(eye=dict(x=-1.5, y=-1.6, z=1.2)),
            aspectratio=dict(x=1, y=1.2, z=0.5),
        ),
        margin=dict(r=10, l=10, b=10, t=10),
        paper_bgcolor=COLOR_BRAND_NAVY,
        plot_bgcolor=COLOR_BRAND_NAVY,
        height=height,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_single_cell_3d_twin(
    target_lat: float,
    target_lon: float,
    risk_score: float,
    risk_tier: str,
    is_alert: bool,
    temp_c: float,
    wind_speed: float,
    elev: float,
    land_cover_name: str,
    mode_label: str = "Manual Input",
    height: int = 460,
):
    """
    Renders the 3D Digital Twin specifically focused on a single evaluated grid cell.
    Visualizes the selected location extruded and highlighted in its risk-tier color,
    against a subtle baseline terrain backdrop of California terrestrial cells.
    Avoids coloring the entire state when only one cell is evaluated.
    """
    # Color mapping for risk tier
    color_map = {
        "Low": [53, 183, 121, 230],
        "Moderate": [246, 183, 60, 240],
        "High": [244, 81, 30, 250],
        "Extreme": [197, 31, 50, 255],
    }
    cell_color = color_map.get(risk_tier, [244, 81, 30, 240])

    # Single cell dataframe
    target_df = pd.DataFrame([{
        "latitude": float(target_lat),
        "longitude": float(target_lon),
        "col_elevation": max(1500.0, float(risk_score) * 35000.0),
        "color_rgba": cell_color,
        "risk_display": f"{risk_score:.1%}",
        "risk_tier": risk_tier,
        "alert_text": "ALERT ELEVATED" if is_alert else "BASELINE NORMAL",
        "temp_display": f"{temp_c:.1f}°C",
        "wind_display": f"{wind_speed:.1f} m/s",
        "elev_display": f"{elev:.0f} m",
        "land_cover_name": land_cover_name,
        "mode_label": mode_label,
    }])

    # Evaluated Target Cell Layer (highlighted 3D column)
    target_column_layer = pdk.Layer(
        "ColumnLayer",
        data=target_df,
        get_position=["longitude", "latitude"],
        get_elevation="col_elevation",
        elevation_scale=1,
        radius=13500,  # ~13.5 km radius matches 0.25 deg spacing
        get_fill_color="color_rgba",
        pickable=True,
        auto_highlight=True,
    )

    # Glowing point pin on top
    pin_layer = pdk.Layer(
        "ScatterplotLayer",
        data=target_df,
        get_position=["longitude", "latitude"],
        get_radius=18000,
        get_fill_color=cell_color,
        pickable=True,
    )

    # Focused California ViewState centered on the target cell
    view_state = pdk.ViewState(
        latitude=target_lat,
        longitude=target_lon,
        zoom=6.8,
        pitch=46.0,
        bearing=-16.0,
    )

    tooltip = {
        "html": """
        <div style="background-color: #071A2C; padding: 12px 14px; border: 1px solid #29465E; border-radius: 8px; color: #F8FAFC; font-family: sans-serif; font-size: 12px; line-height: 1.5; min-width: 220px; box-shadow: 0 4px 16px rgba(0,0,0,0.5);">
            <div style="font-size: 11px; font-weight: 700; color: #FF7A24; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
                {mode_label} // Cell Evaluation
            </div>
            <div style="font-weight: 700; color: #F8FAFC; font-size: 14px; margin-bottom: 6px;">
                {latitude}°N, {longitude}°W
            </div>
            <div style="border-bottom: 1px solid #29465E; padding-bottom: 6px; margin-bottom: 6px;">
                <b>Terrain Elevation:</b> {elev_display}<br/>
                <b>Land Cover:</b> {land_cover_name}<br/>
                <b>Temperature:</b> {temp_display} &nbsp;|&nbsp; <b>Wind:</b> {wind_display}
            </div>
            <div style="font-size: 13px;">
                <b>24h Fire Risk:</b> <span style="font-weight: 800; font-size: 15px;">{risk_display}</span><br/>
                <b>Classification:</b> {risk_tier} ({alert_text})
            </div>
            <div style="font-size: 10px; color: #A9BACB; margin-top: 6px; border-top: 1px solid #29465E; padding-top: 4px;">
                Target: Next-24h satellite detection (fire_next_24h). Not a physical spread simulation.
            </div>
        </div>
        """,
        "style": {"zIndex": 1000},
    }

    deck = pdk.Deck(
        layers=[target_column_layer, pin_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    )

    st.pydeck_chart(deck, height=height)

