"""
Interactive Geospatial Map and 3D Digital Twin for California Wildfire Prediction.
Implements:
1. Real geographic basemaps of California (Realistic Satellite Imagery via Esri,
   Carto Voyager Terrain & Topography, Carto Positron Streets, Carto Dark).
2. Transparent 0.25° grid polygon shapes (~25 km resolution) showing estimated
   24-hour satellite fire detection risk (fire_next_24h) without obscuring geography.
3. 2D Geographic Flat view and 3D Volumetric Extruded view options.
4. Highlighted evaluated cell spotlighting on the statewide California map.
5. Separate, distinct NASA FIRMS thermal anomaly detection pins with separate labeling.
6. Rich cell tooltips with physical parameters, risk probabilities, and clear disclosures.
"""
import os
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
    render_risk_legend,
)

# Standard Basemap Catalog (No API Key Required for Default Providers)
BASEMAP_CATALOG = {
    "Terrain & Geography (Carto Voyager)": {
        "style": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
        "provider": "carto",
        "use_tile_layer": False,
        "description": "Topographic relief, mountain ranges, coastline, highways (I-5, US-101), cities, and forests. Free, reliable default.",
    },
    "Satellite Imagery (Esri World Imagery)": {
        "style": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
        "provider": "esri",
        "use_tile_layer": True,
        "tile_url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "description": "High-resolution satellite photography. Reverts to high-definition terrain basemap if raster tile rendering is blocked.",
    },
    "Streets & Municipal (Carto Positron)": {
        "style": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        "provider": "carto",
        "use_tile_layer": False,
        "description": "Clean, high-legibility street map emphasizing highways, urban centers, and county boundaries. Free, no key required.",
    },
    "Dark Analytical (Carto Dark)": {
        "style": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        "provider": "carto",
        "use_tile_layer": False,
        "description": "High-contrast dark canvas for specialized thermal anomaly inspection. Free, no key required.",
    },
}


def get_color_for_metric(val: float, metric: str, alpha: int = 125) -> list:
    """
    Returns RGBA color quad [R, G, B, A] based on the selected metric.
    Includes transparency (alpha) so underlying California geography remains readable.
    """
    if pd.isna(val):
        return [40, 60, 80, int(alpha * 0.45)]

    if metric == "Model Predicted Risk":
        # Green (low <15%) -> Gold (moderate 15-25.1%) -> Orange (high >=25.1%) -> Crimson (extreme >=45%)
        if val < 0.15:
            return [46, 204, 113, alpha]       # Emerald Green
        elif val < 0.2507:
            return [241, 196, 15, alpha]       # Amber Gold
        elif val < 0.45:
            return [244, 81, 30, min(255, int(alpha * 1.15))]      # Fire Orange (Alert)
        else:
            return [197, 31, 50, min(255, int(alpha * 1.3))]       # Crimson (Extreme)

    elif metric == "Elevation (SRTM, m)":
        norm = np.clip(val / 3500.0, 0.0, 1.0)
        r = int(norm * 240 + 20)
        g = int((1.0 - norm) * 160 + 50)
        b = int((1.0 - norm) * 120 + 30)
        return [r, g, b, alpha]

    elif metric == "Temperature (°C)":
        norm = np.clip((val - 5.0) / 38.0, 0.0, 1.0)
        r = int(norm * 220 + 30)
        g = int((1.0 - abs(norm - 0.5) * 2) * 180 + 30)
        b = int((1.0 - norm) * 220 + 30)
        return [r, g, b, alpha]

    elif metric == "Vapor Pressure Deficit (kPa)":
        norm = np.clip(val / 5.0, 0.0, 1.0)
        r = int(norm * 230 + 25)
        g = int((1.0 - norm) * 150 + 40)
        b = 40
        return [r, g, b, alpha]

    elif metric == "Wind Speed (m/s)":
        norm = np.clip(val / 15.0, 0.0, 1.0)
        r = int(norm * 200 + 40)
        g = int(norm * 180 + 50)
        b = 220
        return [r, g, b, alpha]

    elif metric == "FIRMS Active Fire":
        if val > 0:
            return [255, 30, 30, 240]
        return [25, 45, 65, int(alpha * 0.4)]

    return [30, 80, 130, alpha]


def compute_grid_cell_polygons(df: pd.DataFrame, half_size: float = 0.125) -> pd.DataFrame:
    """
    Creates a 4-point polygon bounding box for each 0.25° x 0.25° grid cell.
    Returns copy with 'polygon' column containing coordinates in [lon, lat] format.
    """
    out = df.copy()
    polys = []
    for _, row in out.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        poly = [
            [lon - half_size, lat - half_size],
            [lon + half_size, lat - half_size],
            [lon + half_size, lat + half_size],
            [lon - half_size, lat + half_size],
        ]
        polys.append(poly)
    out["polygon"] = polys
    return out


def render_pydeck_3d_twin(
    df: pd.DataFrame,
    color_by: str = "Model Predicted Risk",
    basemap_style: str = "Satellite Imagery (Esri World Imagery)",
    view_mode: str = "2D Geographic Map",
    layer_opacity: float = 0.50,
    elevation_scale: float = 2.5,
    pitch: float = 0.0,
    bearing: float = 0.0,
    height: int = 560,
    show_legend: bool = True,
    highlight_cell: tuple = None,  # (lat, lon, label)
):
    """
    Renders the California Wildfire Risk Map using PyDeck with authentic basemaps:
    - Recognizable California coastline, roads, cities, and satellite terrain relief.
    - Transparent 0.25° grid cell shapes displaying estimated 24h satellite fire risk.
    - True 2D geographic or 3D volumetric extrusion modes.
    - Clear distinction between model risk cells and observed NASA FIRMS detections.
    - Optional spotlight highlight on a selected / evaluated target cell.
    """
    if df is None or df.empty:
        st.warning("No spatial data available to render the California map.")
        return

    # 1. Prepare data with 0.25° cell polygons
    plot_df = compute_grid_cell_polygons(df)

    # 2. Compute alpha value from layer_opacity (0.2 to 0.9)
    alpha = int(np.clip(layer_opacity, 0.15, 0.95) * 255)

    # 3. Assign 3D column / extrusion height
    is_3d = (view_mode == "3D Volumetric (Extruded)" or pitch > 10.0)
    if is_3d:
        if color_by == "Model Predicted Risk":
            plot_df["col_elevation"] = (plot_df["model_risk_score"].fillna(0.0) * 10000.0 * (elevation_scale / 2.5)).clip(lower=60)
        else:
            plot_df["col_elevation"] = (plot_df["elevation"].fillna(0.0) * elevation_scale).clip(lower=50)
    else:
        plot_df["col_elevation"] = 0

    # 4. Compute fill colors and stroke colors
    colors = []
    strokes = []
    for _, row in plot_df.iterrows():
        if color_by == "Model Predicted Risk":
            c = get_color_for_metric(row.get("model_risk_score"), color_by, alpha=alpha)
        elif color_by == "Elevation (SRTM, m)":
            c = get_color_for_metric(row.get("elevation"), color_by, alpha=alpha)
        elif color_by == "Temperature (°C)":
            c = get_color_for_metric(row.get("t2m_celsius"), color_by, alpha=alpha)
        elif color_by == "Vapor Pressure Deficit (kPa)":
            c = get_color_for_metric(row.get("vpd"), color_by, alpha=alpha)
        elif color_by == "Wind Speed (m/s)":
            c = get_color_for_metric(row.get("wind_speed"), color_by, alpha=alpha)
        elif color_by == "FIRMS Active Fire":
            c = get_color_for_metric(row.get("fire_now", 0), color_by, alpha=alpha)
        else:
            c = [35, 110, 180, alpha]
        colors.append(c)
        strokes.append([255, 255, 255, min(160, int(alpha * 0.65))])

    plot_df["color_rgba"] = colors
    plot_df["stroke_rgba"] = strokes

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
    plot_df["precip_display"] = plot_df["tp"].apply(
        lambda x: f"{x * 1000:.1f} mm" if pd.notnull(x) else "0.0 mm"
    )

    layers = []

    # Basemap configuration
    bm_info = BASEMAP_CATALOG.get(basemap_style, BASEMAP_CATALOG["Satellite Imagery (Esri World Imagery)"])

    # If satellite imagery selected, inject Deck.gl TileLayer underneath
    if bm_info.get("use_tile_layer"):
        sat_tile_layer = pdk.Layer(
            "TileLayer",
            data=bm_info["tile_url"],
            min_zoom=0,
            max_zoom=19,
            tile_size=256,
        )
        layers.append(sat_tile_layer)

    # 5. Model Risk Grid Polygon Layer (0.25° cells)
    grid_layer = pdk.Layer(
        "PolygonLayer",
        data=plot_df,
        get_polygon="polygon",
        get_fill_color="color_rgba",
        get_line_color="stroke_rgba",
        line_width_min_pixels=1,
        stroked=True,
        filled=True,
        extruded=is_3d,
        get_elevation="col_elevation",
        pickable=True,
        auto_highlight=True,
    )
    layers.append(grid_layer)

    # 6. NASA FIRMS Active Fire Detections Layer (distinct glowing red markers)
    fires_df = plot_df[plot_df["fire_now"] > 0].copy()
    if not fires_df.empty:
        firms_layer = pdk.Layer(
            "ScatterplotLayer",
            data=fires_df,
            get_position=["longitude", "latitude"],
            get_radius=8500,
            get_fill_color=[255, 30, 30, 245],
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=2.5,
            stroked=True,
            pickable=True,
        )
        layers.append(firms_layer)

    # 7. Highlight Evaluated Cell Spotlight Layer (if specified)
    if highlight_cell and len(highlight_cell) >= 2:
        hl_lat, hl_lon = float(highlight_cell[0]), float(highlight_cell[1])
        hl_label = str(highlight_cell[2]) if len(highlight_cell) > 2 else "Selected Target Cell"
        half = 0.125
        hl_poly = [
            [hl_lon - half, hl_lat - half],
            [hl_lon + half, hl_lat - half],
            [hl_lon + half, hl_lat + half],
            [hl_lon - half, hl_lat + half],
        ]
        hl_df = pd.DataFrame([{
            "polygon": hl_poly,
            "latitude": hl_lat,
            "longitude": hl_lon,
            "label": hl_label,
            "fill": [255, 255, 255, 40],
            "stroke": [255, 215, 0, 255],  # Bright gold border
        }])
        hl_poly_layer = pdk.Layer(
            "PolygonLayer",
            data=hl_df,
            get_polygon="polygon",
            get_fill_color="fill",
            get_line_color="stroke",
            line_width_min_pixels=3,
            stroked=True,
            filled=True,
            pickable=False,
        )
        hl_pin_layer = pdk.Layer(
            "ScatterplotLayer",
            data=hl_df,
            get_position=["longitude", "latitude"],
            get_radius=11000,
            get_fill_color=[255, 122, 36, 255],
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=2.5,
            stroked=True,
            pickable=False,
        )
        layers.extend([hl_poly_layer, hl_pin_layer])

    # 8. ViewState centered on California
    eff_pitch = float(pitch) if is_3d else 0.0
    eff_bearing = float(bearing) if is_3d else 0.0
    view_state = pdk.ViewState(
        latitude=37.15,
        longitude=-119.50,
        zoom=5.6,
        pitch=eff_pitch,
        bearing=eff_bearing,
    )

    # 9. Cell Inspector Tooltip
    tooltip = {
        "html": """
        <div style="background-color: #071A2C; padding: 12px 14px; border: 1px solid #29465E; border-radius: 8px; color: #F8FAFC; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 12px; line-height: 1.5; min-width: 230px; box-shadow: 0 6px 20px rgba(0,0,0,0.55);">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; border-bottom: 1px solid #244563; padding-bottom: 4px;">
                <span style="font-weight: 700; color: #FF7A24; font-size: 13px;">0.25° Grid Cell (~25 km)</span>
                <span style="font-family: monospace; font-size: 11px; color: #94A3B8;">{latitude}°N, {longitude}°W</span>
            </div>
            <div style="margin-bottom: 6px;">
                <b>24-Hour Fire Risk:</b> <span style="color: #F4511E; font-size: 14px; font-weight: 800;">{risk_display}</span><br/>
                <b>Operational Status:</b> <span style="font-weight: 600;">{risk_level}</span>
            </div>
            <div style="border-top: 1px solid #244563; padding-top: 5px; margin-bottom: 5px; font-size: 11px; color: #CBD5E1;">
                <b>Terrain Elevation:</b> {elev_display} &nbsp;|&nbsp; <b>Land Cover:</b> {land_cover_name}<br/>
                <b>2m Air Temp:</b> {temp_display} &nbsp;|&nbsp; <b>Wind:</b> {wind_display}<br/>
                <b>Vapor Pressure Deficit:</b> {vpd_display} &nbsp;|&nbsp; <b>Precip:</b> {precip_display}
            </div>
            <div style="margin-top: 6px; font-size: 10px; color: #94A3B8; border-top: 1px dashed #244563; padding-top: 4px; line-height: 1.35;">
                <i>Target: Estimated probability of satellite detection within 24h. Does not simulate fire spread.</i>
            </div>
        </div>
        """,
        "style": {"zIndex": 1000},
    }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=bm_info["style"],
    )

    st.pydeck_chart(deck, height=height)

    if show_legend and color_by == "Model Predicted Risk":
        render_risk_legend(show_note=True)


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
    mode_label: str = "Forecast Evaluation",
    basemap_style: str = "Satellite Imagery (Esri World Imagery)",
    height: int = 420,
):
    """
    Renders a focused California map centered on a single evaluated 0.25° grid cell.
    Displays recognizable California geography with the target cell highlighted
    as a semi-transparent 0.25° square with pin marker and contextual tooltip.
    """
    color_map = {
        "Low": [53, 183, 121, 160],
        "Moderate": [246, 183, 60, 180],
        "High": [244, 81, 30, 200],
        "Extreme": [197, 31, 50, 220],
    }
    cell_fill = color_map.get(risk_tier, [244, 81, 30, 190])

    half_size = 0.125
    target_poly = [
        [target_lon - half_size, target_lat - half_size],
        [target_lon + half_size, target_lat - half_size],
        [target_lon + half_size, target_lat + half_size],
        [target_lon - half_size, target_lat + half_size],
    ]

    target_df = pd.DataFrame([{
        "latitude": float(target_lat),
        "longitude": float(target_lon),
        "polygon": target_poly,
        "col_elevation": max(1500.0, float(risk_score) * 25000.0),
        "fill_color": cell_fill,
        "stroke_color": [255, 255, 255, 240],
        "risk_display": f"{risk_score:.1%}",
        "risk_tier": risk_tier,
        "alert_text": "ELEVATED ALERT" if is_alert else "BASELINE NORMAL",
        "temp_display": f"{temp_c:.1f}°C",
        "wind_display": f"{wind_speed:.1f} m/s",
        "elev_display": f"{elev:.0f} m",
        "land_cover_name": land_cover_name,
        "mode_label": mode_label,
    }])

    layers = []
    bm_info = BASEMAP_CATALOG.get(basemap_style, BASEMAP_CATALOG["Satellite Imagery (Esri World Imagery)"])

    if bm_info.get("use_tile_layer"):
        sat_tile_layer = pdk.Layer(
            "TileLayer",
            data=bm_info["tile_url"],
            min_zoom=0,
            max_zoom=19,
            tile_size=256,
        )
        layers.append(sat_tile_layer)

    target_poly_layer = pdk.Layer(
        "PolygonLayer",
        data=target_df,
        get_polygon="polygon",
        get_fill_color="fill_color",
        get_line_color="stroke_color",
        line_width_min_pixels=3,
        stroked=True,
        filled=True,
        extruded=False,
        pickable=True,
    )
    layers.append(target_poly_layer)

    pin_layer = pdk.Layer(
        "ScatterplotLayer",
        data=target_df,
        get_position=["longitude", "latitude"],
        get_radius=12000,
        get_fill_color=cell_fill[:3] + [240],
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=2,
        stroked=True,
        pickable=True,
    )
    layers.append(pin_layer)

    view_state = pdk.ViewState(
        latitude=target_lat,
        longitude=target_lon,
        zoom=7.4,
        pitch=0.0,
        bearing=0.0,
    )

    tooltip = {
        "html": """
        <div style="background-color: #071A2C; padding: 12px 14px; border: 1px solid #29465E; border-radius: 8px; color: #F8FAFC; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 12px; line-height: 1.5; min-width: 220px; box-shadow: 0 4px 16px rgba(0,0,0,0.5);">
            <div style="font-size: 11px; font-weight: 700; color: #FF7A24; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
                {mode_label} // 0.25° Cell (~25 km)
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
                <b>24h Fire Risk:</b> <span style="font-weight: 800; font-size: 15px; color: #FF7A24;">{risk_display}</span><br/>
                <b>Classification:</b> {risk_tier} ({alert_text})
            </div>
            <div style="font-size: 10px; color: #94A3B8; margin-top: 6px; border-top: 1px solid #29465E; padding-top: 4px;">
                Target: Next-24h satellite detection (fire_next_24h). Not a physical spread simulation.
            </div>
        </div>
        """,
        "style": {"zIndex": 1000},
    }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=bm_info["style"],
    )

    st.pydeck_chart(deck, height=height)


def render_plotly_topographic_twin(
    df: pd.DataFrame,
    color_by: str = "Model Predicted Risk",
    height: int = 580,
):
    """
    Renders a 3D topographic scatter view via Plotly as an alternative orbital perspective.
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
