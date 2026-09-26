"""
Wildfire Prediction Page - California Wildfire Prediction System
Implements EXACTLY two prediction modes:
1. Mode 1: Manual Input (User enters environmental parameters and clicks Predict)
2. Mode 2: Select Date & Predict (User selects date/time & location, real Open-Meteo forecast is retrieved and evaluated)

Both modes are directly integrated with the interactive 3D Digital Twin,
rendering the evaluated grid cell with an elevated, highlighted pillar on California terrain.
"""
from datetime import datetime, date, timedelta, timezone
import streamlit as st
import pandas as pd
import numpy as np
from src.theme import (
    render_main_hero,
    render_section_header,
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
from src.model_loader import (
    load_feature_schema,
    load_pipeline,
    prepare_feature_row,
    predict_wildfire_risk,
    calc_vpd,
    calc_rh,
    LAND_COVER_LABELS,
)
from src.data_loader import (
    fetch_open_meteo_forecast,
    get_cell_static_and_vegetation,
    CALIFORNIA_PRESET_LOCATIONS,
)
from src.twin_3d import render_single_cell_3d_twin


def render_prediction_page():
    # 1. Main Landing Hero Banner
    render_main_hero(
        title="Wildfire Risk Prediction Workspace",
        subtitle="Evaluate next-24-hour wildfire detection probability (fire_next_24h) across California via direct environmental parameters or automated Open-Meteo weather forecasts.",
        eyebrow="🎯 24-HOUR RISK INFERENCE",
    )

    schema, schema_err = load_feature_schema()
    pipeline, pipeline_err = load_pipeline()

    if schema is None or pipeline is None:
        st.error(f"Prediction engine unavailable. Schema: {schema_err} | Pipeline: {pipeline_err}")
        return

    threshold = float(schema.get("decision_threshold", 0.2507))

    # EXACTLY TWO MODES VIA CLEAN TABS
    tab_manual, tab_forecast = st.tabs([
        "📝 Manual Input",
        "📅 Select Date & Predict",
    ])

    # =========================================================================
    # MODE 1: MANUAL INPUT
    # =========================================================================
    with tab_manual:
        render_html(f"""
        <div style="font-size: 0.88rem; color: {COLOR_TEXT_SECONDARY}; margin-bottom: 1.25rem;">
            Enter environmental parameters manually. The application automatically calculates derived meteorological and cyclical features, applies the saved preprocessing pipeline, and renders the result on the 3D Digital Twin.
        </div>
        """)

        col_m_inputs, col_m_results = st.columns([1.05, 0.95], gap="large")

        with col_m_inputs:
            render_section_header("🎯 Environmental Parameter Inputs", tag="MODE 1: MANUAL FORM")

            # 1. Date & Time Context
            render_html(f"""
            <div style="font-family: ui-monospace, monospace; font-size: 0.74rem; font-weight: 700; color: {COLOR_WARM_ORANGE}; letter-spacing: 0.08em; margin: 0.5rem 0 0.4rem 0;">
                1. DATE & TIME CONTEXT
            </div>
            """)

            c_date, c_hr = st.columns(2)
            with c_date:
                m_date = st.date_input(
                    "Observation Date",
                    value=date(2024, 8, 15),
                    help="Determines month and day-of-year seasonality.",
                )
            with c_hr:
                m_hour = st.slider(
                    "Observation Hour (UTC)",
                    min_value=0,
                    max_value=23,
                    value=20,
                    help="Hour in UTC (20:00 UTC = 13:00 PDT peak solar afternoon in California).",
                )

            m_month = m_date.month
            m_doy = m_date.timetuple().tm_yday

            # 2. Location & Terrain
            render_html(f"""
            <div style="font-family: ui-monospace, monospace; font-size: 0.74rem; font-weight: 700; color: {COLOR_WARM_ORANGE}; letter-spacing: 0.08em; margin: 1.25rem 0 0.4rem 0;">
                2. LOCATION & TERRAIN
            </div>
            """)

            c_lat, c_lon = st.columns(2)
            with c_lat:
                m_lat = st.slider("Latitude (°N)", min_value=32.5, max_value=42.0, value=38.50, step=0.25)
            with c_lon:
                m_lon = st.slider("Longitude (°W)", min_value=-124.5, max_value=-114.0, value=-122.50, step=0.25)

            c_elev, c_lc = st.columns(2)
            with c_elev:
                m_elev = st.number_input(
                    "Elevation (SRTM, meters)",
                    min_value=0.0,
                    max_value=3800.0,
                    value=250.0,
                    step=25.0,
                    help="Terrain height above sea level.",
                )
            with c_lc:
                lc_keys = list(LAND_COVER_LABELS.keys())
                lc_options = [f"{k:.0f} - {LAND_COVER_LABELS[k]}" for k in lc_keys]
                m_lc_str = st.selectbox(
                    "MODIS Land Cover",
                    options=lc_options,
                    index=7,  # Grasslands
                    help="IGBP land cover classification from MODIS MCD12Q1.",
                )
                m_lc_val = float(m_lc_str.split(" - ")[0])
                m_lc_name = LAND_COVER_LABELS.get(m_lc_val, "Terrestrial")

            # 3. Weather Conditions
            render_html(f"""
            <div style="font-family: ui-monospace, monospace; font-size: 0.74rem; font-weight: 700; color: {COLOR_WARM_ORANGE}; letter-spacing: 0.08em; margin: 1.25rem 0 0.4rem 0;">
                3. ATMOSPHERIC & METEOROLOGICAL CONDITIONS
            </div>
            """)

            c_temp, c_dp = st.columns(2)
            with c_temp:
                m_temp_c = st.slider("2m Air Temperature (°C)", min_value=-5.0, max_value=48.0, value=34.0, step=0.5)
            with c_dp:
                m_dp_c = st.slider("2m Dewpoint (°C)", min_value=-15.0, max_value=30.0, value=9.0, step=0.5)

            # Interactive thermodynamic preview
            live_vpd = calc_vpd(m_temp_c, m_dp_c)
            live_rh = calc_rh(m_temp_c, m_dp_c)
            st.caption(f"⚡ *Auto-derived:* Relative Humidity: **{live_rh:.1f}%** &nbsp;|&nbsp; Vapor Pressure Deficit (VPD): **{live_vpd:.2f} kPa**")

            c_w_spd, c_w_dir = st.columns(2)
            with c_w_spd:
                m_wind_spd = st.slider("10m Wind Speed (m/s)", min_value=0.0, max_value=25.0, value=5.5, step=0.5)
            with c_w_dir:
                m_wind_dir = st.slider("Wind Direction (° from North)", min_value=0, max_value=360, value=250, step=10, help="Compass direction wind is blowing from.")

            c_sp, c_tp = st.columns(2)
            with c_sp:
                m_sp_hpa = st.number_input("Surface Pressure (hPa)", min_value=700.0, max_value=1050.0, value=995.0, step=5.0)
            with c_tp:
                m_tp_mm = st.number_input("Hourly Precipitation (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)

            # 4. Vegetation & Antecedent Drought Indicators
            render_html(f"""
            <div style="font-family: ui-monospace, monospace; font-size: 0.74rem; font-weight: 700; color: {COLOR_WARM_ORANGE}; letter-spacing: 0.08em; margin: 1.25rem 0 0.4rem 0;">
                4. VEGETATION & ANTECEDENT FUEL DROUGHT
            </div>
            """)

            c_ndvi, c_evi = st.columns(2)
            with c_ndvi:
                m_ndvi_raw = st.slider("MODIS NDVI (Greenness × 10⁴)", min_value=500.0, max_value=9000.0, value=4200.0, step=100.0, help="Vegetation index (4200 = 0.42 NDVI).")
            with c_evi:
                m_evi_raw = st.slider("MODIS EVI (Canopy × 10⁴)", min_value=300.0, max_value=7000.0, value=2800.0, step=100.0)

            c_p24, c_p7 = st.columns(2)
            with c_p24:
                m_tp_24h_mm = st.number_input("24h Accum Precip (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
            with c_p7:
                m_tp_7d_mm = st.number_input("7d Accum Precip (mm)", min_value=0.0, max_value=300.0, value=0.0, step=1.0)

            c_t7m, c_w24m = st.columns(2)
            with c_t7m:
                m_t7_mean = st.number_input("7d Mean Temp (°C)", min_value=-5.0, max_value=45.0, value=30.0, step=0.5)
            with c_w24m:
                m_w24_mean = st.number_input("24h Mean Wind (m/s)", min_value=0.0, max_value=25.0, value=4.5, step=0.5)

            # Action Button
            btn_predict_manual = st.button("⚡ PREDICT WILDFIRE RISK (MANUAL)", type="primary", use_container_width=True)

        with col_m_results:
            render_section_header("📊 Prediction Evaluation Panel", tag="OUTPUT & 3D TWIN")

            # Input validation
            val_errors = []
            if m_dp_c > m_temp_c:
                val_errors.append("Dewpoint temperature cannot exceed air temperature.")

            if val_errors:
                for err in val_errors:
                    st.error(f"Validation Error: {err}")
            else:
                # Mathematical derivation of vector wind
                w_dir_rad = np.radians(m_wind_dir)
                m_u10 = -m_wind_spd * np.sin(w_dir_rad)
                m_v10 = -m_wind_spd * np.cos(w_dir_rad)

                raw_dict_manual = {
                    "u10": m_u10,
                    "v10": m_v10,
                    "t2m": m_temp_c + 273.15,
                    "d2m": m_dp_c + 273.15,
                    "sp": m_sp_hpa * 100.0,
                    "tp": m_tp_mm * 0.001,
                    "land_cover": m_lc_val,
                    "elevation": m_elev,
                    "ndvi": m_ndvi_raw,
                    "evi": m_evi_raw,
                    "pixel_reliability": 0.0,
                    "vi_quality": 2116.0,
                    "month": m_month,
                    "hour": m_hour,
                    "day_of_year": m_doy,
                    "latitude": m_lat,
                    "longitude": m_lon,
                    "tp_sum_24h": m_tp_24h_mm * 0.001,
                    "tp_sum_7d": m_tp_7d_mm * 0.001,
                    "tp_sum_30d": 0.0,
                    "vpd_mean_24h": live_vpd,
                    "vpd_mean_7d": live_vpd,
                    "t2m_mean_7d": m_t7_mean,
                    "wind_mean_24h": m_w24_mean,
                }

                df_features_m = prepare_feature_row(raw_dict_manual, schema)
                res_m = predict_wildfire_risk(pipeline, schema, df_features_m)

                score_m = res_m["raw_score"]
                cal_prob_m = res_m["calibrated_prob"]
                is_alert_m = res_m["is_alert"]
                tier_m = res_m["risk_level"]
                tier_color_m = RISK_TIER_COLORS.get(tier_m, COLOR_TEXT_SECONDARY)
                alert_label_m = "ALERT ELEVATED" if is_alert_m else "BASELINE NORMAL"
                alert_bg_m = COLOR_FIRE_RED if is_alert_m else COLOR_GREEN

                # Result Card
                render_html(f"""
                <div class="app-prediction-panel" style="border: 2px solid {tier_color_m};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <span style="font-family: ui-monospace, monospace; font-size: 0.74rem; color: {COLOR_TEXT_SECONDARY}; text-transform: uppercase;">
                            COORDINATES: {m_lat:.2f}°N, {abs(m_lon):.2f}°W
                        </span>
                        <span style="background-color: {alert_bg_m}; color: #FFFFFF; font-family: ui-monospace, monospace; font-size: 0.74rem; font-weight: 700; padding: 3px 10px; border-radius: 4px; letter-spacing: 0.06em;">
                            {alert_label_m}
                        </span>
                    </div>

                    <div style="margin-bottom: 1rem;">
                        <div style="font-size: 0.72rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace; text-transform: uppercase;">
                            RAW MODEL RISK PROBABILITY
                        </div>
                        <div style="font-size: 3.2rem; font-weight: 700; color: {tier_color_m}; line-height: 1.1; margin: 0.2rem 0;">
                            {score_m:.1%}
                        </div>
                        <div style="font-size: 0.82rem; color: {COLOR_TEXT_SECONDARY};">
                            Operating Threshold: <code>{threshold:.4f}</code> &nbsp;•&nbsp; Status: <b>{'Exceeded' if is_alert_m else 'Below Threshold'}</b>
                        </div>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; border-top: 1px solid {COLOR_BORDER}; padding-top: 0.85rem; margin-bottom: 0.85rem;">
                        <div>
                            <div style="font-size: 0.7rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace;">PRIOR-CORRECTED PROBABILITY</div>
                            <div style="font-size: 1.55rem; font-weight: 700; color: {COLOR_TEXT_PRIMARY};">{cal_prob_m:.1%}</div>
                            <div style="font-size: 0.72rem; color: {COLOR_TEXT_SECONDARY};">Calibrated for 6.04% natural rate</div>
                        </div>
                        <div>
                            <div style="font-size: 0.7rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace;">RISK STRATIFICATION</div>
                            <div style="font-size: 1.55rem; font-weight: 700; color: {tier_color_m};">{tier_m.upper()}</div>
                            <div style="font-size: 0.72rem; color: {COLOR_TEXT_SECONDARY};">Random Forest Classifier</div>
                        </div>
                    </div>

                    <div style="background-color: {COLOR_DARK_NAVY}; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 0.75rem 1rem; font-size: 0.8rem; line-height: 1.5; color: {COLOR_TEXT_SECONDARY};">
                        <b style="color: {COLOR_TEXT_PRIMARY};">Scientific Target Contract:</b>
                        Estimates probability of spaceborne active fire detection in the same 0.25° grid cell during the subsequent 24 hours (<code>fire_next_24h</code>). This is a statistical forecast based on atmospheric conditions and fuel dryness, not a confirmed ignition.
                    </div>
                </div>
                """)

                # 3D Digital Twin Integration for Manual Input
                render_html(f"""
                <div style="font-family: ui-monospace, monospace; font-size: 0.74rem; font-weight: 700; color: {COLOR_WARM_ORANGE}; letter-spacing: 0.08em; margin: 1rem 0 0.5rem 0;">
                    🌐 3D DIGITAL TWIN // TARGET CELL SPOTLIGHT
                </div>
                """)

                render_single_cell_3d_twin(
                    target_lat=m_lat,
                    target_lon=m_lon,
                    risk_score=score_m,
                    risk_tier=tier_m,
                    is_alert=is_alert_m,
                    temp_c=m_temp_c,
                    wind_speed=m_wind_spd,
                    elev=m_elev,
                    land_cover_name=m_lc_name,
                    mode_label="Manual Input Mode",
                    height=440,
                )

    # =========================================================================
    # MODE 2: SELECT DATE & PREDICT (AUTOMATED OPEN-METEO FORECAST)
    # =========================================================================
    with tab_forecast:
        render_html(f"""
        <div style="font-size: 0.88rem; color: {COLOR_TEXT_SECONDARY}; margin-bottom: 1.25rem;">
            Select a California location and date/time. The application queries the <b>Open-Meteo Weather API</b> for real hourly meteorological forecasts, derives 30-day antecedent drought indicators, joins genuine static elevation and MODIS Earth observation layers, and evaluates the trained Random Forest model.
        </div>
        """)

        col_f_inputs, col_f_results = st.columns([1.05, 0.95], gap="large")

        with col_f_inputs:
            render_section_header("📅 Forecast Target Selection", tag="MODE 2: REAL FORECAST")

            # 1. Location Selection
            render_html(f"""
            <div style="font-family: ui-monospace, monospace; font-size: 0.74rem; font-weight: 700; color: {COLOR_WARM_ORANGE}; letter-spacing: 0.08em; margin: 0.5rem 0 0.4rem 0;">
                1. SELECT LOCATION OR GRID CELL
            </div>
            """)

            loc_mode = st.radio(
                "Location Selection Method",
                options=["California Regional Presets", "Custom California Coordinates"],
                horizontal=True,
                label_visibility="collapsed",
            )

            if loc_mode == "California Regional Presets":
                preset_key = st.selectbox(
                    "California Region / Hotspot:",
                    options=list(CALIFORNIA_PRESET_LOCATIONS.keys()),
                    index=0,
                )
                preset_info = CALIFORNIA_PRESET_LOCATIONS[preset_key]
                f_lat = preset_info["latitude"]
                f_lon = preset_info["longitude"]
                st.caption(f"📍 **{preset_key}** ({preset_info['region']}): `{f_lat:.2f}°N, {abs(f_lon):.2f}°W`")
            else:
                c_flat, c_flon = st.columns(2)
                with c_flat:
                    f_lat = st.slider("Forecast Latitude (°N)", min_value=32.5, max_value=42.0, value=38.50, step=0.25)
                with c_flon:
                    f_lon = st.slider("Forecast Longitude (°W)", min_value=-124.5, max_value=-114.0, value=-122.50, step=0.25)

            # 2. Date & Time Selection (Today up to 7 days into future)
            render_html(f"""
            <div style="font-family: ui-monospace, monospace; font-size: 0.74rem; font-weight: 700; color: {COLOR_WARM_ORANGE}; letter-spacing: 0.08em; margin: 1.25rem 0 0.4rem 0;">
                2. SELECT FORECAST DATE & TIME
            </div>
            """)

            now_utc = datetime.now(timezone.utc)
            min_date = now_utc.date()
            max_date = min_date + timedelta(days=7)

            c_fdate, c_fhour = st.columns(2)
            with c_fdate:
                f_date = st.date_input(
                    "Forecast Date (Supported Range: Next 7 Days)",
                    value=min_date + timedelta(days=1),
                    min_value=min_date,
                    max_value=max_date,
                    help="Hourly Open-Meteo global weather model forecasts are available up to 7 days in advance.",
                )
            with c_fhour:
                f_hour = st.slider(
                    "Forecast Hour (UTC)",
                    min_value=0,
                    max_value=23,
                    value=20,
                    help="Hour in UTC (20:00 UTC = 13:00 PDT local afternoon).",
                )

            # Target Timestamp string
            target_ts_str = f"{f_date.strftime('%Y-%m-%d')} {f_hour:02d}:00:00"
            local_pdt_hr = (f_hour - 7) % 24
            st.caption(f"🕒 Target: **{target_ts_str} UTC** (approx. **{local_pdt_hr:02d}:00 PDT** California local time)")

            # Static and Vegetation layer preview
            try:
                static_meta = get_cell_static_and_vegetation(f_lat, f_lon)
                is_water = static_meta["is_water"]
                elev_val = static_meta["elevation"]
                lc_val = static_meta["land_cover"]
                lc_name_val = static_meta["land_cover_name"]
                veg_date_val = static_meta["vegetation_obs_date"]

                if is_water:
                    st.warning(f"⚠️ Selected coordinate (`{f_lat:.2f}°N, {abs(f_lon):.2f}°W`) is classified as open water/ocean. Wildland fire modeling requires terrestrial land cells.")
                else:
                    render_html(f"""
                    <div style="background-color: {COLOR_DEEP_NAVY}; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 10px 12px; margin-top: 8px; font-size: 0.8rem; line-height: 1.5;">
                        <span style="color: {COLOR_GREEN}; font-weight: 600;">✓ TERRESTRIAL GRID CELL RESOLVED</span><br/>
                        <b>Elevation:</b> {elev_val:.0f} m &nbsp;|&nbsp; <b>Land Cover:</b> {lc_name_val}<br/>
                        <b>Latest Spaceborne Vegetation:</b> {veg_date_val} (MODIS MOD13Q1 composite)
                    </div>
                    """)
            except Exception as e:
                st.error(f"Error accessing static project data: {e}")
                is_water = False

            # Primary Forecast Action Button
            btn_predict_forecast = st.button("🌐 GET FORECAST & PREDICT", type="primary", use_container_width=True)

        with col_f_results:
            render_section_header("📊 Forecast Evaluation & 3D Twin", tag="REAL-TIME METEOROLOGY")

            if btn_predict_forecast:
                st.session_state["has_run_forecast"] = True
                st.session_state["f_lat"] = f_lat
                st.session_state["f_lon"] = f_lon
                st.session_state["f_ts_str"] = target_ts_str

            run_f = st.session_state.get("has_run_forecast", False)

            if not run_f:
                render_html(f"""
                <div style="background-color: {COLOR_DEEP_NAVY}; border: 1px dashed {COLOR_BORDER}; border-radius: 8px; padding: 2rem 1.5rem; text-align: center; color: {COLOR_TEXT_SECONDARY};">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🛰️</div>
                    <div style="font-weight: 700; color: {COLOR_TEXT_PRIMARY}; font-size: 1.1rem; margin-bottom: 0.35rem;">
                        Awaiting Forecast Execution
                    </div>
                    <div style="font-size: 0.84rem; max-width: 400px; margin: 0 auto; line-height: 1.5;">
                        Select your location and future forecast date/time on the left, then click <b>GET FORECAST & PREDICT</b> to retrieve real-time meteorology and evaluate next-24h wildfire probability.
                    </div>
                </div>
                """)
            else:
                exec_lat = st.session_state.get("f_lat", f_lat)
                exec_lon = st.session_state.get("f_lon", f_lon)
                exec_ts = st.session_state.get("f_ts_str", target_ts_str)

                with st.spinner("Retrieving real-time meteorology from Open-Meteo & evaluating model..."):
                    try:
                        # 1. Fetch real weather forecast
                        fc_data = fetch_open_meteo_forecast(exec_lat, exec_lon, exec_ts)

                        # 2. Get static & vegetation data
                        cell_meta = get_cell_static_and_vegetation(exec_lat, exec_lon)

                        if cell_meta["is_water"]:
                            st.error(f"Cannot run wildfire prediction for ocean/water coordinate: {exec_lat:.2f}°N, {abs(exec_lon):.2f}°W.")
                        else:
                            # 3. Assemble complete 33-feature raw input
                            target_dt_obj = pd.Timestamp(exec_ts)
                            raw_dict_forecast = {
                                "u10": fc_data["u10"],
                                "v10": fc_data["v10"],
                                "t2m": fc_data["t2m_c"] + 273.15,
                                "d2m": fc_data["d2m_c"] + 273.15,
                                "sp": fc_data["sp_pa"],
                                "tp": fc_data["tp_m"],
                                "land_cover": cell_meta["land_cover"],
                                "elevation": cell_meta["elevation"],
                                "ndvi": cell_meta["ndvi"],
                                "evi": cell_meta["evi"],
                                "pixel_reliability": cell_meta["pixel_reliability"],
                                "vi_quality": cell_meta["vi_quality"],
                                "month": target_dt_obj.month,
                                "hour": target_dt_obj.hour,
                                "day_of_year": target_dt_obj.dayofyear,
                                "latitude": cell_meta["cell_latitude"],
                                "longitude": cell_meta["cell_longitude"],
                                "tp_sum_24h": fc_data["tp_sum_24h"],
                                "tp_sum_7d": fc_data["tp_sum_7d"],
                                "tp_sum_30d": fc_data["tp_sum_30d"],
                                "vpd_mean_24h": fc_data["vpd_mean_24h"],
                                "vpd_mean_7d": fc_data["vpd_mean_7d"],
                                "t2m_mean_7d": fc_data["t2m_mean_7d"],
                                "wind_mean_24h": fc_data["wind_mean_24h"],
                            }

                            # 4. Feature Preparation & Inference
                            df_features_f = prepare_feature_row(raw_dict_forecast, schema)
                            res_f = predict_wildfire_risk(pipeline, schema, df_features_f)

                            score_f = res_f["raw_score"]
                            cal_prob_f = res_f["calibrated_prob"]
                            is_alert_f = res_f["is_alert"]
                            tier_f = res_f["risk_level"]
                            tier_color_f = RISK_TIER_COLORS.get(tier_f, COLOR_TEXT_SECONDARY)
                            alert_label_f = "ALERT ELEVATED" if is_alert_f else "BASELINE NORMAL"
                            alert_bg_f = COLOR_FIRE_RED if is_alert_f else COLOR_GREEN

                            # 5. Display Prediction Result Panel
                            render_html(f"""
                            <div class="app-prediction-panel" style="border: 2px solid {tier_color_f};">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.85rem;">
                                    <span style="font-family: ui-monospace, monospace; font-size: 0.74rem; color: {COLOR_TEXT_SECONDARY}; text-transform: uppercase;">
                                        COORDINATES: {exec_lat:.2f}°N, {abs(exec_lon):.2f}°W
                                    </span>
                                    <span style="background-color: {alert_bg_f}; color: #FFFFFF; font-family: ui-monospace, monospace; font-size: 0.74rem; font-weight: 700; padding: 3px 10px; border-radius: 4px; letter-spacing: 0.06em;">
                                        {alert_label_f}
                                    </span>
                                </div>

                                <div style="margin-bottom: 0.85rem;">
                                    <div style="font-size: 0.72rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace; text-transform: uppercase;">
                                        RAW MODEL RISK PROBABILITY
                                    </div>
                                    <div style="font-size: 3.2rem; font-weight: 700; color: {tier_color_f}; line-height: 1.1; margin: 0.2rem 0;">
                                        {score_f:.1%}
                                    </div>
                                    <div style="font-size: 0.82rem; color: {COLOR_TEXT_SECONDARY};">
                                        Operating Threshold: <code>{threshold:.4f}</code> &nbsp;•&nbsp; Status: <b>{'Exceeded' if is_alert_f else 'Below Threshold'}</b>
                                    </div>
                                </div>

                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; border-top: 1px solid {COLOR_BORDER}; padding-top: 0.85rem; margin-bottom: 0.85rem;">
                                    <div>
                                        <div style="font-size: 0.7rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace;">PRIOR-CORRECTED PROBABILITY</div>
                                        <div style="font-size: 1.55rem; font-weight: 700; color: {COLOR_TEXT_PRIMARY};">{cal_prob_f:.1%}</div>
                                        <div style="font-size: 0.72rem; color: {COLOR_TEXT_SECONDARY};">Natural fire prevalence: 6.04%</div>
                                    </div>
                                    <div>
                                        <div style="font-size: 0.7rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace;">RISK CLASSIFICATION</div>
                                        <div style="font-size: 1.55rem; font-weight: 700; color: {tier_color_f};">{tier_f.upper()}</div>
                                        <div style="font-size: 0.72rem; color: {COLOR_TEXT_SECONDARY};">RandomForest (200 trees)</div>
                                    </div>
                                </div>

                                <div style="background-color: {COLOR_DARK_NAVY}; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 0.75rem 1rem; font-size: 0.8rem; line-height: 1.5; color: {COLOR_TEXT_SECONDARY};">
                                    <b style="color: {COLOR_TEXT_PRIMARY};">Forecast Provenance:</b>
                                    Provider: <b>{fc_data['provider']}</b><br/>
                                    Retrieved: <code>{fc_data['retrieval_time']}</code><br/>
                                    Valid Forecast Timestamp: <code>{fc_data['forecast_time'].strftime('%Y-%m-%d %H:%M UTC')}</code><br/>
                                    Vegetation Layer: <code>{cell_meta['vegetation_source']} ({cell_meta['vegetation_obs_date']})</code>
                                </div>
                            </div>
                            """)

                            # 6. Main Weather Inputs Table
                            render_html(f"""
                            <div style="font-family: ui-monospace, monospace; font-size: 0.74rem; font-weight: 700; color: {COLOR_WARM_ORANGE}; letter-spacing: 0.08em; margin: 1rem 0 0.5rem 0;">
                                🌤️ RETRIEVED METEOROLOGICAL FORECAST VALUES
                            </div>
                            """)

                            w_cols1, w_cols2, w_cols3, w_cols4 = st.columns(4)
                            with w_cols1:
                                st.metric("2m Air Temp", f"{fc_data['t2m_c']:.1f} °C")
                                st.metric("Surface Pressure", f"{fc_data['sp_pa']/100.0:.0f} hPa")
                            with w_cols2:
                                st.metric("2m Dewpoint", f"{fc_data['d2m_c']:.1f} °C")
                                st.metric("Relative Humidity", f"{fc_data['rh']:.0f} %")
                            with w_cols3:
                                st.metric("10m Wind Speed", f"{fc_data['wind_speed']:.1f} m/s")
                                st.metric("VPD (Dryness)", f"{fc_data['vpd_kpa']:.2f} kPa")
                            with w_cols4:
                                st.metric("Hourly Precip", f"{fc_data['tp_m']*1000.0:.1f} mm")
                                st.metric("30d Precip Accum", f"{fc_data['tp_sum_30d']*1000.0:.1f} mm")

                            # 7. 3D Digital Twin Integration for Forecast Mode
                            render_html(f"""
                            <div style="font-family: ui-monospace, monospace; font-size: 0.74rem; font-weight: 700; color: {COLOR_WARM_ORANGE}; letter-spacing: 0.08em; margin: 1.25rem 0 0.5rem 0;">
                                🌐 3D DIGITAL TWIN // FORECAST CELL SPOTLIGHT
                            </div>
                            """)

                            render_single_cell_3d_twin(
                                target_lat=exec_lat,
                                target_lon=exec_lon,
                                risk_score=score_f,
                                risk_tier=tier_f,
                                is_alert=is_alert_f,
                                temp_c=fc_data["t2m_c"],
                                wind_speed=fc_data["wind_speed"],
                                elev=cell_meta["elevation"],
                                land_cover_name=cell_meta["land_cover_name"],
                                mode_label="Select Date & Predict Mode",
                                height=440,
                            )

                    except Exception as e:
                        st.error(f"Inference Blocked: {str(e)}")
                        st.info("The forecast-backed prediction was blocked because required weather data could not be verified from the API. The system refuses to use fake or synthetic fallback data.")

    # Shared Operational Scope Notice
    render_operational_notice()
