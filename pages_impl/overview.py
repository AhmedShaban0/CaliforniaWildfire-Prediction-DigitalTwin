"""
Overview Page - California Wildfire Prediction System
Executive console providing:
1. Prominent hero banner with project identity.
2. Executive Summary & Key Indicators at a glance with brief explanations of the numbers.
3. Statewide California Geographic Risk Map with real basemaps (Terrain, Satellite, Streets),
   transparent 0.25° grid cells (~25 km), clear color legend, and cell inspector tooltip.
4. Quick Live Forecast & Prediction spotlight powered by Open-Meteo and the trained model.
5. Numbered architectural workflow cards.
6. Scientific governance disclosure.
"""
from datetime import datetime, date, timedelta, timezone
import streamlit as st
import pandas as pd
import numpy as np
from src.theme import (
    render_main_hero,
    render_section_header,
    render_intelligence_card,
    render_workflow_card,
    render_operational_notice,
    render_risk_legend,
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
)
from src.data_loader import (
    open_dataset,
    CURATED_SCENARIOS,
    load_spatial_slice_with_predictions,
    fetch_open_meteo_forecast,
    get_cell_static_and_vegetation,
    CALIFORNIA_PRESET_LOCATIONS,
)
from src.twin_3d import (
    render_pydeck_3d_twin,
    render_single_cell_3d_twin,
    BASEMAP_CATALOG,
)


def render_overview_page():
    # 1. Main Landing Hero Banner
    render_main_hero(
        title="California Wildfire Prediction Console",
        subtitle="Forecasting next-24-hour spaceborne active fire detection risk across California's 0.25° grid (~25 km × 25 km) using Copernicus ERA5 reanalysis, MODIS Earth observation, and real-time weather forecasts.",
        eyebrow="🔥 CALIFORNIA WILDFIRE RISK INTELLIGENCE",
    )

    schema, schema_err = load_feature_schema()
    pipeline, pipeline_err = load_pipeline()
    threshold = float(schema.get("decision_threshold", 0.2507)) if schema else 0.2507

    # 2. Executive Summary & Key Indicators At a Glance
    render_section_header("⚡ System Summary & Operational Metrics", tag="EXECUTIVE INDICATORS")

    i1, i2, i3, i4 = st.columns(4)
    with i1:
        render_intelligence_card(
            icon="⏱️",
            eyebrow="FORECAST HORIZON",
            value="24 Hours",
            sub="Advance probability of satellite detection (fire_next_24h) under a zero-leakage causal embargo.",
            border_color=COLOR_BORDER,
        )
    with i2:
        render_intelligence_card(
            icon="🌐",
            eyebrow="GRID RESOLUTION",
            value="0.25° (~25 km)",
            sub="1,677 terrestrial California cells (~625 km² per cell) covering the entire state.",
            border_color=COLOR_WARM_ORANGE,
        )
    with i3:
        render_intelligence_card(
            icon="🎯",
            eyebrow="ALERT THRESHOLD",
            value=f"{threshold:.1%}",
            sub=f"F₂-tuned decision point (≥{threshold:.4f}) prioritizing fire capture (70.8% recall) over false alarms.",
            border_color=COLOR_FIRE_RED,
        )
    with i4:
        render_intelligence_card(
            icon="🛰️",
            eyebrow="LIVE METEOROLOGY",
            value="Open-Meteo",
            sub="Automated real-time forecast retrieval with 24h, 7d, and 30d antecedent drying history.",
            border_color=COLOR_GREEN,
        )

    # 3. Section: Statewide Wildfire Situational Intelligence
    render_section_header("🗺️ Statewide Wildfire Situational Intelligence", tag="GEOGRAPHIC RISK MAP")

    col_map_ctrl1, col_map_ctrl2, col_map_ctrl3, col_map_ctrl4 = st.columns([3, 2.3, 1.8, 1.8], gap="small")
    with col_map_ctrl1:
        scenario_name = st.selectbox(
            "Historical Meteorological Snapshot:",
            options=list(CURATED_SCENARIOS.keys()),
            index=0,
            help="Select verified environmental scenarios from 2024-2025 to explore regional risk distribution.",
        )
    with col_map_ctrl2:
        basemap_choice = st.selectbox(
            "Basemap Style:",
            options=list(BASEMAP_CATALOG.keys()),
            index=0,
            help="Choose a real geographic basemap. All default options are free and require no API key.",
        )
    with col_map_ctrl3:
        view_mode = st.selectbox(
            "Map Perspective:",
            options=["2D Geographic Map", "3D Volumetric (Extruded)"],
            index=0,
            help="Toggle between flat 2D choropleth polygons and 3D extruded terrain blocks.",
        )
    with col_map_ctrl4:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔥 GO TO PREDICTOR", use_container_width=True, type="secondary"):
            st.session_state["nav_target"] = "Wildfire Prediction"
            st.rerun()

    chosen_scenario = CURATED_SCENARIOS[scenario_name]
    ts_str = chosen_scenario["timestamp"]

    with st.spinner("Extracting California grid and computing 24h model risk..."):
        slice_df = load_spatial_slice_with_predictions(ts_str, ts_str, land_only=True)

    if slice_df is not None and not slice_df.empty:
        # Render the pydeck map with real California geography & transparent 0.25 deg polygons
        render_pydeck_3d_twin(
            slice_df,
            color_by="Model Predicted Risk",
            basemap_style=basemap_choice,
            view_mode=view_mode,
            layer_opacity=0.55,
            elevation_scale=2.0,
            pitch=35.0 if view_mode.startswith("3D") else 0.0,
            bearing=-12.0 if view_mode.startswith("3D") else 0.0,
            height=480,
            show_legend=True,
        )

        alerts_count = int(slice_df["is_alert"].sum())
        fires_count = int((slice_df["fire_now"] > 0).sum())
        max_risk = slice_df["model_risk_score"].max()
        mean_risk = slice_df["model_risk_score"].mean()

        # Regional Dossier Strip below map
        html_dossier = f"""
<div style="background-color: {COLOR_DEEP_NAVY}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 0.85rem 1.15rem; margin-top: 0.75rem; margin-bottom: 1.5rem;">
    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.35rem; flex-wrap: wrap;">
        <div style="font-weight: 700; font-size: 0.95rem; color: {COLOR_TEXT_PRIMARY};">{chosen_scenario['region_name']}</div>
        <div style="font-family: ui-monospace, monospace; font-size: 0.75rem; color: {COLOR_WARM_ORANGE};">{ts_str} UTC &nbsp;({chosen_scenario.get('description', '')[:50]}...)</div>
    </div>
    <div style="font-size: 0.82rem; color: {COLOR_TEXT_SECONDARY}; margin-bottom: 0.65rem;">{chosen_scenario['description']}</div>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.65rem; text-align: center; border-top: 1px solid {COLOR_BORDER}; padding-top: 0.6rem;">
        <div>
            <div style="font-size: 0.68rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace; text-transform: uppercase;">ACTIVE LAND CELLS</div>
            <div style="font-weight: 700; color: {COLOR_TEXT_PRIMARY}; font-size: 1.05rem;">{len(slice_df):,}</div>
            <div style="font-size: 0.7rem; color: {COLOR_TEXT_SECONDARY};">0.25° (~25 km) grid</div>
        </div>
        <div>
            <div style="font-size: 0.68rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace; text-transform: uppercase;">ALERTS (≥{threshold:.1%})</div>
            <div style="font-weight: 700; color: {COLOR_FIRE_ORANGE}; font-size: 1.05rem;">{alerts_count} ({alerts_count / len(slice_df):.1%})</div>
            <div style="font-size: 0.7rem; color: {COLOR_TEXT_SECONDARY};">Operational alert state</div>
        </div>
        <div>
            <div style="font-size: 0.68rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace; text-transform: uppercase;">NASA SATELLITE DETECTIONS</div>
            <div style="font-weight: 700; color: {COLOR_CRIMSON if fires_count > 0 else COLOR_GREEN}; font-size: 1.05rem;">{fires_count} active</div>
            <div style="font-size: 0.7rem; color: {COLOR_TEXT_SECONDARY};">MODIS/VIIRS overpasses</div>
        </div>
        <div>
            <div style="font-size: 0.68rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace; text-transform: uppercase;">PEAK CELL RISK</div>
            <div style="font-weight: 700; color: {COLOR_FIRE_RED}; font-size: 1.05rem;">{max_risk:.1%}</div>
            <div style="font-size: 0.7rem; color: {COLOR_TEXT_SECONDARY};">Mean statewide: {mean_risk:.1%}</div>
        </div>
    </div>
</div>
"""
        render_html(html_dossier)
    else:
        st.info("Dataset snapshot loading. If `master_dataset_2024_2025.nc` is absent, the fallback snapshot will be used.")

    # 4. Quick Live Forecast & Predict Section
    render_section_header("🌐 Live Forecast & Risk Inference", tag="REAL-TIME API DISPATCH")

    render_html(f"""
    <div style="font-size: 0.88rem; color: {COLOR_TEXT_SECONDARY}; margin-bottom: 1rem; line-height: 1.55;">
        Dispatch live weather forecasts from the <b>Open-Meteo Global Weather Model</b> for any California region,
        combine with spaceborne vegetation observations, and evaluate the trained model directly.
    </div>
    """)

    col_ov_inputs, col_ov_results = st.columns([1.05, 0.95], gap="large")

    with col_ov_inputs:
        ov_preset_key = st.selectbox(
            "California Region / Benchmark Hotspot:",
            options=list(CALIFORNIA_PRESET_LOCATIONS.keys()),
            index=0,
            key="ov_fc_preset",
        )
        ov_preset_info = CALIFORNIA_PRESET_LOCATIONS[ov_preset_key]
        ov_lat = ov_preset_info["latitude"]
        ov_lon = ov_preset_info["longitude"]

        now_utc = datetime.now(timezone.utc)
        min_fdate = now_utc.date()
        max_fdate = min_fdate + timedelta(days=7)

        c_odate, c_ohour = st.columns(2)
        with c_odate:
            ov_fdate = st.date_input(
                "Forecast Date (Next 7 Days)",
                value=min_fdate + timedelta(days=1),
                min_value=min_fdate,
                max_value=max_fdate,
                key="ov_fc_date",
            )
        with c_ohour:
            ov_fhour = st.slider(
                "Forecast Hour (UTC)",
                min_value=0,
                max_value=23,
                value=20,
                key="ov_fc_hour",
                help="20:00 UTC = 13:00 PDT local solar afternoon.",
            )

        ov_ts_str = f"{ov_fdate.strftime('%Y-%m-%d')} {ov_fhour:02d}:00:00"
        ov_local_hr = (ov_fhour - 7) % 24
        st.caption(f"📍 **{ov_preset_key}** (`{ov_lat:.2f}°N, {abs(ov_lon):.2f}°W`) &nbsp;|&nbsp; 🕒 **{ov_ts_str} UTC** ({ov_local_hr:02d}:00 PDT)")

        btn_ov_predict = st.button("🌐 GET FORECAST & PREDICT RISK", type="primary", use_container_width=True, key="ov_btn_predict")

    with col_ov_results:
        if btn_ov_predict:
            st.session_state["has_run_ov_forecast"] = True
            st.session_state["ov_lat"] = ov_lat
            st.session_state["ov_lon"] = ov_lon
            st.session_state["ov_ts"] = ov_ts_str
            st.session_state["ov_region"] = ov_preset_key

        run_ov = st.session_state.get("has_run_ov_forecast", False)

        if not run_ov:
            html_await = f"""
<div style="background-color: {COLOR_DEEP_NAVY}; border: 1px dashed {COLOR_BORDER}; border-radius: 8px; padding: 2.2rem 1.5rem; text-align: center; color: {COLOR_TEXT_SECONDARY}; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🛰️</div>
    <div style="font-weight: 700; color: {COLOR_TEXT_PRIMARY}; font-size: 1.15rem; margin-bottom: 0.35rem;">
        Ready for Real-Time Forecast
    </div>
    <div style="font-size: 0.85rem; max-width: 380px; margin: 0 auto; line-height: 1.55;">
        Select a California hotspot on the left and click <b>GET FORECAST & PREDICT RISK</b> to retrieve live hourly weather and evaluate next-24h wildfire risk.
    </div>
</div>
"""
            render_html(html_await)
        else:
            exec_lat = st.session_state.get("ov_lat", ov_lat)
            exec_lon = st.session_state.get("ov_lon", ov_lon)
            exec_ts = st.session_state.get("ov_ts", ov_ts_str)
            exec_region = st.session_state.get("ov_region", ov_preset_key)

            with st.spinner("Connecting to Open-Meteo & evaluating 24-hour wildfire risk..."):
                try:
                    fc_data = fetch_open_meteo_forecast(exec_lat, exec_lon, exec_ts)
                    cell_meta = get_cell_static_and_vegetation(exec_lat, exec_lon)

                    if cell_meta["is_water"]:
                        st.error(f"Cannot run wildfire prediction for open water coordinate: {exec_lat:.2f}°N, {abs(exec_lon):.2f}°W.")
                    else:
                        target_dt_obj = pd.Timestamp(exec_ts)
                        raw_dict_ov = {
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

                        df_features_ov = prepare_feature_row(raw_dict_ov, schema)
                        res_ov = predict_wildfire_risk(pipeline, schema, df_features_ov)

                        score_ov = res_ov["raw_score"]
                        cal_prob_ov = res_ov["calibrated_prob"]
                        is_alert_ov = res_ov["is_alert"]
                        tier_ov = res_ov["risk_level"]
                        tier_color_ov = RISK_TIER_COLORS.get(tier_ov, COLOR_TEXT_SECONDARY)
                        alert_label_ov = "ELEVATED ALERT" if is_alert_ov else "NORMAL BASELINE"
                        alert_bg_ov = COLOR_FIRE_RED if is_alert_ov else COLOR_GREEN

                        veg_src_ov = cell_meta.get("vegetation_source", "MODIS MOD13Q1 composite")
                        veg_dt_ov = cell_meta.get("vegetation_obs_date", cell_meta.get("latest_observation", "Latest"))

                        # Prediction result card
                        html_ov_res = f"""
<div class="app-prediction-panel" style="border: 2px solid {tier_color_ov};">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
        <span style="font-family: ui-monospace, monospace; font-size: 0.75rem; color: {COLOR_TEXT_SECONDARY}; text-transform: uppercase;">
            📍 {exec_region} ({exec_lat:.2f}°N, {abs(exec_lon):.2f}°W)
        </span>
        <span style="background-color: {alert_bg_ov}; color: #FFFFFF; font-family: ui-monospace, monospace; font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 4px; letter-spacing: 0.06em;">
            {alert_label_ov}
        </span>
    </div>

    <div style="margin-bottom: 0.75rem;">
        <div style="font-size: 0.72rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace; text-transform: uppercase;">
            ESTIMATED 24-HOUR WILDFIRE DETECTION PROBABILITY
        </div>
        <div style="font-size: 2.8rem; font-weight: 750; color: {tier_color_ov}; line-height: 1.1; margin: 0.2rem 0;">
            {score_ov:.1%}
        </div>
        <div style="font-size: 0.82rem; color: {COLOR_TEXT_SECONDARY};">
            Operational Cutoff: <code>{threshold:.4f}</code> &nbsp;•&nbsp; Status: <b>{'Exceeded (Alert Activated)' if is_alert_ov else 'Below Cutoff (Routine Posture)'}</b>
        </div>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.85rem; border-top: 1px solid {COLOR_BORDER}; padding-top: 0.75rem; margin-bottom: 0.75rem;">
        <div>
            <div style="font-size: 0.7rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace;">PRIOR-CORRECTED RATE</div>
            <div style="font-size: 1.35rem; font-weight: 700; color: {COLOR_TEXT_PRIMARY};">{cal_prob_ov:.1%}</div>
            <div style="font-size: 0.72rem; color: {COLOR_TEXT_SECONDARY};">Natural fire rate: 6.04%</div>
        </div>
        <div>
            <div style="font-size: 0.7rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace;">RISK CLASSIFICATION</div>
            <div style="font-size: 1.35rem; font-weight: 700; color: {tier_color_ov};">{tier_ov.upper()}</div>
            <div style="font-size: 0.72rem; color: {COLOR_TEXT_SECONDARY};">Random Forest (200 trees)</div>
        </div>
    </div>

    <div style="background-color: {COLOR_DARK_NAVY}; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 0.65rem 0.85rem; font-size: 0.78rem; line-height: 1.5; color: {COLOR_TEXT_SECONDARY};">
        <b style="color: {COLOR_TEXT_PRIMARY};">Data Provenance:</b>
        Weather: <b>Open-Meteo</b> ({fc_data['retrieval_time']})<br/>
        Target: <code>{fc_data['forecast_time'].strftime('%Y-%m-%d %H:%M UTC')}</code> &nbsp;|&nbsp;
        Fuels: <code>{veg_src_ov} ({veg_dt_ov})</code>
    </div>
</div>
"""
                        render_html(html_ov_res)

                        # Weather metrics
                        w1, w2, w3, w4 = st.columns(4)
                        with w1:
                            st.metric("Air Temp", f"{fc_data['t2m_c']:.1f} °C")
                        with w2:
                            st.metric("Dewpoint", f"{fc_data['d2m_c']:.1f} °C")
                        with w3:
                            st.metric("Wind Speed", f"{fc_data['wind_speed']:.1f} m/s")
                        with w4:
                            st.metric("VPD Dryness", f"{fc_data['vpd_kpa']:.2f} kPa")

                        # Single-cell geographic map preview
                        render_single_cell_3d_twin(
                            target_lat=exec_lat,
                            target_lon=exec_lon,
                            risk_score=score_ov,
                            risk_tier=tier_ov,
                            is_alert=is_alert_ov,
                            temp_c=fc_data["t2m_c"],
                            wind_speed=fc_data["wind_speed"],
                            elev=cell_meta["elevation"],
                            land_cover_name=cell_meta["land_cover_name"],
                            mode_label="Overview Quick Predict",
                            basemap_style=basemap_choice,
                            height=320,
                        )
                except ValueError as ve:
                    st.error(f"Weather Verification Notice: {ve}")
                    st.info("Live prediction was blocked because required weather data could not be verified from the API.")
                except Exception as e:
                    st.error(f"Inference Blocked: {str(e)}")

    # 5. Numbered Pipeline / Process Workflow Section
    render_section_header("🔄 End-to-End Prediction Architecture", tag="METHODOLOGICAL SPECIFICATION")

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        render_workflow_card(
            step="01 // DATA INGESTION",
            title="Copernicus ERA5 & NASA Sensors",
            description="Hourly reanalysis covering temperature, dewpoint, pressure, and wind vectors fused with NASA FIRMS active fire detections and MODIS biophysical fuel layers.",
        )
    with w2:
        render_workflow_card(
            step="02 // FEATURE ENGINEERING",
            title="33 Spatiotemporal Covariates",
            description="Causal feature extraction including 24h/7d/30d cumulative precipitation deficits, vapor pressure deficit (VPD), wind shear vectors, and cyclical seasonality.",
        )
    with w3:
        render_workflow_card(
            step="03 // MODEL TRAINING",
            title="Balanced Random Forest Pipeline",
            description="Scikit-learn Pipeline with ColumnTransformer, StandardScaler, OneHotEncoder, and 200-tree balanced ensemble trained with a strict 24h causal embargo.",
        )
    with w4:
        render_workflow_card(
            step="04 // OPERATIONAL TUNING",
            title="F₂ Decision Threshold (0.2507)",
            description="Operational threshold optimized to prioritize recall (capturing 70.8% of fires), paired with Bayesian prior-shift calibration to California's natural 6.04% fire rate.",
        )

    # 6. Operational Governance Notice
    render_operational_notice()
