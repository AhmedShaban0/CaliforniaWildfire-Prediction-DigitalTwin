"""
Overview Page - California Wildfire Prediction System
Executive landing console:
1. Hero banner with upper-left light logo tile + pill.
2. Two-column section:
   - Left: Project Objective & Methodology (clean formatted rendering without raw HTML leaks).
   - Right: Statewide Spatial Situational View (historical weather snapshot & 3D preview).
3. Compact Real-Time API Forecast & Prediction section:
   - Directly executes the real Open-Meteo forecast prediction workflow from the Overview page.
   - Reuses shared inference pipeline, displaying forecast provenance, weather metrics, and single-cell 3D spotlight.
4. Numbered pipeline/process section using workflow cards.
5. Operational boundary & scientific governance disclosure.
"""
from datetime import datetime, date, timedelta, timezone
import streamlit as st
import pandas as pd
import numpy as np
from src.theme import (
    render_main_hero,
    render_section_header,
    render_workflow_card,
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
)
from src.data_loader import (
    open_dataset,
    CURATED_SCENARIOS,
    load_spatial_slice_with_predictions,
    fetch_open_meteo_forecast,
    get_cell_static_and_vegetation,
    CALIFORNIA_PRESET_LOCATIONS,
)
from src.twin_3d import render_pydeck_3d_twin, render_single_cell_3d_twin


def render_overview_page():
    # 1. Main Landing Hero Banner
    render_main_hero(
        title="California Wildfire Prediction",
        subtitle="Estimating next-24-hour satellite fire detection probability across California's 0.25° grid (~25 km × 25 km) using Copernicus ERA5 reanalysis, spaceborne Earth observation, and real-time weather forecasting.",
        eyebrow="🔥 CALIFORNIA WILDFIRE INTELLIGENCE",
    )

    schema, schema_err = load_feature_schema()
    pipeline, pipeline_err = load_pipeline()
    threshold = float(schema.get("decision_threshold", 0.2507)) if schema else 0.2507

    # 2. Section: Statewide Wildfire Situational Intelligence & Project Scope
    render_section_header("🔥 Statewide Wildfire Situational Intelligence", tag="SPATIAL RISK & METHODOLOGY")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # Rendered using render_html (dedented) to prevent CommonMark 4-space code block leaks
        html_horizon = f"""
<div class="app-intel-card" style="height: 100%; justify-content: flex-start;">
    <div style="font-family: ui-monospace, monospace; font-size: 0.74rem; color: {COLOR_FIRE_ORANGE}; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.45rem;">
        🎯 PROJECT OBJECTIVE & METHODOLOGY
    </div>
    <h3 style="font-size: 1.25rem; font-weight: 700; color: {COLOR_TEXT_PRIMARY}; margin: 0 0 0.75rem 0;">
        24-Hour Advance Wildfire Forecast Horizon
    </h3>
    <p style="font-size: 0.88rem; color: {COLOR_TEXT_SECONDARY}; line-height: 1.6; margin-bottom: 0.85rem;">
        The California Fire Atlas models the complex relationship between atmospheric thermodynamics, fuel moisture deficit, and spaceborne active fire ignitions across the State of California.
    </p>
    <div style="border-top: 1px solid {COLOR_BORDER}; padding-top: 0.85rem; margin-bottom: 0.85rem;">
        <div style="font-weight: 600; font-size: 0.88rem; color: {COLOR_TEXT_PRIMARY}; margin-bottom: 0.25rem;">
            ⏱️ Zero-Leakage 24-Hour Causal Embargo
        </div>
        <div style="font-size: 0.82rem; color: {COLOR_TEXT_SECONDARY}; line-height: 1.5;">
            To prevent target leakage, all environmental covariates (2m temperature, dewpoint, wind vectors, vapor pressure deficit, surface precipitation) observe a strict causal embargo. The model evaluates whether a thermal detection occurs in hour <code>t+1</code> through <code>t+24</code> strictly using information available at hour <code>t</code>.
        </div>
    </div>
    <div style="border-top: 1px solid {COLOR_BORDER}; padding-top: 0.85rem; margin-bottom: 0.85rem;">
        <div style="font-weight: 600; font-size: 0.88rem; color: {COLOR_TEXT_PRIMARY}; margin-bottom: 0.25rem;">
            🛰️ Multi-Sensor Spatiotemporal Ingestion
        </div>
        <div style="font-size: 0.82rem; color: {COLOR_TEXT_SECONDARY}; line-height: 1.5;">
            Atmospheric state is continuously reconstructed from Copernicus ERA5 hourly reanalysis at 0.25° grid resolution. Vegetation and surface fuels are tracked via MODIS/VIIRS Normalized Difference Vegetation Index (NDVI), Enhanced Vegetation Index (EVI), and Land Cover classifications.
        </div>
    </div>
    <div style="border-top: 1px solid {COLOR_BORDER}; padding-top: 0.85rem;">
        <div style="font-weight: 600; font-size: 0.88rem; color: {COLOR_TEXT_PRIMARY}; margin-bottom: 0.25rem;">
            ⚖️ Imbalanced Learning & F₂ Calibration
        </div>
        <div style="font-size: 0.82rem; color: {COLOR_TEXT_SECONDARY}; line-height: 1.5;">
            With a natural wildfire prevalence of only 6.04% in California, standard 0.50 probability cutoffs fail. The operational cutoff (<code>0.2507</code>) is tuned against the F₂-score, weighting recall twice as heavily as precision to minimize dangerous missed detections in critical terrain.
        </div>
    </div>
</div>
"""
        render_html(html_horizon)

    with col_right:
        # Scenario Selector & Quick Actions
        col_scen, col_act1, col_act2 = st.columns([2.5, 1.3, 1.3])
        with col_scen:
            scenario_name = st.selectbox(
                "Historical Weather Snapshot:",
                options=list(CURATED_SCENARIOS.keys()),
                index=0,
                label_visibility="collapsed",
            )
        with col_act1:
            if st.button("🌐 3D TWIN", use_container_width=True):
                st.session_state["nav_target"] = "3D Digital Twin"
                st.rerun()
        with col_act2:
            if st.button("🔥 PREDICT", use_container_width=True):
                st.session_state["nav_target"] = "Wildfire Prediction"
                st.rerun()

        chosen_scenario = CURATED_SCENARIOS[scenario_name]
        ts_str = chosen_scenario["timestamp"]

        with st.spinner("Extracting spatial slice and evaluating 3D model risk..."):
            slice_df = load_spatial_slice_with_predictions(ts_str, ts_str, land_only=True)

        if slice_df is not None and not slice_df.empty:
            render_pydeck_3d_twin(
                slice_df,
                color_by="Model Predicted Risk",
                elevation_scale=2.0,
                pitch=36.0,
                bearing=-14.0,
                height=320,
            )

            alerts_count = int(slice_df["is_alert"].sum())
            fires_count = int((slice_df["fire_now"] > 0).sum())
            max_risk = slice_df["model_risk_score"].max()

            # Compact Regional Dossier Strip below map (rendered safely)
            html_dossier = f"""
<div style="background-color: {COLOR_DEEP_NAVY}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 0.85rem 1rem; margin-top: 0.75rem;">
    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.35rem;">
        <div style="font-weight: 700; font-size: 0.95rem; color: {COLOR_TEXT_PRIMARY};">{chosen_scenario['region_name']}</div>
        <div style="font-family: ui-monospace, monospace; font-size: 0.72rem; color: {COLOR_WARM_ORANGE};">{ts_str} UTC</div>
    </div>
    <div style="font-size: 0.8rem; color: {COLOR_TEXT_SECONDARY}; margin-bottom: 0.65rem;">{chosen_scenario['description']}</div>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; text-align: center; border-top: 1px solid {COLOR_BORDER}; padding-top: 0.5rem;">
        <div>
            <div style="font-size: 0.68rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace;">LAND CELLS</div>
            <div style="font-weight: 700; color: {COLOR_TEXT_PRIMARY}; font-size: 0.95rem;">{len(slice_df):,}</div>
        </div>
        <div>
            <div style="font-size: 0.68rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace;">ALERTS (≥{threshold:.2f})</div>
            <div style="font-weight: 700; color: {COLOR_FIRE_ORANGE}; font-size: 0.95rem;">{alerts_count} ({alerts_count / len(slice_df):.1%})</div>
        </div>
        <div>
            <div style="font-size: 0.68rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace;">NASA DETECTIONS</div>
            <div style="font-weight: 700; color: {COLOR_CRIMSON if fires_count > 0 else COLOR_GREEN}; font-size: 0.95rem;">{fires_count} active</div>
        </div>
        <div>
            <div style="font-size: 0.68rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace;">PEAK GRID RISK</div>
            <div style="font-weight: 700; color: {COLOR_FIRE_RED}; font-size: 0.95rem;">{max_risk:.1%}</div>
        </div>
    </div>
</div>
"""
            render_html(html_dossier)
        else:
            st.info("Dataset `master_dataset_2024_2025.nc` not loaded. Spatial snapshot is unavailable.")

    # 3. Compact Real-Time API Forecast & Predict Section on Overview Page
    render_section_header("🌐 Live Forecast & Risk Inference", tag="REAL-TIME API DISPATCH")

    render_html(f"""
    <div style="font-size: 0.88rem; color: {COLOR_TEXT_SECONDARY}; margin-bottom: 1rem;">
        Retrieve live hourly meteorology from the <b>Open-Meteo Global Weather Model</b> for any California region, combine with spaceborne Earth observation layers, and evaluate the trained Random Forest model directly from this executive dashboard.
    </div>
    """)

    col_ov_fc_inputs, col_ov_fc_results = st.columns([1.05, 0.95], gap="large")

    with col_ov_fc_inputs:
        ov_preset_key = st.selectbox(
            "California Region / Hotspot:",
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

        btn_ov_predict = st.button("🌐 GET FORECAST & PREDICT", type="primary", use_container_width=True, key="ov_btn_predict")

    with col_ov_fc_results:
        if btn_ov_predict:
            st.session_state["has_run_ov_forecast"] = True
            st.session_state["ov_lat"] = ov_lat
            st.session_state["ov_lon"] = ov_lon
            st.session_state["ov_ts"] = ov_ts_str

        run_ov = st.session_state.get("has_run_ov_forecast", False)

        if not run_ov:
            html_await = f"""
<div style="background-color: {COLOR_DEEP_NAVY}; border: 1px dashed {COLOR_BORDER}; border-radius: 8px; padding: 2rem 1.5rem; text-align: center; color: {COLOR_TEXT_SECONDARY};">
    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🛰️</div>
    <div style="font-weight: 700; color: {COLOR_TEXT_PRIMARY}; font-size: 1.1rem; margin-bottom: 0.35rem;">
        Awaiting Real-Time Forecast Request
    </div>
    <div style="font-size: 0.84rem; max-width: 400px; margin: 0 auto; line-height: 1.5;">
        Select your location and target forecast date on the left, then click <b>GET FORECAST & PREDICT</b> to evaluate next-24h wildfire detection probability.
    </div>
</div>
"""
            render_html(html_await)
        else:
            exec_lat = st.session_state.get("ov_lat", ov_lat)
            exec_lon = st.session_state.get("ov_lon", ov_lon)
            exec_ts = st.session_state.get("ov_ts", ov_ts_str)

            with st.spinner("Connecting to Open-Meteo & computing model inference..."):
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
                        alert_label_ov = "ALERT ELEVATED" if is_alert_ov else "BASELINE NORMAL"
                        alert_bg_ov = COLOR_FIRE_RED if is_alert_ov else COLOR_GREEN

                        veg_src_ov = cell_meta.get("vegetation_source", "MODIS MOD13Q1 (16-day Earth observation composite)")
                        veg_dt_ov = cell_meta.get("vegetation_obs_date", cell_meta.get("latest_observation", "Latest Available"))

                        # Forecast Result Card (rendered cleanly)
                        html_ov_res = f"""
<div class="app-prediction-panel" style="border: 2px solid {tier_color_ov};">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.85rem;">
        <span style="font-family: ui-monospace, monospace; font-size: 0.74rem; color: {COLOR_TEXT_SECONDARY}; text-transform: uppercase;">
            COORDINATES: {exec_lat:.2f}°N, {abs(exec_lon):.2f}°W
        </span>
        <span style="background-color: {alert_bg_ov}; color: #FFFFFF; font-family: ui-monospace, monospace; font-size: 0.74rem; font-weight: 700; padding: 3px 10px; border-radius: 4px; letter-spacing: 0.06em;">
            {alert_label_ov}
        </span>
    </div>

    <div style="margin-bottom: 0.85rem;">
        <div style="font-size: 0.72rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace; text-transform: uppercase;">
            RAW MODEL RISK PROBABILITY
        </div>
        <div style="font-size: 3rem; font-weight: 700; color: {tier_color_ov}; line-height: 1.1; margin: 0.2rem 0;">
            {score_ov:.1%}
        </div>
        <div style="font-size: 0.82rem; color: {COLOR_TEXT_SECONDARY};">
            Operational Threshold: <code>{threshold:.4f}</code> &nbsp;•&nbsp; Status: <b>{'Exceeded' if is_alert_ov else 'Below Threshold'}</b>
        </div>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.85rem; border-top: 1px solid {COLOR_BORDER}; padding-top: 0.85rem; margin-bottom: 0.85rem;">
        <div>
            <div style="font-size: 0.7rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace;">PRIOR-CORRECTED PROBABILITY</div>
            <div style="font-size: 1.45rem; font-weight: 700; color: {COLOR_TEXT_PRIMARY};">{cal_prob_ov:.1%}</div>
            <div style="font-size: 0.72rem; color: {COLOR_TEXT_SECONDARY};">Natural fire rate: 6.04%</div>
        </div>
        <div>
            <div style="font-size: 0.7rem; color: {COLOR_TEXT_SECONDARY}; font-family: ui-monospace, monospace;">RISK CLASSIFICATION</div>
            <div style="font-size: 1.45rem; font-weight: 700; color: {tier_color_ov};">{tier_ov.upper()}</div>
            <div style="font-size: 0.72rem; color: {COLOR_TEXT_SECONDARY};">Random Forest (200 trees)</div>
        </div>
    </div>

    <div style="background-color: {COLOR_DARK_NAVY}; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 0.75rem 1rem; font-size: 0.8rem; line-height: 1.5; color: {COLOR_TEXT_SECONDARY};">
        <b style="color: {COLOR_TEXT_PRIMARY};">Forecast Provenance:</b>
        Provider: <b>{fc_data['provider']}</b> &nbsp;|&nbsp;
        Retrieved: <code>{fc_data['retrieval_time']}</code><br/>
        Valid Target: <code>{fc_data['forecast_time'].strftime('%Y-%m-%d %H:%M UTC')}</code><br/>
        Vegetation Observation: <code>{veg_src_ov} ({veg_dt_ov})</code>
    </div>
</div>
"""
                        render_html(html_ov_res)

                        # Compact weather metrics
                        w1, w2, w3, w4 = st.columns(4)
                        with w1:
                            st.metric("Air Temp", f"{fc_data['t2m_c']:.1f} °C")
                        with w2:
                            st.metric("Dewpoint", f"{fc_data['d2m_c']:.1f} °C")
                        with w3:
                            st.metric("Wind Speed", f"{fc_data['wind_speed']:.1f} m/s")
                        with w4:
                            st.metric("VPD (Dryness)", f"{fc_data['vpd_kpa']:.2f} kPa")

                        # Single-cell 3D spotlight
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
                            mode_label="Overview Live Forecast",
                            height=360,
                        )
                except ValueError as ve:
                    st.error(f"Weather / Input Verification Error: {ve}")
                    st.info("Live prediction was blocked because required weather data could not be verified from the API. The system refuses to use synthetic or placeholder fallback data.")
                except Exception as e:
                    logger.exception(f"Unexpected error in overview live forecast: {e}")
                    st.error(f"Inference Blocked: {str(e)}")

    # 4. Numbered Pipeline / Process Workflow Section
    render_section_header("🔄 Pipeline & Architecture Workflow", tag="END-TO-END SPECIFICATION")

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        render_workflow_card(
            step="01 // DATA INGESTION",
            title="ERA5 & Spaceborne Fusion",
            description="17,544 hourly ERA5 timesteps and spaceborne thermal active fire observations extracted across 1,677 California cells.",
        )
    with w2:
        render_workflow_card(
            step="02 // FEATURE ENGINEERING",
            title="Spatiotemporal Lagging",
            description="33 causal features engineered including 24h/7d/30d antecedent precipitation deficits, VPD, and wind shear vectors.",
        )
    with w3:
        render_workflow_card(
            step="03 // CLASSIFIER TRAINING",
            title="RandomForest Pipeline",
            description="Scikit-learn Pipeline with ColumnTransformer, StandardScaler, OneHotEncoder, and 200-tree balanced RandomForest.",
        )
    with w4:
        render_workflow_card(
            step="04 // RISK STRATIFICATION",
            title="Decision Rule & 3D Twin",
            description="F₂-score threshold optimization (0.2507) and real-time inference dispatch integrated with 3D deck.gl topographic rendering.",
        )

    # 5. Operational Scope & Boundary Disclosure
    render_operational_notice()
