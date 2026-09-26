"""
Data and Methodology Page - California Wildfire Prediction System
Mathematical formulations, satellite integration, and validation design.
Matches reference layout:
- Main hero banner with upper-left logo tile + pill.
- Emoji-led section headings.
- Clean multi-sensor remote sensing table.
- Detailed panels on zero-leakage 24h embargo, odds-ratio prior-shift calibration, and scientific limitations.
- Operational notice.
"""
import streamlit as st
from src.theme import (
    render_main_hero,
    render_section_header,
    render_operational_notice,
    render_html,
    COLOR_BORDER,
    COLOR_CANVAS_BG,
    COLOR_DEEP_NAVY,
    COLOR_DARK_NAVY,
    COLOR_ELEVATED,
    COLOR_FIRE_RED,
    COLOR_FIRE_ORANGE,
    COLOR_WARM_ORANGE,
    COLOR_GREEN,
    COLOR_GOLD,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
)


def render_methodology_page():
    # 1. Main Landing Hero Banner
    render_main_hero(
        title="Data Architecture & Methodology",
        subtitle="Multi-sensor satellite fusion, chronological validation embargoes, and prior-shift odds correction for next-24-hour wildfire risk estimation.",
        eyebrow="🛰️ DATA ARCHITECTURE",
    )

    # 2. Multi-Sensor Integration Table
    render_section_header("🛰️ Multi-Sensor Remote Sensing & Reanalysis Pipeline", tag="INPUT SENSORS")

    render_html(f"""
    <div style="background-color: {COLOR_DEEP_NAVY}; border: 1px solid {COLOR_BORDER}; border-radius: 10px; overflow: hidden; margin-bottom: 1.75rem; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);">
        <table style="width: 100%; border-collapse: collapse; font-size: 0.86rem; color: {COLOR_TEXT_PRIMARY};">
            <thead>
                <tr style="background-color: {COLOR_ELEVATED}; border-bottom: 1px solid {COLOR_BORDER}; text-align: left;">
                    <th style="padding: 12px 16px; font-family: ui-monospace, monospace; font-size: 0.74rem; color: {COLOR_TEXT_SECONDARY}; text-transform: uppercase;">Source Sensor / Product</th>
                    <th style="padding: 12px 16px; font-family: ui-monospace, monospace; font-size: 0.74rem; color: {COLOR_TEXT_SECONDARY}; text-transform: uppercase;">Parameters Extracted</th>
                    <th style="padding: 12px 16px; font-family: ui-monospace, monospace; font-size: 0.74rem; color: {COLOR_TEXT_SECONDARY}; text-transform: uppercase;">Cadence</th>
                    <th style="padding: 12px 16px; font-family: ui-monospace, monospace; font-size: 0.74rem; color: {COLOR_TEXT_SECONDARY}; text-transform: uppercase;">Spatial Alignment</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid {COLOR_BORDER};">
                    <td style="padding: 12px 16px; font-weight: 700;">Copernicus ERA5</td>
                    <td style="padding: 12px 16px;">2m Air Temp (t2m), Dewpoint (d2m), 10m Wind (u10, v10), Surface Pressure (sp), Total Precip (tp)</td>
                    <td style="padding: 12px 16px; color: {COLOR_TEXT_SECONDARY};">Hourly (2024–2025)</td>
                    <td style="padding: 12px 16px; font-family: ui-monospace, monospace;">0.25° (~25 km)</td>
                </tr>
                <tr style="border-bottom: 1px solid {COLOR_BORDER};">
                    <td style="padding: 12px 16px; font-weight: 700;">NASA FIRMS (VIIRS / MODIS)</td>
                    <td style="padding: 12px 16px;">Thermal active fire anomalies, Fire Radiative Power (FRP), scan geometry</td>
                    <td style="padding: 12px 16px; color: {COLOR_TEXT_SECONDARY};">Sub-daily Swath Overpasses</td>
                    <td style="padding: 12px 16px; font-family: ui-monospace, monospace;">Spatially Binned to 0.25°</td>
                </tr>
                <tr style="border-bottom: 1px solid {COLOR_BORDER};">
                    <td style="padding: 12px 16px; font-weight: 700;">MODIS MCD12Q1</td>
                    <td style="padding: 12px 16px;">13 IGBP Land Cover Classes (Forest, Shrubland, Savanna, Grassland, Barren, Cropland)</td>
                    <td style="padding: 12px 16px; color: {COLOR_TEXT_SECONDARY};">Annual Composite</td>
                    <td style="padding: 12px 16px; font-family: ui-monospace, monospace;">Resampled to 0.25°</td>
                </tr>
                <tr style="border-bottom: 1px solid {COLOR_BORDER};">
                    <td style="padding: 12px 16px; font-weight: 700;">MODIS MOD13Q1</td>
                    <td style="padding: 12px 16px;">NDVI & EVI Vegetation Greenness, Canopy Moisture & Foliage Dynamics</td>
                    <td style="padding: 12px 16px; color: {COLOR_TEXT_SECONDARY};">16-day Composite</td>
                    <td style="padding: 12px 16px; font-family: ui-monospace, monospace;">Forward-filled to Hourly</td>
                </tr>
                <tr>
                    <td style="padding: 12px 16px; font-weight: 700;">NASA SRTM</td>
                    <td style="padding: 12px 16px;">Digital Elevation Model (meters above sea level), Slope & Aspect proxies</td>
                    <td style="padding: 12px 16px; color: {COLOR_TEXT_SECONDARY};">Static 1 Arc-Second</td>
                    <td style="padding: 12px 16px; font-family: ui-monospace, monospace;">Averaged to 0.25°</td>
                </tr>
            </tbody>
        </table>
    </div>
    """)

    # 3. Two-Column Analytical Documentation
    col_split, col_prior = st.columns([1, 1], gap="large")

    with col_split:
        render_section_header("⏱️ Chronological Split & 24h Embargo", tag="TEMPORAL DISCIPLINE")

        render_html(f"""
        <div class="app-intel-card" style="height: auto; margin-bottom: 1rem;">
            <h4 style="margin: 0 0 0.5rem 0; font-size: 1.05rem; color: {COLOR_TEXT_PRIMARY};">Zero Target Leakage Architecture</h4>
            <p style="font-size: 0.84rem; color: {COLOR_TEXT_SECONDARY}; line-height: 1.6; margin-bottom: 0.75rem;">
                Standard random cross-validation yields catastrophic optimistic bias when applied to geospatial time-series data due to strong spatial and temporal autocorrelation.
            </p>
            <div style="background-color: {COLOR_DARK_NAVY}; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 0.75rem; font-size: 0.8rem; line-height: 1.5; color: {COLOR_TEXT_SECONDARY};">
                <b style="color: {COLOR_WARM_ORANGE};">Chronological Partitioning:</b><br/>
                • <b>Train:</b> Historical 2024 hourly observations.<br/>
                • <b>Validation:</b> Late 2024 partition used solely for threshold calibration (F₂).<br/>
                • <b>Test:</b> Held-out 2025 sequence (unseen during all training & tuning).
            </div>
            <div style="font-size: 0.82rem; color: {COLOR_TEXT_SECONDARY}; line-height: 1.5;">
                A strict <b>24-hour temporal embargo</b> isolates training horizons from test observation windows, ensuring no antecedent rolling window leaks forward across the partition boundaries.
            </div>
        </div>
        """)

    with col_prior:
        render_section_header("⚖️ Odds-Ratio Prior-Shift Calibration", tag="BAYESIAN ADJUSTMENT")

        render_html(f"""
        <div class="app-intel-card" style="height: auto; margin-bottom: 1rem;">
            <h4 style="margin: 0 0 0.5rem 0; font-size: 1.05rem; color: {COLOR_TEXT_PRIMARY};">Prior Probability Normalization</h4>
            <p style="font-size: 0.84rem; color: {COLOR_TEXT_SECONDARY}; line-height: 1.6; margin-bottom: 0.75rem;">
                During model training, downsampling non-fire observations artificially inflates training prevalence to ~30%. In production, California experiences a natural 24h fire prevalence of approximately <b>6.04%</b> across terrestrial cells.
            </p>
            <div style="background-color: {COLOR_DARK_NAVY}; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 0.75rem; font-family: ui-monospace, monospace; font-size: 0.78rem; line-height: 1.6; color: {COLOR_WARM_ORANGE};">
                odds_raw = p_raw / (1 - p_raw)<br/>
                prior_ratio = (p_natural / (1 - p_natural)) / (p_train / (1 - p_train))<br/>
                odds_cal = odds_raw * prior_ratio<br/>
                p_calibrated = odds_cal / (1 + odds_cal)
            </div>
            <div style="font-size: 0.82rem; color: {COLOR_TEXT_SECONDARY}; line-height: 1.5;">
                This Bayes-optimal adjustment preserves the true ranking of the Random Forest while producing honest posterior probabilities aligned with real California fire occurrence rates.
            </div>
        </div>
        """)

    # 4. Scientific Limitations & Boundaries
    render_section_header("⚠️ Scientific Limitations & Environmental Assumptions", tag="METHODOLOGICAL SCOPE")

    l1, l2, l3 = st.columns(3)
    with l1:
        render_html(f"""
        <div class="app-intel-card">
            <div style="font-weight: 700; color: {COLOR_GOLD}; font-size: 0.95rem; margin-bottom: 0.35rem;">Grid Resolution (0.25°)</div>
            <div style="font-size: 0.82rem; color: {COLOR_TEXT_SECONDARY}; line-height: 1.5;">
                Each grid cell spans approximately 25 km × 25 km (~625 km²). Microclimatic wind channeling through narrow canyons cannot be resolved at this synoptic scale.
            </div>
        </div>
        """)
    with l2:
        render_html(f"""
        <div class="app-intel-card">
            <div style="font-weight: 700; color: {COLOR_GOLD}; font-size: 0.95rem; margin-bottom: 0.35rem;">Satellite Cloud Obscuration</div>
            <div style="font-size: 0.82rem; color: {COLOR_TEXT_SECONDARY}; line-height: 1.5;">
                NASA FIRMS sensors require unobstructed lines of sight. Dense overcast cloud decks or marine fog may delay spaceborne thermal detection by several hours.
            </div>
        </div>
        """)
    with l3:
        render_html(f"""
        <div class="app-intel-card">
            <div style="font-weight: 700; color: {COLOR_GOLD}; font-size: 0.95rem; margin-bottom: 0.35rem;">Human Ignition Exogeneity</div>
            <div style="font-size: 0.82rem; color: {COLOR_TEXT_SECONDARY}; line-height: 1.5;">
                The model quantifies environmental receptivity and fuel flammability. Accidental anthropogenic ignitions (power lines, campfires) remain stochastic exogenous events.
            </div>
        </div>
        """)

    render_operational_notice()
