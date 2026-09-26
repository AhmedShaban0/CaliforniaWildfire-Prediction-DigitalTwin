"""
Model loader, schema validator, and inference pipeline.
Strictly implements the exact feature schema, winsorization, and probability correction.
"""
import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# Land cover mapping for human-readable labels
LAND_COVER_LABELS = {
    1.0: "Evergreen Needleleaf Forest",
    2.0: "Evergreen Broadleaf Forest",
    5.0: "Mixed Forest",
    6.0: "Closed Shrublands",
    7.0: "Open Shrublands",
    8.0: "Woody Savannas",
    9.0: "Savannas",
    10.0: "Grasslands",
    11.0: "Permanent Wetlands",
    12.0: "Croplands",
    13.0: "Urban & Built-up Lands",
    16.0: "Barren",
    17.0: "Water Bodies",
}


def _get_file_mtime(path: str) -> float:
    """Returns file mtime for cache invalidation, or 0.0 if not found."""
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@st.cache_resource(show_spinner=False)
def _load_schema_cached(schema_path: str, mtime: float):
    """Internal cached loader parameterized by file modification timestamp."""
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        return schema, None
    except Exception as e:
        return None, f"Failed to parse schema at '{schema_path}': {str(e)}"


def load_feature_schema(schema_path: str = "feature_schema.json"):
    """Loads and validates the feature schema JSON artifact with mtime-aware cache invalidation."""
    if not os.path.exists(schema_path):
        return None, f"Schema file not found at '{schema_path}'."
    mtime = _get_file_mtime(schema_path)
    return _load_schema_cached(schema_path, mtime)


@st.cache_resource(show_spinner=False)
def _load_pipeline_cached(pipeline_path: str, mtime: float):
    """Internal cached loader parameterized by file modification timestamp."""
    try:
        pipeline = joblib.load(pipeline_path)
        return pipeline, None
    except Exception as e:
        return None, f"Failed to load pipeline at '{pipeline_path}': {str(e)}"


def load_pipeline(pipeline_path: str = "wildfire_model_pipeline.joblib"):
    """Loads the fitted scikit-learn Pipeline artifact with mtime-aware cache invalidation."""
    if not os.path.exists(pipeline_path):
        return None, f"Model pipeline artifact not found at '{pipeline_path}'."
    mtime = _get_file_mtime(pipeline_path)
    return _load_pipeline_cached(pipeline_path, mtime)


def validate_model_schema_compatibility(pipeline, schema: dict) -> tuple[bool, str]:
    """
    Validates that the loaded pipeline and schema are mutually compatible:
    - Pipeline contains 'preprocessor' and 'model' steps.
    - Preprocessor transformers align with schema features.
    - Model provides binary classification predict_proba.
    """
    if pipeline is None:
        return False, "Model pipeline is not loaded (None)."
    if not isinstance(schema, dict):
        return False, "Feature schema is not a valid dictionary."

    features = schema.get("features", [])
    if not features:
        return False, "Schema does not define an expected 'features' list."

    if not hasattr(pipeline, "named_steps"):
        return False, "Loaded pipeline is missing 'named_steps' attribute."

    preprocessor = pipeline.named_steps.get("preprocessor")
    model = pipeline.named_steps.get("model")
    if preprocessor is None or model is None:
        return False, "Pipeline must contain 'preprocessor' and 'model' steps."

    if not hasattr(model, "classes_") or len(model.classes_) != 2:
        return False, f"Model classes must be binary [0, 1], got {getattr(model, 'classes_', None)}."

    return True, f"Pipeline and schema contract verified: {len(features)} features, model={type(model).__name__}."


def calc_es(t_celsius: float) -> float:
    """Calculates saturation vapor pressure in kPa using Tetens-like formula from notebook."""
    return 0.6108 * np.exp(17.27 * t_celsius / (t_celsius + 237.3))


def calc_vpd(t_celsius: float, td_celsius: float) -> float:
    """Calculates Vapor Pressure Deficit (VPD) in kPa, clipped at 0."""
    es_t = calc_es(t_celsius)
    es_td = calc_es(td_celsius)
    return float(np.clip(es_t - es_td, 0.0, None))


def calc_rh(t_celsius: float, td_celsius: float) -> float:
    """Calculates relative humidity in %, clipped to [0, 100]."""
    es_t = calc_es(t_celsius)
    es_td = calc_es(td_celsius)
    return float(np.clip(100.0 * (es_td / np.maximum(es_t, 1e-6)), 0.0, 100.0))


def apply_prior_shift(
    score: float,
    eff_prior: float = 0.2715625,
    natural_prior: float = 0.06044186971301727,
) -> float:
    """
    Re-scales odds from the effective training fire share back to the real-world fire rate.
    Prior-shift odds correction formula specified in the notebook/schema:
    odds = p/(1-p) * (natural_prior/(1-natural_prior)) / (eff_prior/(1-eff_prior))
    p_corrected = odds / (1 + odds)
    """
    p = np.clip(np.asarray(score, dtype="float64"), 1e-9, 1.0 - 1e-9)
    ratio = (natural_prior / (1.0 - natural_prior)) / (eff_prior / (1.0 - eff_prior))
    odds_eff = p / (1.0 - p)
    odds_true = odds_eff * ratio
    return float(odds_true / (1.0 + odds_true))


def categorize_risk(score: float, threshold: float) -> str:
    """
    Categorizes the model estimate into 4 standardized operational risk levels:
    Low, Moderate, High, Extreme.
    Threshold separates Moderate from High (where operational alert fires).
    """
    if score >= 0.45:
        return "Extreme"
    elif score >= threshold:
        return "High"
    elif score >= 0.15:
        return "Moderate"
    else:
        return "Low"


def validate_raw_inputs(raw_inputs: dict) -> list[str]:
    """Validates presence and finite numerical value for essential required inputs."""
    required_keys = ["u10", "v10", "t2m", "d2m", "sp", "tp", "land_cover", "latitude", "longitude"]
    errors = []
    for k in required_keys:
        if k not in raw_inputs:
            errors.append(f"Missing required parameter '{k}'")
        else:
            try:
                val = float(raw_inputs[k])
                if not np.isfinite(val):
                    errors.append(f"Parameter '{k}' must be finite, got {val}")
            except (ValueError, TypeError):
                errors.append(f"Parameter '{k}' must be numeric, got {raw_inputs[k]}")
    return errors


def prepare_feature_row(raw_inputs: dict, schema: dict) -> pd.DataFrame:
    """
    Builds the exact input row required by the schema and scikit-learn pipeline.
    Validates required inputs, applies winsorization, trigonometric and meteorological
    transformations, and returns a single-row DataFrame with the exact column order.
    """
    input_errors = validate_raw_inputs(raw_inputs)
    if input_errors:
        raise ValueError("Invalid model inputs: " + "; ".join(input_errors))

    bounds = schema.get("winsorization_bounds", {})

    # Winsorize primary raw variables if bounds are defined
    u10 = float(raw_inputs.get("u10", 0.0))
    v10 = float(raw_inputs.get("v10", 0.0))
    t2m_k = float(raw_inputs.get("t2m", 295.0))
    d2m_k = float(raw_inputs.get("d2m", 285.0))
    sp = float(raw_inputs.get("sp", 95000.0))
    tp = float(raw_inputs.get("tp", 0.0))

    if "u10" in bounds:
        u10 = float(np.clip(u10, bounds["u10"][0], bounds["u10"][1]))
    if "v10" in bounds:
        v10 = float(np.clip(v10, bounds["v10"][0], bounds["v10"][1]))
    if "t2m" in bounds:
        t2m_k = float(np.clip(t2m_k, bounds["t2m"][0], bounds["t2m"][1]))
    if "d2m" in bounds:
        d2m_k = float(np.clip(d2m_k, bounds["d2m"][0], bounds["d2m"][1]))
    if "sp" in bounds:
        sp = float(np.clip(sp, bounds["sp"][0], bounds["sp"][1]))
    if "tp" in bounds:
        tp = float(np.clip(tp, bounds["tp"][0], bounds["tp"][1]))

    # Derived temperatures in Celsius
    t2m_c = t2m_k - 273.15
    d2m_c = d2m_k - 273.15
    wind_speed = float(np.sqrt(u10**2 + v10**2))
    temp_dewpoint_spread = t2m_c - d2m_c
    rel_humidity = calc_rh(t2m_c, d2m_c)
    vpd = calc_vpd(t2m_c, d2m_c)

    # Cyclical temporal features
    month = int(raw_inputs.get("month", 7))
    hour = int(raw_inputs.get("hour", 20))
    day_of_year = int(raw_inputs.get("day_of_year", 200))

    month_sin = float(np.sin(2 * np.pi * month / 12))
    month_cos = float(np.cos(2 * np.pi * month / 12))
    hour_sin = float(np.sin(2 * np.pi * hour / 24))
    hour_cos = float(np.cos(2 * np.pi * hour / 24))
    doy_sin = float(np.sin(2 * np.pi * day_of_year / 365.25))
    doy_cos = float(np.cos(2 * np.pi * day_of_year / 365.25))

    # Cyclical wind direction (meteorological convention: direction wind is FROM)
    wind_dir_rad = np.arctan2(-u10, -v10)
    wind_dir_sin = float(np.sin(wind_dir_rad))
    wind_dir_cos = float(np.cos(wind_dir_rad))

    # Static / Remote sensing fields
    land_cover = float(raw_inputs.get("land_cover", 10.0))
    elevation = float(raw_inputs.get("elevation", 500.0))
    ndvi = float(raw_inputs.get("ndvi", 4500.0))
    evi = float(raw_inputs.get("evi", 3200.0))
    ndvi_scaled = ndvi * 0.0001
    evi_scaled = evi * 0.0001
    pixel_reliability = float(raw_inputs.get("pixel_reliability", 0.0))
    vi_quality = float(raw_inputs.get("vi_quality", 2116.0))

    # Antecedent rolling features
    tp_sum_24h = float(raw_inputs.get("tp_sum_24h", 0.0))
    tp_sum_7d = float(raw_inputs.get("tp_sum_7d", 0.0))
    tp_sum_30d = float(raw_inputs.get("tp_sum_30d", 0.0))
    vpd_mean_24h = float(raw_inputs.get("vpd_mean_24h", vpd))
    vpd_mean_7d = float(raw_inputs.get("vpd_mean_7d", vpd))
    t2m_mean_7d = float(raw_inputs.get("t2m_mean_7d", t2m_c))
    wind_mean_24h = float(raw_inputs.get("wind_mean_24h", wind_speed))

    latitude = float(raw_inputs.get("latitude", 37.0))
    longitude = float(raw_inputs.get("longitude", -120.0))

    row_data = {
        "u10": u10,
        "v10": v10,
        "t2m_celsius": t2m_c,
        "d2m_celsius": d2m_c,
        "sp": sp,
        "tp": tp,
        "land_cover": land_cover,
        "elevation": elevation,
        "ndvi_scaled": ndvi_scaled,
        "evi_scaled": evi_scaled,
        "pixel_reliability": pixel_reliability,
        "vi_quality": vi_quality,
        "wind_speed": wind_speed,
        "temp_dewpoint_spread": temp_dewpoint_spread,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "rel_humidity": rel_humidity,
        "vpd": vpd,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
        "wind_dir_sin": wind_dir_sin,
        "wind_dir_cos": wind_dir_cos,
        "tp_sum_24h": tp_sum_24h,
        "tp_sum_7d": tp_sum_7d,
        "tp_sum_30d": tp_sum_30d,
        "vpd_mean_24h": vpd_mean_24h,
        "vpd_mean_7d": vpd_mean_7d,
        "t2m_mean_7d": t2m_mean_7d,
        "wind_mean_24h": wind_mean_24h,
        "latitude": latitude,
        "longitude": longitude,
    }

    # Ensure all features in schema are present in exact schema order
    ordered_cols = schema["features"]
    df = pd.DataFrame([row_data])[ordered_cols]
    return df


def predict_wildfire_risk(pipeline, schema: dict, df_row: pd.DataFrame) -> dict:
    """
    Executes inference via the scikit-learn pipeline and returns:
    raw probability, calibrated natural probability, alert state, and risk category.
    """
    threshold = float(schema.get("decision_threshold", 0.2507))
    corr_cfg = schema.get("probability_correction", {})
    eff_prior = float(corr_cfg.get("training_effective_prior", 0.2715625))
    nat_prior = float(corr_cfg.get("natural_fire_rate", 0.06044187))

    # Raw model probability from Random Forest classifier
    raw_score = float(pipeline.predict_proba(df_row)[0, 1])
    calibrated_prob = apply_prior_shift(raw_score, eff_prior, nat_prior)
    is_alert = bool(raw_score >= threshold)
    risk_level = categorize_risk(raw_score, threshold)

    return {
        "raw_score": raw_score,
        "calibrated_prob": calibrated_prob,
        "decision_threshold": threshold,
        "is_alert": is_alert,
        "risk_level": risk_level,
        "features": df_row.to_dict(orient="records")[0],
    }
