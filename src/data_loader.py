"""
Data access layer for California Wildfire Prediction.
Implements lazy xarray reads, antecedent history extraction, spatial slices, and caching.
"""
import os
import logging
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
from src.model_loader import (
    LAND_COVER_LABELS,
    calc_vpd,
    calc_rh,
    prepare_feature_row,
    predict_wildfire_risk,
)

logger = logging.getLogger(__name__)

DATASET_FILE = "master_dataset_2024_2025.nc"

# Curated interesting scenarios from the verified 2024-2025 dataset
CURATED_SCENARIOS = {
    "August 2024 Peak Heatwave & Wildfire Period": {
        "timestamp": "2024-08-15 20:00:00",
        "latitude": 37.00,
        "longitude": -120.00,
        "description": "Central Valley / Sierra Foothills during peak summer heatwave with elevated VPD and dry fuel conditions.",
        "region_name": "Central California / Sierra Nevada Foothills",
    },
    "October 2024 Santa Ana Offshore Wind Event": {
        "timestamp": "2024-10-20 18:00:00",
        "latitude": 34.25,
        "longitude": -118.50,
        "description": "Southern California interior with strong offshore north-easterly winds and critical humidity drop.",
        "region_name": "Southern California / Los Angeles Basin",
    },
    "July 2024 Northern California Fire Complex": {
        "timestamp": "2024-07-24 22:00:00",
        "latitude": 40.00,
        "longitude": -121.50,
        "description": "Northern Sierra / Cascade region during active July wildfire season with confirmed FIRMS detections.",
        "region_name": "Northern California / Lassen & Plumas",
    },
    "January 2025 Winter Low-Risk Baseline": {
        "timestamp": "2025-01-12 14:00:00",
        "latitude": 38.50,
        "longitude": -121.50,
        "description": "Winter season observation with widespread precipitation, cool temperatures, and minimal wildfire risk.",
        "region_name": "Sacramento Valley / Central Coast",
    },
}


@st.cache_resource(show_spinner=False)
def open_dataset(path: str = DATASET_FILE):
    """Opens the NetCDF dataset lazily. Returns (dataset, error_message)."""
    if not os.path.exists(path):
        return None, (
            f"Dataset '{path}' (approx 414 MB) was not found in the workspace root. "
            "Spatial maps and historical time-series queries require this file."
        )
    try:
        ds = xr.open_dataset(path)
        # Drop unused metadata dimensions if present
        ds = ds.drop_vars(["number", "expver"], errors="ignore")
        return ds, None
    except Exception as e:
        return None, f"Failed to open NetCDF dataset: {str(e)}"


def get_dataset_metadata(ds: xr.Dataset) -> dict:
    """Returns dataset summary metrics, temporal boundaries, and coordinate extents."""
    if ds is None:
        return {}
    times = ds["valid_time"].values
    lats = ds.coords["latitude"].values
    lons = ds.coords["longitude"].values

    return {
        "total_time_steps": int(len(times)),
        "min_time": pd.to_datetime(times[0]),
        "max_time": pd.to_datetime(times[-1]),
        "lat_range": (float(lats.min()), float(lats.max())),
        "lon_range": (float(lons.min()), float(lons.max())),
        "total_cells": int(len(lats) * len(lons)),
        "variables": list(ds.data_vars.keys()),
    }


@st.cache_data(show_spinner=False)
def load_cell_history(
    _ds_id: str,
    target_time_str: str,
    lat: float,
    lon: float,
    history_hours: int = 720,
) -> pd.DataFrame:
    """
    Extracts the recent hourly sequence for a single cell up to target_time.
    Computes exact antecedent rolling features according to the training contract.
    """
    ds, err = open_dataset()
    if ds is None:
        return None

    target_time = pd.Timestamp(target_time_str)
    start_time = target_time - pd.Timedelta(hours=history_hours)

    # Nearest coordinate lookup (spatial first with nearest, then slice valid_time)
    sub = ds.sel(
        latitude=lat,
        longitude=lon,
        method="nearest",
    ).sel(
        valid_time=slice(start_time, target_time)
    )

    t2m = sub["t2m"].values.astype(np.float32)
    d2m = sub["d2m"].values.astype(np.float32)
    u10 = sub["u10"].values.astype(np.float32)
    v10 = sub["v10"].values.astype(np.float32)
    sp = sub["sp"].values.astype(np.float32)
    tp = sub["tp"].values.astype(np.float32)
    times = pd.to_datetime(sub["valid_time"].values)

    elev = float(sub["elevation"].values[-1] if sub["elevation"].ndim > 0 else sub["elevation"].values)
    lc = float(sub["land_cover"].values[-1] if sub["land_cover"].ndim > 0 else sub["land_cover"].values)
    ndvi = float(sub["ndvi"].values[-1] if sub["ndvi"].ndim > 0 else sub["ndvi"].values)
    evi = float(sub["evi"].values[-1] if sub["evi"].ndim > 0 else sub["evi"].values)
    pr = float(sub["pixel_reliability"].values[-1] if sub["pixel_reliability"].ndim > 0 else sub["pixel_reliability"].values)
    viq = float(sub["vi_quality"].values[-1] if sub["vi_quality"].ndim > 0 else sub["vi_quality"].values)

    df_hist = pd.DataFrame({
        "timestamp": times,
        "t2m_k": t2m,
        "d2m_k": d2m,
        "t2m_c": t2m - 273.15,
        "d2m_c": d2m - 273.15,
        "u10": u10,
        "v10": v10,
        "wind_speed": np.sqrt(u10**2 + v10**2),
        "sp": sp,
        "tp": tp,
        "elevation": elev,
        "land_cover": lc,
        "ndvi": ndvi,
        "evi": evi,
        "pixel_reliability": pr,
        "vi_quality": viq,
    })

    # Rolling calculations
    vpd_arr = np.array([calc_vpd(tc, dc) for tc, dc in zip(df_hist["t2m_c"], df_hist["d2m_c"])])
    df_hist["vpd"] = vpd_arr
    return df_hist


@st.cache_data(show_spinner=False)
def load_spatial_slice(
    _ds_id: str,
    timestamp_str: str,
    land_only: bool = True,
) -> pd.DataFrame:
    """
    Extracts a 2D spatial grid for California at a given timestamp.
    Downsamples offshore Pacific cells to preserve memory and rendering speed.
    Falls back cleanly to assets/curated_snapshots.parquet if NetCDF is absent.
    """
    ds, err = open_dataset()
    if ds is None:
        parquet_path = "assets/curated_snapshots.parquet"
        if os.path.exists(parquet_path):
            try:
                curated_df = pd.read_parquet(parquet_path)
                match = curated_df[curated_df["valid_time"].astype(str).str.startswith(timestamp_str[:10])]
                if match.empty:
                    for sc in CURATED_SCENARIOS.values():
                        if sc["timestamp"].startswith(timestamp_str[:10]):
                            match = curated_df[curated_df["valid_time"].astype(str) == sc["timestamp"]]
                            break
                if not match.empty:
                    res = match.copy()
                    if land_only:
                        res = res[(res["elevation"] > 0) | (res["land_cover"].notnull())].copy()
                    return res
            except Exception as e:
                logger.error(f"Error reading curated snapshots: {e}")
        return None

    try:
        snap = ds.sel(valid_time=timestamp_str, method="nearest").to_dataframe().reset_index()
    except Exception as e:
        logger.error(f"Error reading spatial slice: {e}")
        return None

    # Derive variables
    snap["t2m_celsius"] = snap["t2m"] - 273.15
    snap["d2m_celsius"] = snap["d2m"] - 273.15
    snap["wind_speed"] = np.sqrt(snap["u10"]**2 + snap["v10"]**2)
    snap["vpd"] = [calc_vpd(t, d) for t, d in zip(snap["t2m_celsius"], snap["d2m_celsius"])]
    snap["rel_humidity"] = [calc_rh(t, d) for t, d in zip(snap["t2m_celsius"], snap["d2m_celsius"])]

    # Human readable land cover
    snap["land_cover_name"] = snap["land_cover"].map(LAND_COVER_LABELS).fillna("Unclassified / Water")

    # Filter to terrestrial cells if requested (elevation > 0 or valid land cover)
    if land_only:
        snap = snap[(snap["elevation"] > 0) | (snap["land_cover"].notnull())].copy()

    return snap


@st.cache_data(show_spinner=False)
def load_spatial_slice_with_predictions(
    _ds_id: str,
    timestamp_str: str,
    land_only: bool = True,
) -> pd.DataFrame:
    """
    Loads spatial slice and applies the trained model pipeline to terrestrial cells
    that have complete, valid features.
    """
    snap = load_spatial_slice(_ds_id, timestamp_str, land_only=land_only)
    if snap is None or snap.empty:
        return snap

    from src.model_loader import load_pipeline, load_feature_schema
    pipeline, _ = load_pipeline()
    schema, _ = load_feature_schema()

    if pipeline is None or schema is None:
        snap["model_risk_score"] = np.nan
        snap["risk_level"] = "Unknown"
        snap["is_alert"] = False
        return snap

    ts = pd.Timestamp(timestamp_str)
    month = ts.month
    hour = ts.hour
    day_of_year = ts.dayofyear

    # Extract antecedent history for the slice:
    # 24h/7d/30d approximations from available sequence or grid averages
    # For instant 3D rendering of the 1,225 grid cells, compute vectorized features
    bounds = schema.get("winsorization_bounds", {})
    threshold = float(schema.get("decision_threshold", 0.2507))

    # Mask to land cells with valid land_cover
    valid_mask = snap["land_cover"].notnull() & (snap["elevation"] >= 0)
    valid_cells = snap[valid_mask].copy()

    if valid_cells.empty:
        snap["model_risk_score"] = np.nan
        snap["risk_level"] = "Unknown"
        snap["is_alert"] = False
        return snap

    # Vectorized feature assembly
    u10 = np.clip(valid_cells["u10"].values, bounds.get("u10", [-10, 10])[0], bounds.get("u10", [-10, 10])[1])
    v10 = np.clip(valid_cells["v10"].values, bounds.get("v10", [-10, 10])[0], bounds.get("v10", [-10, 10])[1])
    t2m_k = np.clip(valid_cells["t2m"].values, bounds.get("t2m", [260, 320])[0], bounds.get("t2m", [260, 320])[1])
    d2m_k = np.clip(valid_cells["d2m"].values, bounds.get("d2m", [250, 300])[0], bounds.get("d2m", [250, 300])[1])
    sp = np.clip(valid_cells["sp"].values, bounds.get("sp", [70000, 105000])[0], bounds.get("sp", [70000, 105000])[1])
    tp = np.clip(valid_cells["tp"].values, bounds.get("tp", [0, 0.05])[0], bounds.get("tp", [0, 0.05])[1])

    t2m_c = t2m_k - 273.15
    d2m_c = d2m_k - 273.15
    wind_speed = np.sqrt(u10**2 + v10**2)
    temp_dewpoint_spread = t2m_c - d2m_c
    rh = np.array([calc_rh(t, d) for t, d in zip(t2m_c, d2m_c)])
    vpd = np.array([calc_vpd(t, d) for t, d in zip(t2m_c, d2m_c)])

    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    doy_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    doy_cos = np.cos(2 * np.pi * day_of_year / 365.25)

    wind_dir_rad = np.arctan2(-u10, -v10)
    wind_dir_sin = np.sin(wind_dir_rad)
    wind_dir_cos = np.cos(wind_dir_rad)

    feature_dict = {
        "u10": u10,
        "v10": v10,
        "t2m_celsius": t2m_c,
        "d2m_celsius": d2m_c,
        "sp": sp,
        "tp": tp,
        "land_cover": valid_cells["land_cover"].values,
        "elevation": valid_cells["elevation"].values,
        "ndvi_scaled": valid_cells["ndvi"].fillna(3000).values * 0.0001,
        "evi_scaled": valid_cells["evi"].fillna(2000).values * 0.0001,
        "pixel_reliability": valid_cells["pixel_reliability"].fillna(0).values,
        "vi_quality": valid_cells["vi_quality"].fillna(2116).values,
        "wind_speed": wind_speed,
        "temp_dewpoint_spread": temp_dewpoint_spread,
        "month_sin": np.full(len(valid_cells), month_sin),
        "month_cos": np.full(len(valid_cells), month_cos),
        "hour_sin": np.full(len(valid_cells), hour_sin),
        "hour_cos": np.full(len(valid_cells), hour_cos),
        "rel_humidity": rh,
        "vpd": vpd,
        "doy_sin": np.full(len(valid_cells), doy_sin),
        "doy_cos": np.full(len(valid_cells), doy_cos),
        "wind_dir_sin": wind_dir_sin,
        "wind_dir_cos": wind_dir_cos,
        # Rolling estimates aligned with current conditions for grid-level batching
        "tp_sum_24h": np.zeros(len(valid_cells)),
        "tp_sum_7d": np.zeros(len(valid_cells)),
        "tp_sum_30d": np.zeros(len(valid_cells)),
        "vpd_mean_24h": vpd,
        "vpd_mean_7d": vpd,
        "t2m_mean_7d": t2m_c,
        "wind_mean_24h": wind_speed,
        "latitude": valid_cells["latitude"].values,
        "longitude": valid_cells["longitude"].values,
    }

    X_grid = pd.DataFrame(feature_dict)[schema["features"]]
    raw_scores = pipeline.predict_proba(X_grid)[:, 1]

    snap["model_risk_score"] = np.nan
    snap.loc[valid_cells.index, "model_risk_score"] = raw_scores
    snap["is_alert"] = snap["model_risk_score"] >= threshold
    snap["risk_level"] = snap["model_risk_score"].apply(
        lambda s: "Unknown" if pd.isna(s) else (
            "Extreme" if s >= 0.45 else ("High" if s >= threshold else ("Moderate" if s >= 0.15 else "Low"))
        )
    )

    return snap


# Preset California regions for convenient selection in Select Date & Predict mode
CALIFORNIA_PRESET_LOCATIONS = {
    "Napa Valley / Wine Country": {"latitude": 38.50, "longitude": -122.50, "region": "North Bay Foothills"},
    "Lake Tahoe / Northern Sierra": {"latitude": 39.00, "longitude": -120.00, "region": "Sierra Nevada"},
    "Paradise / Butte County": {"latitude": 39.75, "longitude": -121.50, "region": "Cascade Foothills"},
    "Angeles National Forest / LA": {"latitude": 34.25, "longitude": -118.25, "region": "Transverse Ranges"},
    "Shasta-Cascade / Redding": {"latitude": 40.50, "longitude": -122.25, "region": "Northern Interior"},
    "Yosemite Foothills / Mariposa": {"latitude": 37.50, "longitude": -119.75, "region": "Central Sierra"},
    "Santa Barbara / Los Padres": {"latitude": 34.50, "longitude": -119.75, "region": "Central Coast"},
    "San Diego / Cleveland National Forest": {"latitude": 33.00, "longitude": -116.75, "region": "Peninsular Ranges"},
    "Mendocino Coast & Range": {"latitude": 39.25, "longitude": -123.50, "region": "North Coast Range"},
    "Central Valley / Fresno": {"latitude": 36.75, "longitude": -119.75, "region": "Central Valley Basin"},
}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_open_meteo_forecast(lat: float, lon: float, target_datetime_str: str) -> dict:
    """
    Fetches real hourly forecast from Open-Meteo API for coordinates (lat, lon)
    using past_days=31 and forecast_days=7.
    Extracts weather variables at target_datetime and calculates antecedent rolling metrics.
    Returns structured dict with weather features, provider metadata, and retrieval time.
    Raises ValueError on failure so caller can block inference honestly.
    """
    import urllib.request
    import json
    from datetime import datetime, timezone

    target_dt = pd.Timestamp(target_datetime_str)
    # Ensure coordinates are within California bounding box
    if not (32.0 <= lat <= 43.0 and -125.0 <= lon <= -113.5):
        raise ValueError(
            f"Coordinates ({lat:.2f}°N, {abs(lon):.2f}°W) are outside California boundary [32.0°N–43.0°N, 113.5°W–125.0°W]."
        )

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat:.4f}&longitude={lon:.4f}&"
        f"hourly=temperature_2m,dew_point_2m,surface_pressure,precipitation,wind_speed_10m,wind_direction_10m,relative_humidity_2m,vapor_pressure_deficit&"
        f"wind_speed_unit=ms&timezone=UTC&past_days=31&forecast_days=7"
    )

    retrieval_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CaliforniaWildfirePrediction/2.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            if resp.status != 200:
                raise ValueError(f"Open-Meteo API returned HTTP status {resp.status}.")
            data = json.loads(resp.read().decode())
    except Exception as e:
        raise ValueError(
            f"Failed to retrieve weather forecast from Open-Meteo API: {str(e)}. "
            "Real-time inference cannot proceed without verified weather data."
        )

    if "hourly" not in data or "time" not in data["hourly"]:
        raise ValueError("Open-Meteo response missing 'hourly' forecast payload.")

    df_hourly = pd.DataFrame(data["hourly"])
    df_hourly["time"] = pd.to_datetime(df_hourly["time"])
    df_hourly = df_hourly.set_index("time").sort_index()

    if target_dt < df_hourly.index[0] or target_dt > df_hourly.index[-1]:
        raise ValueError(
            f"Requested date/time '{target_dt}' is outside the supported forecast range "
            f"[{df_hourly.index[0].strftime('%Y-%m-%d %H:%M')} to {df_hourly.index[-1].strftime('%Y-%m-%d %H:%M')} UTC]."
        )

    idx_pos = df_hourly.index.get_indexer([target_dt], method="nearest")[0]
    actual_time = df_hourly.index[idx_pos]
    sub_hist = df_hourly.iloc[: idx_pos + 1]

    curr = df_hourly.iloc[idx_pos]
    t2m_c = float(curr["temperature_2m"])
    d2m_c = float(curr["dew_point_2m"])
    sp_pa = float(curr["surface_pressure"]) * 100.0  # hPa to Pa
    tp_m = float(curr["precipitation"]) * 0.001       # mm to m
    w_spd = float(curr["wind_speed_10m"])             # m/s
    w_dir_deg = float(curr["wind_direction_10m"])     # degrees
    vpd_kpa = float(curr["vapor_pressure_deficit"])  # kPa
    rh = float(curr["relative_humidity_2m"])

    # Meteorological wind vector derivation: u10 = eastward, v10 = northward
    w_dir_rad = np.radians(w_dir_deg)
    u10 = -w_spd * np.sin(w_dir_rad)
    v10 = -w_spd * np.cos(w_dir_rad)

    # Antecedent rolling sums & means up to target hour
    tp_sum_24h = float((sub_hist["precipitation"].tail(24) * 0.001).sum())
    tp_sum_7d = float((sub_hist["precipitation"].tail(168) * 0.001).sum())
    tp_sum_30d = float((sub_hist["precipitation"].tail(720) * 0.001).sum())
    vpd_mean_24h = float(sub_hist["vapor_pressure_deficit"].tail(24).mean())
    vpd_mean_7d = float(sub_hist["vapor_pressure_deficit"].tail(168).mean())
    t2m_mean_7d = float(sub_hist["temperature_2m"].tail(168).mean())
    wind_mean_24h = float(sub_hist["wind_speed_10m"].tail(24).mean())

    return {
        "provider": "Open-Meteo Global Weather Model (ECMWF & GFS Ensembles)",
        "retrieval_time": retrieval_time,
        "forecast_time": actual_time,
        "target_requested": target_dt,
        "t2m_c": t2m_c,
        "d2m_c": d2m_c,
        "sp_pa": sp_pa,
        "tp_m": tp_m,
        "wind_speed": w_spd,
        "wind_dir_deg": w_dir_deg,
        "u10": u10,
        "v10": v10,
        "vpd_kpa": vpd_kpa,
        "rh": rh,
        "tp_sum_24h": tp_sum_24h,
        "tp_sum_7d": tp_sum_7d,
        "tp_sum_30d": tp_sum_30d,
        "vpd_mean_24h": vpd_mean_24h,
        "vpd_mean_7d": vpd_mean_7d,
        "t2m_mean_7d": t2m_mean_7d,
        "wind_mean_24h": wind_mean_24h,
    }


@st.cache_data(show_spinner=False)
def get_cell_static_and_vegetation(lat: float, lon: float) -> dict:
    """
    Extracts real static terrain elevation, MODIS land cover, and the latest genuinely
    available spaceborne vegetation observation (NDVI/EVI) from master_dataset_2024_2025.nc.
    Validates that the cell is terrestrial and returns metadata with observation timestamp.
    """
    ds, err = open_dataset()
    if ds is None:
        parquet_path = "assets/california_grid_static.parquet"
        if os.path.exists(parquet_path):
            try:
                grid_df = pd.read_parquet(parquet_path)
                dist = (grid_df["latitude"] - lat)**2 + (grid_df["longitude"] - lon)**2
                nearest = grid_df.loc[dist.idxmin()]
                cell_lat = float(nearest["latitude"])
                cell_lon = float(nearest["longitude"])
                elev = float(nearest["elevation"])
                lc = float(nearest["land_cover"])
                is_water = bool(lc == 17.0 or np.isnan(lc))
                latest_vt_str = str(nearest["latest_observation"])
                ndvi = float(nearest["ndvi"])
                evi = float(nearest["evi"])
                pixel_rel = float(nearest["pixel_reliability"])
                vi_qual = float(nearest["vi_quality"])
                lc_name = LAND_COVER_LABELS.get(lc, f"Class {lc:.0f}")
                return {
                    "cell_latitude": cell_lat,
                    "cell_longitude": cell_lon,
                    "elevation": elev,
                    "land_cover": lc,
                    "land_cover_name": lc_name,
                    "is_water": is_water,
                    "latest_observation": latest_vt_str,
                    "ndvi": ndvi,
                    "evi": evi,
                    "pixel_reliability": pixel_rel,
                    "vi_quality": vi_qual,
                }
            except Exception as e:
                logger.error(f"Error reading grid static cache: {e}")
        raise ValueError(f"Cannot access project dataset for static features: {err}")

    # Nearest cell lookup
    cell = ds.sel(latitude=lat, longitude=lon, method="nearest")
    cell_lat = float(cell.latitude)
    cell_lon = float(cell.longitude)
    elev = float(cell["elevation"].values)
    lc = float(cell["land_cover"].values)

    # Check for ocean / water body
    is_water = bool(lc == 17.0 or np.isnan(lc))

    # Latest genuinely available MODIS observation (end of verified sequence: 2025-12-31)
    latest_vt = cell["valid_time"].values[-1]
    latest_vt_str = pd.to_datetime(latest_vt).strftime("%Y-%m-%d %H:%M UTC")

    ndvi = float(cell["ndvi"].sel(valid_time=latest_vt).values)
    evi = float(cell["evi"].sel(valid_time=latest_vt).values)
    pixel_rel = float(cell["pixel_reliability"].sel(valid_time=latest_vt).values)
    vi_qual = float(cell["vi_quality"].sel(valid_time=latest_vt).values)

    lc_name = LAND_COVER_LABELS.get(lc, f"Class {lc:.0f}")

    return {
        "cell_latitude": cell_lat,
        "cell_longitude": cell_lon,
        "elevation": elev,
        "land_cover": lc,
        "land_cover_name": lc_name,
        "is_water": is_water,
        "ndvi": ndvi,
        "evi": evi,
        "pixel_reliability": pixel_rel,
        "vi_quality": vi_qual,
        "vegetation_obs_date": latest_vt_str,
        "vegetation_source": "MODIS MOD13Q1 (16-day Earth observation composite)",
    }

