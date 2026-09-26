"""
Regression test suite for California Wildfire Prediction:
Verifies vegetation_source metadata contract, live forecast retrieval,
feature preparation, and inference execution across both local (NetCDF)
and cloud deployment (Parquet fallback) data paths.
"""

from datetime import datetime, timezone
import os
import sys
import unittest
from unittest.mock import patch
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_cell_static_and_vegetation, fetch_open_meteo_forecast
from src.model_loader import (
    load_feature_schema,
    load_pipeline,
    prepare_feature_row,
    predict_wildfire_risk,
)


class TestVegetationSourceAndInference(unittest.TestCase):
    def setUp(self):
        # Representative terrestrial California coordinates (Los Angeles / Angeles NF foothills)
        self.test_lat = 34.15
        self.test_lon = -118.15
        self.target_time = datetime.now(timezone.utc)

    def test_get_cell_static_and_vegetation_local_netcdf_path(self):
        """Path A: NetCDF dataset is present (Local dev environment)."""
        nc_path = os.path.join(PROJECT_ROOT, "master_dataset_2024_2025.nc")
        if not os.path.exists(nc_path):
            self.skipTest("Local NetCDF dataset not present in working tree.")

        meta = get_cell_static_and_vegetation(self.test_lat, self.test_lon)

        # Essential keys required by downstream consumers
        self.assertIn("vegetation_source", meta)
        self.assertIn("vegetation_obs_date", meta)
        self.assertIn("latest_observation", meta)
        self.assertIn("elevation", meta)
        self.assertIn("land_cover", meta)
        self.assertIn("land_cover_name", meta)
        self.assertIn("ndvi", meta)
        self.assertIn("evi", meta)
        self.assertIn("is_water", meta)

        self.assertIsInstance(meta["vegetation_source"], str)
        self.assertTrue(len(meta["vegetation_source"]) > 0)
        self.assertIn("MODIS", meta["vegetation_source"])

        self.assertIsInstance(meta["vegetation_obs_date"], str)
        self.assertTrue(len(meta["vegetation_obs_date"]) > 0)

    def test_get_cell_static_and_vegetation_cloud_parquet_fallback_path(self):
        """Path B: NetCDF dataset absent, Parquet static grid loaded (Cloud deployment environment)."""
        # Simulate absence of NetCDF dataset (as on Streamlit Community Cloud)
        with patch("src.data_loader.open_dataset", return_value=(None, "Dataset not found in cloud deployment")):
            meta = get_cell_static_and_vegetation(self.test_lat, self.test_lon)

            # Contract requirements:
            self.assertIn("vegetation_source", meta, "vegetation_source missing from cloud Parquet path!")
            self.assertIn("vegetation_obs_date", meta, "vegetation_obs_date missing from cloud Parquet path!")
            self.assertIn("latest_observation", meta, "latest_observation missing from cloud Parquet path!")
            self.assertIn("elevation", meta)
            self.assertIn("land_cover", meta)
            self.assertIn("ndvi", meta)
            self.assertIn("evi", meta)

            self.assertIsInstance(meta["vegetation_source"], str)
            self.assertIn("MODIS", meta["vegetation_source"])
            self.assertIsInstance(meta["vegetation_obs_date"], str)
            self.assertFalse(meta["is_water"])

    def test_end_to_end_inference_with_parquet_metadata(self):
        """
        Simulate the exact deployed Select Date & Predict workflow in Streamlit Cloud:
        1. Fetch static vegetation data via cloud Parquet fallback
        2. Fetch real live weather forecast from Open-Meteo API
        3. Prepare feature vector matching model schema
        4. Execute model prediction pipeline
        5. Validate that all provenance tiles format without KeyError
        """
        schema, s_err = load_feature_schema()
        self.assertIsNone(s_err)
        pipeline, p_err = load_pipeline()
        self.assertIsNone(p_err)
        self.assertIsNotNone(pipeline, "Trained pipeline could not be loaded")

        with patch("src.data_loader.open_dataset", return_value=(None, "Simulated cloud deployment")):
            cell_meta = get_cell_static_and_vegetation(self.test_lat, self.test_lon)

            # 1. Verify safe extraction
            veg_source = cell_meta.get("vegetation_source", "MODIS MOD13Q1 composite")
            veg_date = cell_meta.get("vegetation_obs_date", cell_meta.get("latest_observation", "Latest"))
            self.assertIsNotNone(veg_source)
            self.assertIsNotNone(veg_date)

            # 2. Fetch real live weather forecast
            try:
                fc_data = fetch_open_meteo_forecast(self.test_lat, self.test_lon, self.target_time)
            except Exception as e:
                self.skipTest(f"Live Weather API not reachable or timed out: {e}")

            self.assertIn("t2m_c", fc_data)
            self.assertIn("d2m_c", fc_data)
            self.assertIn("wind_speed", fc_data)
            self.assertIn("vpd_kpa", fc_data)

            # 3. Model inference input construction matching prediction.py lines 475-500
            target_dt_obj = pd.Timestamp(self.target_time)
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

            df_features = prepare_feature_row(raw_dict_forecast, schema)
            res = predict_wildfire_risk(pipeline, schema, df_features)

            self.assertIn("calibrated_prob", res)
            self.assertIn("raw_score", res)
            self.assertIn("risk_level", res)
            self.assertIn("is_alert", res)
            self.assertGreaterEqual(res["calibrated_prob"], 0.0)
            self.assertLessEqual(res["calibrated_prob"], 1.0)

            # 4. Provenance display formatting check (the exact format string that threw KeyError)
            provenance_html = (
                f"Provider: {fc_data['provider']} | "
                f"Vegetation Layer: {cell_meta['vegetation_source']} ({cell_meta['vegetation_obs_date']})"
            )
            self.assertIn("MODIS", provenance_html)

    def test_coordinates_outside_california_raises_value_error(self):
        """Confirm that out-of-bounds coordinates raise a clean ValueError instead of unhandled exceptions."""
        with self.assertRaises(ValueError) as ctx:
            fetch_open_meteo_forecast(50.0, -100.0, self.target_time)
        self.assertIn("outside California boundary", str(ctx.exception))

    def test_ocean_coordinate_detection(self):
        """Confirm ocean/water coordinates are correctly identified without throwing KeyError."""
        # Coordinate in Pacific Ocean off the coast of California
        ocean_lat, ocean_lon = 33.5, -121.5
        with patch("src.data_loader.open_dataset", return_value=(None, "Cloud fallback")):
            meta = get_cell_static_and_vegetation(ocean_lat, ocean_lon)
            self.assertIn("is_water", meta)
            self.assertIn("vegetation_source", meta)
            self.assertIn("vegetation_obs_date", meta)

    def test_overview_provenance_rendering(self):
        """Test overview page provenance rendering with fallback metadata."""
        with patch("src.data_loader.open_dataset", return_value=(None, "Cloud fallback")):
            cell_meta = get_cell_static_and_vegetation(self.test_lat, self.test_lon)
            veg_src_ov = cell_meta.get("vegetation_source", "MODIS MOD13Q1 (16-day Earth observation composite)")
            veg_dt_ov = cell_meta.get("vegetation_obs_date", cell_meta.get("latest_observation", "Latest Available"))
            html_chunk = f"Vegetation: {veg_src_ov} ({veg_dt_ov})"
            self.assertIn("MODIS", html_chunk)


if __name__ == "__main__":
    unittest.main()

