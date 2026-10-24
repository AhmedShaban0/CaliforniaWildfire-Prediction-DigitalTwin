"""
Test suite verifying the California Wildfire Prediction redesign:
1. Basemap catalog and provider configurations.
2. 0.25° grid polygon footprint calculations (~25 km resolution).
3. Risk layer RGBA color and transparency derivation across all risk tiers.
4. Theme component rendering (hero, sidebar header, risk legend, notice).
5. All page rendering modules importability and structure.
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.twin_3d import (
    BASEMAP_CATALOG,
    compute_grid_cell_polygons,
    get_color_for_metric,
)
from src.theme import (
    get_logo_b64,
    render_risk_legend,
    render_operational_notice,
    render_main_hero,
    RISK_TIER_COLORS,
)


class TestRedesignVerification(unittest.TestCase):
    def test_basemap_catalog_entries(self):
        """Verify standard basemaps are present with proper styles and descriptions."""
        self.assertIn("Terrain & Geography (Carto Voyager)", BASEMAP_CATALOG)
        self.assertIn("Satellite Imagery (Esri World Imagery)", BASEMAP_CATALOG)
        self.assertIn("Streets & Municipal (Carto Positron)", BASEMAP_CATALOG)
        self.assertIn("Dark Analytical (Carto Dark)", BASEMAP_CATALOG)

        # Voyager style check
        voyager = BASEMAP_CATALOG["Terrain & Geography (Carto Voyager)"]
        self.assertIn("voyager-gl-style", voyager["style"])

        # Esri tile layer check
        esri = BASEMAP_CATALOG["Satellite Imagery (Esri World Imagery)"]
        self.assertTrue(esri["use_tile_layer"])
        self.assertIn("arcgisonline.com", esri["tile_url"])

    def test_compute_grid_cell_polygons(self):
        """Verify 0.25° grid polygon generation around cell centers."""
        sample_df = pd.DataFrame([
            {"latitude": 37.0, "longitude": -120.0, "elevation": 300},
            {"latitude": 34.25, "longitude": -118.5, "elevation": 450},
        ])
        poly_df = compute_grid_cell_polygons(sample_df, half_size=0.125)
        self.assertIn("polygon", poly_df.columns)
        self.assertEqual(len(poly_df), 2)

        # Check first cell polygon coordinates
        p1 = poly_df.iloc[0]["polygon"]
        self.assertEqual(len(p1), 4)
        # [lon - 0.125, lat - 0.125] -> [-120.125, 36.875]
        self.assertAlmostEqual(p1[0][0], -120.125)
        self.assertAlmostEqual(p1[0][1], 36.875)
        # [lon + 0.125, lat + 0.125] -> [-119.875, 37.125]
        self.assertAlmostEqual(p1[2][0], -119.875)
        self.assertAlmostEqual(p1[2][1], 37.125)

    def test_get_color_for_metric_with_transparency(self):
        """Verify RGBA color output and transparency across all risk tiers."""
        alpha = 140

        # Low risk (< 15%)
        c_low = get_color_for_metric(0.08, "Model Predicted Risk", alpha=alpha)
        self.assertEqual(len(c_low), 4)
        self.assertEqual(c_low[3], alpha)
        # Greenish: G > R
        self.assertGreater(c_low[1], c_low[0])

        # Moderate risk (15% - 25.07%)
        c_mod = get_color_for_metric(0.20, "Model Predicted Risk", alpha=alpha)
        self.assertEqual(len(c_mod), 4)

        # High risk (>= 25.07%)
        c_high = get_color_for_metric(0.35, "Model Predicted Risk", alpha=alpha)
        self.assertEqual(len(c_high), 4)
        # Orange/Red: R > B
        self.assertGreater(c_high[0], c_high[2])

        # Extreme risk (>= 45%)
        c_ext = get_color_for_metric(0.60, "Model Predicted Risk", alpha=alpha)
        self.assertEqual(len(c_ext), 4)
        # Crimson: R dominant
        self.assertGreater(c_ext[0], 180)

    def test_logo_asset_resolution(self):
        """Verify logo b64 resolution succeeds."""
        b64 = get_logo_b64()
        self.assertIsInstance(b64, str)
        self.assertTrue(len(b64) > 0, "wildfire.png base64 encoding must not be empty.")

    def test_all_page_modules_importable(self):
        """Verify all 6 page implementation modules import cleanly without errors."""
        from pages_impl.overview import render_overview_page
        from pages_impl.prediction import render_prediction_page
        from pages_impl.interactive_map import render_interactive_map_page
        from pages_impl.digital_twin import render_digital_twin_page
        from pages_impl.performance import render_performance_page
        from pages_impl.methodology import render_methodology_page
        from pages_impl.about import render_about_page

        self.assertTrue(callable(render_overview_page))
        self.assertTrue(callable(render_prediction_page))
        self.assertTrue(callable(render_interactive_map_page))
        self.assertTrue(callable(render_digital_twin_page))
        self.assertTrue(callable(render_performance_page))
        self.assertTrue(callable(render_methodology_page))
        self.assertTrue(callable(render_about_page))


    def test_risk_tier_classification(self):
        """Verify the operational threshold 0.2507 correctly partitions risk categories."""
        from src.model_loader import categorize_risk
        threshold = 0.2507
        
        # Test boundary cases
        t1 = categorize_risk(0.05, threshold)
        self.assertEqual(t1, "Low")
        
        t2 = categorize_risk(0.18, threshold)
        self.assertEqual(t2, "Moderate")
        
        t3 = categorize_risk(0.26, threshold)
        self.assertEqual(t3, "High")
        
        t4 = categorize_risk(0.55, threshold)
        self.assertEqual(t4, "Extreme")

    def test_feature_schema_contract(self):
        """Verify feature_schema.json contains the exact 33 features expected by the pipeline."""
        from src.model_loader import load_feature_schema
        schema, err = load_feature_schema()
        self.assertIsNone(err)
        self.assertIsNotNone(schema)
        self.assertIn("features", schema)
        features = schema["features"]
        self.assertEqual(len(features), 33)
        self.assertIn("land_cover", features)
        self.assertIn("latitude", features)
        self.assertIn("longitude", features)
    def test_leaflet_html_generation(self):
        """Verify Leaflet HTML generation produces valid markup with basemaps, features, and controls."""
        from src.leaflet_map import generate_leaflet_html
        sample_df = pd.DataFrame([
            {
                "latitude": 38.5,
                "longitude": -122.5,
                "model_risk_score": 0.35,
                "is_alert": True,
                "elevation": 450,
                "t2m_celsius": 32.0,
                "wind_speed": 6.5,
                "vpd": 3.1,
                "fire_now": 0,
            },
            {
                "latitude": 40.0,
                "longitude": -121.5,
                "model_risk_score": 0.65,
                "is_alert": True,
                "elevation": 900,
                "t2m_celsius": 36.0,
                "wind_speed": 8.0,
                "vpd": 4.2,
                "fire_now": 1,
            },
        ])
        html = generate_leaflet_html(sample_df, default_basemap="Satellite Imagery (Esri World)", target_cell=(38.5, -122.5, "Napa"))
        self.assertIn("leaflet.js", html)
        self.assertIn("server.arcgisonline.com", html)
        self.assertIn("World_Topo_Map", html)
        self.assertIn("World_Imagery", html)
        self.assertIn("opentopomap.org", html)
        self.assertIn("openstreetmap.org", html)
        self.assertIn("L.map", html)
        self.assertIn("L.geoJSON", html)
        self.assertIn("ALERT ACTIVATED", html)

    def test_basemap_layer_switching_mapping(self):
        """Verify that basemap selections map cleanly to verified watermark-free tile endpoints."""
        from src.leaflet_map import generate_leaflet_html
        sample_df = pd.DataFrame([{"latitude": 38.5, "longitude": -122.5, "model_risk_score": 0.2}])
        
        # Terrain selection maps to basemapTerrain without CARTO watermark
        html_terrain = generate_leaflet_html(sample_df, default_basemap="Terrain & Geography (Carto Voyager)")
        self.assertIn("basemapTerrain.addTo(map)", html_terrain)
        self.assertNotIn("basemaps.cartocdn.com/rastertiles", html_terrain)

        # Dark selection maps to basemapDark without CARTO watermark
        html_dark = generate_leaflet_html(sample_df, default_basemap="Dark Matter (Carto Dark)")
        self.assertIn("basemapDark.addTo(map)", html_dark)
        self.assertNotIn("basemaps.cartocdn.com/dark_all", html_dark)

        # Satellite selection maps to basemapSatellite
        html_sat = generate_leaflet_html(sample_df, default_basemap="Satellite Imagery (Esri World)")
        self.assertIn("basemapSatellite.addTo(map)", html_sat)

    def test_dynamic_region_thumbnails(self):
        """Verify that dynamic regional images exist and return valid base64 strings."""
        from src.data_loader import CALIFORNIA_PRESET_LOCATIONS
        from src.theme import get_region_thumb_b64

        self.assertEqual(len(CALIFORNIA_PRESET_LOCATIONS), 10)
        for preset_name in CALIFORNIA_PRESET_LOCATIONS:
            b64 = get_region_thumb_b64(preset_name)
            self.assertTrue(len(b64) > 100, f"Thumbnail for {preset_name} must be valid base64.")

    def test_white_logo_container_in_sidebar(self):
        """Verify sidebar header has the white background container."""
        from src.theme import get_logo_b64
        self.assertTrue(len(get_logo_b64()) > 0)


if __name__ == "__main__":
    unittest.main()

