# California Wildfire Prediction — Deployment Audit & Model/Data Contract

**Date:** September 2026  
**Auditor:** Senior ML Engineer, Geospatial Specialist, & Deployment Engineer  
**Workspace:** `c:\Users\ahmed\Downloads\California_WildFire_Prediction`

---

## 1. Discovered File Tree & Streamlit Entry Point

### Workspace Contents
```text
California_WildFire_Prediction/
├── .venv/                                      # Python 3.14 virtual environment
├── assets/
│   └── wildfire.jpg                            # Supplied project logo (520 KB)
├── wildfire.jpg                                # Root project logo (520 KB)
├── feature_schema.json                         # Actual model & feature contract (4.33 KB)
├── wildfire_model_pipeline.joblib              # Fitted scikit-learn Pipeline (91.01 MB)
├── master_dataset_2024_2025.nc                 # Primary NetCDF-4 dataset (434.71 MB)
├── master_dataset_2024_2025.ncci63wf9y.part    # Incomplete partial download artifact (14.16 MB)
└── California_Wildfire_ML_final_5_improved.ipynb # End-to-end ML notebook (1.49 MB)
```

- **Existing App Entry Point:** No prior `.py` or Streamlit files existed in the root.
- **Designated Production Entry Point:** `app.py` in the workspace root, structured modularly with pages and utilities (`src/`).

---

## 2. Model, Schema & Preprocessing Artifacts

### Model Artifact: `wildfire_model_pipeline.joblib`
- **File Size:** 91.01 MB (compressed level 3).
- **Structure:** Scikit-learn `Pipeline` with 2 named steps:
  1. `preprocessor`: `ColumnTransformer`
     - `numeric`: `StandardScaler` applied to 32 numeric features.
     - `categorical`: `OneHotEncoder(handle_unknown='ignore', dtype=np.float32)` applied to `land_cover`.
  2. `model`: `RandomForestClassifier(max_features=0.5, max_samples=0.3, min_samples_leaf=5, n_estimators=200, n_jobs=-1, random_state=42)`
- **Origin:** Refit on balanced training + validation data using the top-performing model from Section 13/14 of `California_Wildfire_ML_final_5_improved.ipynb`.

### Schema Artifact: `feature_schema.json`
- **Target:** `fire_next_24h` (binary detection in same 0.25° grid cell during following 24 hours).
- **Model Architecture:** `RandomForestClassifier`.
- **Selected Feature Set:** `+ location` (33 features total: 32 numeric, 1 categorical).
- **Land Cover Categories:** `[1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 16.0, 17.0]`.
- **Decision Threshold:** `0.2507388167388168` (tuned using $F_\beta$ with $\beta=2.0$ on chronological validation set).
- **Probability Correction:**
  - Method: `prior_shift_odds`
  - Effective Training Prior: `0.2715625` (due to balanced sampling)
  - Natural Fire Rate: `0.06044186971301727` (~6.04% natural occurrence in training period)
  - Formula: $\text{odds}_{\text{true}} = \text{odds}_{\text{model}} \times \frac{\pi_{\text{nat}} / (1 - \pi_{\text{nat}})}{\pi_{\text{eff}} / (1 - \pi_{\text{eff}})}$
- **Validated Performance Metrics (Held-out Test Period):**
  - PR-AUC: `0.4863` (vs. no-skill baseline of `~0.060`)
  - ROC-AUC: `0.8844`
  - Precision: `0.2908`
  - Recall: `0.7078`
  - F1-Score: `0.4123`
  - $F_2$-Score: `0.5501`
  - Accuracy: `0.8820`
  - Operational Alert Rate: `14.23%`

---

## 3. Data Artifacts & Access Contract

### Primary Dataset: `master_dataset_2024_2025.nc`
- **Format:** NetCDF-4 / HDF5.
- **File Size:** 434,712,590 bytes (~414.57 MB).
- **Dimensions:**
  - `valid_time`: 17,544 steps (hourly: 2024-01-01 00:00:00 to 2025-12-31 23:00:00 UTC).
  - `latitude`: 39 coordinates ($32.5^\circ\text{N}$ to $42.0^\circ\text{N}$, $0.25^\circ$ spacing).
  - `longitude`: 43 coordinates ($124.5^\circ\text{W}$ to $114.0^\circ\text{W}$, $0.25^\circ$ spacing).
  - Total spatial cells: 1,677 cells covering all of California and immediate borders.
  - Total space-time points: $17,544 \times 1,677 = 29,421,288$ (~29.4M records).
- **Variables:**
  - ERA5 Weather: `u10` (10m eastward wind, m/s), `v10` (10m northward wind, m/s), `t2m` (2m temp, K), `d2m` (2m dewpoint, K), `sp` (surface pressure, Pa), `tp` (total precipitation, m).
  - FIRMS Fire Activity: `fire_now` (active fire detection), `fire_next_24h` (future fire target).
  - MODIS / Static: `land_cover` (MODIS MCD12Q1 category), `elevation` (SRTM, m), `ndvi` (scaled $\times 10^4$), `evi` (scaled $\times 10^4$), `pixel_reliability`, `vi_quality`.
- **Loading Behavior & Constraints:**
  - Loading the entire file into memory as a pandas DataFrame requires $>12\text{ GB}$ of RAM and causes out-of-memory errors on standard environments.
  - Slicing temporally and spatially via `xarray` lazy loading reads only requested subsets in $<100\text{ ms}$ with minimal memory overhead ($<50\text{ MB}$).

---

## 4. Feature Engineering Contract

### Feature List (Exact Order Required by Schema):
1. `u10`
2. `v10`
3. `t2m_celsius` ($T_{2m} - 273.15$)
4. `d2m_celsius` ($D_{2m} - 273.15$)
5. `sp`
6. `tp`
7. `land_cover`
8. `elevation`
9. `ndvi_scaled` ($\text{ndvi} \times 0.0001$)
10. `evi_scaled` ($\text{evi} \times 0.0001$)
11. `pixel_reliability`
12. `vi_quality`
13. `wind_speed` ($\sqrt{u_{10}^2 + v_{10}^2}$)
14. `temp_dewpoint_spread` ($T_{\text{celsius}} - D_{\text{celsius}}$)
15. `month_sin` ($\sin(2\pi \cdot \text{month} / 12)$)
16. `month_cos` ($\cos(2\pi \cdot \text{month} / 12)$)
17. `hour_sin` ($\sin(2\pi \cdot \text{hour}_{\text{UTC}} / 24)$)
18. `hour_cos` ($\cos(2\pi \cdot \text{hour}_{\text{UTC}} / 24)$)
19. `rel_humidity` ($100 \times e_s(D) / e_s(T)$, clipped to $[0, 100]$)
20. `vpd` ($e_s(T) - e_s(D)$ in kPa, clipped at 0; $e_s(T) = 0.6108 \cdot \exp(17.27T / (T+237.3))$)
21. `doy_sin` ($\sin(2\pi \cdot \text{DOY} / 365.25)$)
22. `doy_cos` ($\cos(2\pi \cdot \text{DOY} / 365.25)$)
23. `wind_dir_sin` ($\sin(\text{atan2}(-u_{10}, -v_{10}))$)
24. `wind_dir_cos` ($\cos(\text{atan2}(-u_{10}, -v_{10}))$)
25. `tp_sum_24h` (rolling 24-hour sum of precipitation)
26. `tp_sum_7d` (rolling 168-hour sum of precipitation)
27. `tp_sum_30d` (rolling 720-hour sum of precipitation)
28. `vpd_mean_24h` (rolling 24-hour mean of VPD)
29. `vpd_mean_7d` (rolling 168-hour mean of VPD)
30. `t2m_mean_7d` (rolling 168-hour mean of 2m temperature in °C)
31. `wind_mean_24h` (rolling 24-hour mean of wind speed)
32. `latitude`
33. `longitude`

### Winsorization Bounds:
- `u10`: $[-5.1401, 8.3054]$
- `v10`: $[-6.9422, 7.4034]$
- `d2m`: $[251.7611, 292.8153]$ (K)
- `t2m`: $[262.0136, 317.7478]$ (K)
- `sp`: $[71800.37, 102297.00]$ (Pa)
- `tp`: $[0.0, 0.003897]$ (m)

### Antecedent History Requirement:
- Features 25–31 require 30 days (720 hourly steps) of past weather history for the specific grid cell.
- Because `master_dataset_2024_2025.nc` contains full 2024–2025 hourly records, real antecedent history **is available and computable** for historical point predictions and temporal snapshots across the dataset.
- In manual "single-observation" prediction mode without historical time series, antecedent features cannot be fabricated or arbitrarily guessed. The UI must allow users to either (a) select an actual historical space-time coordinate to load real antecedent history, or (b) adjust recent antecedent indicators with transparent explanations of what history data is being supplied.

---

## 5. Technology Stack & Visualization Architecture

1. **Streamlit**: Multi-page app structure with persistent sidebar branding, caching (`@st.cache_resource`, `@st.cache_data`).
2. **PyDeck / Deck.gl & Plotly**:
   - PyDeck for the **3D Digital Twin**: High-performance GPU-accelerated 3D geographic column / terrain visualizations over Mapbox/Carto basemaps. Allows genuine 3D pitch, bearing, rotation, and elevation extrusion.
   - Plotly for interactive environmental profile charts, ROC/PR curves, feature importance charts, and 2D spatial cross-sections.
3. **Branding & Theme**:
   - Exact brand tokens: Deep Navy (`#102A43`), Main Background (`#081827`), Card Background (`#102A43`), Elevated Surface (`#183B56`), Borders (`#29465E`), Fire Accents (`#E52B24`, `#F4511E`, `#FF7A24`), Crisp typography (`Inter`, `Segoe UI`).
   - Project logo: Solely `wildfire.jpg`, cleanly placed in sidebar and header containers preserving original aspect ratio. All external/NTI logos removed.

---

## 6. Supported vs. Unsupported Capabilities

| Capability | Status | Implementation Details |
|---|---|---|
| **Overview & Executive Dashboard** | Supported | System status, key metrics, model architecture summary, geographic scope. |
| **Wildfire Risk Prediction (Cell-level)** | Supported | Point prediction with complete 33-feature pipeline, threshold comparison, calibrated probability, antecedent history loading. |
| **3D Digital Twin Geographic View** | Supported | Interactive 3D California map (PyDeck) displaying real spatial layers (SRTM elevation, land cover, NDVI, temperature, wind vectors, model risk). Downsampled to active grid for high FPS. |
| **Model Performance & Evaluation** | Supported | Exact held-out test and validation metrics from `feature_schema.json` and notebook (PR-AUC, ROC-AUC, threshold curves, confusion matrix, permutation importance). |
| **Data & Environmental Methodology** | Supported | Deep dive into ERA5, FIRMS, MODIS, SRTM integration, 24h embargo, causal rolling windows, and prior shift calibration. |
| **Physical Wildfire Spread Simulation** | Not Supported | The dataset and model predict same-cell 24-hour occurrence, not thermodynamic fire front propagation or CFD flame physics. App explicitly clarifies this distinction. |
| **Live Satellite Telemetry Streaming** | Not Supported | Offline 2024–2025 reanalysis dataset. App clearly documents temporal boundaries. |

---

## 7. Implementation Plan

1. **Core Utilities (`src/`):**
   - `src/theme.py`: Design tokens, dark mode CSS injection, custom component styling.
   - `src/model_loader.py`: Safe cached loading of `wildfire_model_pipeline.joblib` and `feature_schema.json`, score computation, odds calibration, risk classification.
   - `src/data_loader.py`: Efficient `xarray` lazy loader, time/space slice extraction, antecedent history computation, grid aggregation.
2. **Interactive 3D Digital Twin (`src/twin_3d.py`):**
   - PyDeck 3D grid column layer with real SRTM elevation extrusion, risk color-coding, interactive pitch/rotation controls, tooltip inspection, and multi-layer overlay (Elevation, Land Cover, Temperature, Wind, Risk).
3. **Streamlit Multi-Page Interface (`app.py` and navigation):**
   - Page 1: **Overview** — System overview, California map framing, KPI summary, operational alerts.
   - Page 2: **Wildfire Prediction** — Cell-level inference, preset historical scenarios, real antecedent calculation, risk gauge, factor contribution breakdown.
   - Page 3: **3D Digital Twin** — Full-screen interactive 3D spatial twin with layer toggles, timestamp selection, and spatial filters.
   - Page 4: **Model Performance** — Test & validation benchmarks, PR/ROC curves, threshold analysis, permutation feature importance.
   - Page 5: **Data & Methodology** — Multi-sensor integration details, causal rolling windows, prior odds correction.
   - Page 6: **About & Operational Scope** — System architecture, ethical AI guidelines, and operational boundaries.
4. **Deployment & Packaging:**
   - `requirements.txt`: Pin compatible versions.
   - `.streamlit/config.toml`: Dark theme matching brand palette.
   - `.gitignore`: Ignore large raw files, caches, virtual environments.
   - `README.md` & `DEPLOYMENT.md`: Comprehensive local setup and hosting guide.
