# Model Update Audit & Deployment Contract

**Project:** California Wildfire Prediction System & 3D Digital Twin  
**Date:** September 26, 2026  
**Auditor:** Senior ML Engineer & Deployment Specialist  
**Workspace:** `c:\Users\ahmed\Downloads\California_WildFire_Prediction`  
**Status:** Audit & Deployment Integration Complete

---

## 1. Notebook Identification & Audit

### Source Notebook Candidates Evaluated
1. **`California_Wildfire_ML_final_5_improved.ipynb`** (Project Root, 1,690,384 bytes, Modified 2026-09-26 04:06:37):
   - **Verdict: CANONICAL SOURCE OF TRUTH.**
   - Complete 99-cell ML workflow with end-to-end training, time-series embargoed splits, ablation experiments, hyperparameter search, probability calibration, SHAP interpretation, and artifact serialization.
   - Refits the winning tuned `RandomForestClassifier` on 1,200,000 train+validation rows and evaluates on a strictly held-out chronological test partition (400,000 rows).
   - Exports the pipeline to `wildfire_model_pipeline.joblib` and metadata schema to `feature_schema.json`.

2. **`../Projeyy/California_Wildfire_ML_improved_v2.ipynb`** (Sibling directory, 1,479,180 bytes):
   - Older iteration lacking the refined clean feature definitions and SHAP analysis. Included experimental multi-scale history features (`fire_freq_90d`, `fire_freq_90d_smooth`) that were removed in the final project iteration due to data leakage / operational latency considerations.

3. **`../wildfire_ml(1).ipynb` & `../wildfire_ml(1) - Copy copy 2.ipynb`**:
   - Earlier exploratory notebooks focusing on baseline models and preliminary XGBoost experiments without the rigorous prior-shift odds calibration or 3D Digital Twin coordinate alignment.

---

## 2. Updated Model Contract & Architecture

### Artifact Specifications
- **Pipeline Artifact:** [`wildfire_model_pipeline.joblib`](file:///c:/Users/ahmed/Downloads/California_WildFire_Prediction/wildfire_model_pipeline.joblib) (~91.02 MB).
- **Schema Artifact:** [`feature_schema.json`](file:///c:/Users/ahmed/Downloads/California_WildFire_Prediction/feature_schema.json) (~4.33 KB).
- **Archived Previous Artifacts:** Preserved in [`models_backup/`](file:///c:/Users/ahmed/Downloads/California_WildFire_Prediction/models_backup/).

### Pipeline Composition
The serialized artifact is an all-inclusive scikit-learn `Pipeline` bundling preprocessing and classification:
```text
Pipeline(steps=[
    ('preprocessor', ColumnTransformer(transformers=[
        ('numeric', StandardScaler(), 32 features),
        ('categorical', OneHotEncoder(categories=[...], dtype=float32, handle_unknown='ignore'), ['land_cover'])
    ])),
    ('model', RandomForestClassifier(
        n_estimators=200,
        max_features=0.5,
        max_samples=0.3,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ))
])
```

### Input Feature Contract (Exact Schema Order — 33 Features)
1. `u10`: 10m U-wind component (m/s)
2. `v10`: 10m V-wind component (m/s)
3. `t2m_celsius`: 2m temperature in °C ($T_{2m} - 273.15$)
4. `d2m_celsius`: 2m dewpoint temperature in °C ($D_{2m} - 273.15$)
5. `sp`: Surface air pressure (Pa)
6. `tp`: Hourly total precipitation (m)
7. `land_cover`: MODIS MCD12Q1 land cover classification category (1.0 to 17.0)
8. `elevation`: SRTM digital elevation (m)
9. `ndvi_scaled`: Normalized Difference Vegetation Index ($\text{raw} \times 10^{-4}$)
10. `evi_scaled`: Enhanced Vegetation Index ($\text{raw} \times 10^{-4}$)
11. `pixel_reliability`: MODIS pixel quality assessment flag
12. `vi_quality`: MODIS VI quality bitmask
13. `wind_speed`: Magnitude $\sqrt{u_{10}^2 + v_{10}^2}$ (m/s)
14. `temp_dewpoint_spread`: $T_{2m} - D_{2m}$ (°C)
15. `month_sin`: $\sin(2\pi \cdot \text{month} / 12)$
16. `month_cos`: $\cos(2\pi \cdot \text{month} / 12)$
17. `hour_sin`: $\sin(2\pi \cdot \text{hour}_{\text{UTC}} / 24)$
18. `hour_cos`: $\cos(2\pi \cdot \text{hour}_{\text{UTC}} / 24)$
19. `rel_humidity`: Relative humidity (%), $100 \times e_s(D_{2m})/e_s(T_{2m})$ clipped to $[0, 100]$
20. `vpd`: Vapor Pressure Deficit (kPa), $\max(0, e_s(T_{2m}) - e_s(D_{2m}))$
21. `doy_sin`: $\sin(2\pi \cdot \text{DOY} / 365.25)$
22. `doy_cos`: $\cos(2\pi \cdot \text{DOY} / 365.25)$
23. `wind_dir_sin`: $\sin(\text{atan2}(-u_{10}, -v_{10}))$
24. `wind_dir_cos`: $\cos(\text{atan2}(-u_{10}, -v_{10}))$
25. `tp_sum_24h`: 24-hour antecedent rainfall accumulation (m)
26. `tp_sum_7d`: 7-day antecedent rainfall accumulation (m)
27. `tp_sum_30d`: 30-day antecedent rainfall accumulation (m)
28. `vpd_mean_24h`: 24-hour rolling mean VPD (kPa)
29. `vpd_mean_7d`: 7-day rolling mean VPD (kPa)
30. `t2m_mean_7d`: 7-day rolling mean 2m temperature (°C)
31. `wind_mean_24h`: 24-hour rolling mean wind speed (m/s)
32. `latitude`: Grid cell centroid latitude (°N)
33. `longitude`: Grid cell centroid longitude (°W)

### Winsorization Bounds (Training Distribution 0.1% – 99.9% Quantiles)
- `u10`: `[-5.1401, 8.3054]`
- `v10`: `[-6.9422, 7.4034]`
- `d2m`: `[251.7611, 292.8153]` (K)
- `t2m`: `[262.0136, 317.7478]` (K)
- `sp`: `[71800.37, 102297.00]` (Pa)
- `tp`: `[0.0, 0.0039]` (m)

### Target & Decision Threshold
- **Target:** `fire_next_24h` (Binary: 1 if active satellite thermal anomaly detected in the 0.25° cell during the subsequent 24 hours, 0 otherwise).
- **Decision Threshold:** `0.2507388167388168`.
- **Threshold Selection Strategy:** $F_\beta$-score maximization ($\beta=2.0$) on the chronological validation split to prioritize recall (identifying actual fire outbreaks) while controlling false alarm volume.

### Probability Calibration: Prior-Shift Odds Correction
Because training balances fire positive and negative rows (~22.6% - 27.2% effective fire prevalence), the raw model probabilities overestimate natural real-world occurrence (~6.04% natural fire rate).
The deployment code applies the exact prior-shift odds re-scaling:
$$\text{odds}_{\text{eff}} = \frac{p_{\text{raw}}}{1 - p_{\text{raw}}}$$
$$\text{odds}_{\text{true}} = \text{odds}_{\text{eff}} \times \frac{\pi_{\text{nat}} / (1 - \pi_{\text{nat}})}{\pi_{\text{eff}} / (1 - \pi_{\text{eff}})}$$
$$p_{\text{calibrated}} = \frac{\text{odds}_{\text{true}}}{1 + \text{odds}_{\text{true}}}$$
- **Training Effective Prior ($\pi_{\text{eff}}$):** `0.2715625`
- **Natural Fire Rate ($\pi_{\text{nat}}$):** `0.06044186971301727`

---

## 3. Validated Benchmark Metrics (Held-Out Test Partition)

Evaluation strictly conducted on 400,000 held-out test rows following a 24-hour chronological causal embargo:

| Metric | Score (Tuned $\tau=0.2507$) | Baseline ($\tau=0.5000$) |
|---|---|---|
| **PR-AUC (Average Precision)** | **0.4863** | 0.4863 (Invariant) |
| **ROC-AUC** | **0.8844** | 0.8844 (Invariant) |
| **Recall (Sensitivity)** | **70.78%** | 51.63% |
| **Precision** | **29.08%** | 49.56% |
| **$F_1$-Score** | **0.4123** | 0.5058 |
| **$F_2$-Score** | **0.5501** | 0.5121 |
| **Accuracy** | **88.20%** | 94.10% |
| **Alert Rate** | **14.23%** | 6.09% |
| **No-Skill PR-AUC Baseline** | **0.0585** | 0.0585 |

---

## 4. Deployment Integration & Enhancements

1. **Clean Scikit-Learn 1.9.1 Serialization:**
   - Re-serialized [`wildfire_model_pipeline.joblib`](file:///c:/Users/ahmed/Downloads/California_WildFire_Prediction/wildfire_model_pipeline.joblib) using the active virtual environment's scikit-learn version, resolving all `InconsistentVersionWarning` unpickling messages.
2. **Dynamic Mtime Cache Invalidation:**
   - Updated [`src/model_loader.py`](file:///c:/Users/ahmed/Downloads/California_WildFire_Prediction/src/model_loader.py) with `_get_file_mtime()`-aware `@st.cache_resource` decorators. When model or schema files on disk are modified, Streamlit automatically purges stale cached pipelines and loads the new artifacts.
3. **Mutual Compatibility Validator:**
   - Implemented `validate_model_schema_compatibility(pipeline, schema)` ensuring that pipeline steps, preprocessor column alignments, and binary class contracts are verified at load time.
4. **Strict Input Validation:**
   - Added `validate_raw_inputs()` within `prepare_feature_row()`. Rejects missing required parameters or non-finite values (`NaN`, `inf`) with descriptive `ValueError` exceptions instead of silently filling invented defaults.
5. **Preservation of UI & Layout:**
   - All visual styling, dark theme tokens, sidebar compact navigation, PNG logo, and 3D deck.gl Digital Twin integration remain 100% intact.

---

## 5. Verification Test Log

- **Test 1 — Model & Schema Loading:** Clean import with 0 unpickling warnings (`PASS`).
- **Test 2 — Schema Compatibility:** Validated 33 features against `ColumnTransformer` (`PASS`).
- **Test 3 — Real Row Generation:** Extracted slice from [`master_dataset_2024_2025.nc`](file:///c:/Users/ahmed/Downloads/California_WildFire_Prediction/master_dataset_2024_2025.nc) and engineered 33 features (`PASS`).
- **Test 4 — Real Inference:** Output `raw_score=0.2635`, `calibrated_prob=0.0582`, `is_alert=True`, `risk_level='High'` (`PASS`).
- **Test 5 — Threshold & Prior Shift:** Calibrated odds and categorical bins verified across operating points (`PASS`).
- **Test 6 — Missing File Graceful Degradation:** Clean UI error messages returned on missing files or invalid inputs (`PASS`).
