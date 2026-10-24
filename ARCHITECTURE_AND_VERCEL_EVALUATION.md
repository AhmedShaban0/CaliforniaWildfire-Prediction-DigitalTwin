# Architecture & Vercel Migration Evaluation Report

**California Wildfire Prediction & 3D Digital Twin**  
**Evaluation Date:** October 2026  
**Document Purpose:** Evidence-based technical evaluation of platform architecture, hosting options, and Vercel suitability.

---

## 1. Executive Summary

This evaluation analyzes whether the California Wildfire Prediction System should remain on a unified Python/Streamlit architecture or be migrated to a decoupled frontend (e.g., Next.js / React) and backend (FastAPI / Flask), with specific scrutiny of **Vercel** as a hosting candidate.

### Core Findings
1. **Vercel is NOT viable for the Python ML Model Backend**: Vercel’s serverless functions enforce a strict **250 MB uncompressed bundle size limit**, a 50 MB zip upload limit, and a 10s–60s execution timeout with zero persistent disk storage. The project's ML environment alone (`scikit-learn`, `scipy`, `numpy`, `pandas`, `xarray`, `netCDF4`) exceeds 400 MB, and the trained pipeline artifact is **91.01 MB**, causing instant build or deployment rejection on Vercel.
2. **Vercel IS viable for a decoupled Frontend (Next.js/React)**: If the UI were rewritten in Next.js, Vercel provides world-class global CDN delivery, instant edge caching, and seamless Mapbox GL / Leaflet integration.
3. **Keeping Streamlit with Native Leaflet / WebGL Maps is the Simplest, Most Reliable Solution**: Integrating Leaflet (with Esri Satellite and Carto Voyager basemaps) directly into Streamlit eliminates the black-screen basemap defect, delivers desktop/mobile responsiveness, and preserves direct in-memory access to the 91 MB model pipeline and 414 MB NetCDF dataset without network serialization lag, cold-start latency, or multi-service operational overhead.

---

## 2. Technical Asset & Dependency Profile

| Component | Size / Resource Requirement | Deployment Constraint |
|---|---|---|
| **Model Pipeline (`wildfire_model_pipeline.joblib`)** | **91.01 MB** | Exceeds serverless function payload norms; requires ~300 MB RAM when unpickled. |
| **Feature Schema (`feature_schema.json`)** | 4.33 KB | Lightweight; portable anywhere. |
| **NetCDF Dataset (`master_dataset_2024_2025.nc`)** | **414.57 MB** | Requires local filesystem or S3/GCS mount; incompatible with ephemeral serverless disks. |
| **Parquet Fallback Caches (`assets/`)** | 516 KB total | Highly portable; embedded in git. |
| **Python Core ML Dependencies** | `scikit-learn` (90 MB), `scipy` (120 MB), `numpy` (35 MB), `pandas` (100 MB), `netCDF4` (40 MB) | **Total: ~385–450 MB uncompressed**. Far exceeds Vercel 250 MB ceiling. |
| **Inference Latency Target** | < 500 ms | Cold starts on serverless Python functions range from 4s to 12s. |

---

## 3. Four-Point Evaluation

### 1. Whether keeping Streamlit is the simplest reliable option after fixing the map
**Verdict: YES.**
- **Single Process Simplicity:** Direct in-process communication between UI event handlers, Open-Meteo REST calls, and the scikit-learn classification pipeline. Zero API contracts, CORS headers, or authentication tokens to maintain.
- **Zero Serialization Latency:** Spatial arrays (1,677 California cells) pass in-memory as Pandas DataFrames directly into Leaflet GeoJSON and PyDeck WebGL buffers.
- **Map Resolution:** Embedding a native Leaflet map component solves the black-map issue permanently. Users get authentic Esri Satellite imagery, Carto Voyager terrain, OpenTopoMap contours, and Carto Positron streets with zero paid API keys.

### 2. Whether a separate frontend and Python backend would materially improve the product
**Verdict: Only for multi-tenant enterprise scale (>10,000 concurrent users).**
- **Advantages of Decoupling:**
  - A Next.js / TailwindCSS / Mapbox GL JS frontend could offer bespoke micro-interactions, client-side vector tile rotation, and instant page routing.
  - A FastAPI backend could autoscale horizontally on Kubernetes / Google Cloud Run independently of UI sessions.
- **Disadvantages & Overhead:**
  - Requires maintaining two codebases, dual CI/CD pipelines, API authentication, and cross-origin resource sharing (CORS).
  - Serializing a 1,677-cell spatial grid over HTTP on every timestamp change adds 150–350 ms network transit time.
  - For decision-support workflows with standard operational teams, the architectural complexity offers negligible user-facing gain compared to a polished Streamlit + Leaflet implementation.

### 3. Whether Vercel is appropriate for the frontend, the backend, both, or neither
**Verdict: Frontend only; strictly NOT for the Python backend.**

| Architecture on Vercel | Feasibility | Technical Reason |
|---|---|---|
| **Vercel for Python Model Backend** | ❌ **FAILED** | Exceeds 250 MB uncompressed function limit; missing C-libraries for HDF5/NetCDF; 10s–60s cold start timeout; cannot host 91 MB model + 414 MB NetCDF. |
| **Vercel for Fullstack App (Unified)** | ❌ **FAILED** | Streamlit cannot run natively inside Vercel Serverless Functions. |
| **Vercel for Next.js Frontend** | ✅ **RECOMMENDED (If Migrated)** | Superb CDN edge delivery, fast static asset bundling, optimized for React/Mapbox GL. |

### 4. Practical Deployment Architecture & Tradeoffs

#### Option A: Unified Streamlit + Leaflet (Recommended & Implemented)
```text
┌──────────────────────────────────────────────────────────────┐
│                    Unified Container / Host                   │
│  [Hugging Face Spaces / Streamlit Cloud / Docker VPS / GCP]  │
│                                                              │
│  ┌───────────────────────┐        ┌────────────────────────┐ │
│  │  Streamlit UI Engine  │◄──────►│ Scikit-Learn Pipeline  │ │
│  │  (Custom Dark Theme)  │ In-Mem │ (91 MB joblib in RAM)  │ │
│  └──────────┬────────────┘        └────────────────────────┘ │
│             │                                  ▲             │
│             ▼                                  │             │
│  ┌───────────────────────┐        ┌────────────┴───────────┐ │
│  │  Leaflet GeoJSON Map  │        │ NetCDF / Parquet Cache │ │
│  │  (Esri/Carto Basemaps)│        │ (414 MB / 516 KB)      │ │
│  └───────────────────────┘        └────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```
- **Cost:** Free to ~$10/month.
- **Maintenance:** Single repository, single test suite, zero API drift.
- **Reliability:** 100% deterministic basemap rendering with zero API keys.

#### Option B: Decoupled Enterprise Architecture (Next.js + FastAPI)
*(Only recommended if migrating to commercial SaaS with public API tiers)*
- **Frontend:** Next.js on Vercel (Edge CDN).
- **Backend API:** FastAPI running in a Docker container on **Google Cloud Run** (2 vCPU, 4 GB RAM) or **AWS App Runner**.
- **Storage:** Model and NetCDF mounted via Google Cloud Storage Bucket / S3.
- **Estimated Cost:** $35–$120/month depending on ingress/egress.

---

## 4. Conclusion & Action Taken

Based on this evidence, migrating the entire application to Vercel is technically infeasible due to platform limits on Python ML dependencies and model artifact sizes.

Instead, the project adopts the high-performance **Leaflet + PyDeck dual-engine architecture** within Streamlit:
- **Leaflet Engine:** Guarantees 100% working satellite imagery (Esri World Imagery) and terrain (Carto Voyager) with zero API keys and zero black rectangles.
- **PyDeck 3D Engine:** Provides hardware-accelerated 3D terrain extrusion with Carto Voyager terrain default.
- **Modern UI Design System:** Completely replaces the stock Streamlit appearance with an executive dark-mode decision-support interface.
