# Deployment & Production Architecture Guide

**Project:** California Wildfire Prediction System  
**Version:** 1.0.0 (Production-Ready)

---

## 1. Hosting Platform Evaluation

### Recommended Hosting Options

| Hosting Platform | Suitability | RAM Limit | Large File Handling | Notes |
|---|---|---|---|---|
| **Streamlit Community Cloud** | Suitable with Git LFS / External Storage | ~1 GB RAM | Git LFS or S3 download | Fits model (91 MB) and schema (4 KB). For NetCDF (414 MB), uses embedded parquet fallbacks or external blob storage. |
| **Hugging Face Spaces (Docker)** | Highly Recommended | 16 GB RAM (Free) | Direct upload or Git LFS | Effortlessly loads full 414 MB NetCDF in memory without resource starvation. |
| **Google Cloud Run / AWS ECS** | Recommended for Enterprise | Configurable (2–8 GB) | Mounted Cloud Storage Bucket | Predictable latency, autoscaling, containerized isolation. |
| **Self-Hosted VPS (Linux/Ubuntu)** | Highly Recommended | 4+ GB RAM | Local disk storage | Direct checkout, instant execution, zero cloud memory caps. |
| **Vercel (Serverless)** | ❌ Not Suitable for ML Backend | 250 MB Function Cap | Not Supported | Bundle limit (<250 MB) rejects scikit-learn+scipy+pandas (~450 MB) and 91 MB model. See `ARCHITECTURE_AND_VERCEL_EVALUATION.md`. |

*(For a comprehensive technical analysis of decoupled architectures and Vercel hosting constraints, see [ARCHITECTURE_AND_VERCEL_EVALUATION.md](ARCHITECTURE_AND_VERCEL_EVALUATION.md)).*

---

## 2. Model & Data Asset Checklist

Before launching in any target environment, verify that the following files exist in the project root:

```text
├── feature_schema.json               # Required (4.33 KB) - Model feature contract & metadata
├── wildfire_model_pipeline.joblib    # Required (91.01 MB) - Trained scikit-learn pipeline
├── wildfire.png / wildfire.jpg       # Required - Official project branding logo
├── assets/
│   ├── curated_snapshots.parquet     # Required (492 KB) - 3D Digital Twin snapshot fallback
│   └── california_grid_static.parquet# Required (24 KB) - Grid elevation & static features fallback
└── master_dataset_2024_2025.nc       # Optional (414.57 MB) - Full NetCDF dataset (temporal queries)
```

*Note: If `master_dataset_2024_2025.nc` is absent on lean deployments (such as Streamlit Community Cloud with strict memory limits), the app utilizes `assets/curated_snapshots.parquet` and `assets/california_grid_static.parquet` for immediate statewide risk rendering and Open-Meteo API for real-time predictions without crashing or missing data.*

---

## 3. Docker Deployment (Containerization)

A production-grade `Dockerfile` is provided for containerized deployments:

```dockerfile
FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

WORKDIR /app

# Install system dependencies for NetCDF & C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    libnetcdf-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and assets
COPY assets/ ./assets/
COPY src/ ./src/
COPY pages_impl/ ./pages_impl/
COPY .streamlit/ ./.streamlit/
COPY app.py feature_schema.json wildfire_model_pipeline.joblib wildfire.png wildfire.jpg ./

# Optionally copy NetCDF dataset (or mount via volume)
COPY master_dataset_2024_2025.nc ./

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Build & Run Commands
```bash
# Build Docker image
docker build -t california-wildfire-app:latest .

# Run container locally on port 8501
docker run -p 8501:8501 -v $(pwd)/master_dataset_2024_2025.nc:/app/master_dataset_2024_2025.nc california-wildfire-app:latest
```

---

## 4. Local Execution & Validation

### Start Command
```powershell
& .\.venv\Scripts\streamlit.exe run app.py
```
Or with standard PATH configuration:
```bash
streamlit run app.py
```

### Verification Endpoints
- **Application UI:** `http://localhost:8501`
- **Streamlit Healthcheck:** `http://localhost:8501/_stcore/health`

---

## 5. Memory & Performance Safeguards

1. **Lazy xarray Loading:** The dataset is never dumped as a monolithic 29M-row dataframe. Slices are extracted using indexers (`.sel()`), materializing only 1,225 terrestrial cells per timestamp.
2. **Immutable Caching:**
   - Model and schema are loaded once per process lifecycle via `@st.cache_resource`.
   - Spatial slices and 30-day antecedent sequences are cached via `@st.cache_data`.
3. **GPU WebGL Acceleration:** PyDeck utilizes client-side WebGL shaders for high-frame-rate 3D column extrusion, offloading geospatial rendering from server CPU/memory.

---

## 6. Map Tile Providers, Attribution & Zero-Key Configuration

The application is engineered to operate out-of-the-box with **zero paid API keys or proprietary map accounts**.

### Supported Basemap Providers

| Basemap Style | Default Provider | Requirement | Attribution & Fair-Use Terms |
|---|---|---|---|
| **Satellite Imagery** | Esri World Imagery (ArcGIS MapServer) | **No Key Required** | © Esri, Maxar, Earthstar Geographics, USDA FSA, USGS, Aerogrid, IGN, IGP, and GIS User Community. Free for public non-commercial web visualization. |
| **Terrain & Roads (Voyager)** | CARTO Voyager (Vector/Raster GL) | **No Key Required** | © [CARTO](https://carto.com/attributions), © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors. |
| **Carto Positron (Light)** | CARTO Positron | **No Key Required** | © CARTO, © OpenStreetMap contributors. |
| **Carto Dark** | CARTO Dark Matter | **No Key Required** | © CARTO, © OpenStreetMap contributors. |
| **Custom Mapbox** | Mapbox GL Styles | *Optional API Key* | If set via `MAPBOX_API_KEY` environment variable or `.streamlit/secrets.toml`, Mapbox custom vector styles will be loaded. Not required. |

### Environment Variables & Secrets Policy
- **Never commit private API keys:** No API keys are hard-coded in the repository or committed to git.
- **Graceful Fallbacks:** If network connectivity to tile servers is restricted, PyDeck renders the geospatial geometry grid on a neutral dark coordinate canvas without crashing. Open-Meteo queries fallback gracefully to historical cached distributions.
