# Deployment & Production Architecture Guide

**Project:** California Wildfire Prediction System  
**Version:** 1.0.0 (Production-Ready)

---

## 1. Hosting Platform Evaluation

### Recommended Hosting Options

| Hosting Platform | Suitability | RAM Limit | Large File Handling | Notes |
|---|---|---|---|---|
| **Streamlit Community Cloud** | Suitable with Git LFS / External Storage | ~1 GB RAM | Git LFS or S3 download | Fits model (91 MB) and schema (4 KB). For NetCDF (414 MB), use external blob storage (AWS S3 or GCS) or compact extract. |
| **Hugging Face Spaces (Docker)** | Highly Recommended | 16 GB RAM (Free) | Direct upload or Git LFS | Effortlessly loads full 414 MB NetCDF in memory without resource starvation. |
| **Google Cloud Run / AWS ECS** | Recommended for Enterprise | Configurable (2–8 GB) | Mounted Cloud Storage Bucket | Predictable latency, autoscaling, containerized isolation. |
| **Self-Hosted VPS (Linux/Ubuntu)** | Highly Recommended | 4+ GB RAM | Local disk storage | Direct checkout, instant execution, zero cloud memory caps. |

---

## 2. Model & Data Asset Checklist

Before launching in any target environment, verify that the following files exist in the project root:

```text
├── feature_schema.json               # Required (4.33 KB) - Model feature contract & metadata
├── wildfire_model_pipeline.joblib    # Required (91.01 MB) - Trained scikit-learn pipeline
├── wildfire.jpg                      # Required (520 KB) - Official project logo
└── master_dataset_2024_2025.nc       # Recommended (414.57 MB) - Required for 3D Twin & spatial maps
```

*Note: If `master_dataset_2024_2025.nc` is absent on lean deployments, the app degrades gracefully: point predictions, model evaluation metrics, and methodology remain fully functional, while the 3D twin and overview map display an informative missing-dataset guide.*

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
COPY src/ ./src/
COPY pages_impl/ ./pages_impl/
COPY .streamlit/ ./.streamlit/
COPY app.py feature_schema.json wildfire_model_pipeline.joblib wildfire.jpg ./

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
