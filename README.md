# Wildfire Intelligence & Early Warning Dashboard

An enterprise-grade, high-performance geospatial and AI analytics dashboard built with **Python Dash + Plotly** for monitoring national wildfire patterns, pre-fire environmental danger indices, and historical fire-severity dynamics.

---

## 🌟 Complete Architecture Overview

The system spans 6 integrated operational modules:

1. **Command Center**:
   - **Executive KPI Cards**: Real-time aggregation of Total Fire Incidents ($73,221$), Total Burned Acreage, Largest Single Incident, Escaped / Large Fire % ($\ge 10$ acres), Mean Pre-Fire Burning Index, and Active Satellite Detections.
   - **Interactive Geospatial Wildfire Map**: High-resolution US map showing historical wildfire incidents color-coded by operational severity (*Low: <10 ac*, *Moderate: 10–99 ac*, *Severe: ≥100 ac*) with circle size dynamically scaled to fire acreage.
   - **Satellite Thermal Hotspot Layer**: Suomi VIIRS Near Real Time (NRT) satellite thermal anomaly overlay displaying Brightness Temperature ($T_{i4}$) and Fire Radiative Power (FRP in MW).
   - **Temporal Fire Siege Trends**: Monthly breakdown of wildfire volume against total consumed acreage, highlighting summer-autumn peak periods.
   - **Ignition Cause Breakdown**: Frequency and acreage impact sorted by NWCG general causes (Arson, Lightning, Equipment, Debris Burning, etc.).
   - **Interactive Multi-Filter Toolbar**: Filter dynamically by State, Severity Tier, Ignition Cause, and toggle satellite overlays.

2. **Environmental & Fire Impact Explorer**:
   - **Heavy-Tailed Burned Area Distribution**: Log-scale histogram illustrating power-law wildfire dynamics (where $<3\%$ of fires cause $>90\%$ of acreage damage).
   - **Pre-Fire Danger Indices Boxplots**: Comparative distributions of pre-fire meteorological variables (Burning Index `bi`, Energy Release Component `erc`, Temperature, Vapor Pressure Deficit `vpd`, Wind Speed `vs`, Relative Humidity `rmin`) across severity tiers.
   - **Spearman Correlation Matrix**: Non-linear correlation analysis between environmental indices and fire size.
   - **Scientific Methodology Distinction**: Clear operational boundary defining historical severity classification vs. future occurrence forecasting.

3. **AI Model Comparison**:
   - **Chronological Leakage-Free Split**: Training on Jan 1 – Aug 31, 2020 ($56,669$ incidents); held-out evaluation on Sep 1 – Dec 31, 2020 ($16,552$ incidents).
   - **3 Advanced Architectures**: Side-by-side performance benchmarks for **LightGBM**, **XGBoost**, and **CatBoost**.
   - **Normalized Confusion Matrices**: Class-by-class evaluation revealing true positives, containment errors, and severe escape recall.
   - **Primary Metric Justification**: Prioritizes **Macro F1 ($0.4228$)** over raw accuracy ($80-81\%$) due to the 89% class imbalance of small fires.

4. **Interactive AI Prediction Simulator**:
   - **Pre-Fire Environmental Sliders**: Temperature, wind speed, relative humidity, Burning Index, Energy Release Component, and Vapor Pressure Deficit.
   - **Real-Time Severity Inference**: Instant probability breakdown across Low, Moderate, and Severe classes.
   - **Operational Containment Advisory**: Tactical guidance based on growth potential.

5. **Model Explainability & TreeSHAP**:
   - **Global Feature Importance**: Identifies top meteorological drivers across the test set.
   - **Beeswarm Summary Plot**: Illustrates direction of effect (how high BI and dry fuel drive severe fire escalation).
   - **Local Instance Waterfall**: Real-time decomposition of positive and negative drivers for the current simulation profile.

6. **Pretrained YOLO Fire & Smoke Vision Detection**:
   - **Ultralytics Weights Integration**: Direct surveillance inference powered by [`models/best.pt`](file:///C:/Users/palva/.gemini/antigravity/scratch/wildfire_dashboard/models/best.pt) (classes: `smoke`, `fire`).
   - **Dual Ingestion**: Preset aerial and mountain surveillance scenarios or drag-and-drop custom image uploads.
   - **Bounding Box Annotation**: Fire highlighted in red, smoke highlighted in cyan with confidence percentages and automated hazard alerts.

---

## 📁 Directory Structure

```
wildfire_dashboard/
├── assets/
│   ├── style.css                   # Glassmorphic dark theme & responsive layout
│   └── samples/                    # Preloaded high-resolution field surveillance images
├── data/
│   ├── wildfires_processed.csv     # Cleaned full 73,221-incident FPA FOD 2020 dataset
│   ├── wildfires_map_sample.csv    # Stratified high-performance map sample (15,914 incidents)
│   └── viirs_hotspots.csv          # Suomi VIIRS NRT active satellite detections (460 hotspots)
├── models/
│   ├── best.pt                     # Pretrained YOLO fire/smoke weights
│   ├── lightgbm_model.joblib       # Best trained tabular model (Macro F1: 0.4228)
│   ├── xgboost_model.joblib        # Trained XGBoost classifier
│   ├── catboost_model.joblib       # Trained CatBoost classifier
│   ├── feature_encoder.joblib      # Categorical encoder
│   ├── model_metrics.json          # Cached empirical evaluation metrics & confusion matrices
│   └── shap_cache.json             # Precomputed TreeSHAP values for instant rendering
├── src/
│   ├── data_loader.py              # Ingestion, cleaning, feature engineering, and caching
│   ├── models_trainer.py           # Chronological training, evaluation & inference pipeline
│   ├── explainability.py           # TreeSHAP computation & visualization generators
│   ├── yolo_detector.py            # Computer vision detection & bounding box annotator
│   └── visualizations.py           # Plotly geospatial maps, charts, and metrics
├── app.py                          # Master Dash application entry point
├── requirements.txt                # Python package dependencies
└── README.md                       # Comprehensive guide
```

---

## 🚀 Windows Installation & Launch

Open PowerShell:

```powershell
# 1. Navigate to workspace
cd C:\Users\palva\.gemini\antigravity\scratch\wildfire_dashboard

# 2. Launch the application
python app.py
```

Open your browser and navigate to:
**`http://127.0.0.1:8050/`**
