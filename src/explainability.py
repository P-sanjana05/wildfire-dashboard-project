"""
Wildfire Intelligence & Early Warning Dashboard - Model Explainability & SHAP
Computes and visualizes TreeSHAP values for feature importance and local instance contributions.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
DATA_PATH = os.path.join(BASE_DIR, "data", "wildfires_processed.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SHAP_CACHE_PATH = os.path.join(MODELS_DIR, "shap_cache.json")
ENCODER_PATH = os.path.join(MODELS_DIR, "feature_encoder.joblib")

from src.models_trainer import FEATURES, CLASS_NAMES

FEATURE_LABELS = {
    'bi': 'Burning Index (BI)',
    'erc': 'Energy Release Component (ERC)',
    'temp_max_c': 'Maximum Temperature (°C)',
    'vpd': 'Vapor Pressure Deficit (VPD)',
    'vs': 'Wind Speed (m/s)',
    'rmin': 'Minimum Relative Humidity (%)',
    'rmax': 'Maximum Relative Humidity (%)',
    'fm100': '100-Hour Fuel Moisture (%)',
    'fm1000': '1000-Hour Fuel Moisture (%)',
    'Elevation': 'Elevation (m)',
    'Slope': 'Slope (°)',
    'Aspect': 'Aspect (°)',
    'TRI': 'Terrain Ruggedness (TRI)',
    'TPI': 'Topographic Position (TPI)',
    'pr': 'Precipitation (mm)',
    'temp_min_c': 'Minimum Temperature (°C)',
    'temp_range_c': 'Diurnal Temperature Range (°C)',
    'DISCOVERY_DOY': 'Day of Year',
    'Month': 'Month',
    'STATE': 'State',
    'NWCG_GENERAL_CAUSE': 'Ignition Cause'
}

def compute_and_cache_shap(sample_size=350, force_recompute=False):
    """
    Computes TreeSHAP values on held-out test data and caches the results
    so Dash renders the explainability dashboard in milliseconds.
    """
    if os.path.exists(SHAP_CACHE_PATH) and not force_recompute:
        with open(SHAP_CACHE_PATH, 'r') as f:
            return json.load(f)

    print(f"[SHAP] Computing TreeSHAP values on {sample_size} test incidents...")
    df = pd.read_csv(DATA_PATH)
    test_df = df[df['DISCOVERY_DATE'] >= '2020-09-01'].sample(n=min(sample_size, len(df[df['DISCOVERY_DATE'] >= '2020-09-01'])), random_state=42)

    encoder = joblib.load(ENCODER_PATH)
    test_df[['STATE', 'NWCG_GENERAL_CAUSE']] = encoder.transform(test_df[['STATE', 'NWCG_GENERAL_CAUSE']])

    X_sample = test_df[FEATURES]
    model = joblib.load(os.path.join(MODELS_DIR, "lightgbm_model.joblib"))

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)

    # shap_vals has shape (n_samples, n_features, n_classes)
    if isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
        # Class 2 is Severe (>=100 acres)
        severe_shap = shap_vals[:, :, 2]
        # Mean absolute SHAP across all classes
        mean_abs_all = np.mean(np.abs(shap_vals), axis=(0, 2))
        mean_abs_severe = np.mean(np.abs(severe_shap), axis=0)
    elif isinstance(shap_vals, list):
        severe_shap = shap_vals[2]
        mean_abs_all = np.mean([np.mean(np.abs(s), axis=0) for s in shap_vals], axis=0)
        mean_abs_severe = np.mean(np.abs(severe_shap), axis=0)
    else:
        severe_shap = shap_vals
        mean_abs_all = np.mean(np.abs(shap_vals), axis=0)
        mean_abs_severe = mean_abs_all

    # Sort features by overall importance
    sorted_idx = np.argsort(mean_abs_all)[::-1]
    top_features = [FEATURES[i] for i in sorted_idx]
    top_importance = [float(mean_abs_all[i]) for i in sorted_idx]
    top_severe_importance = [float(mean_abs_severe[i]) for i in sorted_idx]

    # Prepare beeswarm data points for top 10 features
    beeswarm_data = {}
    for idx in sorted_idx[:10]:
        feat = FEATURES[idx]
        vals = X_sample[feat].values
        # Normalize feature values to [0, 1] for color-coding
        min_v, max_v = vals.min(), vals.max()
        norm_v = ((vals - min_v) / (max_v - min_v + 1e-8)).tolist()
        shap_pts = severe_shap[:, idx].tolist()
        beeswarm_data[feat] = {
            'label': FEATURE_LABELS.get(feat, feat),
            'shap_values': [round(x, 4) for x in shap_pts],
            'normalized_values': [round(x, 3) for x in norm_v],
            'raw_values': [round(float(x), 2) for x in vals]
        }

    cache = {
        'top_features': top_features,
        'feature_labels': [FEATURE_LABELS.get(f, f) for f in top_features],
        'mean_abs_importance': [round(v, 4) for v in top_importance],
        'severe_importance': [round(v, 4) for v in top_severe_importance],
        'beeswarm_data': beeswarm_data
    }

    with open(SHAP_CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"[SHAP] Successfully cached SHAP summary to {SHAP_CACHE_PATH}")
    return cache

def build_shap_importance_chart(shap_cache):
    """Horizontal bar chart showing global mean absolute SHAP values."""
    labels = shap_cache['feature_labels'][:10]
    vals = shap_cache['mean_abs_importance'][:10]

    # Reverse so top feature is at the top of the chart
    labels = labels[::-1]
    vals = vals[::-1]

    fig = go.Figure(go.Bar(
        x=vals,
        y=labels,
        orientation='h',
        marker=dict(
            color=vals,
            colorscale='YlOrRd',
            showscale=False
        ),
        text=[f"{v:.4f}" for v in vals],
        textposition='auto',
        textfont=dict(color='white', size=11)
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        margin=dict(l=20, r=20, t=15, b=25),
        xaxis=dict(title='Mean |SHAP Value| (Impact on Model Severity Output)', gridcolor='#1e293b'),
        yaxis=dict(autorange=True),
        height=320
    )
    return fig

def build_shap_beeswarm_plot(shap_cache):
    """
    Simulated Plotly Beeswarm scatter plot showing direction of feature effect on Severe fire risk.
    Red = High Feature Value, Blue = Low Feature Value.
    """
    beeswarm = shap_cache.get('beeswarm_data', {})
    fig = go.Figure()

    y_pos = 0
    y_ticks = []
    y_labels = []

    for feat, data in list(beeswarm.items())[::-1]:
        label = data['label']
        shap_pts = data['shap_values']
        norm_vals = data['normalized_values']
        raw_vals = data['raw_values']

        # Add slight random vertical jitter to simulate beeswarm dispersion
        np.random.seed(42)
        jitter = np.random.normal(0, 0.08, size=len(shap_pts))
        y_coords = [y_pos + j for j in jitter]

        fig.add_trace(go.Scatter(
            x=shap_pts,
            y=y_coords,
            mode='markers',
            name=label,
            marker=dict(
                size=5,
                color=norm_vals,
                colorscale=[[0, '#38bdf8'], [0.5, '#facc15'], [1, '#ef4444']], # Blue to Red
                showscale=(y_pos == 0),
                colorbar=dict(
                    title='Feature Value',
                    titleside='top',
                    tickvals=[0, 1],
                    ticktext=['Low', 'High'],
                    len=0.7,
                    thickness=12,
                    x=1.02
                ) if y_pos == 0 else None,
                opacity=0.7
            ),
            customdata=np.stack((raw_vals, [label]*len(raw_vals)), axis=-1),
            hovertemplate="<b>%{customdata[1]}</b><br>Value: %{customdata[0]}<br>SHAP Impact: %{x:+.4f}<extra></extra>",
            showlegend=False
        ))

        y_ticks.append(y_pos)
        y_labels.append(label)
        y_pos += 1

    fig.add_vline(x=0, line_dash='dash', line_color='#94a3b8', line_width=1)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc', size=11),
        margin=dict(l=20, r=40, t=20, b=30),
        xaxis=dict(title='SHAP Value (Impact on Severe Fire Growth: Left=Reduces, Right=Increases)', gridcolor='#1e293b'),
        yaxis=dict(tickvals=y_ticks, ticktext=y_labels, gridcolor='#1e293b'),
        height=330
    )
    return fig

# Cached explainer and model for fast local inference
_cached_explainer = None
_cached_model = None
_cached_encoder = None

def get_cached_shap_resources():
    global _cached_explainer, _cached_model, _cached_encoder
    if _cached_model is None:
        _cached_model = joblib.load(os.path.join(MODELS_DIR, "lightgbm_model.joblib"))
        _cached_encoder = joblib.load(ENCODER_PATH)
        _cached_explainer = shap.TreeExplainer(_cached_model)
    return _cached_model, _cached_encoder, _cached_explainer

def build_local_waterfall_chart(input_features_dict=None):
    """
    Computes local feature contribution waterfall for a single simulation profile.
    Shows the top positive drivers (pushing severity up) and negative drivers (pushing down).
    Includes safe fallback handling to prevent empty or broken plots.
    """
    if input_features_dict is None:
        # Default representative scenario
        input_features_dict = {
            'temp_max_c': 34.0, 'temp_min_c': 22.0, 'temp_range_c': 12.0,
            'rmin': 12.0, 'rmax': 47.0, 'vs': 9.5, 'pr': 0.0,
            'bi': 65.0, 'erc': 75.0, 'vpd': 3.2, 'fm100': 6.0, 'fm1000': 8.0,
            'Elevation': 1100, 'Slope': 20, 'Aspect': 180, 'TRI': 40.0, 'TPI': 10.0,
            'DISCOVERY_DOY': 240, 'Month': 8, 'STATE': 'CA', 'NWCG_GENERAL_CAUSE': 'Lightning'
        }

    try:
        model, encoder, explainer = get_cached_shap_resources()

        row_df = pd.DataFrame([input_features_dict])
        for c in ['STATE', 'NWCG_GENERAL_CAUSE']:
            if c not in row_df:
                row_df[c] = 'Unknown'
        row_df[['STATE', 'NWCG_GENERAL_CAUSE']] = encoder.transform(row_df[['STATE', 'NWCG_GENERAL_CAUSE']])

        for f in FEATURES:
            if f not in row_df:
                row_df[f] = 0.0

        X_in = row_df[FEATURES]
        local_shap = explainer.shap_values(X_in)

        # Class 2 = Severe Fire (>=100 acres)
        if isinstance(local_shap, np.ndarray) and len(local_shap.shape) == 3:
            vals = local_shap[0, :, 2]
        elif isinstance(local_shap, list):
            vals = local_shap[2][0]
        else:
            vals = local_shap[0]

        # Select top 8 largest absolute contributors
        top_indices = np.argsort(np.abs(vals))[::-1][:8]
        top_feats = [FEATURE_LABELS.get(FEATURES[i], FEATURES[i]) for i in top_indices]
        top_vals = [float(vals[i]) for i in top_indices]

        # Order for vertical bar display (largest impact at top)
        top_feats = top_feats[::-1]
        top_vals = top_vals[::-1]

        colors = ['#ef4444' if v > 0 else '#38bdf8' for v in top_vals]

        fig = go.Figure(go.Bar(
            x=top_vals,
            y=top_feats,
            orientation='h',
            marker=dict(color=colors, line=dict(color='#0f172a', width=1)),
            text=[f"{v:+.3f}" for v in top_vals],
            textposition='auto',
            textfont=dict(color='white', size=11),
            hovertemplate="<b>%{y}</b><br>SHAP Contribution: %{x:+.4f}<extra></extra>"
        ))

        fig.add_vline(x=0, line_dash='dash', line_color='#94a3b8', line_width=1.5)

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc', size=11),
            margin=dict(l=20, r=20, t=10, b=25),
            xaxis=dict(
                title='Contribution to Severe Fire Log-Odds (Red = Escalates Risk, Blue = Mitigates Risk)',
                gridcolor='#1e293b',
                zeroline=True,
                zerolinecolor='#94a3b8'
            ),
            yaxis=dict(autorange=True),
            height=260
        )
        return fig

    except Exception as e:
        # Fallback message chart
        fig = go.Figure()
        fig.add_annotation(
            text=f"SHAP explanation unavailable: {str(e)[:80]}<br>Please ensure input parameters are valid.",
            showarrow=False,
            font=dict(color='#f59e0b', size=12)
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=200,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        return fig

if __name__ == "__main__":
    cache = compute_and_cache_shap()
    print("SHAP computation test passed.")
    print("Top 5 features:", cache['top_features'][:5])
