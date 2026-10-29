"""
Wildfire Intelligence & Early Warning Dashboard
Interactive Python Dash + Plotly Web Application
Optimized Production Application:
- Command Center: KPIs, Albers Map, VIIRS Satellite Hotspots, Click Telemetry Inspector, Date Filter & Reset
- Environmental & Fire Impact: Log-scale Burned Area, Sample Counts (n=...), Pre-fire Indices, Data Interpretations
- AI Model Comparison: XGBoost, LightGBM, CatBoost Chronological Evaluation, Severe Recall/Precision/F1 & Imbalance Warnings
- Interactive AI Prediction Simulator: Red Flag / Summer / Mild Presets, Feature Store, Live Probabilities, Tactical Advisory
- Model Explainability & TreeSHAP: Global Feature Importance, Beeswarm Summary, Local Waterfall Decomposition
- Pretrained YOLO Fire & Smoke Vision: Dual Original/Annotated View, Latency Timing, Bounding Coordinates Table, Disclaimers
"""

import os
import sys
import json
import base64
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# Add local modules to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.data_loader import load_processed_data, load_map_sample, load_viirs_data
from src.models_trainer import (
    train_and_evaluate_models,
    load_best_model,
    predict_single_fire,
    METRICS_PATH,
    CLASS_NAMES
)
from src.explainability import (
    compute_and_cache_shap,
    build_shap_importance_chart,
    build_shap_beeswarm_plot,
    build_local_waterfall_chart,
    SHAP_CACHE_PATH
)
from src.yolo_detector import (
    run_yolo_detection,
    get_sample_image_b64,
    SAMPLES_DIR
)
from src.visualizations import (
    build_wildfire_map,
    build_temporal_trend_chart,
    build_cause_breakdown_chart,
    build_burned_area_distribution,
    build_environmental_boxplots,
    build_correlation_heatmap,
    build_model_metrics_comparison_chart,
    build_confusion_matrix_heatmap,
    build_prediction_probability_chart
)

# Initialize Dash application with dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP],
    title="Wildfire Intelligence & Early Warning Dashboard",
    suppress_callback_exceptions=True
)
server = app.server

# ---------------------------------------------------------
# DATA & CACHED ARTIFACTS
# ---------------------------------------------------------
print("[App] Loading dataset cache...")
df_full = load_processed_data()
df_map = load_map_sample()
viirs_df = load_viirs_data()
print(f"[App] Data loaded: {len(df_full):,} full records, {len(df_map):,} map sample, {len(viirs_df):,} VIIRS hotspots.")

# Load Model Metrics
if not os.path.exists(METRICS_PATH):
    print("[App] Model metrics not found. Training models...")
    metrics_summary = train_and_evaluate_models()
else:
    with open(METRICS_PATH, 'r') as f:
        metrics_summary = json.load(f)

best_model_name = metrics_summary.get('best_model', 'LightGBM')

# Load SHAP Cache
if not os.path.exists(SHAP_CACHE_PATH):
    print("[App] Caching SHAP values...")
    shap_cache = compute_and_cache_shap()
else:
    with open(SHAP_CACHE_PATH, 'r') as f:
        shap_cache = json.load(f)

# Unique filter options
ALL_STATES = sorted(df_full['STATE'].dropna().unique().tolist())
ALL_CAUSES = sorted(df_full['NWCG_GENERAL_CAUSE'].dropna().unique().tolist())
SEVERITY_OPTIONS = [
    {'label': 'All Severity Tiers', 'value': 'ALL'},
    {'label': 'Low (Class A-B: < 10 acres)', 'value': 'Low (Class A-B, <10 ac)'},
    {'label': 'Moderate (Class C: 10 - 99 acres)', 'value': 'Moderate (Class C, 10-99 ac)'},
    {'label': 'Severe (Class D-G: >= 100 acres)', 'value': 'Severe (Class D-G, >=100 ac)'}
]

# ---------------------------------------------------------
# HEADER COMPONENT
# ---------------------------------------------------------
def build_header():
    return html.Div(
        className="dashboard-header d-flex flex-wrap justify-content-between align-items-center",
        children=[
            html.Div([
                html.Div([
                    html.Span("🔥", style={"fontSize": "1.7rem", "marginRight": "10px"}),
                    html.H3("Wildfire Intelligence & Early Warning Dashboard", className="d-inline text-white fw-bold mb-0")
                ], className="d-flex align-items-center"),
                html.P("National Historical Wildfire Severity Analytics (FPA-FOD 2020), NASA VIIRS Thermal Hotspots & Pretrained YOLO Vision",
                       className="text-muted small mb-0 mt-1")
            ]),
            html.Div([
                html.Div(
                    [
                        html.Span("●", style={"color": "#10b981", "fontSize": "1.1rem"}),
                        html.Span(f" 2020 Archive: {len(df_full):,} Incidents | Satellite Hotspots: {len(viirs_df):,}", className="text-light small")
                    ],
                    className="header-badge me-2"
                ),
                html.Span(f"🏆 Best AI: {best_model_name}", className="badge bg-primary text-light py-2 px-3 fw-bold")
            ], className="d-flex align-items-center mt-2 mt-md-0")
        ]
    )

# ---------------------------------------------------------
# PAGE 1: COMMAND CENTER
# ---------------------------------------------------------
def build_command_center_tab():
    return html.Div([
        # Filter Bar
        dbc.Row([
            dbc.Col([
                html.Label("Filter by State", className="small text-muted fw-bold mb-1"),
                dcc.Dropdown(
                    id="filter-state",
                    options=[{'label': 'All States (National)', 'value': 'ALL'}] + [{'label': s, 'value': s} for s in ALL_STATES],
                    value='ALL',
                    clearable=False,
                    className="text-dark small"
                )
            ], xs=12, sm=6, md=3, className="mb-2"),

            dbc.Col([
                html.Label("Fire Severity Tier", className="small text-muted fw-bold mb-1"),
                dcc.Dropdown(
                    id="filter-severity",
                    options=SEVERITY_OPTIONS,
                    value='ALL',
                    clearable=False,
                    className="text-dark small"
                )
            ], xs=12, sm=6, md=3, className="mb-2"),

            dbc.Col([
                html.Label("Ignition Cause", className="small text-muted fw-bold mb-1"),
                dcc.Dropdown(
                    id="filter-cause",
                    options=[{'label': 'All Ignition Causes', 'value': 'ALL'}] + [{'label': c, 'value': c} for c in ALL_CAUSES],
                    value='ALL',
                    clearable=False,
                    className="text-dark small"
                )
            ], xs=12, sm=6, md=2, className="mb-2"),

            dbc.Col([
                html.Label("Satellite Hotspots (VIIRS)", className="small text-muted fw-bold mb-1"),
                dbc.Checklist(
                    id="toggle-viirs",
                    options=[{"label": " Overlay Satellite Feed", "value": "ENABLE"}],
                    value=["ENABLE"],
                    switch=True,
                    className="mt-1 text-info small fw-bold"
                )
            ], xs=8, sm=4, md=2, className="mb-2"),

            dbc.Col([
                html.Label("Controls", className="small text-muted fw-bold mb-1 d-none d-md-block"),
                dbc.Button("↺ Reset Filters", id="btn-reset-filters", color="secondary", outline=True, size="sm", className="w-100")
            ], xs=4, sm=2, md=2, className="mb-2 d-flex flex-column justify-content-end")
        ], className="custom-card mb-2 py-2"),

        # Date Range Filter Row
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("📅 Discovery Month Range (Historical 2020 Archive):", className="small text-muted fw-bold me-3"),
                    html.Div([
                        dcc.RangeSlider(
                            id="filter-month-range",
                            min=1,
                            max=12,
                            step=1,
                            value=[1, 12],
                            marks={1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'},
                            className="w-100"
                        )
                    ], className="flex-grow-1")
                ], className="d-flex align-items-center px-2")
            ], xs=12)
        ], className="custom-card mb-3 py-2"),

        # KPI Metric Row
        dbc.Row(id="kpi-container", className="mb-3 g-2"),

        # Main Visualization Grid
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.H5("🗺️ Geospatial Fire Incidents & Satellite Thermal Detections", className="text-white fw-bold mb-0"),
                        html.Small("Historical FPA-FOD fires (scaled by acreage) with NASA VIIRS satellite thermal hotspots overlay", className="text-muted")
                    ], className="d-flex flex-column mb-2"),
                    dcc.Loading(
                        id="loading-map",
                        type="circle",
                        color="#ef4444",
                        children=dcc.Graph(id="wildfire-map", config={'displayModeBar': True, 'scrollZoom': True})
                    ),
                    # Selected Incident Telemetry Inspector
                    html.Div(id="selected-incident-details", className="mt-2")
                ], className="custom-card h-100")
            ], xs=12, lg=7, className="mb-3 mb-lg-0"),

            dbc.Col([
                html.Div([
                    html.Div([
                        html.H6("📈 Temporal Fire Volume & Burned Acreage", className="text-white fw-bold mb-0"),
                        html.Small("Monthly fire volume vs total acres consumed (highlighting peak summer-autumn season)", className="text-muted")
                    ], className="mb-2"),
                    dcc.Graph(id="chart-temporal-trend", config={'displayModeBar': False})
                ], className="custom-card mb-3"),

                html.Div([
                    html.Div([
                        html.H6("🔍 Primary Wildfire Ignition Causes", className="text-white fw-bold mb-0"),
                        html.Small("Top reported causes by total frequency and acreage impact", className="text-muted")
                    ], className="mb-2"),
                    dcc.Graph(id="chart-cause-breakdown", config={'displayModeBar': False})
                ], className="custom-card")
            ], xs=12, lg=5)
        ])
    ])

# ---------------------------------------------------------
# PAGE 2: ENVIRONMENTAL & FIRE IMPACT
# ---------------------------------------------------------
def build_environmental_tab():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("📊 Heavy-Tailed Burned Area Distribution", className="text-white fw-bold mb-1"),
                    html.P("Wildfire acreage exhibits extreme power-law characteristics. Over 90% of total acreage is consumed by fewer than 3% of fires.", className="text-muted small mb-2"),
                    dcc.Graph(figure=build_burned_area_distribution(df_full), config={'displayModeBar': False}),
                    html.Div([
                        html.Small("📌 Data Interpretation: Power-law / Pareto dynamics dictate that small fires (89.2% under 10 acres) represent normal containment, whereas rare runaway escapes (2.8% above 100 acres) dominate ecological and economic destruction.", className="text-muted")
                    ], className="mt-2 p-2 rounded bg-dark border border-secondary")
                ], className="custom-card h-100")
            ], xs=12, lg=6, className="mb-3"),

            dbc.Col([
                html.Div([
                    html.Div([
                        html.H5("🌡️ Pre-Fire Danger Indices vs Severity", className="text-white fw-bold mb-1"),
                        html.P("Inspect how meteorological and fuel moisture indicators shift between low, moderate, and severe fires.", className="text-muted small")
                    ]),
                    dbc.RadioItems(
                        id="env-var-selector",
                        options=[
                            {"label": "Burning Index (BI)", "value": "bi"},
                            {"label": "Energy Release (ERC)", "value": "erc"},
                            {"label": "Max Temperature (°C)", "value": "temp_max_c"},
                            {"label": "Vapor Pressure Deficit (VPD)", "value": "vpd"},
                            {"label": "Wind Speed (m/s)", "value": "vs"},
                            {"label": "Min Relative Humidity (%)", "value": "rmin"}
                        ],
                        value="bi",
                        inline=True,
                        className="mb-2 small text-light"
                    ),
                    dcc.Graph(id="env-boxplot", config={'displayModeBar': False}),
                    html.Div([
                        html.Small("📌 Data Interpretation: Higher Burning Index (BI) and Energy Release Component (ERC) values directly shift distributions upward in Severe classes, indicating that pre-fire fuel desiccation is the primary enabler of rapid flame propagation.", className="text-muted")
                    ], className="mt-2 p-2 rounded bg-dark border border-secondary")
                ], className="custom-card h-100")
            ], xs=12, lg=6, className="mb-3"),
        ]),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("🔥 Pre-Fire Feature Correlation Matrix", className="text-white fw-bold mb-1"),
                    html.P("Spearman non-linear rank correlation between pre-fire environmental variables and log-scale fire acreage.", className="text-muted small"),
                    dcc.Graph(figure=build_correlation_heatmap(df_full), config={'displayModeBar': False}),
                    html.Div([
                        html.Small("📌 Data Interpretation: Max temperature, VPD, and Burning Index show consistent positive rank correlations with fire size, whereas fuel moisture (fm100) exhibits negative correlation, acting as an operational buffer.", className="text-muted")
                    ], className="mt-2 p-2 rounded bg-dark border border-secondary")
                ], className="custom-card h-100")
            ], xs=12, lg=7, className="mb-3"),

            dbc.Col([
                html.Div([
                    html.H5("⚖️ Scientific Rigor & Forecasting Scope Boundary", className="text-warning fw-bold mb-2"),
                    html.P(
                        "This dashboard models Historical Wildfire Severity Classification — estimating the potential scale and containment difficulty of a CONFIRMED fire ignition given pre-fire fuel moisture, topography, and weather.",
                        className="text-light small"
                    ),
                    html.Hr(style={"borderColor": "#334155"}),
                    html.P(
                        "Important distinction: Severity classification does NOT predict whether a future wildfire will ignite at an arbitrary coordinate. Operational ignition forecasting requires numerical weather forecasts and a balanced sample of non-fire background controls.",
                        className="text-muted small"
                    ),
                    html.Ul([
                        html.Li("Pre-fire features reflect physical fire behavior potential at time of discovery.", className="text-muted small"),
                        html.Li("Chronological splits ensure no future weather states leak into earlier fire models.", className="text-muted small"),
                        html.Li("YOLO vision integration inspects imagery directly without confounding tabular features.", className="text-muted small")
                    ], className="mb-0")
                ], className="custom-card h-100 border-warning")
            ], xs=12, lg=5, className="mb-3")
        ])
    ])

# ---------------------------------------------------------
# PAGE 3: AI MODEL COMPARISON
# ---------------------------------------------------------
def build_model_comparison_tab():
    models_info = metrics_summary.get('models', {})
    split_info = metrics_summary.get('split_info', {})

    table_rows = []
    for m_name, m_stats in models_info.items():
        is_best = (m_name == best_model_name)
        row_style = {"backgroundColor": "rgba(56, 189, 248, 0.12)"} if is_best else {}
        badge = html.Span(" ★ Selected Best", className="badge bg-primary ms-2") if is_best else None

        # Compute severe class recall and precision from confusion matrix
        cm = m_stats.get('confusion_matrix', [[0,0,0],[0,0,0],[0,0,0]])
        counts = m_stats.get('confusion_counts', [[0,0,0],[0,0,0],[0,0,0]])
        s_rec = cm[2][2] if len(cm) > 2 else 0.0
        c_sum = (counts[0][2] + counts[1][2] + counts[2][2]) if len(counts) > 2 else 0
        s_prec = (counts[2][2] / c_sum) if c_sum > 0 else 0.0
        s_f1 = (2 * s_rec * s_prec / (s_rec + s_prec)) if (s_rec + s_prec) > 0 else 0.0

        table_rows.append(html.Tr([
            html.Td([html.Strong(m_name, className="text-light"), badge]),
            html.Td(f"{m_stats['macro_f1']:.4f}", className="text-info fw-bold"),
            html.Td(f"{s_rec*100:.1f}%", className="text-danger fw-bold"),
            html.Td(f"{s_prec*100:.1f}%"),
            html.Td(f"{s_f1*100:.1f}%"),
            html.Td(f"{m_stats['accuracy']*100:.2f}%"),
            html.Td(f"{m_stats['weighted_f1']:.4f}"),
        ], style=row_style))

    return html.Div([
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("🤖 Advanced Tabular AI Model Comparison", className="text-white fw-bold mb-1"),
                    html.P("Empirical benchmarking of XGBoost, LightGBM, and CatBoost on historical wildfire severity using a strict chronological split.", className="text-muted small mb-0")
                ], xs=12, md=6),
                dbc.Col([
                    html.Div([
                        html.Span("⏱️ Split Protocol: ", className="text-muted small"),
                        html.Span(f"Train: Jan-Aug ({split_info.get('train_size', 56669):,}) | Test: Sep-Dec ({split_info.get('test_size', 16552):,})", className="badge bg-dark border border-secondary text-light small")
                    ], className="text-md-end mt-2 mt-md-0")
                ], xs=12, md=6)
            ])
        ], className="custom-card mb-2 py-3"),

        # Class counts banner
        dbc.Alert(
            [
                html.Span("📊 Held-Out Test Set Class Distribution (N=16,552 Incidents): ", className="fw-bold"),
                html.Span("Low (<10 ac): 14,703 (88.8%) | Moderate (10-99 ac): 1,426 (8.6%) | Severe (≥100 ac): 423 (2.6%)", className="small")
            ],
            color="info",
            className="mb-3 py-2"
        ),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H6("📋 Performance Leaderboard (Held-Out Test Set)", className="text-white fw-bold mb-2"),
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Model Architecture"),
                            html.Th("Macro F1 (Primary)"),
                            html.Th("Severe Recall"),
                            html.Th("Severe Precision"),
                            html.Th("Severe F1"),
                            html.Th("Accuracy"),
                            html.Th("Weighted F1")
                        ], className="text-secondary small")),
                        html.Tbody(table_rows, className="small text-light")
                    ], responsive=True, hover=True, className="align-middle mb-0"),

                    html.Div([
                        html.Small("💡 Macro F1 treats all three classes equally, preventing small fires (88.8% majority) from masking catastrophic fire escapes.", className="text-muted")
                    ], className="mt-2 p-2 rounded bg-dark border border-secondary")
                ], className="custom-card h-100")
            ], xs=12, lg=6, className="mb-3"),

            dbc.Col([
                html.Div([
                    html.H6("📊 Multi-Metric Model Comparison", className="text-white fw-bold mb-2"),
                    dcc.Graph(figure=build_model_metrics_comparison_chart(metrics_summary), config={'displayModeBar': False})
                ], className="custom-card h-100")
            ], xs=12, lg=6, className="mb-3")
        ]),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.H6("🔲 Normalized Confusion Matrix", className="text-white fw-bold mb-0"),
                        html.Small("Rows represent Actual ground truth classes; columns represent Model predictions", className="text-muted")
                    ], className="mb-2"),
                    dbc.RadioItems(
                        id="cm-model-selector",
                        options=[{"label": m, "value": m} for m in models_info.keys()],
                        value=best_model_name,
                        inline=True,
                        className="mb-2 small text-light"
                    ),
                    dcc.Graph(id="confusion-matrix-chart", config={'displayModeBar': False})
                ], className="custom-card h-100")
            ], xs=12, lg=6, className="mb-3"),

            dbc.Col([
                html.Div([
                    html.H6("🔬 Operational Decision Guidance & Class Imbalance Analysis", className="text-white fw-bold mb-2"),
                    dbc.Alert(
                        [
                            html.Strong("⚠️ Operational Imbalance Warning: ", className="text-warning"),
                            "Severe fires constitute only 2.6% of test incidents. Predicting 100% 'Low' yields 88.8% accuracy but fails on all severe fires. Severe fire recall ranges from 19.9% (LightGBM) to 40.7% (CatBoost)."
                        ],
                        color="warning",
                        className="py-2 small mb-2"
                    ),
                    html.Ul([
                        html.Li([
                            html.Strong("Why LightGBM Won: ", className="text-warning"),
                            f"LightGBM achieved the highest test Macro F1 ({models_info.get('LightGBM', {}).get('macro_f1', 0.4228):.4f}) with 80.00% accuracy and the best balance across Moderate (Class C: 27.0% recall) and Severe fires."
                        ], className="text-light small mb-2"),
                        html.Li([
                            html.Strong("CatBoost Severe Sensitivity: ", className="text-danger"),
                            "CatBoost attained the highest recall on severe fires (40.7%), but at the expense of a 91.5% false alarm rate on small fires, yielding lower overall precision (8.5%)."
                        ], className="text-light small mb-2"),
                        html.Li([
                            html.Strong("Chronological Leakage Protection: ", className="text-success"),
                            "Evaluating strictly on late-season fires (Sept-Dec) ensures authentic performance during the peak wildfire siege."
                        ], className="text-light small mb-0")
                    ], className="mb-0")
                ], className="custom-card h-100")
            ], xs=12, lg=6, className="mb-3")
        ])
    ])

# ---------------------------------------------------------
# PAGE 4: INTERACTIVE AI PREDICTION SIMULATOR
# ---------------------------------------------------------
def build_ai_prediction_tab():
    return html.Div([
        html.Div([
            html.H4("🎯 Interactive Fire Severity Prediction Simulator", className="text-white fw-bold mb-1"),
            html.P(f"Simulate pre-fire meteorological, topographic, and fuel conditions to evaluate predicted fire growth potential using the trained {best_model_name} model.", className="text-muted small mb-0")
        ], className="custom-card mb-3 py-3"),

        # Scope Notice
        dbc.Alert(
            [
                html.Strong("⚠️ Scope Note: ", className="text-warning"),
                "This simulator evaluates fire growth severity and containment difficulty assuming an ignition has occurred. It does NOT predict future wildfire ignition probability. Containment guidance is illustrative for decision-support modeling."
            ],
            color="secondary",
            className="py-2 small mb-3"
        ),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.H6("⚙️ Pre-Fire Environmental Inputs", className="text-white fw-bold mb-0"),
                        html.Small("Preset Scenarios:", className="text-muted small me-2"),
                        html.Div([
                            dbc.Button("🚨 Red Flag", id="btn-preset-redflag", color="danger", size="sm", className="me-1 py-0 px-2 small"),
                            dbc.Button("☀️ Summer", id="btn-preset-summer", color="warning", size="sm", className="me-1 py-0 px-2 small"),
                            dbc.Button("🌧️ Mild", id="btn-preset-mild", color="info", size="sm", className="me-1 py-0 px-2 small"),
                            dbc.Button("↺ Reset", id="btn-reset-sim", color="secondary", outline=True, size="sm", className="py-0 px-2 small")
                        ], className="d-inline-flex")
                    ], className="d-flex justify-content-between align-items-center mb-3"),

                    dbc.Row([
                        dbc.Col([
                            html.Label("Target State", className="small text-muted fw-bold"),
                            dcc.Dropdown(
                                id="sim-state",
                                options=[{'label': s, 'value': s} for s in ALL_STATES],
                                value='CA',
                                clearable=False,
                                className="text-dark small"
                            )
                        ], xs=6, className="mb-3"),

                        dbc.Col([
                            html.Label("Ignition Cause", className="small text-muted fw-bold"),
                            dcc.Dropdown(
                                id="sim-cause",
                                options=[{'label': c, 'value': c} for c in ALL_CAUSES],
                                value='Lightning',
                                clearable=False,
                                className="text-dark small"
                            )
                        ], xs=6, className="mb-3"),
                    ]),

                    html.Div([
                        html.Div([
                            html.Label("Maximum Temperature (°C):", className="small text-light fw-bold"),
                            html.Span(id="val-sim-temp", className="text-danger fw-bold small ms-2")
                        ], className="d-flex justify-content-between"),
                        dcc.Slider(id="sim-temp", min=5, max=45, step=1, value=34,
                                   marks={10: '10°C', 20: '20°C', 30: '30°C', 40: '40°C'}, className="mb-2")
                    ], className="mb-3"),

                    html.Div([
                        html.Div([
                            html.Label("Minimum Relative Humidity (%):", className="small text-light fw-bold"),
                            html.Span(id="val-sim-rh", className="text-info fw-bold small ms-2")
                        ], className="d-flex justify-content-between"),
                        dcc.Slider(id="sim-rh", min=5, max=80, step=1, value=12,
                                   marks={10: '10%', 25: '25%', 50: '50%', 75: '75%'}, className="mb-2")
                    ], className="mb-3"),

                    html.Div([
                        html.Div([
                            html.Label("Wind Speed (m/s):", className="small text-light fw-bold"),
                            html.Span(id="val-sim-wind", className="text-warning fw-bold small ms-2")
                        ], className="d-flex justify-content-between"),
                        dcc.Slider(id="sim-wind", min=1, max=25, step=0.5, value=9.5,
                                   marks={2: '2', 8: '8', 15: '15', 22: '22'}, className="mb-2")
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.Label("Burning Index (BI):", className="small text-light fw-bold"),
                                    html.Span(id="val-sim-bi", className="text-danger fw-bold small ms-1")
                                ], className="d-flex justify-content-between"),
                                dcc.Slider(id="sim-bi", min=0, max=100, step=2, value=65, marks={0:'0', 50:'50', 100:'100'})
                            ], className="mb-3")
                        ], xs=6),

                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.Label("Energy Release (ERC):", className="small text-light fw-bold"),
                                    html.Span(id="val-sim-erc", className="text-warning fw-bold small ms-1")
                                ], className="d-flex justify-content-between"),
                                dcc.Slider(id="sim-erc", min=0, max=100, step=2, value=75, marks={0:'0', 50:'50', 100:'100'})
                            ], className="mb-3")
                        ], xs=6),
                    ]),

                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.Label("VPD (kPa):", className="small text-light fw-bold"),
                                    html.Span(id="val-sim-vpd", className="text-info fw-bold small ms-1")
                                ], className="d-flex justify-content-between"),
                                dcc.Slider(id="sim-vpd", min=0.2, max=6.0, step=0.2, value=3.2, marks={1:'1', 3:'3', 5:'5'})
                            ], className="mb-3")
                        ], xs=6),

                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.Label("100h Moisture (%):", className="small text-light fw-bold"),
                                    html.Span(id="val-sim-fm100", className="text-success fw-bold small ms-1")
                                ], className="d-flex justify-content-between"),
                                dcc.Slider(id="sim-fm100", min=3, max=30, step=1, value=6, marks={5:'5%', 15:'15%', 25:'25%'})
                            ], className="mb-3")
                        ], xs=6),
                    ]),

                    dbc.Button("🔥 Run Severity Prediction", id="btn-predict-severity", color="danger", className="w-100 fw-bold py-2 mt-2")
                ], className="custom-card h-100")
            ], xs=12, lg=6, className="mb-3"),

            dbc.Col([
                html.Div([
                    html.H6("📊 AI Inference Output & Containment Advisory", className="text-white fw-bold mb-3"),
                    html.Div(id="prediction-result-container")
                ], className="custom-card h-100")
            ], xs=12, lg=6, className="mb-3")
        ])
    ])

# ---------------------------------------------------------
# PAGE 5: MODEL EXPLAINABILITY & SHAP
# ---------------------------------------------------------
def build_explainability_tab():
    return html.Div([
        html.Div([
            html.H4("🔍 Model Explainability & TreeSHAP Impact Analysis", className="text-white fw-bold mb-1"),
            html.P("Game-theoretic TreeSHAP feature contributions revealing how pre-fire danger indices influence fire-severity classifications.", className="text-muted small mb-0")
        ], className="custom-card mb-3 py-3"),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H6("🌟 Global Mean Absolute Feature Importance", className="text-white fw-bold mb-1"),
                    html.Small("Mean |SHAP| magnitude across all historical classes on the held-out test set", className="text-muted"),
                    dcc.Graph(figure=build_shap_importance_chart(shap_cache), config={'displayModeBar': False})
                ], className="custom-card h-100")
            ], xs=12, lg=6, className="mb-3"),

            dbc.Col([
                html.Div([
                    html.H6("🐝 SHAP Beeswarm Summary (Severe Fire Impact)", className="text-white fw-bold mb-1"),
                    html.Small("Feature values (Red=High, Blue=Low) vs impact on Severe Fire growth (Class D-G, >=100 ac)", className="text-muted"),
                    dcc.Graph(figure=build_shap_beeswarm_plot(shap_cache), config={'displayModeBar': False})
                ], className="custom-card h-100")
            ], xs=12, lg=6, className="mb-3")
        ]),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H6("🔬 Local Feature Contribution (Active Simulator Scenario)", className="text-white fw-bold mb-1"),
                    html.Small("Decomposition of positive drivers (Red = escalating risk) vs negative drivers (Blue = mitigating risk)", className="text-muted"),
                    dcc.Graph(id="shap-local-waterfall", figure=build_local_waterfall_chart(), config={'displayModeBar': False})
                ], className="custom-card h-100")
            ], xs=12, lg=6, className="mb-3"),

            dbc.Col([
                html.Div([
                    html.H6("💡 Meteorological & Fuel Explainability Insights", className="text-white fw-bold mb-2"),
                    html.Ul([
                        html.Li([
                            html.Strong("Burning Index (BI) & ERC Dominance: ", className="text-danger"),
                            "The Burning Index and Energy Release Component consistently register as the strongest positive contributors to severe fire escapes. High values shift odds towards large fires."
                        ], className="text-light small mb-2"),
                        html.Li([
                            html.Strong("Fuel Moisture Buffer (fm100): ", className="text-info"),
                            "100-hour fuel moisture acts as a major negative driver when elevated (>15%), dampening flame velocity and allowing initial suppression forces to achieve containment."
                        ], className="text-light small mb-2"),
                        html.Li([
                            html.Strong("Relative Humidity Thresholds: ", className="text-warning"),
                            "Minimum relative humidity below 15% combined with wind speed above 8 m/s triggers rapid non-linear escalations in model-predicted severity."
                        ], className="text-light small mb-2"),
                        html.Li([
                            html.Strong("Non-Linear Interactivity: ", className="text-success"),
                            "TreeSHAP captures the complex multiplicative interaction between wind, terrain slope, and dry fuel that linear models fail to detect."
                        ], className="text-light small mb-0")
                    ])
                ], className="custom-card h-100")
            ], xs=12, lg=6, className="mb-3")
        ])
    ])

# ---------------------------------------------------------
# PAGE 6: PRETRAINED YOLO FIRE & SMOKE VISION DETECTION
# ---------------------------------------------------------
def build_yolo_tab():
    return html.Div([
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("👁️ Pretrained YOLO Fire & Smoke Vision Surveillance", className="text-white fw-bold mb-1"),
                    html.P("Real-time computer vision pipeline powered by pretrained Ultralytics weights (best.pt) to identify active flames and smoke plumes.", className="text-muted small mb-0")
                ], xs=12, md=8),
                dbc.Col([
                    html.Div([
                        html.Span("Weights: ", className="text-muted small"),
                        html.Span("models/best.pt", className="badge bg-danger text-light small me-2"),
                        html.Span("Classes: Smoke (0), Fire (1)", className="badge bg-dark border border-secondary text-info small")
                    ], className="text-md-end mt-2 mt-md-0")
                ], xs=12, md=4)
            ])
        ], className="custom-card mb-2 py-3"),

        # Operational disclaimer
        dbc.Alert(
            [
                html.Strong("⚠️ Field Surveillance Notice: ", className="text-warning"),
                "Automated vision detections indicate visual smoke or flame cues within the analyzed image. Bounding box detections do not confirm a real-world wildfire emergency without ground or multi-spectral confirmation."
            ],
            color="secondary",
            className="py-2 small mb-3"
        ),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H6("📸 Imagery Input Source", className="text-white fw-bold mb-2"),

                    html.Label("Select Test Scenario:", className="small text-muted fw-bold mb-1"),
                    dbc.RadioItems(
                        id="yolo-sample-selector",
                        options=[
                            {"label": " Sample 1: Aerial Forest Fire (Active Flames & Smoke Plume)", "value": "wildfire_aerial_sample.jpg"},
                            {"label": " Sample 2: Mountain Surveillance (Rising Smoke Column)", "value": "wildfire_smoke_plume.jpg"},
                            {"label": " Upload Custom Field Image", "value": "CUSTOM"}
                        ],
                        value="wildfire_aerial_sample.jpg",
                        className="small text-light mb-3"
                    ),

                    html.Div(id="yolo-upload-wrapper", children=[
                        dcc.Upload(
                            id="yolo-file-upload",
                            children=html.Div([
                                html.Span("📁 Drag & Drop or "),
                                html.A("Select Fire/Smoke Image", className="text-primary fw-bold")
                            ], className="small py-3 text-center"),
                            style={
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderColor": "#334155",
                                "borderRadius": "8px",
                                "backgroundColor": "#0b1220",
                                "cursor": "pointer"
                            },
                            multiple=False
                        )
                    ], className="mb-3"),

                    html.Div([
                        html.Div([
                            html.Label("Detection Confidence Threshold:", className="small text-light fw-bold"),
                            html.Span(id="val-yolo-conf", className="text-info fw-bold small ms-2")
                        ], className="d-flex justify-content-between"),
                        dcc.Slider(
                            id="yolo-conf-slider",
                            min=0.10,
                            max=0.80,
                            step=0.05,
                            value=0.20,
                            marks={0.15: '15%', 0.3: '30%', 0.5: '50%', 0.7: '70%'},
                            className="mb-3"
                        )
                    ]),

                    dbc.Button("👁️ Execute YOLO Detection", id="btn-run-yolo", color="danger", className="w-100 fw-bold py-2 mt-2")
                ], className="custom-card h-100")
            ], xs=12, lg=5, className="mb-3"),

            dbc.Col([
                html.Div([
                    html.H6("🎯 Vision Analytics & Detection Results", className="text-white fw-bold mb-3"),
                    dcc.Loading(
                        id="loading-yolo",
                        type="circle",
                        color="#ef4444",
                        children=html.Div(id="yolo-output-container")
                    )
                ], className="custom-card h-100")
            ], xs=12, lg=7, className="mb-3")
        ])
    ])

# ---------------------------------------------------------
# MASTER LAYOUT & STORES
# ---------------------------------------------------------
app.layout = html.Div([
    # Shared Client-Side Stores
    dcc.Store(id="simulated-scenario-store"),

    build_header(),
    html.Div([
        dcc.Tabs(id="dashboard-tabs", value="tab-command-center", className="mb-3", children=[
            dcc.Tab(label="Command Center", value="tab-command-center"),
            dcc.Tab(label="Environmental & Fire Impact", value="tab-environmental"),
            dcc.Tab(label="AI Model Comparison", value="tab-model-comparison"),
            dcc.Tab(label="Interactive AI Prediction", value="tab-ai-prediction"),
            dcc.Tab(label="Explainability & SHAP", value="tab-shap"),
            dcc.Tab(label="YOLO Fire & Smoke Vision", value="tab-yolo")
        ]),
        html.Div(id="tab-content", className="px-3 pb-4")
    ])
])

# ---------------------------------------------------------
# INTERACTIVE CALLBACKS
# ---------------------------------------------------------

@app.callback(
    Output("tab-content", "children"),
    Input("dashboard-tabs", "value")
)
def render_tab_content(tab_name):
    if tab_name == "tab-command-center":
        return build_command_center_tab()
    elif tab_name == "tab-environmental":
        return build_environmental_tab()
    elif tab_name == "tab-model-comparison":
        return build_model_comparison_tab()
    elif tab_name == "tab-ai-prediction":
        return build_ai_prediction_tab()
    elif tab_name == "tab-shap":
        return build_explainability_tab()
    elif tab_name == "tab-yolo":
        return build_yolo_tab()
    return html.Div("Page Not Found", className="text-white")

# Command Center Filters Reset Callback
@app.callback(
    [
        Output("filter-state", "value"),
        Output("filter-severity", "value"),
        Output("filter-cause", "value"),
        Output("filter-month-range", "value"),
        Output("toggle-viirs", "value")
    ],
    Input("btn-reset-filters", "n_clicks"),
    prevent_initial_call=True
)
def reset_command_center_filters(n_clicks):
    return 'ALL', 'ALL', 'ALL', [1, 12], ["ENABLE"]

# Command Center Primary Update Callback
@app.callback(
    [
        Output("kpi-container", "children"),
        Output("wildfire-map", "figure"),
        Output("chart-temporal-trend", "figure"),
        Output("chart-cause-breakdown", "figure")
    ],
    [
        Input("filter-state", "value"),
        Input("filter-severity", "value"),
        Input("filter-cause", "value"),
        Input("filter-month-range", "value"),
        Input("toggle-viirs", "value")
    ]
)
def update_command_center(selected_state, selected_severity, selected_cause, month_range, viirs_toggle):
    dff = df_full.copy()
    if selected_state and selected_state != 'ALL':
        dff = dff[dff['STATE'] == selected_state]
    if selected_severity and selected_severity != 'ALL':
        dff = dff[dff['Severity_Category'] == selected_severity]
    if selected_cause and selected_cause != 'ALL':
        dff = dff[dff['NWCG_GENERAL_CAUSE'] == selected_cause]
    if month_range and len(month_range) == 2:
        dff = dff[(dff['Month'] >= month_range[0]) & (dff['Month'] <= month_range[1])]

    dff_map = df_map.copy()
    if selected_state and selected_state != 'ALL':
        dff_map = dff_map[dff_map['STATE'] == selected_state]
    if selected_severity and selected_severity != 'ALL':
        dff_map = dff_map[dff_map['Severity_Category'] == selected_severity]
    if selected_cause and selected_cause != 'ALL':
        dff_map = dff_map[dff_map['NWCG_GENERAL_CAUSE'] == selected_cause]
    if month_range and len(month_range) == 2:
        dff_map = dff_map[(dff_map['Month'] >= month_range[0]) & (dff_map['Month'] <= month_range[1])]

    total_fires = len(dff)
    total_acres = dff['FIRE_SIZE'].sum() if total_fires > 0 else 0
    severe_count = (dff['FIRE_SIZE'] >= 10.0).sum() if total_fires > 0 else 0
    severe_pct = (severe_count / total_fires * 100) if total_fires > 0 else 0.0

    if total_fires > 0 and not dff['FIRE_SIZE'].empty:
        max_fire_row = dff.loc[dff['FIRE_SIZE'].idxmax()]
        max_fire_acres = f"{max_fire_row['FIRE_SIZE']:,.0f} ac"
        max_fire_name = f"{max_fire_row['FIRE_NAME']} ({max_fire_row['STATE']})"
        avg_bi = f"{dff['bi'].mean():.1f}"
    else:
        max_fire_acres = "0 ac"
        max_fire_name = "N/A"
        avg_bi = "0"

    show_viirs = ("ENABLE" in viirs_toggle) if viirs_toggle else False
    active_hotspots = len(viirs_df) if show_viirs else 0

    kpi_cards = [
        dbc.Col([
            html.Div([
                html.Div("Total Fire Incidents", className="kpi-title"),
                html.Div(f"{total_fires:,}", className="kpi-value text-white"),
                html.Div("2020 Historical Archive", className="kpi-subtext")
            ], className="kpi-card kpi-orange")
        ], xs=6, sm=4, md=2),

        dbc.Col([
            html.Div([
                html.Div("Total Burned Land", className="kpi-title"),
                html.Div(f"{total_acres:,.0f} ac", className="kpi-value text-white"),
                html.Div(f"{(total_acres/1e6):.2f}M total acres", className="kpi-subtext")
            ], className="kpi-card kpi-fire")
        ], xs=6, sm=4, md=2),

        dbc.Col([
            html.Div([
                html.Div("Largest Single Fire", className="kpi-title"),
                html.Div(max_fire_acres, className="kpi-value text-white"),
                html.Div(max_fire_name[:24], className="kpi-subtext text-truncate")
            ], className="kpi-card kpi-amber")
        ], xs=6, sm=4, md=2),

        dbc.Col([
            html.Div([
                html.Div("Escaped / Large Fire %", className="kpi-title"),
                html.Div(f"{severe_pct:.1f}%", className="kpi-value text-white"),
                html.Div(f"{severe_count:,} fires >= 10 acres", className="kpi-subtext")
            ], className="kpi-card kpi-blue")
        ], xs=6, sm=4, md=2),

        dbc.Col([
            html.Div([
                html.Div("Mean Burning Index", className="kpi-title"),
                html.Div(avg_bi, className="kpi-value text-white"),
                html.Div("Potential fire intensity", className="kpi-subtext")
            ], className="kpi-card kpi-green")
        ], xs=6, sm=4, md=2),

        dbc.Col([
            html.Div([
                html.Div("Active Satellite Hotspots", className="kpi-title"),
                html.Div(f"{active_hotspots:,}", className="kpi-value text-info"),
                html.Div("Suomi VIIRS NRT Snapshot", className="kpi-subtext")
            ], className="kpi-card kpi-blue")
        ], xs=6, sm=4, md=2),
    ]

    map_fig = build_wildfire_map(dff_map, viirs_df=viirs_df, show_viirs=show_viirs)
    temporal_fig = build_temporal_trend_chart(dff)
    cause_fig = build_cause_breakdown_chart(dff)

    return kpi_cards, map_fig, temporal_fig, cause_fig

# Map Click Telemetry Inspector Callback
@app.callback(
    Output("selected-incident-details", "children"),
    Input("wildfire-map", "clickData")
)
def display_map_click_telemetry(click_data):
    if not click_data or 'points' not in click_data or len(click_data['points']) == 0:
        return html.Div([
            html.Small("📌 Click on any historical fire incident or satellite thermal hotspot on the map to inspect its full incident telemetry.", className="text-muted fst-italic")
        ], className="p-2 rounded bg-dark border border-secondary text-center")

    pt = click_data['points'][0]
    custom = pt.get('customdata', [])

    # Check if point is satellite hotspot or historical wildfire
    if len(custom) >= 8:
        # Historical Fire: [FIRE_NAME, COUNTY, STATE, FIRE_SIZE, DISCOVERY_DATE, NWCG_GENERAL_CAUSE, temp_max_c, bi, erc, vs, rmin]
        name = custom[0]
        county = custom[1]
        state = custom[2]
        size = custom[3]
        date = custom[4]
        cause = custom[5]
        temp = custom[6]
        bi = custom[7]
        erc = custom[8] if len(custom) > 8 else "N/A"
        wind = custom[9] if len(custom) > 9 else "N/A"
        rh = custom[10] if len(custom) > 10 else "N/A"

        return dbc.Card([
            dbc.CardHeader([
                html.Span("🔥 Selected Historical Incident: ", className="text-muted small fw-bold"),
                html.Strong(f"{name} ({county}, {state})", className="text-white")
            ], className="py-2 bg-dark border-secondary"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div("Discovery Date:", className="small text-muted"),
                        html.Div(f"{date}", className="small text-light fw-bold")
                    ], xs=6, md=3),
                    dbc.Col([
                        html.Div("Final Burned Area:", className="small text-muted"),
                        html.Div(f"{size} acres", className="small text-warning fw-bold")
                    ], xs=6, md=3),
                    dbc.Col([
                        html.Div("Ignition Cause:", className="small text-muted"),
                        html.Div(f"{cause}", className="small text-light fw-bold")
                    ], xs=6, md=3),
                    dbc.Col([
                        html.Div("Pre-Fire Environment:", className="small text-muted"),
                        html.Div(f"Temp: {temp} | BI: {bi} | ERC: {erc} | Wind: {wind} | RH: {rh}", className="small text-info fw-bold")
                    ], xs=12, md=3),
                ])
            ], className="py-2 bg-dark")
        ], className="border-secondary")

    elif len(custom) >= 4:
        # Satellite Hotspot: [bright_ti4, frp, confidence, acq_date, satellite, daynight]
        b_temp = custom[0]
        frp = custom[1]
        conf = custom[2]
        acq = custom[3]
        sat = custom[4] if len(custom) > 4 else "Suomi NPP"
        orbit = custom[5] if len(custom) > 5 else "N"

        return dbc.Card([
            dbc.CardHeader([
                html.Span("🛰️ Selected Satellite Anomaly: ", className="text-muted small fw-bold"),
                html.Strong(f"Suomi VIIRS Hotspot ({sat})", className="text-info")
            ], className="py-2 bg-dark border-secondary"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div("Acquired Timestamp:", className="small text-muted"),
                        html.Div(f"{acq} ({orbit} orbit)", className="small text-light fw-bold")
                    ], xs=6, md=3),
                    dbc.Col([
                        html.Div("Brightness Temp (Ti4):", className="small text-muted"),
                        html.Div(f"{b_temp}", className="small text-warning fw-bold")
                    ], xs=6, md=3),
                    dbc.Col([
                        html.Div("Fire Radiative Power (FRP):", className="small text-muted"),
                        html.Div(f"{frp}", className="small text-danger fw-bold")
                    ], xs=6, md=3),
                    dbc.Col([
                        html.Div("Detection Confidence:", className="small text-muted"),
                        html.Div(f"{conf.capitalize()}", className="small text-light fw-bold")
                    ], xs=6, md=3),
                ])
            ], className="py-2 bg-dark")
        ], className="border-secondary")

    return dash.no_update

@app.callback(
    Output("env-boxplot", "figure"),
    Input("env-var-selector", "value")
)
def update_environmental_boxplot(selected_var):
    return build_environmental_boxplots(df_full, variable=selected_var)

@app.callback(
    Output("confusion-matrix-chart", "figure"),
    Input("cm-model-selector", "value")
)
def update_confusion_matrix(selected_model):
    model_data = metrics_summary.get('models', {}).get(selected_model, {})
    cm = model_data.get('confusion_matrix', [[0,0,0],[0,0,0],[0,0,0]])
    return build_confusion_matrix_heatmap(cm, CLASS_NAMES, selected_model)

# Simulator Preset Scenarios and Reset Callback
@app.callback(
    [
        Output("sim-state", "value"),
        Output("sim-cause", "value"),
        Output("sim-temp", "value"),
        Output("sim-rh", "value"),
        Output("sim-wind", "value"),
        Output("sim-bi", "value"),
        Output("sim-erc", "value"),
        Output("sim-vpd", "value"),
        Output("sim-fm100", "value")
    ],
    [
        Input("btn-preset-redflag", "n_clicks"),
        Input("btn-preset-summer", "n_clicks"),
        Input("btn-preset-mild", "n_clicks"),
        Input("btn-reset-sim", "n_clicks")
    ],
    prevent_initial_call=True
)
def apply_simulator_presets(btn_rf, btn_sum, btn_mild, btn_reset):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    btn_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if btn_id == "btn-preset-redflag":
        # Critical fire weather
        return 'CA', 'Lightning', 38, 9, 14.0, 85, 90, 4.2, 4
    elif btn_id == "btn-preset-summer":
        # Moderate summer fire
        return 'OR', 'Equipment Use', 29, 24, 6.5, 48, 58, 2.4, 9
    elif btn_id == "btn-preset-mild":
        # Low risk moist conditions
        return 'WA', 'Debris Burning', 17, 58, 3.5, 14, 22, 0.9, 18
    elif btn_id == "btn-reset-sim":
        # Default scenario
        return 'CA', 'Lightning', 34, 12, 9.5, 65, 75, 3.2, 6

    return dash.no_update

# Real-time Slider Value Readouts
@app.callback(
    [
        Output("val-sim-temp", "children"),
        Output("val-sim-rh", "children"),
        Output("val-sim-wind", "children"),
        Output("val-sim-bi", "children"),
        Output("val-sim-erc", "children"),
        Output("val-sim-vpd", "children"),
        Output("val-sim-fm100", "children")
    ],
    [
        Input("sim-temp", "value"),
        Input("sim-rh", "value"),
        Input("sim-wind", "value"),
        Input("sim-bi", "value"),
        Input("sim-erc", "value"),
        Input("sim-vpd", "value"),
        Input("sim-fm100", "value")
    ]
)
def update_slider_labels(temp, rh, wind, bi, erc, vpd, fm100):
    return (
        f"{temp}°C",
        f"{rh}%",
        f"{wind} m/s",
        f"{bi}",
        f"{erc}",
        f"{vpd} kPa",
        f"{fm100}%"
    )

# Run Live Severity Prediction & Update Scenario Store
@app.callback(
    [
        Output("prediction-result-container", "children"),
        Output("simulated-scenario-store", "data")
    ],
    [
        Input("btn-predict-severity", "n_clicks"),
        Input("dashboard-tabs", "value")
    ],
    [
        State("sim-state", "value"),
        State("sim-cause", "value"),
        State("sim-temp", "value"),
        State("sim-rh", "value"),
        State("sim-wind", "value"),
        State("sim-bi", "value"),
        State("sim-erc", "value"),
        State("sim-vpd", "value"),
        State("sim-fm100", "value")
    ]
)
def run_live_prediction(n_clicks, tab_value, state, cause, temp, rh, wind, bi, erc, vpd, fm100):
    state = state or 'CA'
    cause = cause or 'Lightning'
    temp = temp if temp is not None else 34.0
    rh = rh if rh is not None else 12.0
    wind = wind if wind is not None else 9.5
    bi = bi if bi is not None else 65.0
    erc = erc if erc is not None else 75.0
    vpd = vpd if vpd is not None else 3.2
    fm100 = fm100 if fm100 is not None else 6.0

    input_payload = {
        'temp_max_c': float(temp),
        'temp_min_c': float(temp) - 12.0,
        'temp_range_c': 12.0,
        'rmin': float(rh),
        'rmax': min(float(rh) + 35.0, 100.0),
        'vs': float(wind),
        'pr': 0.0,
        'bi': float(bi),
        'erc': float(erc),
        'vpd': float(vpd),
        'fm100': float(fm100),
        'fm1000': float(fm100) + 2.0,
        'Elevation': 1100,
        'Slope': 20,
        'Aspect': 180,
        'TRI': 40.0,
        'TPI': 10.0,
        'DISCOVERY_DOY': 240,
        'Month': 8,
        'STATE': state,
        'NWCG_GENERAL_CAUSE': cause
    }

    result = predict_single_fire(input_payload)
    prob_chart = build_prediction_probability_chart(result['probabilities'])

    badge_theme = result['badge_color']
    pred_label = result['predicted_class']

    pred_ui = html.Div([
        html.Div([
            html.Div([
                html.Span("Predicted Growth Severity Class:", className="text-muted small text-uppercase"),
                html.H4(pred_label, className=f"text-{badge_theme} fw-bold mt-1 mb-0")
            ]),
            html.Span(f"Trained AI: {result['model_used']}", className="badge bg-secondary")
        ], className="d-flex justify-content-between align-items-center p-3 rounded mb-3", style={"backgroundColor": "#0b1220", "border": f"1px solid var(--border-color)"}),

        html.Div([
            html.Label("Class Confidence Probabilities", className="text-light small fw-bold mb-1"),
            dcc.Graph(figure=prob_chart, config={'displayModeBar': False})
        ], className="mb-3"),

        dbc.Alert(
            [
                html.H6("🚨 Tactical Guidance (Illustrative Scenario Modeling)", className="alert-heading fw-bold mb-1"),
                html.P(result['advisory'], className="mb-0 small")
            ],
            color=badge_theme,
            className="mb-3"
        ),

        html.Div([
            html.Small(f"Simulation Profile: State={state} | Cause={cause} | Temp={temp}°C | RH={rh}% | Wind={wind} m/s | BI={bi} | ERC={erc}", className="text-muted")
        ], className="text-end")
    ])

    return pred_ui, input_payload

# Update Local SHAP Waterfall from Simulated Scenario Store
@app.callback(
    Output("shap-local-waterfall", "figure"),
    [
        Input("simulated-scenario-store", "data"),
        Input("dashboard-tabs", "value")
    ]
)
def update_shap_waterfall(stored_scenario, tab_value):
    return build_local_waterfall_chart(stored_scenario)

# YOLO Callbacks
@app.callback(
    Output("val-yolo-conf", "children"),
    Input("yolo-conf-slider", "value")
)
def update_yolo_conf_label(conf):
    return f"{int(conf*100)}%"

@app.callback(
    Output("yolo-output-container", "children"),
    [
        Input("btn-run-yolo", "n_clicks"),
        Input("yolo-sample-selector", "value"),
        Input("dashboard-tabs", "value")
    ],
    [
        State("yolo-file-upload", "contents"),
        State("yolo-conf-slider", "value")
    ]
)
def run_yolo_inference(n_clicks, sample_choice, tab_value, upload_contents, conf_thresh):
    conf = conf_thresh if conf_thresh is not None else 0.20

    # Determine image source
    if sample_choice == "CUSTOM" and upload_contents:
        img_source = upload_contents
    else:
        sample_file = sample_choice if sample_choice != "CUSTOM" else "wildfire_aerial_sample.jpg"
        img_source = os.path.join(SAMPLES_DIR, sample_file)

    try:
        res = run_yolo_detection(img_source, conf_threshold=conf)
    except Exception as e:
        return dbc.Alert(f"YOLO Processing Error: {str(e)}", color="danger")

    # Metrics row with inference latency
    metric_cards = dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("Active Flames", className="kpi-title"),
                html.Div(f"{res['fire_count']}", className="kpi-value text-danger"),
                html.Div("Bounding detections", className="kpi-subtext")
            ], className="kpi-card kpi-fire")
        ], xs=6, md=3),

        dbc.Col([
            html.Div([
                html.Div("Smoke Plumes", className="kpi-title"),
                html.Div(f"{res['smoke_count']}", className="kpi-value text-info"),
                html.Div("Bounding detections", className="kpi-subtext")
            ], className="kpi-card kpi-blue")
        ], xs=6, md=3),

        dbc.Col([
            html.Div([
                html.Div("Peak Confidence", className="kpi-title"),
                html.Div(f"{res['max_confidence']:.1f}%", className="kpi-value text-warning"),
                html.Div("Top detector score", className="kpi-subtext")
            ], className="kpi-card kpi-amber")
        ], xs=6, md=3),

        dbc.Col([
            html.Div([
                html.Div("Inference Latency", className="kpi-title"),
                html.Div(f"{res['inference_time_ms']} ms", className="kpi-value text-light"),
                html.Div("PyTorch CPU execution", className="kpi-subtext")
            ], className="kpi-card kpi-green")
        ], xs=6, md=3),
    ], className="g-2 mb-3")

    # Threat Alert
    threat_alert = dbc.Alert([
        html.H6(f"🚨 Hazard Status: {res['hazard_level']}", className="alert-heading fw-bold mb-1"),
        html.P(res['hazard_msg'], className="mb-0 small")
    ], color=res['alert_theme'], className="mb-3")

    # Dual view: Original vs Annotated Side by Side
    image_display = dbc.Row([
        dbc.Col([
            html.Div([
                html.Small("📷 Original Field Frame", className="text-muted fw-bold d-block mb-1"),
                html.Img(
                    src=res.get('orig_image_uri', res['image_uri']),
                    style={"width": "100%", "borderRadius": "8px", "border": "1px solid #334155"}
                )
            ])
        ], xs=12, md=6, className="mb-2"),

        dbc.Col([
            html.Div([
                html.Small("🎯 YOLO Annotated Frame (Bounding Boxes & Confidences)", className="text-info fw-bold d-block mb-1"),
                html.Img(
                    src=res['image_uri'],
                    style={"width": "100%", "borderRadius": "8px", "border": "1px solid #38bdf8", "boxShadow": "0 4px 20px rgba(0,0,0,0.5)"}
                )
            ])
        ], xs=12, md=6, className="mb-2")
    ], className="mb-3")

    # Bounding Box Coordinate Table
    table_rows = []
    for idx, d in enumerate(res.get('detections', [])):
        b_class = d['class']
        badge_col = "danger" if b_class == "FIRE" else "info"
        coords_str = f"[{d['bbox'][0]}, {d['bbox'][1]}, {d['bbox'][2]}, {d['bbox'][3]}]"

        table_rows.append(html.Tr([
            html.Td(f"#{idx+1}", className="text-secondary"),
            html.Td(html.Span(b_class, className=f"badge bg-{badge_col}")),
            html.Td(f"{d['confidence']:.1f}%", className="text-light fw-bold"),
            html.Td(coords_str, className="text-muted font-monospace small")
        ]))

    detection_table = None
    if table_rows:
        detection_table = html.Div([
            html.H6("📋 Bounding Box Telemetry & Coordinates [x1, y1, x2, y2]", className="text-white fw-bold mb-2 small"),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("#"), html.Th("Object Class"), html.Th("Confidence"), html.Th("Box Coordinates [x1, y1, x2, y2]")
                ], className="small text-secondary")),
                html.Tbody(table_rows, className="small text-light")
            ], responsive=True, hover=True, className="align-middle mb-0")
        ], className="p-3 rounded bg-dark border border-secondary")

    return html.Div([
        threat_alert,
        metric_cards,
        image_display,
        detection_table
    ])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"\n=======================================================")
    print(f"🔥 Wildfire Dashboard Launching on http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(debug=False, host="127.0.0.1", port=port)
