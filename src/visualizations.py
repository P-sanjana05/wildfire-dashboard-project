"""
Wildfire Intelligence & Early Warning Dashboard - Visualization Components
Provides interactive Plotly charts, maps, and KPI card generators.
Designed with a sleek dark command-center theme.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Professional Color Palette
PALETTE = {
    'bg': '#0f172a',        # Deep Slate 900
    'card_bg': '#1e293b',   # Slate 800
    'border': '#334155',    # Slate 700
    'text': '#f8fafc',      # Slate 50
    'muted': '#94a3b8',     # Slate 400
    'accent_fire': '#ef4444', # Fire Red
    'accent_orange': '#f97316', # Orange
    'accent_yellow': '#eab308', # Amber / Yellow
    'accent_blue': '#38bdf8',  # Sky Blue
    'accent_green': '#10b981', # Emerald
    'purple': '#a855f7'
}

SEVERITY_COLORS = {
    'Low (Class A-B, <10 ac)': '#10b981',      # Emerald Green
    'Moderate (Class C, 10-99 ac)': '#f59e0b', # Amber
    'Severe (Class D-G, ≥100 ac)': '#ef4444'   # Fire Red
}

def create_kpi_card(title, value, subtext=None, border_color="#ef4444", icon=None):
    """Creates a stylized metric card dictionary for HTML/Dash layout."""
    return {
        'title': title,
        'value': value,
        'subtext': subtext,
        'border_color': border_color,
        'icon': icon
    }

def build_wildfire_map(df, viirs_df=None, show_viirs=True):
    """
    Builds the primary interactive geospatial wildfire map.
    Renders historical wildfire incidents colored by severity, with optional VIIRS satellite overlay.
    """
    fig = go.Figure()

    if df is not None and not df.empty:
        # Sort so severe fires render on top
        df_sorted = df.sort_values(by='FIRE_SIZE', ascending=True)

        for severity, color in SEVERITY_COLORS.items():
            sub = df_sorted[df_sorted['Severity_Category'] == severity]
            if sub.empty:
                continue

            # Scale marker size reasonably for map display
            if severity.startswith('Severe'):
                marker_size = np.clip(np.log1p(sub['FIRE_SIZE']) * 1.8, 7, 22)
                opacity = 0.85
            elif severity.startswith('Moderate'):
                marker_size = 5.5
                opacity = 0.7
            else:
                marker_size = 3.5
                opacity = 0.5

            customdata = np.stack((
                sub['FIRE_NAME'],
                sub['COUNTY'],
                sub['STATE'],
                sub['FIRE_SIZE'].apply(lambda x: f"{x:,.1f}"),
                sub['DISCOVERY_DATE'].astype(str),
                sub['NWCG_GENERAL_CAUSE'],
                sub['temp_max_c'].apply(lambda x: f"{x:.1f}°C"),
                sub['bi'].apply(lambda x: f"{x:.0f}"),
                sub['erc'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "N/A"),
                sub['vs'].apply(lambda x: f"{x:.1f} m/s" if pd.notnull(x) else "N/A"),
                sub['rmin'].apply(lambda x: f"{x:.0f}%" if pd.notnull(x) else "N/A")
            ), axis=-1)

            legend_label = f"Historical Incident: {severity}"

            fig.add_trace(go.Scattergeo(
                lon=sub['LONGITUDE'],
                lat=sub['LATITUDE'],
                mode='markers',
                name=legend_label,
                marker=dict(
                    size=marker_size,
                    color=color,
                    opacity=opacity,
                    line=dict(width=0.5, color='#000000')
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>🔥 Historical Wildfire (2020 Archive)</b><br>"
                    "<b>Incident:</b> %{customdata[0]}<br>"
                    "<b>Location:</b> %{customdata[1]}, %{customdata[2]}<br>"
                    "<b>Discovery Date:</b> %{customdata[4]}<br>"
                    "<b>Fire Size:</b> %{customdata[3]} acres<br>"
                    "<b>Cause:</b> %{customdata[5]}<br>"
                    "<b>Max Temp:</b> %{customdata[6]} | <b>Burning Index:</b> %{customdata[7]}<br>"
                    "<i>Click marker on map to view full incident details below</i>"
                    "<extra></extra>"
                )
            ))

    # Add VIIRS Satellite Thermal Hotspots if enabled
    if show_viirs and viirs_df is not None and not viirs_df.empty:
        v_customdata = np.stack((
            viirs_df['bright_ti4'].apply(lambda x: f"{x:.1f} K"),
            viirs_df['frp'].apply(lambda x: f"{x:.2f} MW"),
            viirs_df['confidence'].astype(str),
            viirs_df['acq_date'].astype(str) + " " + viirs_df['acq_time'].astype(str),
            viirs_df['satellite'].astype(str),
            viirs_df['daynight'].astype(str)
        ), axis=-1)

        fig.add_trace(go.Scattergeo(
            lon=viirs_df['longitude'],
            lat=viirs_df['latitude'],
            mode='markers',
            name='Satellite Hotspots: Suomi VIIRS NRT (Active Thermal Anomalies)',
            marker=dict(
                size=6.5,
                color='#38bdf8',  # Neon Cyan/Blue
                symbol='diamond',
                opacity=0.9,
                line=dict(width=1, color='#ffffff')
            ),
            customdata=v_customdata,
            hovertemplate=(
                "<b>🛰️ Satellite Thermal Detection (VIIRS NRT)</b><br>"
                "<b>Brightness Temp (Ti4):</b> %{customdata[0]}<br>"
                "<b>Fire Radiative Power:</b> %{customdata[1]}<br>"
                "<b>Confidence:</b> %{customdata[2]}<br>"
                "<b>Acquired:</b> %{customdata[3]} (%{customdata[5]} orbit)<br>"
                "<b>Satellite:</b> Suomi NPP (%{customdata[4]})"
                "<extra></extra>"
            )
        ))

    # Configure US Geographic scope & dark basemap style
    fig.update_layout(
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showland=True,
            landcolor='#1e293b',
            showlakes=True,
            lakecolor='#0f172a',
            showsubunits=True,
            subunitcolor='#334155',
            countrycolor='#475569',
            showcountries=True,
            bgcolor='#0f172a'
        ),
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=0.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(15, 23, 42, 0.85)',
            bordercolor='#334155',
            borderwidth=1,
            font=dict(color='#f8fafc', size=11)
        ),
        paper_bgcolor='#0f172a',
        plot_bgcolor='#0f172a',
        height=520
    )
    return fig

def build_temporal_trend_chart(df):
    """Builds dual-axis chart: Monthly Incident Count vs Total Acres Burned."""
    if df is None or df.empty:
        return go.Figure()

    monthly = df.groupby('Month').agg(
        Incident_Count=('FOD_ID', 'count'),
        Total_Acres=('FIRE_SIZE', 'sum'),
        Severe_Count=('Is_Severe', 'sum')
    ).reset_index()

    month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                   7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    monthly['Month_Label'] = monthly['Month'].map(month_names)

    fig = go.Figure()

    # Bar chart for Total Acres Burned
    fig.add_trace(go.Bar(
        x=monthly['Month_Label'],
        y=monthly['Total_Acres'],
        name='Total Acres Burned',
        marker=dict(color='#f97316', opacity=0.8),
        yaxis='y1',
        hovertemplate='<b>%{x}</b><br>Burned: %{y:,.0f} acres<extra></extra>'
    ))

    # Line chart for Incident Count
    fig.add_trace(go.Scatter(
        x=monthly['Month_Label'],
        y=monthly['Incident_Count'],
        name='Total Fire Incidents',
        mode='lines+markers',
        line=dict(color='#38bdf8', width=3),
        marker=dict(size=7, color='#38bdf8'),
        yaxis='y2',
        hovertemplate='<b>%{x}</b><br>Fires: %{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(
            title='Month (2020)',
            gridcolor='#1e293b',
            showgrid=True,
            linecolor='#334155'
        ),
        yaxis=dict(
            title='Burned Area (Acres)',
            titlefont=dict(color='#f97316'),
            tickfont=dict(color='#f97316'),
            gridcolor='#1e293b',
            showgrid=True
        ),
        yaxis2=dict(
            title='Incident Count',
            titlefont=dict(color='#38bdf8'),
            tickfont=dict(color='#38bdf8'),
            overlaying='y',
            side='right',
            showgrid=False
        ),
        height=320
    )
    return fig

def build_cause_breakdown_chart(df):
    """Horizontal bar chart showing fires by NWCG general cause."""
    if df is None or df.empty:
        return go.Figure()

    cause_df = df.groupby('NWCG_GENERAL_CAUSE').agg(
        Count=('FOD_ID', 'count'),
        Acres=('FIRE_SIZE', 'sum')
    ).reset_index().sort_values(by='Count', ascending=True).tail(8)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cause_df['NWCG_GENERAL_CAUSE'],
        x=cause_df['Count'],
        orientation='h',
        marker=dict(
            color=cause_df['Count'],
            colorscale='OrRd',
            showscale=False
        ),
        customdata=cause_df['Acres'].apply(lambda x: f"{x:,.0f}"),
        hovertemplate='<b>%{y}</b><br>Fires: %{x:,}<br>Acres Burned: %{customdata}<extra></extra>'
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc', size=11),
        margin=dict(l=10, r=20, t=20, b=20),
        xaxis=dict(title='Recorded Incidents', gridcolor='#1e293b', showgrid=True),
        yaxis=dict(autorange='reversed', linecolor='#334155'),
        height=320
    )
    return fig

def build_burned_area_distribution(df):
    """Log-scale histogram/distribution of historical fire sizes with explicit data interpretation."""
    if df is None or df.empty:
        return go.Figure()

    log_sizes = np.log10(np.clip(df['FIRE_SIZE'], 0.001, None))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=log_sizes,
        nbinsx=42,
        marker=dict(color='#f97316', line=dict(color='#0f172a', width=1)),
        hovertemplate='Log10(Acres): %{x:.2f}<br>Count: %{y:,}<extra></extra>'
    ))

    # Add vertical reference lines for class boundaries
    fig.add_vline(x=1.0, line_dash='dash', line_color='#f59e0b',
                  annotation_text='Class C Threshold (10 ac)', annotation_position='top left',
                  annotation_font=dict(color='#f59e0b', size=11))
    fig.add_vline(x=2.0, line_dash='dash', line_color='#ef4444',
                  annotation_text='Large Fire Threshold (100 ac)', annotation_position='top right',
                  annotation_font=dict(color='#ef4444', size=11))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        margin=dict(l=40, r=30, t=30, b=45),
        xaxis=dict(
            title='Log₁₀(Acreage) Scale: 0=1 ac | 1=10 ac (Class C) | 2=100 ac (Severe Fire) | 4=10,000 ac',
            gridcolor='#1e293b',
            showgrid=True
        ),
        yaxis=dict(title='Incident Count', gridcolor='#1e293b', showgrid=True),
        height=320
    )
    return fig

def build_environmental_boxplots(df, variable='bi'):
    """Boxplot of pre-fire danger indices with explicit class sample counts (n=...) and insights."""
    if df is None or df.empty:
        return go.Figure()

    var_labels = {
        'bi': 'Burning Index (BI - Potential Flame Front Intensity)',
        'erc': 'Energy Release Component (ERC - Fuel Heat Release Potential)',
        'temp_max_c': 'Maximum Temperature (°C)',
        'vpd': 'Vapor Pressure Deficit (VPD - Atmospheric Drying Demand, kPa)',
        'vs': 'Wind Speed (m/s)',
        'rmin': 'Minimum Relative Humidity (%)'
    }

    title = var_labels.get(variable, variable)
    counts = df['Severity_Category'].value_counts()

    fig = go.Figure()
    for severity, color in SEVERITY_COLORS.items():
        sub = df[df['Severity_Category'] == severity]
        if sub.empty:
            continue
        c_n = counts.get(severity, len(sub))
        cat_short = severity.split(' ')[0]

        fig.add_trace(go.Box(
            y=sub[variable],
            name=f"{cat_short} (n={c_n:,})",
            marker_color=color,
            boxmean=True,
            hovertemplate=f"<b>{severity}</b><br>Sample Count: {c_n:,}<br>{title}: %{{y:.2f}}<extra></extra>"
        ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        margin=dict(l=40, r=20, t=20, b=30),
        yaxis=dict(title=title, gridcolor='#1e293b', showgrid=True),
        xaxis=dict(title='Historical Severity Class (Sample Counts Shown in Legend)', gridcolor='#1e293b'),
        showlegend=False,
        height=320
    )
    return fig

def build_correlation_heatmap(df):
    """Pre-fire environmental correlation matrix heatmap."""
    cols = ['FIRE_SIZE', 'temp_max_c', 'rmin', 'vs', 'bi', 'erc', 'vpd', 'fm100', 'Elevation']
    labels = ['Fire Size', 'Max Temp', 'Min RH', 'Wind Spd', 'Burning Idx', 'Energy Release', 'VPD', '100h Moisture', 'Elevation']

    sub = df[cols].dropna()
    sub['Log_Fire_Size'] = np.log10(np.clip(sub['FIRE_SIZE'], 0.001, None))
    corr_cols = ['Log_Fire_Size', 'temp_max_c', 'rmin', 'vs', 'bi', 'erc', 'vpd', 'fm100', 'Elevation']
    corr = sub[corr_cols].corr(method='spearman').round(2)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        text=corr.values,
        texttemplate="%{text}",
        textfont={"size": 10, "color": "white"}
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc', size=11),
        margin=dict(l=40, r=20, t=20, b=20),
        height=340
    )
    return fig

def build_model_metrics_comparison_chart(metrics_summary):
    """Grouped bar chart highlighting Macro F1, Severe Fire Recall, Precision, and Accuracy."""
    models_dict = metrics_summary.get('models', {})
    model_names = list(models_dict.keys())

    macro_f1s = [models_dict[m]['macro_f1'] for m in model_names]
    accuracies = [models_dict[m]['accuracy'] for m in model_names]

    sev_recalls = []
    sev_precisions = []
    for m in model_names:
        cm = models_dict[m].get('confusion_matrix', [[0,0,0],[0,0,0],[0,0,0]])
        counts = models_dict[m].get('confusion_counts', [[0,0,0],[0,0,0],[0,0,0]])
        s_rec = cm[2][2] if len(cm) > 2 else 0.0
        c_sum = (counts[0][2] + counts[1][2] + counts[2][2]) if len(counts) > 2 else 0
        s_prec = (counts[2][2] / c_sum) if c_sum > 0 else 0.0
        sev_recalls.append(round(s_rec, 4))
        sev_precisions.append(round(s_prec, 4))

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Macro F1 (Primary Selection)', x=model_names, y=macro_f1s, marker_color='#38bdf8', text=[f"{v:.4f}" for v in macro_f1s], textposition='auto'))
    fig.add_trace(go.Bar(name='Severe Fire Recall (≥100 ac)', x=model_names, y=sev_recalls, marker_color='#ef4444', text=[f"{v*100:.1f}%" for v in sev_recalls], textposition='auto'))
    fig.add_trace(go.Bar(name='Severe Fire Precision', x=model_names, y=sev_precisions, marker_color='#f59e0b', text=[f"{v*100:.1f}%" for v in sev_precisions], textposition='auto'))
    fig.add_trace(go.Bar(name='Test Accuracy (Overall)', x=model_names, y=accuracies, marker_color='#10b981', text=[f"{v*100:.1f}%" for v in accuracies], textposition='auto'))

    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        margin=dict(l=30, r=20, t=25, b=25),
        yaxis=dict(title='Metric Score (0.0 - 1.0)', range=[0, 1.0], gridcolor='#1e293b'),
        xaxis=dict(gridcolor='#1e293b'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        height=340
    )
    return fig

def build_confusion_matrix_heatmap(cm_matrix, class_names, model_name):
    """Normalized Confusion Matrix Heatmap for specified model."""
    cm = np.array(cm_matrix)
    text_labels = [[f"{val*100:.1f}%" for val in row] for row in cm]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Pred: {c}" for c in class_names],
        y=[f"True: {c}" for c in class_names],
        colorscale='YlOrRd',
        zmin=0.0,
        zmax=1.0,
        text=text_labels,
        texttemplate="%{text}",
        textfont={"size": 13, "color": "white"}
    ))

    fig.update_layout(
        title=dict(text=f"{model_name} Normalized Confusion Matrix (Test Set)", font=dict(color='#f8fafc', size=13)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        margin=dict(l=30, r=20, t=40, b=30),
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed'),
        height=320
    )
    return fig

def build_prediction_probability_chart(probabilities_dict):
    """Horizontal probability distribution chart for live severity inference."""
    classes = list(probabilities_dict.keys())
    probs = [probabilities_dict[c] * 100 for c in classes]
    colors = ['#10b981', '#f59e0b', '#ef4444']

    fig = go.Figure(go.Bar(
        x=probs,
        y=classes,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{p:.1f}%" for p in probs],
        textposition='auto',
        textfont=dict(color='white', size=12)
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        margin=dict(l=10, r=20, t=10, b=10),
        xaxis=dict(title='Confidence Probability (%)', range=[0, 100], gridcolor='#1e293b'),
        yaxis=dict(autorange='reversed'),
        height=190
    )
    return fig

