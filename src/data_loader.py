"""
Wildfire Intelligence & Early Warning Dashboard - Data Loading & Feature Pipeline
Loads, cleans, and prepares FPA FOD Wildfire records and VIIRS satellite active fire data.
Ensures leakage-free temporal ordering and robust error handling.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Default paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
RAW_FPA_CSV = os.path.join(DOWNLOADS_DIR, "2020_FPA_FOD_cons.csv")
PROCESSED_CSV = os.path.join(DATA_DIR, "wildfires_processed.csv")
SAMPLE_CSV = os.path.join(DATA_DIR, "wildfires_map_sample.csv")
VIIRS_CSV = os.path.join(DATA_DIR, "viirs_hotspots.csv")

TARGET_COLS = [
    'FOD_ID', 'FIRE_NAME', 'DISCOVERY_DATE', 'DISCOVERY_DOY',
    'LATITUDE', 'LONGITUDE', 'STATE', 'COUNTY', 'NWCG_GENERAL_CAUSE',
    'FIRE_SIZE', 'FIRE_SIZE_CLASS',
    'Elevation', 'Slope', 'Aspect', 'TRI', 'TPI',
    'tmmx', 'tmmn', 'rmin', 'rmax', 'vs', 'pr', 'bi', 'erc', 'vpd', 'fm100', 'fm1000'
]

# Valid pre-fire environmental & topographical features (No target leakage!)
FEATURE_COLUMNS = [
    'Elevation', 'Slope', 'Aspect', 'TRI', 'TPI',
    'temp_max_c', 'temp_min_c', 'temp_range_c',
    'rmin', 'rmax', 'vs', 'pr',
    'bi', 'erc', 'vpd', 'fm100', 'fm1000',
    'DISCOVERY_DOY', 'Month'
]

CATEGORICAL_FEATURES = ['STATE', 'NWCG_GENERAL_CAUSE']

def categorize_severity(size):
    """Categorize fire size into standard operational severity classes."""
    if size < 10.0:
        return 'Low (Class A-B, <10 ac)'
    elif size < 100.0:
        return 'Moderate (Class C, 10-99 ac)'
    else:
        return 'Severe (Class D-G, ≥100 ac)'

def categorize_severity_binary(size):
    """Binary severity: 1 if escaped initial attack / significant (>=10 ac), 0 if contained small (<10 ac)."""
    return 1 if size >= 10.0 else 0

def prepare_and_clean_data(raw_csv_path=RAW_FPA_CSV, force_recompute=False):
    """Reads raw 2020 FPA FOD CSV, creates derived features, and caches cleaned data."""
    if os.path.exists(PROCESSED_CSV) and not force_recompute:
        print(f"[DataLoader] Using existing processed data: {PROCESSED_CSV}")
        return pd.read_csv(PROCESSED_CSV)

    if not os.path.exists(raw_csv_path):
        # Look in workspace data folder if not in Downloads
        alt_path = os.path.join(DATA_DIR, "2020_FPA_FOD_cons.csv")
        if os.path.exists(alt_path):
            raw_csv_path = alt_path
        else:
            raise FileNotFoundError(f"Could not find raw dataset at {raw_csv_path} or {alt_path}")

    print(f"[DataLoader] Reading raw dataset from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path, usecols=lambda c: c in TARGET_COLS)

    # Clean text columns
    df['FIRE_NAME'] = df['FIRE_NAME'].fillna('Unnamed Incident').astype(str).str.strip()
    df['COUNTY'] = df['COUNTY'].fillna('Unknown').astype(str).str.strip()
    df['NWCG_GENERAL_CAUSE'] = df['NWCG_GENERAL_CAUSE'].fillna('Missing data/not specified/undetermined').astype(str)

    # Convert Discovery Date
    df['DISCOVERY_DATE'] = pd.to_datetime(df['DISCOVERY_DATE'])
    df['Month'] = df['DISCOVERY_DATE'].dt.month
    df['Month_Name'] = df['DISCOVERY_DATE'].dt.strftime('%b')
    df['DayOfWeek'] = df['DISCOVERY_DATE'].dt.day_name()

    # Environmental feature engineering (temperatures from Kelvin to Celsius)
    df['temp_max_c'] = np.round(df['tmmx'] - 273.15, 2)
    df['temp_min_c'] = np.round(df['tmmn'] - 273.15, 2)
    df['temp_range_c'] = np.round(df['temp_max_c'] - df['temp_min_c'], 2)

    # Handle negligible missing values in environmental continuous columns (median impute)
    env_cols = ['TRI', 'TPI', 'pr', 'tmmn', 'tmmx', 'rmin', 'rmax', 'vs', 'fm100', 'fm1000', 'bi', 'vpd', 'erc',
                'temp_max_c', 'temp_min_c', 'temp_range_c']
    for col in env_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Severity targets
    df['Severity_Category'] = df['FIRE_SIZE'].apply(categorize_severity)
    df['Is_Severe'] = df['FIRE_SIZE'].apply(categorize_severity_binary)
    # Severity 3-class numeric (0: Low, 1: Moderate, 2: Severe)
    severity_map = {
        'Low (Class A-B, <10 ac)': 0,
        'Moderate (Class C, 10-99 ac)': 1,
        'Severe (Class D-G, ≥100 ac)': 2
    }
    df['Severity_Class_Code'] = df['Severity_Category'].map(severity_map)

    # Sort chronologically (leakage-free ordering)
    df = df.sort_values(by=['DISCOVERY_DATE', 'DISCOVERY_DOY']).reset_index(drop=True)

    # Ensure output dir exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Save cleaned processed CSV
    df.to_csv(PROCESSED_CSV, index=False)
    print(f"[DataLoader] Successfully saved processed dataset: {len(df):,} records to {PROCESSED_CSV}")

    # Create stratified map sample for blazing-fast Dash map rendering
    # Keep 100% of severe fires (>= 100 acres: 2,072) and moderate fires (10-99 acres: 5,842)
    # and sample 8,000 small fires for total ~15,900 points
    large_fires = df[df['FIRE_SIZE'] >= 10.0]
    small_fires = df[df['FIRE_SIZE'] < 10.0].sample(n=min(8000, len(df[df['FIRE_SIZE'] < 10.0])), random_state=42)
    map_sample = pd.concat([large_fires, small_fires]).sort_values(by='DISCOVERY_DATE').reset_index(drop=True)
    map_sample.to_csv(SAMPLE_CSV, index=False)
    print(f"[DataLoader] Successfully saved map sample: {len(map_sample):,} records to {SAMPLE_CSV}")

    return df

def load_processed_data():
    """Loads processed data from disk or runs preparation if not cached."""
    if os.path.exists(PROCESSED_CSV):
        df = pd.read_csv(PROCESSED_CSV)
        df['DISCOVERY_DATE'] = pd.to_datetime(df['DISCOVERY_DATE'])
        return df
    return prepare_and_clean_data()

def load_map_sample():
    """Loads map-ready sample for high-performance interactive Plotly rendering."""
    if os.path.exists(SAMPLE_CSV):
        df = pd.read_csv(SAMPLE_CSV)
        df['DISCOVERY_DATE'] = pd.to_datetime(df['DISCOVERY_DATE'])
        return df
    full_df = load_processed_data()
    if os.path.exists(SAMPLE_CSV):
        return pd.read_csv(SAMPLE_CSV)
    return full_df.sample(n=min(15000, len(full_df)), random_state=42)

def load_viirs_data():
    """Loads Suomi VIIRS Near Real Time satellite fire detections if available."""
    if os.path.exists(VIIRS_CSV):
        try:
            viirs = pd.read_csv(VIIRS_CSV)
            # Filter valid coordinates
            viirs = viirs.dropna(subset=['latitude', 'longitude'])
            return viirs
        except Exception as e:
            print(f"[DataLoader] Warning loading VIIRS data: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

if __name__ == "__main__":
    df = prepare_and_clean_data()
    print("Done. Sample preview:")
    print(df[['DISCOVERY_DATE', 'STATE', 'FIRE_SIZE', 'Severity_Category', 'temp_max_c', 'bi', 'erc']].head(5))
