"""
Wildfire Intelligence & Early Warning Dashboard - Advanced Model Training & Evaluation
Trains and compares XGBoost, LightGBM, and CatBoost for historical wildfire severity classification.
Enforces a strict chronological leakage-free train/test split.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "wildfires_processed.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")
ENCODER_PATH = os.path.join(MODELS_DIR, "feature_encoder.joblib")

FEATURES = [
    'temp_max_c', 'temp_min_c', 'temp_range_c', 'rmin', 'rmax', 'vs', 'pr',
    'bi', 'erc', 'vpd', 'fm100', 'fm1000',
    'Elevation', 'Slope', 'Aspect', 'TRI', 'TPI',
    'DISCOVERY_DOY', 'Month', 'STATE', 'NWCG_GENERAL_CAUSE'
]

CLASS_NAMES = ['Low (<10 ac)', 'Moderate (10-99 ac)', 'Severe (>=100 ac)']

def train_and_evaluate_models(force_retrain=False):
    """
    Trains XGBoost, LightGBM, and CatBoost with a chronological split (Jan-Aug 2020 Train, Sep-Dec 2020 Test).
    Saves models and metrics to disk.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.exists(METRICS_PATH) and not force_retrain:
        print(f"[ModelTrainer] Found existing model metrics at {METRICS_PATH}.")
        with open(METRICS_PATH, 'r') as f:
            return json.load(f)

    print(f"[ModelTrainer] Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # Chronological Split (No Temporal Leakage!)
    # Jan 1 - Aug 31: Training
    # Sep 1 - Dec 31: Testing (peak autumn wildfire siege)
    train_df = df[df['DISCOVERY_DATE'] <= '2020-08-31'].copy()
    test_df = df[df['DISCOVERY_DATE'] >= '2020-09-01'].copy()

    print(f"[ModelTrainer] Chronological Split: Train={len(train_df):,} | Test={len(test_df):,}")

    # Encode Categoricals
    cat_cols = ['STATE', 'NWCG_GENERAL_CAUSE']
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train_df[cat_cols] = encoder.fit_transform(train_df[cat_cols])
    test_df[cat_cols] = encoder.transform(test_df[cat_cols])

    # Save encoder
    joblib.dump(encoder, ENCODER_PATH)

    X_train = train_df[FEATURES]
    y_train = train_df['Severity_Class_Code'].values
    X_test = test_df[FEATURES]
    y_test = test_df['Severity_Class_Code'].values

    # Sample weights for class balancing
    sample_weights_train = compute_sample_weight('balanced', y_train)

    models = {
        'XGBoost': XGBClassifier(
            n_estimators=180,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=180,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.08,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=180,
            depth=6,
            learning_rate=0.08,
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=0,
            thread_count=-1
        )
    }

    results = {}
    best_macro_f1 = -1
    best_model_name = None

    for name, model in models.items():
        print(f"\n[ModelTrainer] Training {name} on {len(X_train):,} incidents...")
        if name == 'XGBoost':
            model.fit(X_train, y_train, sample_weight=sample_weights_train)
        else:
            model.fit(X_train, y_train)

        # Predictions on chronological test set
        y_pred = model.predict(X_test)
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()

        y_proba = model.predict_proba(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average='macro'))
        weighted_f1 = float(f1_score(y_test, y_pred, average='weighted'))
        macro_rec = float(recall_score(y_test, y_pred, average='macro'))
        macro_prec = float(precision_score(y_test, y_pred, average='macro', zero_division=0))

        # Normalized confusion matrix (percentage of actual class)
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        cm_counts = confusion_matrix(y_test, y_pred)

        # Save model artifact
        model_save_path = os.path.join(MODELS_DIR, f"{name.lower()}_model.joblib")
        joblib.dump(model, model_save_path)

        results[name] = {
            'accuracy': round(acc, 4),
            'macro_f1': round(macro_f1, 4),
            'weighted_f1': round(weighted_f1, 4),
            'macro_recall': round(macro_rec, 4),
            'macro_precision': round(macro_prec, 4),
            'confusion_matrix': cm.round(4).tolist(),
            'confusion_counts': cm_counts.tolist(),
            'model_path': model_save_path
        }
        print(f"[{name}] Test Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | Macro Recall: {macro_rec:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model_name = name

    summary = {
        'split_info': {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_period': '2020-01-01 to 2020-08-31',
            'test_period': '2020-09-01 to 2020-12-31'
        },
        'best_model': best_model_name,
        'features': FEATURES,
        'class_names': CLASS_NAMES,
        'models': results
    }

    with open(METRICS_PATH, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[ModelTrainer] Comparison Complete! Best Model: {best_model_name} (Macro F1 = {best_macro_f1:.4f})")
    print(f"[ModelTrainer] Metrics saved to {METRICS_PATH}")
    return summary

def load_best_model():
    """Loads the highest-scoring model and encoder for live inference."""
    if not os.path.exists(METRICS_PATH):
        train_and_evaluate_models()

    with open(METRICS_PATH, 'r') as f:
        meta = json.load(f)

    best_name = meta['best_model']
    model_path = os.path.join(MODELS_DIR, f"{best_name.lower()}_model.joblib")
    model = joblib.load(model_path)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder, best_name, meta

def predict_single_fire(input_data):
    """
    Takes an input dictionary with features and returns predicted severity,
    class probabilities, and operational risk advice.
    """
    model, encoder, best_name, meta = load_best_model()

    row_df = pd.DataFrame([input_data])
    # Encode categorical columns
    cat_cols = ['STATE', 'NWCG_GENERAL_CAUSE']
    for col in cat_cols:
        if col not in row_df.columns:
            row_df[col] = 'Unknown'
    row_df[cat_cols] = encoder.transform(row_df[cat_cols])

    # Ensure all required features exist
    for f in FEATURES:
        if f not in row_df.columns:
            row_df[f] = 0.0

    X_in = row_df[FEATURES]
    pred_code = int(model.predict(X_in)[0])
    probas = model.predict_proba(X_in)[0]

    severity_label = CLASS_NAMES[pred_code]

    # Advisory recommendation
    if pred_code == 2:
        advisory = "CRITICAL: High probability of rapid escape (>=100 acres). High-intensity suppression assets & multi-agency containment recommended."
        badge_color = "danger"
    elif pred_code == 1:
        advisory = "ELEVATED: Moderate fire growth potential (10-99 acres). Early initial attack within 60 minutes critical to prevent crowning."
        badge_color = "warning"
    else:
        advisory = "CONTROLLED: High likelihood of rapid initial containment (<10 acres) under current meteorological and fuel conditions."
        badge_color = "success"

    return {
        'model_used': best_name,
        'predicted_class': severity_label,
        'class_code': pred_code,
        'probabilities': {
            'Low (<10 ac)': round(float(probas[0]), 3),
            'Moderate (10-99 ac)': round(float(probas[1]), 3),
            'Severe (>=100 ac)': round(float(probas[2]), 3)
        },
        'advisory': advisory,
        'badge_color': badge_color
    }

if __name__ == "__main__":
    train_and_evaluate_models()
