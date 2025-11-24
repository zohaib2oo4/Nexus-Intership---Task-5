
import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, \
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------
# CONFIG
# -----------------------
ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
LOG_FILE = ROOT / "pipeline_log.txt"

UPLOADED_PATH = Path("/mnt/data/SuperMarket Analysis.csv")  # initial uploaded copy

for p in (DATA_DIR, MODELS_DIR, OUTPUT_DIR):
    p.mkdir(parents=True, exist_ok=True)

# REQUIRED columns (must match your dataset exactly)
REQUIRED = {
    "date": "Date",
    "time": "Time",
    "target": "Sales",
    "numeric": [
        "Unit price", "Quantity", "Tax 5%", "cogs", "gross income", "Rating"
    ],
    "categorical": [
        "Branch", "City", "Customer type", "Gender", "Product line", "Payment"
    ],
    # additional derived features will be Year, Month, Day, Hour, Minute
}

# Copy uploaded file into data/ if data/ empty and uploaded exists
if UPLOADED_PATH.exists() and not any(DATA_DIR.glob("*.csv")):
    try:
        shutil.copy(UPLOADED_PATH, DATA_DIR / UPLOADED_PATH.name)
        print(f"Copied uploaded dataset to data/: {UPLOADED_PATH.name}")
    except Exception as e:
        print(f"Warning: failed to copy uploaded dataset: {e}")

# -----------------------
# Helpers
# -----------------------
def log(message):
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"[{ts}] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def list_csvs():
    return sorted(DATA_DIR.glob("*.csv"))

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed reading CSV {path}: {e}")

def validate_columns(df):
    missing = []
    # required date & time & target
    if REQUIRED["date"] not in df.columns:
        missing.append(REQUIRED["date"])
    if REQUIRED["time"] not in df.columns:
        missing.append(REQUIRED["time"])
    if REQUIRED["target"] not in df.columns:
        missing.append(REQUIRED["target"])
    for c in REQUIRED["numeric"] + REQUIRED["categorical"]:
        if c not in df.columns:
            missing.append(c)
    return missing

def extract_datetime_features(df):
    # create Year, Month, Day, Hour, Minute
    df = df.copy()
    # Date parsing; coerce errors to NaT
    df[REQUIRED["date"]] = pd.to_datetime(df[REQUIRED["date"]], errors="coerce")
    # Time parsing: if time column exists as string like '1:08:00 PM', parse with to_datetime
    df[REQUIRED["time"]] = pd.to_datetime(df[REQUIRED["time"]], format="%I:%M:%S %p", errors="coerce").dt.time
    # Some datasets may have time parse issues; handle gracefully
    # Combine date + time into a datetime if time parsed
    try:
        df["__datetime__"] = df[REQUIRED["date"]].dt.date.astype("datetime64[ns]")
        if df[REQUIRED["time"]].notna().any():
            # convert time to timedelta hours/minutes
            df["Hour"] = df[REQUIRED["time"]].apply(lambda t: t.hour if not pd.isna(t) else np.nan)
            df["Minute"] = df[REQUIRED["time"]].apply(lambda t: t.minute if not pd.isna(t) else np.nan)
        else:
            df["Hour"] = np.nan
            df["Minute"] = np.nan
    except Exception:
        df["Hour"] = np.nan
        df["Minute"] = np.nan

    df["Year"] = df[REQUIRED["date"]].dt.year
    df["Month"] = df[REQUIRED["date"]].dt.month
    df["Day"] = df[REQUIRED["date"]].dt.day

    return df

# -----------------------
# Build sklearn pipeline
# -----------------------
def build_pipeline(numeric_features, categorical_features):
    # numeric: median impute + scale
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    # categorical: most_frequent + onehot
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="drop")
    # final pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))
    ])
    return pipeline

# -----------------------
# Metrics
# -----------------------
def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "MSE": float(mse), "RMSE": rmse, "R2": float(r2)}

def classification_metrics_from_regression(y_true, y_pred, threshold):
    y_true_bin = (y_true > threshold).astype(int)
    y_pred_bin = (y_pred > threshold).astype(int)
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    cm = confusion_matrix(y_true_bin, y_pred_bin).tolist()
    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "confusion_matrix": cm}

# -----------------------
# Main pipeline run
# -----------------------
def run_pipeline_once():
    csvs = list_csvs()
    if not csvs:
        log("No CSV files found in data/. Place CSV(s) there and rerun.")
        return

    train_path = csvs[0]
    new_paths = csvs[1:]

    log(f"Training CSV: {train_path.name}")
    if new_paths:
        log(f"Found {len(new_paths)} new CSV(s): {[p.name for p in new_paths]}")
    else:
        log("No additional CSVs found. Will train and evaluate on training CSV only.")

    # Read train CSV and validate
    df_train = safe_read_csv(train_path)
    missing = validate_columns(df_train)
    if missing:
        log(f"ERROR: Training CSV is missing required columns: {missing}")
        return

    # Extract date/time features
    df_train = extract_datetime_features(df_train)
    # Drop rows with missing date (if any)
    n_before = len(df_train)
    df_train = df_train.dropna(subset=[REQUIRED["date"]])
    if len(df_train) < n_before:
        log(f"Dropped {n_before - len(df_train)} rows with invalid Date in training CSV")

    # Build feature lists
    numeric_feats = REQUIRED["numeric"] + ["Year", "Month", "Day", "Hour", "Minute"]
    categorical_feats = REQUIRED["categorical"]

    # Ensure numeric columns exist; coerce to numeric where possible
    for col in numeric_feats:
        if col in df_train.columns:
            df_train[col] = pd.to_numeric(df_train[col], errors="coerce")

    # Drop rows with missing target
    df_train = df_train.dropna(subset=[REQUIRED["target"]])
    y_train_full = df_train[REQUIRED["target"]].astype(float)

    # Prepare X
    # If Hour/Minute are entirely missing, keep them (imputer will handle)
    X_train_full = df_train[[c for c in numeric_feats + categorical_feats if c in df_train.columns]]

    # internal holdout split (time-respecting: keep order; here we use simple 80/20 split without shuffle)
    split_idx = int(0.8 * len(X_train_full))
    X_tr = X_train_full.iloc[:split_idx]
    y_tr = y_train_full.iloc[:split_idx]
    X_hold = X_train_full.iloc[split_idx:]
    y_hold = y_train_full.iloc[split_idx:]

    log(f"Training rows: {len(X_tr)}, holdout rows: {len(X_hold)}")

    pipeline = build_pipeline(numeric_feats, categorical_feats)

    log("Fitting pipeline...")
    pipeline.fit(X_tr, y_tr)

    # Evaluate holdout
    log("Predicting on holdout...")
    y_hold_pred = pipeline.predict(X_hold)
    reg_hold = regression_metrics(y_hold, y_hold_pred)
    threshold = float(y_train_full.median())
    class_hold = classification_metrics_from_regression(y_hold, y_hold_pred, threshold)

    # Save model pipeline
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = MODELS_DIR / f"pipeline_model_{ts}.joblib"
    joblib.dump(pipeline, model_file)
    log(f"Saved pipeline to {model_file}")

    output = {
        "run_time": ts,
        "train_csv": train_path.name,
        "holdout_regression": reg_hold,
        "holdout_classification": class_hold,
        "threshold_from_train_median": threshold,
        "new_csvs": {}
    }

    # Process new CSVs if any
    for p in new_paths:
        try:
            log(f"Processing new CSV: {p.name}")
            df_new = safe_read_csv(p)
            missing_new = validate_columns(df_new)
            if missing_new:
                log(f"Skipping {p.name}: missing columns {missing_new}")
                output["new_csvs"][p.name] = {"error": f"missing_columns_{missing_new}"}
                continue
            df_new = extract_datetime_features(df_new)
            df_new = df_new.dropna(subset=[REQUIRED["date"]])
            for col in numeric_feats:
                if col in df_new.columns:
                    df_new[col] = pd.to_numeric(df_new[col], errors="coerce")
            df_new = df_new.dropna(subset=[REQUIRED["target"]])
            X_new = df_new[[c for c in numeric_feats + categorical_feats if c in df_new.columns]]
            y_new = df_new[REQUIRED["target"]].astype(float)
            log(f"Predicting on {p.name} (n={len(X_new)})")
            y_new_pred = pipeline.predict(X_new)
            output["new_csvs"][p.name] = {
                "n_rows": int(len(X_new)),
                "regression": regression_metrics(y_new, y_new_pred),
                "classification": classification_metrics_from_regression(y_new, y_new_pred, threshold)
            }
        except Exception as e:
            log(f"Error processing {p.name}: {e}")
            output["new_csvs"][p.name] = {"error": str(e)}

    # Save evaluation JSON
    out_file = OUTPUT_DIR / f"eval_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    log(f"Saved evaluation output to {out_file}")
    # append short summary
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\nSUMMARY: " + json.dumps({
            "run_time": ts, "model": str(model_file), "train_csv": train_path.name,
            "holdout": reg_hold
        }) + "\n")

    log("Pipeline run completed successfully.")


if __name__ == "__main__":
    try:
        run_pipeline_once()
    except Exception as exc:
        log(f"FATAL: {exc}")
        raise
