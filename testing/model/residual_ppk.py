"""
PPK Position-Based LGBM Residual Learning
==========================================

Trains LightGBM models using ONLY PPK .pos files to predict ground truth.
Uses only position data from PPK processing - no other sensors.

Approach:
- Load PPK .pos files for baseline positions
- Load ground_truth.csv for GT positions
- Extract ONLY position-based features (lat, lon, height)
- Train LGBM to predict residuals

Usage:
    python residual_ppk.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DATA_ROOT = Path(r"C:\Users\avnee\Downloads\smartphone-decimeter-2022\train")
PPK_OUTPUT_DIR = Path(r"C:\Users\avnee\Downloads\pos_output_final-20251112T124635Z-1-001\pos_output_final")
OUTPUT_DIR = Path("model/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_SIZE = 0.2
RANDOM_STATE = 42

# LightGBM parameters
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 6,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.5,
    'verbose': -1,
    'random_state': RANDOM_STATE
}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50


# ============================================================
# DATA LOADING
# ============================================================

def read_pos_file(pos_file_path: str) -> pd.DataFrame:
    """
    Read PPK .pos file (RTKLIB format).

    Format:
    %  GPST                  latitude(deg) longitude(deg)  height(m)   Q  ns   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m) age(s)  ratio
    """
    try:
        # Read .pos file (skip header lines starting with %)
        df = pd.read_csv(
            pos_file_path,
            delim_whitespace=True,
            comment='%',
            header=None,
            names=['date', 'time', 'latitude', 'longitude', 'height',
                   'Q', 'ns', 'sdn', 'sde', 'sdu', 'sdne', 'sdeu', 'sdun', 'age', 'ratio']
        )

        # Combine date and time into datetime
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

        # Convert to GPS milliseconds
        # GPS epoch: 1980-01-06 00:00:00 UTC
        gps_epoch = pd.Timestamp('1980-01-06 00:00:00', tz='UTC')
        df['millisSinceGpsEpoch'] = ((df['datetime'] - gps_epoch).dt.total_seconds() * 1000).astype(np.int64)

        # Keep only relevant columns
        df = df[['millisSinceGpsEpoch', 'latitude', 'longitude', 'height', 'Q', 'ns', 'sdn', 'sde', 'sdu']]

        return df

    except Exception as e:
        print(f"Error reading .pos file: {e}")
        return pd.DataFrame()


def load_drive_data(drive_path: Path) -> pd.DataFrame:
    """
    Load data from a single drive folder.

    Returns DataFrame with:
    - PPK positions from .pos files
    - Ground truth positions from ground_truth.csv
    - Only position-based features (no sensors)
    """
    drive_id = drive_path.name
    phone_dirs = list(drive_path.iterdir())

    if not phone_dirs:
        return None

    phone_dir = phone_dirs[0]
    phone_id = phone_dir.name

    # Load ground truth
    gt_path = phone_dir / "ground_truth.csv"
    if not gt_path.exists():
        return None

    # Load PPK .pos file
    pos_file_name = f"{drive_id}-{phone_id}-pos.txt"
    pos_file_path = PPK_OUTPUT_DIR / pos_file_name

    if not pos_file_path.exists():
        print(f"  Warning: No .pos file found for {drive_id}/{phone_id}")
        return None

    # Read data
    ppk_df = read_pos_file(str(pos_file_path))
    gt_df = pd.read_csv(gt_path)

    if ppk_df.empty:
        return None

    # Convert GT timestamps
    GPS_EPOCH_OFFSET_MILLIS = 315964800000
    gt_df['millisSinceGpsEpoch'] = gt_df['UnixTimeMillis'] - GPS_EPOCH_OFFSET_MILLIS
    gt_df['millisSinceGpsEpoch'] = (gt_df['millisSinceGpsEpoch'] / 10).round() * 10
    gt_df['millisSinceGpsEpoch'] = gt_df['millisSinceGpsEpoch'].astype(np.int64)

    # Round PPK timestamps to nearest 10ms for matching
    ppk_df['millisSinceGpsEpoch'] = (ppk_df['millisSinceGpsEpoch'] / 10).round() * 10
    ppk_df['millisSinceGpsEpoch'] = ppk_df['millisSinceGpsEpoch'].astype(np.int64)

    # Sort both
    ppk_df = ppk_df.sort_values('millisSinceGpsEpoch').reset_index(drop=True)
    gt_df = gt_df.sort_values('millisSinceGpsEpoch').reset_index(drop=True)

    # Merge PPK with GT using merge_asof for nearest timestamp
    merged = pd.merge_asof(
        ppk_df,
        gt_df[['millisSinceGpsEpoch', 'LatitudeDegrees', 'LongitudeDegrees']],
        on='millisSinceGpsEpoch',
        direction='nearest',
        tolerance=100  # 100ms tolerance
    )

    # Remove rows without GT
    merged = merged.dropna(subset=['LatitudeDegrees', 'LongitudeDegrees'])

    if len(merged) == 0:
        return None

    # Rename for consistency
    merged = merged.rename(columns={
        'latitude': 'ppk_lat',
        'longitude': 'ppk_lon',
        'height': 'ppk_height',
        'LatitudeDegrees': 'gt_lat',
        'LongitudeDegrees': 'gt_lon'
    })

    # Add drive identifier
    merged['drive_id'] = drive_id

    return merged


def load_all_training_data(train_root: Path) -> pd.DataFrame:
    """Load data from all drive folders."""
    all_data = []

    drive_folders = sorted([d for d in train_root.iterdir() if d.is_dir()])

    print(f"Loading data from {len(drive_folders)} drives...")

    for drive_path in tqdm(drive_folders, desc="Loading drives"):
        drive_df = load_drive_data(drive_path)
        if drive_df is not None:
            all_data.append(drive_df)

    if not all_data:
        raise ValueError("No data loaded!")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined):,} samples from {len(all_data)} drives")

    return combined


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def extract_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from PPK position data ONLY.

    Features:
    - Raw PPK lat/lon/height
    - Relative position from mean
    - Position quality indicators (sdn, sde, sdu)

    NO other sensors allowed!
    """
    features = pd.DataFrame()

    # Raw PPK position
    features['ppk_lat'] = df['ppk_lat']
    features['ppk_lon'] = df['ppk_lon']
    features['ppk_height'] = df['ppk_height']

    # Relative position (removes absolute coordinates, keeps patterns)
    features['ppk_lat_relative'] = df['ppk_lat'] - df['ppk_lat'].mean()
    features['ppk_lon_relative'] = df['ppk_lon'] - df['ppk_lon'].mean()
    features['ppk_height_relative'] = df['ppk_height'] - df['ppk_height'].mean()

    # PPK quality indicators (from .pos file)
    features['ppk_sdn'] = df['sdn'].fillna(0)  # North position std
    features['ppk_sde'] = df['sde'].fillna(0)  # East position std
    features['ppk_sdu'] = df['sdu'].fillna(0)  # Up position std
    features['ppk_Q'] = df['Q'].fillna(5)      # Solution quality
    features['ppk_ns'] = df['ns'].fillna(0)    # Number of satellites

    return features


def compute_position_error(lat1: np.ndarray, lon1: np.ndarray,
                          lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Compute 2D position error in meters using Haversine formula."""
    R = 6371000  # Earth radius in meters

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c


# ============================================================
# EVALUATION METRICS
# ============================================================

def compute_metrics(errors: np.ndarray) -> Dict:
    """Compute comprehensive error metrics."""
    mean = np.mean(errors)
    p50 = np.percentile(errors, 50)
    p95 = np.percentile(errors, 95)

    return {
        'Mean': mean,
        'Median/P50': p50,
        'P95': p95,
        'P99': np.percentile(errors, 99),
        'RMSE': np.sqrt(np.mean(errors**2)),
        'Std': np.std(errors),
        'Max': np.max(errors),
        '🎯 TARGET (Mean+P50+P95)/3': (mean + p50 + p95) / 3,
        'Target (P50+P95)/2': (p50 + p95) / 2,
    }


def display_metrics_comparison(baseline_metrics: Dict, lgbm_metrics: Dict):
    """Display side-by-side metrics comparison."""
    print("\n" + "="*100)
    print(f"{'Metric':<35} {'PPK Baseline (m)':>20} {'PPK+LGBM (m)':>20} {'Improvement':>15}")
    print("-"*100)

    for metric in baseline_metrics.keys():
        baseline_val = baseline_metrics[metric]
        lgbm_val = lgbm_metrics[metric]
        improvement = (baseline_val - lgbm_val) / baseline_val * 100 if baseline_val > 0 else 0

        # Highlight target metrics
        marker = "🎯 " if "TARGET" in metric or metric == "Median/P50" else "   "

        print(f"{marker}{metric:<32} {baseline_val:>20.4f} {lgbm_val:>20.4f} {improvement:>14.1f}%")

    print("="*100)


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("="*100)
    print("PPK POSITION-BASED LGBM: Predict GT from PPK .pos files ONLY")
    print("="*100)

    # ============================================================
    # STEP 1: LOAD DATA
    # ============================================================
    print("\n[1/6] Loading PPK .pos files and ground truth...")

    df = load_all_training_data(TRAIN_DATA_ROOT)

    print(f"  Total samples: {len(df):,}")
    print(f"  Unique drives: {df['drive_id'].nunique()}")

    # ============================================================
    # STEP 2: EXTRACT FEATURES
    # ============================================================
    print("\n[2/6] Extracting position-based features (PPK ONLY)...")
    print("  Features: PPK lat/lon/height + quality indicators")
    print("  NO GNSS raw data, NO IMU, NO EKF - ONLY PPK positions!")

    X = extract_position_features(df)

    print(f"  Feature shape: {X.shape}")
    print(f"  Features: {list(X.columns)}")

    # ============================================================
    # STEP 3: PREPARE TARGETS
    # ============================================================
    print("\n[3/6] Preparing targets (residuals: GT - PPK)...")

    # Ground truth
    gt_lat = df['gt_lat'].values
    gt_lon = df['gt_lon'].values

    # PPK baseline
    ppk_lat = df['ppk_lat'].values
    ppk_lon = df['ppk_lon'].values

    # Target: predict residuals in meters (not absolute positions)
    lat_residual_deg = gt_lat - ppk_lat
    lon_residual_deg = gt_lon - ppk_lon

    # Convert to meters
    METERS_PER_DEGREE_LAT = 111320.0
    y_lat = lat_residual_deg * METERS_PER_DEGREE_LAT
    y_lon = lon_residual_deg * METERS_PER_DEGREE_LAT * np.cos(np.radians(gt_lat))

    print(f"  Residual stats:")
    print(f"    Lat residual: mean={np.nanmean(y_lat):.2f}m, std={np.nanstd(y_lat):.2f}m")
    print(f"    Lon residual: mean={np.nanmean(y_lon):.2f}m, std={np.nanstd(y_lon):.2f}m")

    # Remove NaN rows
    valid_mask = ~(np.isnan(gt_lat) | np.isnan(gt_lon) |
                   np.isnan(ppk_lat) | np.isnan(ppk_lon) |
                   X.isna().any(axis=1))

    X = X[valid_mask].values
    y_lat = y_lat[valid_mask]
    y_lon = y_lon[valid_mask]
    ppk_lat = ppk_lat[valid_mask]
    ppk_lon = ppk_lon[valid_mask]
    gt_lat = gt_lat[valid_mask]
    gt_lon = gt_lon[valid_mask]

    print(f"  Valid samples: {len(X):,}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ============================================================
    # STEP 4: TRAIN/VAL SPLIT
    # ============================================================
    print("\n[4/6] Splitting data...")

    X_train, X_val, \
    y_lat_train, y_lat_val, \
    y_lon_train, y_lon_val, \
    ppk_lat_train, ppk_lat_val, \
    ppk_lon_train, ppk_lon_val, \
    gt_lat_train, gt_lat_val, \
    gt_lon_train, gt_lon_val = train_test_split(
        X_scaled, y_lat, y_lon,
        ppk_lat, ppk_lon,
        gt_lat, gt_lon,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")

    # ============================================================
    # STEP 5: TRAIN MODELS
    # ============================================================
    print("\n[5/6] Training LightGBM models...")
    print(f"  Parameters: leaves={LGBM_PARAMS['num_leaves']}, depth={LGBM_PARAMS['max_depth']}, lr={LGBM_PARAMS['learning_rate']}")

    # Latitude model
    print("\n  Training Latitude model...")
    train_data_lat = lgb.Dataset(X_train, label=y_lat_train)
    val_data_lat = lgb.Dataset(X_val, label=y_lat_val, reference=train_data_lat)

    model_lat = lgb.train(
        LGBM_PARAMS,
        train_data_lat,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data_lat, val_data_lat],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=50)
        ]
    )

    print(f"  ✓ Latitude model: {model_lat.num_trees()} trees (stopped at {model_lat.best_iteration})")

    # Longitude model
    print("\n  Training Longitude model...")
    train_data_lon = lgb.Dataset(X_train, label=y_lon_train)
    val_data_lon = lgb.Dataset(X_val, label=y_lon_val, reference=train_data_lon)

    model_lon = lgb.train(
        LGBM_PARAMS,
        train_data_lon,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data_lon, val_data_lon],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=50)
        ]
    )

    print(f"  ✓ Longitude model: {model_lon.num_trees()} trees (stopped at {model_lon.best_iteration})")

    # ============================================================
    # STEP 6: EVALUATE
    # ============================================================
    print("\n[6/6] Evaluating performance...")

    # Predict residuals in meters
    pred_lat_residual_m = model_lat.predict(X_val, num_iteration=model_lat.best_iteration)
    pred_lon_residual_m = model_lon.predict(X_val, num_iteration=model_lon.best_iteration)

    # Convert residuals back to degrees and apply correction
    METERS_PER_DEGREE_LAT = 111320.0
    pred_lat_residual_deg = pred_lat_residual_m / METERS_PER_DEGREE_LAT
    pred_lon_residual_deg = pred_lon_residual_m / (METERS_PER_DEGREE_LAT * np.cos(np.radians(ppk_lat_val)))

    # Apply corrections to PPK baseline
    corrected_lat = ppk_lat_val + pred_lat_residual_deg
    corrected_lon = ppk_lon_val + pred_lon_residual_deg

    # Compute errors
    baseline_error = compute_position_error(
        ppk_lat_val, ppk_lon_val, gt_lat_val, gt_lon_val
    )

    lgbm_error = compute_position_error(
        corrected_lat, corrected_lon, gt_lat_val, gt_lon_val
    )

    # Compute metrics
    baseline_metrics = compute_metrics(baseline_error)
    lgbm_metrics = compute_metrics(lgbm_error)

    # Display comparison
    display_metrics_comparison(baseline_metrics, lgbm_metrics)

    # Feature importance
    print("\n" + "="*100)
    print("FEATURE IMPORTANCE (Latitude model)")
    print("="*100)
    importance = model_lat.feature_importance(importance_type='gain')
    feature_names = X.columns if hasattr(X, 'columns') else [
        'ppk_lat', 'ppk_lon', 'ppk_height',
        'ppk_lat_relative', 'ppk_lon_relative', 'ppk_height_relative',
        'ppk_sdn', 'ppk_sde', 'ppk_sdu', 'ppk_Q', 'ppk_ns'
    ]

    for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
        print(f"  {name:<30} {imp:>15.1f}")

    # ============================================================
    # SAVE MODELS
    # ============================================================
    print("\n" + "="*100)
    print("SAVING MODELS")
    print("="*100)

    model_lat_path = OUTPUT_DIR / f"ppk_pos_lgbm_lat_{timestamp_str}.pkl"
    model_lon_path = OUTPUT_DIR / f"ppk_pos_lgbm_lon_{timestamp_str}.pkl"
    scaler_path = OUTPUT_DIR / f"ppk_pos_lgbm_scaler_{timestamp_str}.pkl"

    with open(model_lat_path, 'wb') as f:
        pickle.dump(model_lat, f)
    with open(model_lon_path, 'wb') as f:
        pickle.dump(model_lon, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Save metrics
    metrics_df = pd.DataFrame({
        'Metric': list(baseline_metrics.keys()),
        'PPK_Baseline': list(baseline_metrics.values()),
        'PPK_LGBM': list(lgbm_metrics.values())
    })
    metrics_path = OUTPUT_DIR / f"ppk_pos_lgbm_metrics_{timestamp_str}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"  ✓ Latitude model: {model_lat_path.name}")
    print(f"  ✓ Longitude model: {model_lon_path.name}")
    print(f"  ✓ Scaler: {scaler_path.name}")
    print(f"  ✓ Metrics: {metrics_path.name}")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "="*100)
    print("EXPERIMENT SUMMARY")
    print("="*100)

    target_metric = baseline_metrics['🎯 TARGET (Mean+P50+P95)/3']
    lgbm_target = lgbm_metrics['🎯 TARGET (Mean+P50+P95)/3']
    improvement = (target_metric - lgbm_target) / target_metric * 100

    print(f"\n🎯 PRIMARY METRIC: (Mean+P50+P95)/3")
    print(f"   PPK Baseline: {target_metric:.4f}m")
    print(f"   PPK + LGBM:   {lgbm_target:.4f}m")
    print(f"   Improvement:  {improvement:.1f}%")

    median_baseline = baseline_metrics['Median/P50']
    median_lgbm = lgbm_metrics['Median/P50']
    median_improvement = (median_baseline - median_lgbm) / median_baseline * 100

    print(f"\n📊 MEDIAN/P50:")
    print(f"   PPK Baseline: {median_baseline:.4f}m")
    print(f"   PPK + LGBM:   {median_lgbm:.4f}m")
    print(f"   Improvement:  {median_improvement:.1f}%")

    if improvement > 0:
        print("\n✅ SUCCESS! LGBM improves upon PPK positions")
        print("   PPK .pos data contains patterns that can be learned")
    else:
        print("\n⚠️  LGBM did not improve over baseline")
        print("   PPK positions alone may not have sufficient patterns")

    print("="*100)


if __name__ == "__main__":
    main()
