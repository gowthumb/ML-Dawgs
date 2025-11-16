import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import os
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
import warnings
import optuna
from functools import partial

print(">>> RUNNING ULTIMATE K-FOLD OPTUNA (150+ Features + GroupKFold Split) <<<")

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
tqdm.pandas()

# ============================================================
# CONFIGURATION
# ============================================================

# --- 1. Set these paths to your folders ---
TRAIN_DATA_ROOT = Path(r"C:\Users\gauth\Documents\SC4000_proj\train")
PPK_OUTPUT_DIR = Path(r"C:\Users\gauth\Documents\SC4000_proj\pos_output_final")
# ----------------------------------------

OUTPUT_DIR = Path("model/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 67
N_TRIALS = 35 # 35 trials. Each trial trains 5 models.
N_SPLITS = 5  # 5-Fold Cross-Validation

NUM_BOOST_ROUND = 1200
EARLY_STOPPING_ROUNDS = 100

METERS_PER_DEGREE_LAT = 111320.0
GPS_UTC_OFFSET_SECONDS = 18
GPS_EPOCH_OFFSET_MILLIS = 315964800000


# ============================================================
# DATA LOADING (Your robust functions)
# ============================================================

def read_pos_file(file_path: str) -> pd.DataFrame:
    """Read PPK .pos file."""
    if not os.path.exists(file_path):
        return pd.DataFrame()

    data_lines = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('%') or not line:
                    continue
                parts = line.split()
                if len(parts) < 14:
                    continue
                try:
                    gps_week = int(parts[0])
                    gps_seconds = float(parts[1])
                    millis_since_gps_epoch = int((gps_week * 604800 + gps_seconds - GPS_UTC_OFFSET_SECONDS) * 1000)
                    millis_since_gps_epoch = round(millis_since_gps_epoch / 10) * 10
                    data_lines.append({
                        'millisSinceGpsEpoch': millis_since_gps_epoch,
                        'latitude': float(parts[2]),
                        'longitude': float(parts[3]),
                        'height': float(parts[4]),
                        'quality': int(parts[5]),
                        'num_satellites': int(parts[6]),
                        'sdn': float(parts[7]),
                        'sde': float(parts[8]),
                        'sdu': float(parts[9]),
                        'sdne': float(parts[10]),
                        'sdeu': float(parts[11]),
                        'sdun': float(parts[12]),
                        'age': float(parts[13]),
                        'ratio': float(parts[14]) if len(parts) > 14 else 0.0
                    })
                except (ValueError, IndexError):
                    continue
        return pd.DataFrame(data_lines) if data_lines else pd.DataFrame()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()


def load_drive_data(drive_path: Path) -> pd.DataFrame:
    """Load data from a single drive (all phones)."""
    drive_id = drive_path.name
    all_phone_data = []

    for phone_dir in drive_path.iterdir():
        if not phone_dir.is_dir():
            continue
            
        phone_id = phone_dir.name
        gt_path = phone_dir / "ground_truth.csv"
        if not gt_path.exists():
            continue

        pos_file_name = f"{drive_id}-{phone_id}-pos.txt"
        pos_file_path = PPK_OUTPUT_DIR / pos_file_name

        if not pos_file_path.exists():
            continue

        ppk_df = read_pos_file(str(pos_file_path))
        gt_df = pd.read_csv(gt_path)

        if ppk_df.empty or gt_df.empty:
            continue

        gt_df['millisSinceGpsEpoch'] = gt_df['UnixTimeMillis'] - GPS_EPOCH_OFFSET_MILLIS
        gt_df['millisSinceGpsEpoch'] = (gt_df['millisSinceGpsEpoch'] / 10).round() * 10
        gt_df['millisSinceGpsEpoch'] = gt_df['millisSinceGpsEpoch'].astype(np.int64)

        ppk_df['millisSinceGpsEpoch'] = (ppk_df['millisSinceGpsEpoch'] / 10).round() * 10
        ppk_df['millisSinceGpsEpoch'] = ppk_df['millisSinceGpsEpoch'].astype(np.int64)

        ppk_df = ppk_df.sort_values('millisSinceGpsEpoch').reset_index(drop=True)
        gt_df = gt_df.sort_values('millisSinceGpsEpoch').reset_index(drop=True)

        merged = pd.merge_asof(
            ppk_df,
            gt_df[['millisSinceGpsEpoch', 'LatitudeDegrees', 'LongitudeDegrees']],
            on='millisSinceGpsEpoch',
            direction='nearest',
            tolerance=100
        )

        merged = merged.dropna(subset=['LatitudeDegrees', 'LongitudeDegrees'])
        if len(merged) == 0:
            continue

        merged = merged.rename(columns={
            'latitude': 'ppk_lat',
            'longitude': 'ppk_lon',
            'height': 'ppk_height',
            'LatitudeDegrees': 'gt_lat',
            'LongitudeDegrees': 'gt_lon'
        })

        merged['drive_id'] = drive_id
        merged['phone_id'] = phone_id
        all_phone_data.append(merged)
    
    return pd.concat(all_phone_data, ignore_index=True) if all_phone_data else None


def load_all_training_data(train_root: Path) -> pd.DataFrame:
    """Load all training data."""
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
    print(f"Loaded {len(combined):,} samples from {len(combined['drive_id'].unique())} drives")
    return combined


# ============================================================
# FEATURE ENGINEERING (Your ULTIMATE features)
# ============================================================

def extract_position_features(df_group):
    """
    Extract ALL features (Base + Physics + Advanced + ULTRA).
    This function is now safe to use with groupby().apply()
    """
    df = df_group.copy()
    features = pd.DataFrame(index=df.index)
    
    # Add back original columns needed for splitting/targets
    features['drive_id'] = df['drive_id']
    features['phone_id'] = df['phone_id']
    features['ppk_lat_base'] = df['ppk_lat'] # Keep base ppk for eval
    features['ppk_lon_base'] = df['ppk_lon']
    features['gt_lat_base'] = df['gt_lat']
    features['gt_lon_base'] = df['gt_lon']

    # ========== YOUR BASE FEATURES (72) ==========
    features['ppk_lat'] = df['ppk_lat']
    features['ppk_lon'] = df['ppk_lon']
    features['ppk_height'] = df['ppk_height']
    features['ppk_lat_relative'] = df['ppk_lat'] - df['ppk_lat'].mean()
    features['ppk_lon_relative'] = df['ppk_lon'] - df['ppk_lon'].mean()
    features['ppk_height_relative'] = df['ppk_height'] - df['ppk_height'].mean()

    features['ppk_sdn'] = df['sdn'].fillna(0)
    features['ppk_sde'] = df['sde'].fillna(0)
    features['ppk_sdu'] = df['sdu'].fillna(0)
    features['ppk_quality'] = df['quality'].fillna(5)
    features['ppk_num_satellites'] = df['num_satellites'].fillna(0)
    features['ppk_ratio'] = df['ratio'].fillna(0)
    features['ppk_age'] = df['age'].fillna(0)

    features['position_uncertainty'] = np.sqrt(
        features['ppk_sdn']**2 + features['ppk_sde']**2 + features['ppk_sdu']**2
    )
    features['horizontal_uncertainty'] = np.sqrt(
        features['ppk_sdn']**2 + features['ppk_sde']**2
    )

    time_diff = df['millisSinceGpsEpoch'].diff().fillna(0) / 1000.0
    features['time_delta'] = time_diff.clip(0, 10).replace(0, 0.1)

    lat_diff = df['ppk_lat'].diff().fillna(0)
    lon_diff = df['ppk_lon'].diff().fillna(0)
    height_diff = df['ppk_height'].diff().fillna(0)

    lat_diff_m = lat_diff * METERS_PER_DEGREE_LAT
    lon_diff_m = lon_diff * METERS_PER_DEGREE_LAT * np.cos(np.radians(df['ppk_lat']))

    distance_2d_m = np.sqrt(lat_diff_m**2 + lon_diff_m**2)

    velocity_2d = (distance_2d_m / features['time_delta']).replace([np.inf, -np.inf], 0).fillna(0)
    features['velocity'] = velocity_2d.clip(0, 50)

    features['velocity_lat'] = (lat_diff_m / features['time_delta']).replace([np.inf, -np.inf], 0).fillna(0).clip(-50, 50)
    features['velocity_lon'] = (lon_diff_m / features['time_delta']).replace([np.inf, -np.inf], 0).fillna(0).clip(-50, 50)
    features['velocity_height'] = (height_diff / features['time_delta']).replace([np.inf, -np.inf], 0).fillna(0).clip(-20, 20)
    features['velocity_height_abs'] = features['velocity_height'].abs()

    heading = np.arctan2(lon_diff_m, lat_diff_m)
    features['heading_sin'] = np.sin(heading).fillna(0)
    features['heading_cos'] = np.cos(heading).fillna(0)

    heading_change = heading.diff().fillna(0)
    heading_change = np.arctan2(np.sin(heading_change), np.cos(heading_change))
    features['heading_change'] = heading_change

    turn_rate = heading_change / features['time_delta']
    features['turn_rate'] = turn_rate.replace([np.inf, -np.inf], 0).fillna(0).clip(-5, 5)
    features['turn_rate_abs'] = features['turn_rate'].abs()

    curvature = features['turn_rate'] / (features['velocity'] + 1e-6)
    features['curvature'] = curvature.replace([np.inf, -np.inf], 0).fillna(0).clip(-1, 1)

    accel = features['velocity'].diff().fillna(0) / features['time_delta']
    features['acceleration'] = accel.replace([np.inf, -np.inf], 0).fillna(0).clip(-10, 10)
    features['acceleration_abs'] = features['acceleration'].abs()

    accel_lat = features['velocity_lat'].diff().fillna(0) / features['time_delta']
    accel_lon = features['velocity_lon'].diff().fillna(0) / features['time_delta']
    features['acceleration_lat'] = accel_lat.replace([np.inf, -np.inf], 0).fillna(0).clip(-10, 10)
    features['acceleration_lon'] = accel_lon.replace([np.inf, -np.inf], 0).fillna(0).clip(-10, 10)

    features['centripetal_acceleration'] = ((features['velocity']**2) * features['curvature'].abs()).clip(0, 10)

    jerk = features['acceleration'].diff().fillna(0) / features['time_delta']
    features['jerk'] = jerk.replace([np.inf, -np.inf], 0).fillna(0).clip(-20, 20)
    features['jerk_abs'] = features['jerk'].abs()
    
    # --- NEW: ULTRA Features (from ultimate_ppk.py) ---
    features['jerk_lat'] = (features['acceleration_lat'].diff().fillna(0) / features['time_delta']).clip(-20, 20)
    features['jerk_lon'] = (features['acceleration_lon'].diff().fillna(0) / features['time_delta']).clip(-20, 20)

    features['snap'] = (features['jerk'].diff().fillna(0) / features['time_delta']).clip(-50, 50)
    features['snap_abs'] = features['snap'].abs()
    
    features['snap_lat'] = (features['jerk_lat'].diff().fillna(0) / features['time_delta']).clip(-50, 50)
    features['snap_lon'] = (features['jerk_lon'].diff().fillna(0) / features['time_delta']).clip(-50, 50)
    
    features['crackle'] = (features['snap'].diff().fillna(0) / features['time_delta']).clip(-100, 100)
    features['crackle_abs'] = features['crackle'].abs()

    features['high_order_smoothness'] = (
        features['jerk_abs'] * 0.4 +
        features['snap_abs'] * 0.3 +
        features['crackle_abs'] * 0.3
    ).clip(0, 100)
    # --- End ULTRA ---

    features['is_turning'] = (features['turn_rate_abs'] > 0.3).astype(int)
    features['is_accelerating'] = (features['acceleration'] > 0.5).astype(int)
    features['is_braking'] = (features['acceleration'] < -0.5).astype(int)
    features['is_stationary'] = (features['velocity'] < 0.5).astype(int)
    features['high_jerk'] = (features['jerk_abs'] > 2).astype(int)

    for lag in [1, 2, 3]:
        features[f'ppk_lat_lag{lag}'] = df['ppk_lat'].shift(lag).fillna(df['ppk_lat'])
        features[f'ppk_lon_lag{lag}'] = df['ppk_lon'].shift(lag).fillna(df['ppk_lon'])
        features[f'velocity_lag{lag}'] = features['velocity'].shift(lag).fillna(features['velocity'])

    features['lat_change_lag1'] = df['ppk_lat'] - features['ppk_lat_lag1']
    features['lon_change_lag1'] = df['ppk_lon'] - features['ppk_lon_lag1']
    features['velocity_change_lag1'] = features['velocity'] - features['velocity_lag1']

    features['position_jump'] = (distance_2d_m > 5.0).astype(int)

    features['is_climbing'] = (features['velocity_height'] > 1.0).astype(int)
    features['is_descending'] = (features['velocity_height'] < -1.0).astype(int)

    features['poor_quality'] = (features['ppk_quality'] > 2).astype(int)
    features['low_satellites'] = (features['ppk_num_satellites'] < 8).astype(int)
    features['implausible_velocity'] = (features['velocity'] > 40.0).astype(int)
    features['implausible_acceleration'] = (features['acceleration_abs'] > 8.0).astype(int)

    # Temporal consistency
    features['lat_trend_3'] = (df['ppk_lat'] - features['ppk_lat_lag3']) / 3
    features['lon_trend_3'] = (df['ppk_lon'] - features['ppk_lon_lag3']) / 3
    
    features['lat_deviation_from_trend'] = (df['ppk_lat'] - features['ppk_lat_lag1']) - features['lat_trend_3']
    features['lon_deviation_from_trend'] = (df['ppk_lon'] - features['ppk_lon_lag1']) - features['lon_trend_3']
    
    features['velocity_expected'] = features['velocity_lag1'] + features['acceleration'].shift(1).fillna(0) * features['time_delta']
    features['velocity_expected'] = features['velocity_expected'].fillna(features['velocity'])
    
    features['velocity_surprise'] = features['velocity'] - features['velocity_expected']
    features['velocity_surprise'] = features['velocity_surprise'].replace([np.inf, -np.inf], 0).fillna(0).clip(-10, 10)
    
    lat_consistency = features['lat_deviation_from_trend'].abs()
    lon_consistency = features['lon_deviation_from_trend'].abs()
    features['position_consistency_score'] = np.sqrt(lat_consistency**2 + lon_consistency**2)
    features['position_consistency_score'] = features['position_consistency_score'].replace([np.inf, -np.inf], 0).fillna(0)
    
    velocity_change_rate = features['velocity'].diff() / features['time_delta']
    features['velocity_consistency'] = velocity_change_rate.diff().abs()
    features['velocity_consistency'] = features['velocity_consistency'].replace([np.inf, -np.inf], 0).fillna(0).clip(0, 20)
    
    features['trajectory_smoothness'] = features['position_consistency_score'] + features['velocity_consistency']
    features['trajectory_smoothness'] = features['trajectory_smoothness'].clip(0, 50)

    # ========== PHYSICS FEATURES (25) ==========
    uncertainty_weight = 1.0 / (features['horizontal_uncertainty'] + 0.1)
    uncertainty_weight_norm = uncertainty_weight / uncertainty_weight.rolling(10, min_periods=1).sum()
    
    features['weighted_lat'] = (df['ppk_lat'] * uncertainty_weight_norm).rolling(5, min_periods=1).sum()
    features['weighted_lon'] = (df['ppk_lon'] * uncertainty_weight_norm).rolling(5, min_periods=1).sum()
    features['lat_deviation_from_weighted'] = df['ppk_lat'] - features['weighted_lat']
    features['lon_deviation_from_weighted'] = df['ppk_lon'] - features['weighted_lon']
    
    quality_score = (
        (1.0 - features['ppk_quality'] / 5.0).clip(0, 1) * 0.3 +
        (features['ppk_num_satellites'] / 15.0).clip(0, 1) * 0.3 +
        (features['ppk_ratio'] / 100.0).clip(0, 1) * 0.2 +
        (1.0 - features['ppk_age'] / 10.0).clip(0, 1) * 0.2
    )
    features['signal_quality_score'] = quality_score
    features['signal_quality_change'] = quality_score.diff().abs().fillna(0)
    
    features['uncertainty_trend'] = features['horizontal_uncertainty'].diff().fillna(0)
    features['uncertainty_acceleration'] = features['uncertainty_trend'].diff().fillna(0)
    
    features['expected_error_from_quality'] = (
        features['horizontal_uncertainty'] * (1.0 + features['velocity'] / 10.0) *
        (1.0 + features['ppk_age'] / 5.0)
    ).clip(0, 10)
    
    for window in [3, 7, 15]:
        vel_rolling = features['velocity'].rolling(window, min_periods=1)
        features[f'velocity_std_{window}'] = vel_rolling.std().fillna(0)
        features[f'velocity_range_{window}'] = (vel_rolling.max() - vel_rolling.min()).fillna(0)
    
    features['velocity_consistency_3_7'] = (features['velocity_std_3'] - features['velocity_std_7']).abs()
    features['velocity_consistency_7_15'] = (features['velocity_std_7'] - features['velocity_std_15']).abs()
    
    max_reasonable_accel = 5.0
    features['accel_violation'] = (features['acceleration_abs'] > max_reasonable_accel).astype(float)
    
    min_turn_radius = 5.0
    implied_radius = features['velocity']**2 / (features['centripetal_acceleration'] + 1e-6)
    features['turn_violation'] = (implied_radius < min_turn_radius).astype(float)
    
    features['physics_violation_score'] = features['accel_violation'] + features['turn_violation']
    
    lat_acf_1 = features['lat_change_lag1'] * features['lat_change_lag1'].shift(1)
    lon_acf_1 = features['lon_change_lag1'] * features['lon_change_lag1'].shift(1)
    features['position_autocorr'] = (lat_acf_1 + lon_acf_1).fillna(0).clip(-1, 1)
    
    vel_acf = features['velocity'] * features['velocity_lag1']
    features['velocity_autocorr'] = (vel_acf.fillna(0) / (features['velocity'].std()**2 + 1e-6)).clip(-1, 1)
    
    lat_diff_2nd = features['lat_change_lag1'].diff().abs()
    lon_diff_2nd = features['lon_change_lag1'].diff().abs()
    features['position_high_freq_energy'] = (lat_diff_2nd + lon_diff_2nd).rolling(5, min_periods=1).sum()
    
    vel_diff_2nd = features['velocity'].diff().diff().abs()
    features['velocity_smoothness_score'] = (1.0 / (vel_diff_2nd.rolling(5, min_periods=1).mean() + 0.01)).clip(0, 10)
    
    features['overall_confidence'] = (
        features['signal_quality_score'] * 0.4 +
        (1.0 - features['position_consistency_score'] / 10.0).clip(0, 1) * 0.3 +
        (1.0 - features['physics_violation_score']).clip(0, 1) * 0.3
    )
    
    features['suggested_correction_weight'] = (1.0 - features['overall_confidence']) * features['horizontal_uncertainty']

    # ========== ADVANCED FEATURES (15) ==========
    window = 7
    lat_poly_residual = []
    lon_poly_residual = []
    
    for i in range(len(df)):
        start = max(0, i - window // 2)
        end = min(len(df), i + window // 2 + 1)
        
        if end - start >= 3:
            x = np.arange(end - start)
            try:
                lat_coeffs = np.polyfit(x, df['ppk_lat'].iloc[start:end], 2)
                lon_coeffs = np.polyfit(x, df['ppk_lon'].iloc[start:end], 2)
                
                mid_idx = i - start
                lat_fitted = np.polyval(lat_coeffs, mid_idx)
                lon_fitted = np.polyval(lon_coeffs, mid_idx)
                
                lat_poly_residual.append(df['ppk_lat'].iloc[i] - lat_fitted)
                lon_poly_residual.append(df['ppk_lon'].iloc[i] - lon_fitted)
            except:
                lat_poly_residual.append(0)
                lon_poly_residual.append(0)
        else:
            lat_poly_residual.append(0)
            lon_poly_residual.append(0)
    
    features['lat_poly_residual'] = lat_poly_residual
    features['lon_poly_residual'] = lon_poly_residual
    features['poly_residual_magnitude'] = np.sqrt(
        np.array(lat_poly_residual)**2 + np.array(lon_poly_residual)**2
    )
    
    try:
        if len(df) >= 11:
            lat_smooth = savgol_filter(df['ppk_lat'].fillna(method='ffill').fillna(method='bfill'), 
                                      window_length=11, polyorder=2)
            lon_smooth = savgol_filter(df['ppk_lon'].fillna(method='ffill').fillna(method='bfill'), 
                                      window_length=11, polyorder=2)
            
            features['lat_smooth_residual'] = df['ppk_lat'] - lat_smooth
            features['lon_smooth_residual'] = df['ppk_lon'] - lon_smooth
        else:
            features['lat_smooth_residual'] = 0
            features['lon_smooth_residual'] = 0
    except:
        features['lat_smooth_residual'] = 0
        features['lon_smooth_residual'] = 0
    
    window = 10
    local_density = []
    
    for i in range(len(df)):
        start = max(0, i - window)
        end = min(len(df), i + window + 1)
        
        if end - start > 1:
            local_lats = df['ppk_lat'].iloc[start:end]
            local_lons = df['ppk_lon'].iloc[start:end]
            
            centroid_lat = local_lats.mean()
            centroid_lon = local_lons.mean()
            
            dist_to_centroid = np.sqrt(
                (df['ppk_lat'].iloc[i] - centroid_lat)**2 + 
                (df['ppk_lon'].iloc[i] - centroid_lon)**2
            )
            
            local_std = np.sqrt(local_lats.std()**2 + local_lons.std()**2)
            normalized_dist = dist_to_centroid / (local_std + 1e-8)
            local_density.append(normalized_dist)
        else:
            local_density.append(0)
    
    features['local_outlier_score'] = local_density
    features['is_local_outlier'] = (np.array(local_density) > 2.5).astype(int)
    
    features['uncertainty_velocity'] = (features['horizontal_uncertainty'].diff().fillna(0) / features['time_delta']).clip(-5, 5)
    
    for window in [5, 10, 20]:
        lat_ma = df['ppk_lat'].rolling(window, min_periods=1).mean()
        lon_ma = df['ppk_lon'].rolling(window, min_periods=1).mean()
        
        features[f'lat_dev_ma{window}'] = df['ppk_lat'] - lat_ma
        features[f'lon_dev_ma{window}'] = df['ppk_lon'] - lon_ma
    
    features['cumulative_lat_change'] = features['lat_change_lag1'].cumsum()
    features['cumulative_lon_change'] = features['lon_change_lag1'].cumsum()
    
    try:
        features['lat_change_zscore'] = zscore(features['lat_change_lag1'].fillna(0), nan_policy='omit')
        features['lon_change_zscore'] = zscore(features['lon_change_lag1'].fillna(0), nan_policy='omit')
        features['lat_change_zscore'] = features['lat_change_zscore'].fillna(0).clip(-5, 5)
        features['lon_change_zscore'] = features['lon_change_zscore'].fillna(0).clip(-5, 5)
    except:
        features['lat_change_zscore'] = 0
        features['lon_change_zscore'] = 0
        
    # --- NEW: ULTRA Features (from ultimate_ppk.py, 25+) ---
    curvature_change = features['curvature'].diff()
    features['curvature_change'] = curvature_change.fillna(0).clip(-0.5, 0.5)
    features['curvature_acceleration'] = curvature_change.diff().fillna(0).clip(-1, 1)
    features['curvature_jerk'] = features['curvature_acceleration'].diff().fillna(0).clip(-2, 2)
    features['curvature_smoothness'] = features['curvature'].rolling(7, min_periods=1).std().fillna(0)
    features['is_sharp_turn'] = (features['curvature'].abs() > 0.3).astype(int)

    for window in [3, 7, 15]:
        accel_rolling = features['acceleration'].rolling(window, min_periods=1)
        features[f'accel_mean_{window}'] = accel_rolling.mean().fillna(0)
        features[f'accel_std_{window}'] = accel_rolling.std().fillna(0)

    features['uncertainty_ratio'] = features['ppk_sdn'] / (features['ppk_sde'] + 1e-6)
    features['uncertainty_ratio'] = features['uncertainty_ratio'].replace([np.inf, -np.inf], 1.0).fillna(1.0).clip(0.1, 10)

    features['uncertainty_anisotropy'] = (features['ppk_sdn'] - features['ppk_sde']).abs()
    features['vertical_uncertainty_ratio'] = features['ppk_sdu'] / (features['horizontal_uncertainty'] + 1e-6)
    features['vertical_uncertainty_ratio'] = features['vertical_uncertainty_ratio'].replace([np.inf, -np.inf], 1.0).fillna(1.0).clip(0.1, 10)

    features['uncertainty_momentum'] = features['horizontal_uncertainty'].rolling(5, min_periods=1).mean()
    features['uncertainty_surprise'] = features['horizontal_uncertainty'] - features['uncertainty_momentum']

    heading_raw = np.arctan2(lon_diff_m, lat_diff_m)
    features['heading_stability'] = heading_raw.rolling(10, min_periods=1).std().fillna(0)
    features['heading_reversal'] = (features['heading_change'].abs() > 2.5).astype(int)

    heading_accel = features['turn_rate'].diff() / features['time_delta']
    features['heading_acceleration'] = heading_accel.replace([np.inf, -np.inf], 0).fillna(0).clip(-10, 10)
    features['heading_consistency'] = features['heading_acceleration'].rolling(7, min_periods=1).std().fillna(0)

    features['error_momentum_lat'] = features['lat_deviation_from_trend'].rolling(10, min_periods=1).sum()
    features['error_momentum_lon'] = features['lon_deviation_from_trend'].rolling(10, min_periods=1).sum()
    features['error_momentum_magnitude'] = np.sqrt(
        features['error_momentum_lat']**2 + features['error_momentum_lon']**2
    )

    features['error_acceleration'] = features['error_momentum_magnitude'].diff().fillna(0)
    features['error_direction_change'] = (
        np.sign(features['error_momentum_lat']) != np.sign(features['error_momentum_lat'].shift(1))
    ).astype(int)

    features['quality_transition'] = features['ppk_quality'].diff().abs().fillna(0)
    features['satellite_gain_loss'] = features['ppk_num_satellites'].diff().fillna(0)
    features['ratio_jump'] = features['ppk_ratio'].diff().abs().fillna(0)

    features['lat_lon_correlation'] = (
        features['lat_change_lag1'] * features['lon_change_lag1']
    ).rolling(10, min_periods=1).mean().fillna(0)

    features['movement_bias_ns'] = features['lat_change_lag1'].rolling(20, min_periods=1).sum()
    features['movement_bias_ew'] = features['lon_change_lag1'].rolling(20, min_periods=1).sum()
    features['movement_directionality'] = np.sqrt(
        features['movement_bias_ns']**2 + features['movement_bias_ew']**2
    )

    try:
        if len(df) >= 15:
            lat_gaussian = gaussian_filter1d(df['ppk_lat'].fillna(method='ffill').fillna(method='bfill'), sigma=2)
            lon_gaussian = gaussian_filter1d(df['ppk_lon'].fillna(method='ffill').fillna(method='bfill'), sigma=2)
            features['lat_gaussian_residual'] = df['ppk_lat'] - lat_gaussian
            features['lon_gaussian_residual'] = df['ppk_lon'] - lon_gaussian
        else:
            features['lat_gaussian_residual'] = 0
            features['lon_gaussian_residual'] = 0
    except:
        features['lat_gaussian_residual'] = 0
        features['lon_gaussian_residual'] = 0

    features['velocity_quality_interaction'] = features['velocity'] * features['signal_quality_score']
    features['uncertainty_velocity_interaction'] = features['horizontal_uncertainty'] * features['velocity']
    features['acceleration_quality_interaction'] = features['acceleration_abs'] * features['signal_quality_score']
    features['curvature_velocity_interaction'] = features['curvature'].abs() * features['velocity']

    features['jerk_uncertainty_interaction'] = features['jerk_abs'] * features['horizontal_uncertainty']
    features['turn_quality_interaction'] = features['turn_rate_abs'] * features['signal_quality_score']
    features['age_uncertainty_interaction'] = features['ppk_age'] * features['horizontal_uncertainty']
    features['satellite_ratio_interaction'] = features['ppk_num_satellites'] * features['ppk_ratio']
    # --- End ULTRA ---

    return features.fillna(0)


# ============================================================
# METRICS & EVALUATION
# ============================================================

def compute_position_error(lat1, lon1, lat2, lon2):
    """Compute 2D error using Haversine."""
    R = 6371000
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def compute_metrics(errors):
    """Compute error metrics."""
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
    }

def display_metrics_comparison(baseline_metrics, lgbm_metrics):
    """Display metrics."""
    print("\n" + "="*100)
    print(f"{'Metric':<35} {'PPK Baseline (m)':>20} {'PPK+LGBM (m)':>20} {'Improvement':>15}")
    print("-"*100)

    for metric in baseline_metrics.keys():
        baseline_val = baseline_metrics[metric]
        lgbm_val = lgbm_metrics[metric]
        improvement = (baseline_val - lgbm_val) / baseline_val * 100 if baseline_val > 0 else 0
        marker = "🎯 " if "TARGET" in metric or metric == "Median/P50" else "   "
        print(f"{marker}{metric:<32} {baseline_val:>20.4f} {lgbm_val:>20.4f} {improvement:>14.1f}%")
    print("="*100)
    
# ============================================================
# OPTUNA OBJECTIVE FUNCTION (K-FOLD)
# ============================================================

def objective(
    trial: optuna.trial.Trial,
    X_scaled, y_lat, y_lon,
    ppk_lat, ppk_lon, gt_lat, gt_lon,
    groups, feature_names, categorical_indices
) -> float:
    
    # --- 1. Define the hyperparameter search space ---
    trial_params = {
        'objective': 'regression_l1', # Use L1 for robustness
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'device': 'cpu', # Change to 'gpu' if needed
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 30, 120),
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.95),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }

    callbacks_list = [lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)]
    
    kf = GroupKFold(n_splits=N_SPLITS)
    fold_errors = []

    # --- 2. Train and Validate on each K-Fold ---
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y_lat, groups)):
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_lat_train, y_lat_val = y_lat[train_idx], y_lat[val_idx]
        y_lon_train, y_lon_val = y_lon[train_idx], y_lon[val_idx]
        
        ppk_lat_val = ppk_lat[val_idx]
        ppk_lon_val = ppk_lon[val_idx]
        gt_lat_val = gt_lat[val_idx]
        gt_lon_val = gt_lon[val_idx]

        # --- Lat Model (Fold {fold}) ---
        train_data_lat = lgb.Dataset(X_train, label=y_lat_train, feature_name=feature_names, 
                                     categorical_feature=categorical_indices)
        val_data_lat = lgb.Dataset(X_val, label=y_lat_val, reference=train_data_lat)
        
        model_lat = lgb.train(
            trial_params, train_data_lat, num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_data_lat], valid_names=['valid'],
            callbacks=callbacks_list
        )

        # --- Lon Model (Fold {fold}) ---
        train_data_lon = lgb.Dataset(X_train, label=y_lon_train, feature_name=feature_names,
                                     categorical_feature=categorical_indices)
        val_data_lon = lgb.Dataset(X_val, label=y_lon_val, reference=train_data_lon)
        
        model_lon = lgb.train(
            trial_params, train_data_lon, num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_data_lon], valid_names=['valid'],
            callbacks=callbacks_list
        )

        # --- 3. Evaluate (Fold {fold}) ---
        pred_lat_residual_m = model_lat.predict(X_val, num_iteration=model_lat.best_iteration)
        pred_lon_residual_m = model_lon.predict(X_val, num_iteration=model_lon.best_iteration)
        
        pred_lat_residual_deg = pred_lat_residual_m / METERS_PER_DEGREE_LAT
        pred_lon_residual_deg = pred_lon_residual_m / (METERS_PER_DEGREE_LAT * np.cos(np.radians(ppk_lat_val)))
        
        corrected_lat = ppk_lat_val + pred_lat_residual_deg
        corrected_lon = ppk_lon_val + pred_lon_residual_deg
        
        lgbm_error = compute_position_error(corrected_lat, corrected_lon, gt_lat_val, gt_lon_val)
        
        metrics = compute_metrics(lgbm_error)
        fold_errors.append(metrics['🎯 TARGET (Mean+P50+P95)/3'])
    
    # --- 4. Return the average score across all 5 folds ---
    return np.mean(fold_errors)


# ============================================================
# MAIN
# ============================================================

def main():
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("="*100)
    print(f"ROBUST PPK LGBM: Advanced Features + {N_SPLITS}-Fold KFold Optuna")
    print("="*100)

    print("\n[1/7] Loading training data...")
    df = load_all_training_data(TRAIN_DATA_ROOT)
    
    phone_encoder = LabelEncoder()
    df['phone_id_encoded'] = phone_encoder.fit_transform(df['phone_id'])

    print("\n[2/7] Extracting features (Base + Physics + Advanced)...")
    tqdm.pandas(desc="Calculating features")
    X_features = df.groupby(['drive_id', 'phone_id']).progress_apply(extract_position_features)
    X_features = X_features.reset_index(drop=True)
    print(f"  Total features: {X_features.shape[1]}")

    print("\n[3/7] Preparing targets...")
    gt_lat = X_features['gt_lat_base'].values
    gt_lon = X_features['gt_lon_base'].values
    ppk_lat = X_features['ppk_lat_base'].values
    ppk_lon = X_features['ppk_lon_base'].values
    
    # --- This is our Grouping key for K-Fold ---
    groups = X_features['drive_id'].values 

    y_lat = (gt_lat - ppk_lat) * METERS_PER_DEGREE_LAT
    y_lon = (gt_lon - ppk_lon) * METERS_PER_DEGREE_LAT * np.cos(np.radians(gt_lat))

    drop_cols = ['drive_id', 'phone_id', 'ppk_lat_base', 'ppk_lon_base', 
                   'gt_lat_base', 'gt_lon_base', 'ppk_lat', 'ppk_lon']
    feature_names = [col for col in X_features.columns if col not in drop_cols]
    
    X_features['phone_id_encoded'] = phone_encoder.transform(X_features['phone_id'])
    feature_names.append('phone_id_encoded')
    
    X = X_features[feature_names]
    X = X.sort_index(axis=1)
    feature_names = X.columns.tolist()

    print("\n[4/7] Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\n[5/7] Preparing K-Fold split...")
    # Splitting happens inside the Optuna objective
    categorical_cols = ['ppk_quality', 'is_turning', 'is_accelerating', 'is_braking', 
                        'is_stationary', 'high_jerk', 'position_jump', 'is_climbing', 
                        'is_descending', 'poor_quality', 'low_satellites', 
                        'implausible_velocity', 'implausible_acceleration', 
                        'accel_violation', 'turn_violation', 'physics_violation_score',
                        'is_local_outlier', 'phone_id_encoded']
    
    categorical_indices = [i for i, col in enumerate(feature_names) if col in categorical_cols]
    print(f"  Found {len(categorical_indices)} categorical features.")

    print(f"\n[6/7] Running Optuna search ({N_TRIALS} TRIALS, {N_SPLITS} FOLDS each)...")
    
    optuna_objective = partial(
        objective,
        X_scaled=X_scaled, y_lat=y_lat, y_lon=y_lon,
        ppk_lat=ppk_lat, ppk_lon=ppk_lon, 
        gt_lat=gt_lat, gt_lon=gt_lon,
        groups=groups,
        feature_names=feature_names, categorical_indices=categorical_indices
    )
    
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2) # Prune after 2 folds
    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(optuna_objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    print(f"\nOptuna search complete. Best Avg Score (Target Metric): {study.best_trial.value:.4f} m")
    print("Best parameters found:")
    print(best_params)
    
    final_params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'device': 'cpu',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
    }
    final_params.update(best_params)
    
    print("\n[7/7] Training FINAL models on ALL data with best parameters...")
    
    # --- We now train the 5-Fold models for ensembling ---
    kf = GroupKFold(n_splits=N_SPLITS)
    all_models_lat = []
    all_models_lon = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y_lat, groups)):
        print(f"--- Training Fold {fold+1}/{N_SPLITS} ---")
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_lat_train, y_lat_val = y_lat[train_idx], y_lat[val_idx]
        y_lon_train, y_lon_val = y_lon[train_idx], y_lon[val_idx]

        # --- Lat Model ---
        train_data_lat = lgb.Dataset(X_train, label=y_lat_train, feature_name=feature_names, 
                                     categorical_feature=categorical_indices)
        val_data_lat = lgb.Dataset(X_val, label=y_lat_val, reference=train_data_lat)
        
        model_lat = lgb.train(
            final_params, train_data_lat, num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_data_lat], valid_names=['valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS), lgb.log_evaluation(period=200)]
        )
        all_models_lat.append(model_lat)

        # --- Lon Model ---
        train_data_lon = lgb.Dataset(X_train, label=y_lon_train, feature_name=feature_names,
                                     categorical_feature=categorical_indices)
        val_data_lon = lgb.Dataset(X_val, label=y_lon_val, reference=train_data_lon)
        
        model_lon = lgb.train(
            final_params, train_data_lon, num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_data_lon], valid_names=['valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS), lgb.log_evaluation(period=200)]
        )
        all_models_lon.append(model_lon)

    print("\n" + "="*100)
    print(f"SAVING FINAL {N_SPLITS}-FOLD ENSEMBLE MODELS")
    print("="*100)

    # Save all 5 models for lat and lon
    for i in range(N_SPLITS):
        with open(OUTPUT_DIR / f"kfold_model_lat_fold{i}_{timestamp_str}.pkl", 'wb') as f:
            pickle.dump(all_models_lat[i], f)
        with open(OUTPUT_DIR / f"kfold_model_lon_fold{i}_{timestamp_str}.pkl", 'wb') as f:
            pickle.dump(all_models_lon[i], f)
    
    # Save the scaler, encoder, and feature list (we only need one of each)
    scaler_path = OUTPUT_DIR / f"kfold_scaler_{timestamp_str}.pkl"
    encoder_path = OUTPUT_DIR / f"kfold_encoder_{timestamp_str}.pkl"
    features_path = OUTPUT_DIR / f"kfold_features_{timestamp_str}.txt"

    with open(scaler_path, 'wb') as f: pickle.dump(scaler, f)
    with open(encoder_path, 'wb') as f: pickle.dump(phone_encoder, f)
    with open(features_path, 'w') as f: f.write('\n'.join(feature_names))

    print(f"  ✓ Saved {N_SPLITS} Lat models (e.g., kfold_model_lat_fold0_{timestamp_str}.pkl)")
    print(f"  ✓ Saved {N_SPLITS} Lon models (e.g., kfold_model_lon_fold0_{timestamp_str}.pkl)")
    print(f"  ✓ Saved: {scaler_path.name}")
    print(f"  ✓ Saved: {encoder_path.name}")
    print(f"  ✓ Saved: {features_path.name}")
    print("="*100)

if __name__ == "__main__":
    main()