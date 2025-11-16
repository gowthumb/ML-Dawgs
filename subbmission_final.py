import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import os
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
import warnings

print(">>> RUNNING K-FOLD ENSEMBLE SUBMISSION (for Advanced Optuna Model) <<<")

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
tqdm.pandas()

# ============================================================
# CONFIGURATION
# ============================================================

# --- 1. Set these paths to your folders ---
PPK_TEST_DIR = Path(r"C:\Users\gauth\Documents\SC4000_proj\pos_output_test")
SAMPLE_SUBMISSION_PATH = Path(r"C:\Users\gauth\Documents\SC4000_proj\sample_submission.csv")
MODEL_DIR = Path("model/outputs")
OUTPUT_DIR = Path("model/submissions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ----------------------------------------

# --- 2. 🛑 UPDATE THIS TIMESTAMP 🛑 ---
# Copy/paste the timestamp (e.g., 20251114_195400) from your training script output
TIMESTAMP = "20251114_225717" # <-- REPLACE THIS
# ----------------------------------------

# --- 3. Model file paths will be generated from the timestamp ---
N_SPLITS = 5
MODEL_PATHS_LAT = [MODEL_DIR / f"kfold_model_lat_fold{i}_{TIMESTAMP}.pkl" for i in range(N_SPLITS)]
MODEL_PATHS_LON = [MODEL_DIR / f"kfold_model_lon_fold{i}_{TIMESTAMP}.pkl" for i in range(N_SPLITS)]
SCALER_PATH = MODEL_DIR / f"kfold_scaler_{TIMESTAMP}.pkl"
ENCODER_PATH = MODEL_DIR / f"kfold_encoder_{TIMESTAMP}.pkl"
FEATURES_PATH = MODEL_DIR / f"kfold_features_{TIMESTAMP}.txt"
# ----------------------------------------

METERS_PER_DEGREE_LAT = 111320.0
GPS_UTC_OFFSET_SECONDS = 18
GPS_EPOCH_OFFSET_MILLIS = 315964800000

# ============================================================
# DATA LOADING (Copied from training script)
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

def load_ppk_for_phone_trip(trip_id: str, phone_name: str) -> pd.DataFrame:
    """Load PPK data for a single test trip."""
    pos_file_name = f"{trip_id}_{phone_name}-pos.txt"
    pos_file_path = PPK_TEST_DIR / pos_file_name
    
    if not pos_file_path.exists():
        pos_file_name = f"{trip_id}-{phone_name}-pos.txt"
        pos_file_path = PPK_TEST_DIR / pos_file_name
    
    if not pos_file_path.exists():
        print(f"Warning: No POS file found for {trip_id}/{phone_name}")
        return None

    ppk_df = read_pos_file(str(pos_file_path))
    if ppk_df.empty:
        return None
        
    ppk_df['drive_id'] = trip_id # Add drive_id for groupby
    ppk_df['phone_id'] = phone_name # Add phone_id for groupby

    ppk_df = ppk_df.sort_values('millisSinceGpsEpoch').reset_index(drop=True)
    return ppk_df

def parse_tripid_phone(tripid_col: str) -> tuple:
    """Parse tripId."""
    parts = tripid_col.rsplit('/', 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (tripid_col, "Unknown")
    
# ============================================================
# FEATURE ENGINEERING (EXACT MATCH from training script)
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
    features['ppk_lat_base'] = df['latitude'] # Keep base ppk for eval
    features['ppk_lon_base'] = df['longitude']
    
    # --- RENAME ---
    df = df.rename(columns={
        'latitude': 'ppk_lat',
        'longitude': 'ppk_lon',
        'height': 'ppk_height',
        'quality': 'ppk_quality',
        'num_satellites': 'ppk_num_satellites',
        'sdn': 'ppk_sdn',
        'sde': 'ppk_sde',
        'sdu': 'ppk_sdu',
        'ratio': 'ppk_ratio',
        'age': 'ppk_age'
    })
    
    # --- ADD RENAMED COLS TO FEATURES ---
    features['ppk_lat'] = df['ppk_lat']
    features['ppk_lon'] = df['ppk_lon']
    features['ppk_height'] = df['ppk_height']
    features['ppk_lat_relative'] = df['ppk_lat'] - df['ppk_lat'].mean()
    features['ppk_lon_relative'] = df['ppk_lon'] - df['ppk_lon'].mean()
    features['ppk_height_relative'] = df['ppk_height'] - df['ppk_height'].mean()

    features['ppk_sdn'] = df['ppk_sdn'].fillna(0)
    features['ppk_sde'] = df['ppk_sde'].fillna(0)
    features['ppk_sdu'] = df['ppk_sdu'].fillna(0)
    features['ppk_quality'] = df['ppk_quality'].fillna(5)
    features['ppk_num_satellites'] = df['ppk_num_satellites'].fillna(0)
    features['ppk_ratio'] = df['ppk_ratio'].fillna(0)
    features['ppk_age'] = df['ppk_age'].fillna(0)

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
# PREDICTION
# ============================================================

def generate_predictions(models_lat, models_lon, scaler, sample_submission: pd.DataFrame, phone_encoder, feature_names: list) -> pd.DataFrame:
    """Generate predictions by ensembling all K-Fold models."""
    GPS_EPOCH_OFFSET_MILLIS = 315964800000

    sample_submission['trip_name'] = sample_submission['tripId'].apply(lambda x: parse_tripid_phone(x)[0])
    sample_submission['phone_name'] = sample_submission['tripId'].apply(lambda x: parse_tripid_phone(x)[1])
    unique_combos = sample_submission[['tripId', 'trip_name', 'phone_name']].drop_duplicates()

    print(f"\nProcessing {len(unique_combos)} unique (trip, phone) combinations...")
    predictions_dict = {}

    for _, row in tqdm(unique_combos.iterrows(), total=len(unique_combos), desc="Generating predictions"):
        tripid_col = row['tripId']
        
        ppk_df = load_ppk_for_phone_trip(row['trip_name'], row['phone_name'])
        
        if ppk_df is None or len(ppk_df) == 0:
            continue

        # --- APPLY OUR FEATURE ENGINEERING ---
        X_df = extract_position_features(ppk_df)
        
        # Add phone_id_encoded
        try:
            X_df['phone_id_encoded'] = phone_encoder.transform([row['phone_name']])[0]
        except ValueError:
             # This phone was not in the training set
            X_df['phone_id_encoded'] = 0 # Use 0 as a default "unknown" class

        # --- Select the feature columns (MUST MATCH TRAINING) ---
        # 1. Ensure all columns exist, fill missing with 0
        for col in feature_names:
            if col not in X_df.columns:
                X_df[col] = 0
                
        # 2. Select and reorder (using the sorted list from training)
        X = X_df[feature_names]

        # Scale features
        X_scaled = scaler.transform(X)

        # --- ENSEMBLING: Predict with all 5 models ---
        all_preds_lat = []
        all_preds_lon = []
        for model_lat, model_lon in zip(models_lat, models_lon):
            all_preds_lat.append(model_lat.predict(X_scaled))
            all_preds_lon.append(model_lon.predict(X_scaled))

        # Average the predictions
        pred_lat_residual_m = np.mean(all_preds_lat, axis=0)
        pred_lon_residual_m = np.mean(all_preds_lon, axis=0)
        # --- END ENSEMBLING ---

        # Convert to degrees
        pred_lat_residual_deg = pred_lat_residual_m / METERS_PER_DEGREE_LAT
        pred_lon_residual_deg = pred_lon_residual_m / (
            METERS_PER_DEGREE_LAT * np.cos(np.radians(X_df['ppk_lat']))
        )

        # Apply corrections
        corrected_lat = X_df['ppk_lat_base'] + pred_lat_residual_deg # Use ppk_lat_base
        corrected_lon = X_df['ppk_lon_base'] + pred_lon_residual_deg # Use ppk_lon_base

        # Convert timestamps
        X_df['UnixTimeMillis'] = ppk_df['millisSinceGpsEpoch'] + GPS_EPOCH_OFFSET_MILLIS

        # Store predictions
        for idx in range(len(X_df)):
            key = (tripid_col, X_df['UnixTimeMillis'].iloc[idx])
            predictions_dict[key] = (corrected_lat.iloc[idx], corrected_lon.iloc[idx])

    # Match to sample submission
    print("\nMatching predictions to required timestamps...")
    results = []
    missing_count = 0

    for _, row in tqdm(sample_submission.iterrows(), total=len(sample_submission), desc="Filling submission"):
        tripid_col = row['tripId']
        unix_time = row['UnixTimeMillis']
        key = (tripid_col, unix_time)

        if key in predictions_dict:
            lat, lon = predictions_dict[key]
        else:
            matching_keys = [(k, v) for k, v in predictions_dict.items() if k[0] == tripid_col]
            if matching_keys:
                nearest_key = min(matching_keys, key=lambda x: abs(x[0][1] - unix_time))
                lat, lon = nearest_key[1]
            else:
                lat, lon = 37.4, -122.1 # Kaggle default
                missing_count += 1

        results.append({
            'tripId': tripid_col,
            'UnixTimeMillis': unix_time,
            'LatitudeDegrees': lat,
            'LongitudeDegrees': lon
        })

    if missing_count > 0:
        print(f"  ⚠  {missing_count} timestamps had no PPK data at all for their trip.")

    return pd.DataFrame(results)


def main():
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("="*80)
    print("GENERATING SUBMISSION - K-Fold Ensemble Model")
    print("="*80)

    # Load models
    print(f"\n[1/3] Loading {N_SPLITS}-Fold Ensemble models...")
    
    # Check for all files
    all_files_exist = True
    all_paths = MODEL_PATHS_LAT + MODEL_PATHS_LON + [SCALER_PATH, ENCODER_PATH, FEATURES_PATH]
    for p in all_paths:
        if not p.exists():
            print(f"  ❌ ERROR: File not found: {p.name}")
            all_files_exist = False
            
    if not all_files_exist:
        print("\nPlease update the TIMESTAMP in the script (lines 29-33) to match your saved models.")
        return

    models_lat = [pickle.load(open(p, 'rb')) for p in MODEL_PATHS_LAT]
    models_lon = [pickle.load(open(p, 'rb')) for p in MODEL_PATHS_LON]
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    phone_encoder = pickle.load(open(ENCODER_PATH, 'rb'))
    with open(FEATURES_PATH, 'r') as f: feature_names = f.read().splitlines()
    
    print(f"  ✓ Loaded {len(models_lat)} Lat models")
    print(f"  ✓ Loaded {len(models_lon)} Lon models")
    print(f"  ✓ Loaded Scaler, Encoder, and {len(feature_names)} features")

    # Load sample submission
    print("\n[2/3] Loading sample submission...")
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    print(f"  ✓ {len(sample_submission):,} rows required")

    # Generate predictions
    print("\n[3/3] Generating predictions (averaging 5 models)...")
    submission_df = generate_predictions(models_lat, models_lon, scaler, sample_submission.copy(), phone_encoder, feature_names)

    # Save
    submission_df = submission_df[['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']]
    submission_path = OUTPUT_DIR / f"submission_kfold_ensemble_{timestamp_str}.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"\n✅ Submission saved: {submission_path}")
    print(f"  Rows: {len(submission_df):,}")
    print("\n📤 Ready to upload to Kaggle!")
    print("="*80)

if __name__ == "__main__":
    main()