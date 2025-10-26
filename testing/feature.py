import pandas as pd
import numpy as np
from typing import List # Added List import for clarity, though not strictly necessary in modern Python for simple list type hints

# ==========================================
# ----------- CRITICAL FIXES APPLIED ------
# ==========================================
# This file has been updated to work with the smartphone-decimeter-2022 dataset:
#
# 1. IMU COLUMN NAMES: The original code used hardcoded column names like 'accel_x/y/z'
#    and 'gyro_x/y/z', but the actual CSV files use standardized names from the schema:
#    - 'UncalAccel.X/Y/Z' for accelerometer data
#    - 'UncalGyro.X/Y/Z' for gyroscope data
#
# 2. SCHEMA COMPATIBILITY: The code now uses the correct column names that match
#    the updated schemas in loader.py, ensuring proper feature extraction.

# --- Feature Extraction Functions ---

def extract_raw_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts features from the cleaned GNSS Raw data (aggregated by synced time).
    
    Features include signal quality metrics (CN0) and time integrity checks (sv_time_diff).
    """
    if raw_df.empty: 
        return pd.DataFrame({})
    
    features = raw_df.copy()
    
    # Ensure necessary columns are present after schema application
    if 'Cn0DbHz' not in features.columns or 'ReceivedSvTimeInNanos' not in features.columns:
        print("Warning: Missing critical columns for raw feature extraction.")
        return pd.DataFrame({})

    # Feature 1: Carrier Frequency Normalized CN0
    # Use a typical maximum CN0 of 45 dBHz for normalization
    features['cn0_norm'] = features['Cn0DbHz'] / 45.0 
    
    # Feature 2: Time Difference between reception and broadcast (in ms)
    # ReceivedSvTimeInNanos is the time the signal left the SV
    features['ReceivedSvTimeMillis'] = features['ReceivedSvTimeInNanos'] / 1_000_000
    # 'millisSinceGpsEpoch' is the time the signal was received by the phone (from data_uitiles.py)
    features['sv_time_diff'] = features['millisSinceGpsEpoch'] - features['ReceivedSvTimeMillis']
    
    # Group and aggregate features by the common time step (10ms resolution)
    # Note: 'millisSinceGpsEpoch' is the aggregated time column created in data_uitiles.py
    raw_features = features.groupby('millisSinceGpsEpoch').agg(
        num_sats=('Svid', 'nunique'),
        mean_cn0=('Cn0DbHz', 'mean'),
        std_cn0=('Cn0DbHz', 'std'),
        max_sv_time_diff=('sv_time_diff', 'max'),
        mean_cn0_norm=('cn0_norm', 'mean')
    ).reset_index()
    
    # The output column name is already correct if the input column was 'millisSinceGpsEpoch'
    # raw_features.rename(columns={'millisSinceGpsEpoch_synced': 'millisSinceGpsEpoch'}, inplace=True) # Removed rename, relying on correct time column name
    
    return raw_features

def extract_status_features(status_df: pd.DataFrame) -> pd.DataFrame:
    """Extracts features from the cleaned GNSS Status data."""
    if status_df.empty: 
        return pd.DataFrame({})
    
    features = status_df.copy()

    # Rename key columns for simplicity and consistency
    features.rename(columns={'WlsPositionVelocity.WlsPosition.WlsPositionStatus': 'wls_status',
                             'NumSatellitesUsed': 'sats_used',
                             'WlsPositionVelocity.WlsPosition.HaeMeters': 'hae_meters'}, 
                             inplace=True)
    
    # Feature 3: Status-based features (e.g., number of satellites used, solution quality)
    status_features = features.groupby('millisSinceGpsEpoch').agg(
        # Position solution quality indicator (0=No fix, 1=2D fix, 2=3D fix, etc.)
        mean_wls_status=('wls_status', 'mean'), 
        max_sats_used=('sats_used', 'max'),
        hae_std=('hae_meters', 'std') # Std dev of Height Above Ellipsoid is a good quality metric
    ).reset_index()

    return status_features

def extract_imu_features(imu_accel_df: pd.DataFrame, imu_gyro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts magnitude and rolling mean features from both Accel and Gyro data and 
    merges them by 'millisSinceBoot'.
    
    Note: Time synchronization to GPS time happens in fuser.py after this step.
    
    Args:
        imu_accel_df: DataFrame containing pre-processed accelerometer readings.
        imu_gyro_df: DataFrame containing pre-processed gyroscope readings.
    """
    all_imu_features: List[pd.DataFrame] = []
    window_size = 50 # 50 samples, roughly 500ms at ~100Hz IMU rate

    # 1. ACCELEROMETER FEATURES
    # CRITICAL FIX: Using correct column names from the updated schema
    # The CSV files use 'UncalAccel.X/Y/Z' after standardization, not 'accel_x/y/z'
    if not imu_accel_df.empty and all(col in imu_accel_df.columns for col in ['UncalAccel.X', 'UncalAccel.Y', 'UncalAccel.Z']):
        accel_features = imu_accel_df.copy()
        # Magnitude calculation using the correct standardized column names
        accel_features['accel_mag'] = np.sqrt(accel_features['UncalAccel.X']**2 + accel_features['UncalAccel.Y']**2 + accel_features['UncalAccel.Z']**2)
        # Rolling mean of magnitude
        accel_features['accel_mag_roll_mean'] = accel_features['accel_mag'].rolling(window=window_size, min_periods=1).mean()
        
        all_imu_features.append(accel_features[['millisSinceBoot', 'accel_mag', 'accel_mag_roll_mean']].drop_duplicates(subset=['millisSinceBoot']))

    # 2. GYROSCOPE FEATURES
    # CRITICAL FIX: Using correct column names from the updated schema
    # The CSV files use 'UncalGyro.X/Y/Z' after standardization, not 'gyro_x/y/z'
    if not imu_gyro_df.empty and all(col in imu_gyro_df.columns for col in ['UncalGyro.X', 'UncalGyro.Y', 'UncalGyro.Z']):
        gyro_features = imu_gyro_df.copy()
        # Magnitude calculation using the correct standardized column names
        gyro_features['gyro_mag'] = np.sqrt(gyro_features['UncalGyro.X']**2 + gyro_features['UncalGyro.Y']**2 + gyro_features['UncalGyro.Z']**2)
        # Rolling mean of magnitude
        gyro_features['gyro_mag_roll_mean'] = gyro_features['gyro_mag'].rolling(window=window_size, min_periods=1).mean()
        
        all_imu_features.append(gyro_features[['millisSinceBoot', 'gyro_mag', 'gyro_mag_roll_mean']].drop_duplicates(subset=['millisSinceBoot']))

    # Combine Accel and Gyro features by 'millisSinceBoot'
    if len(all_imu_features) == 2:
        imu_merged = pd.merge(all_imu_features[0], all_imu_features[1], on='millisSinceBoot', how='outer')
        return imu_merged.sort_values('millisSinceBoot').reset_index(drop=True)
    elif len(all_imu_features) == 1:
        return all_imu_features[0].sort_values('millisSinceBoot').reset_index(drop=True)
    else:
        return pd.DataFrame({})

def add_aggregate_features(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds final aggregate features based on the fully merged GNSS+IMU data.
    These features typically require data points over time for aggregation.
    """
    if merged_df.empty: 
        return merged_df
    
    features = merged_df.copy()
    
    # Feature 4: 1-second rolling standard deviation of IMU magnitude
    # Assuming the merged_df now has a 10ms resolution (100Hz equivalent), 
    # a 1-second window is 100 data points.
    window_size = 100 
    
    if 'accel_mag_mean' in features.columns:
        # Standard deviation of the mean acceleration magnitude over a 1-second window
        features['accel_mag_std_1s_roll'] = features['accel_mag_mean'].rolling(window=window_size, min_periods=1).std()
    
    # Feature 5: Rolling mean of Height Above Ellipsoid (HAE) Std Dev
    # This feature smooths out the confidence measure
    if 'hae_meters_std' in features.columns:
        features['hae_meters_std_roll'] = features['hae_meters_std'].rolling(window=20, min_periods=1).mean()

    return features
