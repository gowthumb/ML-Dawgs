# feature.py
"""
GNSS and IMU Feature Extraction
Enhanced version combining teammate's aggregation + original detailed features
"""

import pandas as pd
import numpy as np
from typing import List


# ============================================================================
# GNSS RAW FEATURE EXTRACTION
# ============================================================================

def extract_raw_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from the cleaned GNSS Raw data (aggregated by synced time).
    
    Features include:
    - Signal quality metrics (CN0)
    - Time integrity checks (sv_time_diff)
    - Pseudorange and carrier phase features
    - Doppler shift features
    """
    if raw_df.empty: 
        return pd.DataFrame({})
    
    features = raw_df.copy()
    
    # Ensure necessary columns are present
    if 'Cn0DbHz' not in features.columns:
        print("Warning: Missing Cn0DbHz column for raw feature extraction.")
        return pd.DataFrame({})

    # === PER-MEASUREMENT FEATURES ===
    
    # Feature 1: Carrier Frequency Normalized CN0
    features['cn0_norm'] = features['Cn0DbHz'] / 45.0  # Normalize by typical max
    
    # Feature 2: Time Difference (signal travel time)
    if 'ReceivedSvTimeInNanos' in features.columns:
        features['ReceivedSvTimeMillis'] = features['ReceivedSvTimeInNanos'] / 1_000_000
        features['sv_time_diff'] = features['millisSinceGpsEpoch'] - features['ReceivedSvTimeMillis']
    
    # Feature 3: CN0 smoothing and rate of change (per satellite)
    features = features.sort_values(['Svid', 'millisSinceGpsEpoch'])
    features['Cn0DbHz_smooth'] = features.groupby('Svid')['Cn0DbHz'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    features['Cn0DbHz_rate'] = features.groupby('Svid')['Cn0DbHz'].diff().fillna(0)
    
    # Feature 4: Carrier Phase features
    if 'AccumulatedDeltaRangeMeters' in features.columns:
        features['carrier_phase_m'] = features['AccumulatedDeltaRangeMeters']
        features['carrier_phase_rate'] = features.groupby('Svid')['carrier_phase_m'].diff().fillna(0)
    
    # Feature 5: Pseudorange computation (if possible)
    if all(col in features.columns for col in ['TimeNanos', 'ReceivedSvTimeNanos']):
        SPEED_OF_LIGHT = 299792458  # m/s
        features['travel_time_ns'] = features['TimeNanos'] - features['ReceivedSvTimeNanos']
        features['pseudorange_m'] = (features['travel_time_ns'] * SPEED_OF_LIGHT / 1e9)
        
        # Sanity filter for pseudorange (20,000 km to 30,000 km typical)
        features.loc[features['pseudorange_m'] < 20000000, 'pseudorange_m'] = np.nan
        features.loc[features['pseudorange_m'] > 30000000, 'pseudorange_m'] = np.nan
    
    # Feature 6: Doppler shift
    if 'PseudorangeRateMetersPerSecond' in features.columns:
        features['doppler_mps'] = features['PseudorangeRateMetersPerSecond']
    
    # === AGGREGATE FEATURES (per time epoch) ===
    
    raw_features = features.groupby('millisSinceGpsEpoch').agg(
        num_sats=('Svid', 'nunique'),
        mean_cn0=('Cn0DbHz', 'mean'),
        std_cn0=('Cn0DbHz', 'std'),
        max_cn0=('Cn0DbHz', 'max'),
        min_cn0=('Cn0DbHz', 'min'),
        mean_cn0_norm=('cn0_norm', 'mean'),
        max_sv_time_diff=('sv_time_diff', 'max') if 'sv_time_diff' in features.columns else ('Svid', 'count'),
        mean_cn0_smooth=('Cn0DbHz_smooth', 'mean'),
        std_cn0_rate=('Cn0DbHz_rate', 'std'),
        mean_pseudorange=('pseudorange_m', 'mean') if 'pseudorange_m' in features.columns else ('Svid', 'count'),
        std_pseudorange=('pseudorange_m', 'std') if 'pseudorange_m' in features.columns else ('Svid', 'count'),
        mean_doppler=('doppler_mps', 'mean') if 'doppler_mps' in features.columns else ('Svid', 'count'),
        num_cycle_slips=('cycle_slip_flag', 'sum') if 'cycle_slip_flag' in features.columns else ('Svid', 'count')
    ).reset_index()
    
    # Fill NaNs in std columns with 0
    for col in raw_features.columns:
        if 'std' in col:
            raw_features[col] = raw_features[col].fillna(0)
    
    return raw_features


# ============================================================================
# GNSS STATUS FEATURE EXTRACTION
# ============================================================================

def extract_status_features(status_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from the cleaned GNSS Status data.
    Includes satellite geometry and position quality metrics.
    """
    if status_df.empty: 
        return pd.DataFrame({})
    
    features = status_df.copy()

    # Rename key columns for simplicity
    rename_map = {
        'WlsPositionVelocity.WlsPosition.WlsPositionStatus': 'wls_status',
        'NumSatellitesUsed': 'sats_used',
        'WlsPositionVelocity.WlsPosition.HaeMeters': 'hae_meters'
    }
    
    for old_name, new_name in rename_map.items():
        if old_name in features.columns:
            features.rename(columns={old_name: new_name}, inplace=True)
    
    # === SATELLITE GEOMETRY FEATURES ===
    
    # Elevation-based features
    if 'Elevation' in features.columns:
        features['elevation_weight'] = np.clip(features['Elevation'] / 90, 0.1, 1.0)
        features['is_high_elevation'] = (features['Elevation'] > 30).astype(int)
    
    # CN0 weighted by elevation (higher elevation = more reliable)
    if 'Cn0DbHz' in features.columns and 'elevation_weight' in features.columns:
        features['weighted_cn0'] = features['Cn0DbHz'] * features['elevation_weight']
    
    # === AGGREGATE FEATURES (per time epoch) ===
    
    agg_dict = {
        'Svid': 'count'  # Number of satellites visible
    }
    
    if 'wls_status' in features.columns:
        agg_dict['wls_status'] = 'mean'
    
    if 'sats_used' in features.columns:
        agg_dict['sats_used'] = 'max'
    
    if 'hae_meters' in features.columns:
        agg_dict['hae_meters'] = 'std'
    
    if 'Elevation' in features.columns:
        agg_dict['Elevation'] = ['mean', 'std', 'max', 'min']
    
    if 'Azimuth' in features.columns:
        agg_dict['Azimuth'] = 'std'  # Azimuth spread = geometry quality
    
    if 'weighted_cn0' in features.columns:
        agg_dict['weighted_cn0'] = 'mean'
    
    if 'is_high_elevation' in features.columns:
        agg_dict['is_high_elevation'] = 'sum'  # Count of high elevation sats
    
    status_features = features.groupby('millisSinceGpsEpoch').agg(agg_dict).reset_index()
    
    # Flatten multi-level columns
    status_features.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                               for col in status_features.columns.values]
    
    # Rename for clarity
    rename_map = {
        'Svid_count': 'num_sats_status',
        'wls_status_mean': 'mean_wls_status',
        'sats_used_max': 'max_sats_used',
        'hae_meters_std': 'hae_std',
        'Elevation_mean': 'mean_elevation',
        'Elevation_std': 'std_elevation',
        'Elevation_max': 'max_elevation',
        'Elevation_min': 'min_elevation',
        'Azimuth_std': 'azimuth_spread',
        'weighted_cn0_mean': 'mean_weighted_cn0',
        'is_high_elevation_sum': 'num_high_elev_sats'
    }
    
    for old_name, new_name in rename_map.items():
        if old_name in status_features.columns:
            status_features.rename(columns={old_name: new_name}, inplace=True)
    
    # Fill NaN std values with 0
    for col in status_features.columns:
        if 'std' in col or 'hae' in col:
            status_features[col] = status_features[col].fillna(0)
    
    return status_features


# ============================================================================
# IMU FEATURE EXTRACTION
# ============================================================================

def extract_imu_features(imu_accel_df: pd.DataFrame, imu_gyro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract magnitude and rolling mean features from both Accel and Gyro data.
    
    Note: Time synchronization to GPS time happens in fuser.py after this step.
    
    Args:
        imu_accel_df: DataFrame with 'millisSinceBoot' and accel columns
        imu_gyro_df: DataFrame with 'millisSinceBoot' and gyro columns
    """
    all_imu_features: List[pd.DataFrame] = []
    window_size = 50  # 50 samples, roughly 500ms at ~100Hz IMU rate

    # 1. ACCELEROMETER FEATURES
    # Handle both column naming conventions
    accel_cols = None
    if not imu_accel_df.empty:
        if all(col in imu_accel_df.columns for col in ['accel_x', 'accel_y', 'accel_z']):
            accel_cols = ('accel_x', 'accel_y', 'accel_z')
        elif all(col in imu_accel_df.columns for col in ['UncalAccel.X', 'UncalAccel.Y', 'UncalAccel.Z']):
            accel_cols = ('UncalAccel.X', 'UncalAccel.Y', 'UncalAccel.Z')
    
    if accel_cols:
        accel_features = imu_accel_df.copy()
        x_col, y_col, z_col = accel_cols
        
        # Magnitude
        accel_features['accel_mag'] = np.sqrt(
            accel_features[x_col]**2 + 
            accel_features[y_col]**2 + 
            accel_features[z_col]**2
        )
        
        # Magnitude without gravity (approximate)
        accel_features['accel_mag_no_gravity'] = np.sqrt(
            accel_features[x_col]**2 + 
            accel_features[y_col]**2 + 
            (accel_features[z_col] - 9.81)**2
        )
        
        # Rolling mean of magnitude
        accel_features['accel_mag_roll_mean'] = accel_features['accel_mag'].rolling(
            window=window_size, min_periods=1
        ).mean()
        
        # Variance (measure of motion intensity)
        accel_features['accel_variance'] = accel_features['accel_mag'].rolling(
            window=window_size, min_periods=1
        ).std().fillna(0)
        
        # Jerk (rate of change of acceleration)
        accel_features['accel_jerk'] = accel_features['accel_mag'].diff().fillna(0)
        
        all_imu_features.append(
            accel_features[[
                'millisSinceBoot', 'accel_mag', 'accel_mag_roll_mean', 
                'accel_variance', 'accel_jerk', 'accel_mag_no_gravity'
            ]].drop_duplicates(subset=['millisSinceBoot'])
        )

    # 2. GYROSCOPE FEATURES
    # Handle both column naming conventions
    gyro_cols = None
    if not imu_gyro_df.empty:
        if all(col in imu_gyro_df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
            gyro_cols = ('gyro_x', 'gyro_y', 'gyro_z')
        elif all(col in imu_gyro_df.columns for col in ['UncalGyro.X', 'UncalGyro.Y', 'UncalGyro.Z']):
            gyro_cols = ('UncalGyro.X', 'UncalGyro.Y', 'UncalGyro.Z')
    
    if gyro_cols:
        gyro_features = imu_gyro_df.copy()
        x_col, y_col, z_col = gyro_cols
        
        # Magnitude (angular velocity)
        gyro_features['gyro_mag'] = np.sqrt(
            gyro_features[x_col]**2 + 
            gyro_features[y_col]**2 + 
            gyro_features[z_col]**2
        )
        
        # Rolling mean
        gyro_features['gyro_mag_roll_mean'] = gyro_features['gyro_mag'].rolling(
            window=window_size, min_periods=1
        ).mean()
        
        # Variance (measure of rotation intensity)
        gyro_features['gyro_variance'] = gyro_features['gyro_mag'].rolling(
            window=window_size, min_periods=1
        ).std().fillna(0)
        
        # Angular jerk
        gyro_features['gyro_jerk'] = gyro_features['gyro_mag'].diff().fillna(0)
        
        all_imu_features.append(
            gyro_features[[
                'millisSinceBoot', 'gyro_mag', 'gyro_mag_roll_mean',
                'gyro_variance', 'gyro_jerk'
            ]].drop_duplicates(subset=['millisSinceBoot'])
        )

    # 3. COMBINE ACCEL AND GYRO
    if len(all_imu_features) == 2:
        imu_merged = pd.merge(all_imu_features[0], all_imu_features[1], on='millisSinceBoot', how='outer')
        return imu_merged.sort_values('millisSinceBoot').reset_index(drop=True)
    elif len(all_imu_features) == 1:
        return all_imu_features[0].sort_values('millisSinceBoot').reset_index(drop=True)
    else:
        return pd.DataFrame({})


# ============================================================================
# AGGREGATE FEATURES (Post-Merge)
# ============================================================================

def add_aggregate_features(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add final aggregate features based on the fully merged GNSS+IMU data.
    These features typically require data points over time for aggregation.
    """
    if merged_df.empty: 
        return merged_df
    
    features = merged_df.copy()
    
    # Assuming 10ms resolution, a 1-second window is 100 data points
    window_1s = 100 
    window_5s = 500
    
    # Feature 1: Rolling std of IMU magnitude (motion stability)
    if 'accel_mag_mean' in features.columns:
        features['accel_mag_std_1s_roll'] = features['accel_mag_mean'].rolling(
            window=window_1s, min_periods=1
        ).std().fillna(0)
    
    # Feature 2: Rolling mean of HAE std (position confidence)
    if 'hae_std' in features.columns:
        features['hae_std_roll'] = features['hae_std'].rolling(
            window=20, min_periods=1
        ).mean().fillna(0)
    
    # Feature 3: CN0 trend (improving or degrading signal)
    if 'mean_cn0' in features.columns:
        features['cn0_trend'] = features['mean_cn0'].diff().rolling(
            window=10, min_periods=1
        ).mean().fillna(0)
    
    # Feature 4: Satellite count stability
    if 'num_sats' in features.columns:
        features['num_sats_std_5s'] = features['num_sats'].rolling(
            window=window_5s, min_periods=1
        ).std().fillna(0)
    
    # Feature 5: Motion state indicator (stationary vs moving)
    if 'accel_variance' in features.columns:
        # Low variance + low magnitude = stationary
        features['is_stationary'] = (
            (features['accel_variance'] < 0.1) & 
            (features.get('accel_mag_mean', 10) < 10.5)
        ).astype(int)
    
    # Feature 6: High-quality satellite ratio
    if 'num_high_elev_sats' in features.columns and 'num_sats_status' in features.columns:
        features['high_qual_sat_ratio'] = features['num_high_elev_sats'] / (features['num_sats_status'] + 1)
    
    return features