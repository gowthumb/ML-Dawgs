import pandas as pd
import numpy as np


# ============================================================================
# GNSS FEATURE EXTRACTION
# ============================================================================

def extract_raw_features(raw_df):
    """Extract features from Raw GNSS measurements"""
    print("\n=== Extracting Raw Features ===")
    print(f"Input: {len(raw_df)} measurements")
    
    df = raw_df.copy()
    
    numeric_cols = [
        'Svid', 'Cn0DbHz', 'AccumulatedDeltaRangeMeters',
        'AccumulatedDeltaRangeUncertaintyMeters', 'PseudorangeRateMetersPerSecond',
        'PseudorangeRateUncertaintyMetersPerSecond', 'TimeNanos',
        'ReceivedSvTimeNanos', 'ConstellationType'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'TimeNanos' in df.columns:
        df['rawTimeMillis'] = (df['TimeNanos'] / 1e6).astype(np.int64)
    
    if 'Cn0DbHz' not in df.columns:
        print("WARNING: Cn0DbHz not found!")
    
    df = df.sort_values(['Svid', 'TimeNanos'])
    df['Cn0DbHz_smooth'] = df.groupby('Svid')['Cn0DbHz'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    df['Cn0DbHz_rate'] = df.groupby('Svid')['Cn0DbHz'].diff()
    df['Cn0DbHz_rate'] = df['Cn0DbHz_rate'].fillna(0)
    
    if 'AccumulatedDeltaRangeMeters' in df.columns:
        df['carrier_phase_m'] = df['AccumulatedDeltaRangeMeters']
        df['carrier_phase_rate'] = df.groupby('Svid')['carrier_phase_m'].diff()
        df['carrier_phase_rate'] = df['carrier_phase_rate'].fillna(0)
        
        if 'AccumulatedDeltaRangeUncertaintyMeters' in df.columns:
            df['carrier_phase_uncertainty'] = df['AccumulatedDeltaRangeUncertaintyMeters']
    
    if 'ReceivedSvTimeNanos' in df.columns and 'TimeNanos' in df.columns:
        SPEED_OF_LIGHT = 299792458
        df['travel_time_ns'] = df['TimeNanos'] - df['ReceivedSvTimeNanos']
        df['pseudorange_m'] = (df['travel_time_ns'] * SPEED_OF_LIGHT / 1e9)
        df.loc[df['pseudorange_m'] < 0, 'pseudorange_m'] = np.nan
        df.loc[df['pseudorange_m'] > 30000000, 'pseudorange_m'] = np.nan
    
    if 'PseudorangeRateMetersPerSecond' in df.columns:
        df['doppler_mps'] = df['PseudorangeRateMetersPerSecond']
        if 'PseudorangeRateUncertaintyMetersPerSecond' in df.columns:
            df['doppler_uncertainty'] = df['PseudorangeRateUncertaintyMetersPerSecond']
    
    if 'ConstellationType' in df.columns:
        df['constellation'] = df['ConstellationType']
    df['satellite_id'] = df['Svid']
    
    if 'cycle_slip_flag' not in df.columns:
        df['cycle_slip_flag'] = False
    
    feature_cols = [
        'rawTimeMillis', 'satellite_id', 'constellation',
        'Cn0DbHz', 'Cn0DbHz_smooth', 'Cn0DbHz_rate',
        'carrier_phase_m', 'carrier_phase_rate', 'carrier_phase_uncertainty',
        'pseudorange_m', 'doppler_mps', 'doppler_uncertainty',
        'cycle_slip_flag'
    ]
    
    feature_cols = [col for col in feature_cols if col in df.columns]
    features_df = df[feature_cols].copy()
    
    print(f"Output: {len(features_df)} measurements with {len(feature_cols)} features")
    return features_df


def extract_status_features(status_df):
    """Extract features from Status (satellite geometry) data"""
    print("\n=== Extracting Status Features ===")
    print(f"Input: {len(status_df)} measurements")
    
    df = status_df.copy()
    
    numeric_cols = ['Svid', 'Cn0DbHz', 'Elevation', 'Azimuth', 
                    'UnixTimeMillis', 'ConstellationType']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'UnixTimeMillis' in df.columns:
        GPS_EPOCH_UNIX_MS = 315964800000
        df['millisSinceGpsEpoch'] = (df['UnixTimeMillis'] - GPS_EPOCH_UNIX_MS).astype(np.int64)
        df = df[df['millisSinceGpsEpoch'] > 0].copy()
    
    if 'Elevation' in df.columns:
        df['elevation_deg'] = df['Elevation']
    
    if 'Azimuth' in df.columns:
        df['azimuth_deg'] = df['Azimuth']
    
    if 'Cn0DbHz' in df.columns:
        df['signal_strength'] = df['Cn0DbHz']
    
    df['satellite_id'] = df['Svid']
    if 'ConstellationType' in df.columns:
        df['constellation'] = df['ConstellationType']
    
    feature_cols = [
        'millisSinceGpsEpoch', 'satellite_id', 'constellation',
        'elevation_deg', 'azimuth_deg', 'signal_strength'
    ]
    
    feature_cols = [col for col in feature_cols if col in df.columns]
    features_df = df[feature_cols].copy()
    
    print(f"Output: {len(features_df)} measurements with {len(feature_cols)} features")
    return features_df


def sync_raw_to_status_time(raw_features, status_features):
    """Synchronize Raw data's relative time to Status data's absolute GPS time"""
    print("\n=== Synchronizing Raw to Status Time ===")
    
    raw_df = raw_features.copy()
    status_df = status_features.copy()
    
    raw_time_min = raw_df['rawTimeMillis'].min()
    raw_time_max = raw_df['rawTimeMillis'].max()
    status_time_min = status_df['millisSinceGpsEpoch'].min()
    status_time_max = status_df['millisSinceGpsEpoch'].max()
    
    time_offset = status_time_min - raw_time_min
    print(f"Time offset: {time_offset} ms")
    
    raw_df['millisSinceGpsEpoch'] = raw_df['rawTimeMillis'] + time_offset
    
    overlap_duration = min(raw_df['millisSinceGpsEpoch'].max(), status_time_max) - max(raw_df['millisSinceGpsEpoch'].min(), status_time_min)
    print(f"✓ Time overlap: {overlap_duration} ms ({overlap_duration/1000:.1f} seconds)")
    
    raw_df = raw_df.drop(columns=['rawTimeMillis'])
    
    return raw_df, status_df


def merge_raw_status_features(raw_features, status_features, time_tolerance_ms=500):
    """Merge Raw and Status features"""
    print("\n=== Merging Raw + Status Features ===")
    
    raw_clean = raw_features.dropna(subset=['millisSinceGpsEpoch', 'satellite_id']).copy()
    status_clean = status_features.dropna(subset=['millisSinceGpsEpoch', 'satellite_id']).copy()
    
    raw_clean['millisSinceGpsEpoch'] = raw_clean['millisSinceGpsEpoch'].astype(np.int64)
    status_clean['millisSinceGpsEpoch'] = status_clean['millisSinceGpsEpoch'].astype(np.int64)
    raw_clean['satellite_id'] = raw_clean['satellite_id'].astype(int)
    status_clean['satellite_id'] = status_clean['satellite_id'].astype(int)
    
    raw_sorted = raw_clean.sort_values(['satellite_id', 'millisSinceGpsEpoch']).reset_index(drop=True)
    status_sorted = status_clean.sort_values(['satellite_id', 'millisSinceGpsEpoch']).reset_index(drop=True)
    
    common_sats = set(raw_sorted['satellite_id'].unique()) & set(status_sorted['satellite_id'].unique())
    print(f"Common satellites: {len(common_sats)}")
    
    merged_list = []
    for sat_id in sorted(common_sats):
        raw_sat = raw_sorted[raw_sorted['satellite_id'] == sat_id].copy()
        status_sat = status_sorted[status_sorted['satellite_id'] == sat_id].copy()
        
        merged_sat = pd.merge_asof(
            raw_sat, status_sat,
            on='millisSinceGpsEpoch',
            direction='nearest',
            tolerance=time_tolerance_ms,
            suffixes=('_raw', '_status')
        )
        merged_list.append(merged_sat)
    
    merged = pd.concat(merged_list, ignore_index=True)
    
    status_cols = [col for col in merged.columns if col.endswith('_status') and col != 'satellite_id_status']
    if status_cols:
        merged = merged.dropna(subset=[status_cols[0]])
    
    print(f"✓ Merged: {len(merged)} matches")
    
    # Clean up duplicate columns
    if 'constellation_raw' in merged.columns and 'constellation_status' in merged.columns:
        merged['constellation'] = merged['constellation_raw'].fillna(merged['constellation_status'])
        merged = merged.drop(columns=['constellation_raw', 'constellation_status'])
    
    if 'satellite_id_status' in merged.columns:
        merged = merged.drop(columns=['satellite_id_status'])
    if 'satellite_id' not in merged.columns and 'satellite_id_raw' in merged.columns:
        merged['satellite_id'] = merged['satellite_id_raw']
        merged = merged.drop(columns=['satellite_id_raw'])
    
    if 'Cn0DbHz' in merged.columns and 'signal_strength' in merged.columns:
        merged['Cn0DbHz_combined'] = (merged['Cn0DbHz'] + merged['signal_strength']) / 2
    
    return merged


def add_gnss_aggregate_features(features_df):
    """Add aggregate features per timestamp"""
    print("\n=== Adding GNSS Aggregate Features ===")
    
    if len(features_df) == 0:
        return features_df
    
    df = features_df.copy()
    
    df['num_satellites'] = df.groupby('millisSinceGpsEpoch')['satellite_id'].transform('count')
    
    if 'Cn0DbHz' in df.columns:
        df['mean_cn0'] = df.groupby('millisSinceGpsEpoch')['Cn0DbHz'].transform('mean')
        df['std_cn0'] = df.groupby('millisSinceGpsEpoch')['Cn0DbHz'].transform('std').fillna(0)
    
    if 'elevation_deg' in df.columns:
        df['mean_elevation'] = df.groupby('millisSinceGpsEpoch')['elevation_deg'].transform('mean')
        df['max_elevation'] = df.groupby('millisSinceGpsEpoch')['elevation_deg'].transform('max')
        df['num_high_elevation'] = df.groupby('millisSinceGpsEpoch')['elevation_deg'].transform(lambda x: (x > 30).sum())
    
    print(f"Total columns: {len(df.columns)}")
    return df


# ============================================================================
# IMU FEATURE EXTRACTION
# ============================================================================

def extract_accelerometer_features(accel_df):
    """Extract features from accelerometer data"""
    df = accel_df.copy()
    
    numeric_cols = ['utcTimeMillis', 'UncalAccelXMps2', 'UncalAccelYMps2', 'UncalAccelZMps2']
    if 'UncalAccelXMps2' not in df.columns and 'AccelXMps2' in df.columns:
        numeric_cols = ['utcTimeMillis', 'AccelXMps2', 'AccelYMps2', 'AccelZMps2']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.sort_values('utcTimeMillis').reset_index(drop=True)
    
    if 'UncalAccelXMps2' in df.columns:
        x_col, y_col, z_col = 'UncalAccelXMps2', 'UncalAccelYMps2', 'UncalAccelZMps2'
    else:
        x_col, y_col, z_col = 'AccelXMps2', 'AccelYMps2', 'AccelZMps2'
    
    features = pd.DataFrame()
    features['utcTimeMillis'] = df['utcTimeMillis']
    features['accel_x'] = df[x_col]
    features['accel_y'] = df[y_col]
    features['accel_z'] = df[z_col]
    
    features['accel_magnitude'] = np.sqrt(df[x_col]**2 + df[y_col]**2 + df[z_col]**2)
    features['accel_magnitude_no_gravity'] = np.sqrt(df[x_col]**2 + df[y_col]**2 + (df[z_col] - 9.81)**2)
    
    window = 10
    features['accel_magnitude_smooth'] = features['accel_magnitude'].rolling(window=window, min_periods=1).mean()
    features['accel_variance'] = features['accel_magnitude'].rolling(window=window, min_periods=1).std().fillna(0)
    features['accel_jerk'] = features['accel_magnitude'].diff().fillna(0)
    
    return features


def extract_gyroscope_features(gyro_df):
    """Extract features from gyroscope data"""
    df = gyro_df.copy()
    
    numeric_cols = ['utcTimeMillis', 'UncalGyroXRadPerSec', 'UncalGyroYRadPerSec', 'UncalGyroZRadPerSec']
    if 'UncalGyroXRadPerSec' not in df.columns and 'GyroXRadPerSec' in df.columns:
        numeric_cols = ['utcTimeMillis', 'GyroXRadPerSec', 'GyroYRadPerSec', 'GyroZRadPerSec']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.sort_values('utcTimeMillis').reset_index(drop=True)
    
    if 'UncalGyroXRadPerSec' in df.columns:
        x_col, y_col, z_col = 'UncalGyroXRadPerSec', 'UncalGyroYRadPerSec', 'UncalGyroZRadPerSec'
    else:
        x_col, y_col, z_col = 'GyroXRadPerSec', 'GyroYRadPerSec', 'GyroZRadPerSec'
    
    features = pd.DataFrame()
    features['utcTimeMillis'] = df['utcTimeMillis']
    features['gyro_x'] = df[x_col]
    features['gyro_y'] = df[y_col]
    features['gyro_z'] = df[z_col]
    
    features['gyro_magnitude'] = np.sqrt(df[x_col]**2 + df[y_col]**2 + df[z_col]**2)
    
    window = 10
    features['gyro_magnitude_smooth'] = features['gyro_magnitude'].rolling(window=window, min_periods=1).mean()
    features['gyro_variance'] = features['gyro_magnitude'].rolling(window=window, min_periods=1).std().fillna(0)
    features['gyro_jerk'] = features['gyro_magnitude'].diff().fillna(0)
    
    return features


def extract_imu_features(accel_df, gyro_df, orient_df=None):
    """Extract features from IMU (Inertial Measurement Unit) data"""
    print("\n=== Extracting IMU Features ===")
    
    features_list = []
    
    if accel_df is not None and len(accel_df) > 0:
        print(f"Processing {len(accel_df)} accelerometer measurements...")
        accel_features = extract_accelerometer_features(accel_df)
        features_list.append(accel_features)
    
    if gyro_df is not None and len(gyro_df) > 0:
        print(f"Processing {len(gyro_df)} gyroscope measurements...")
        gyro_features = extract_gyroscope_features(gyro_df)
        features_list.append(gyro_features)
    
    if len(features_list) == 0:
        print("⚠️ No IMU data available")
        return pd.DataFrame()
    
    merged = features_list[0]
    for features in features_list[1:]:
        merged = pd.merge(merged, features, on='utcTimeMillis', how='outer', suffixes=('', '_dup'))
        merged = merged[[c for c in merged.columns if not c.endswith('_dup')]]
    
    merged = merged.sort_values('utcTimeMillis').reset_index(drop=True)
    merged = merged.fillna(method='ffill').fillna(method='bfill')
    
    print(f"✓ IMU features: {len(merged)} rows, {len(merged.columns)} features")
    return merged


# ============================================================================
# COMBINED PIPELINE
# ============================================================================

def merge_gnss_imu_features(gnss_features, imu_features):
    """
    Merge GNSS and IMU features based on timestamps
    
    Challenge: GNSS uses GPS time (millisSinceGpsEpoch)
               IMU uses Unix time (utcTimeMillis)
    
    Solution: Convert GPS time to Unix time for merging
    """
    print("\n=== Merging GNSS + IMU Features ===")
    
    if len(gnss_features) == 0 or len(imu_features) == 0:
        print("⚠️ Cannot merge: missing GNSS or IMU data")
        return gnss_features if len(gnss_features) > 0 else imu_features
    
    gnss = gnss_features.copy()
    imu = imu_features.copy()
    
    # Convert GPS time to Unix time for merging
    GPS_EPOCH_UNIX_MS = 315964800000
    gnss['utcTimeMillis'] = gnss['millisSinceGpsEpoch'] + GPS_EPOCH_UNIX_MS
    
    print(f"GNSS time range: {gnss['utcTimeMillis'].min()} to {gnss['utcTimeMillis'].max()}")
    print(f"IMU time range: {imu['utcTimeMillis'].min()} to {imu['utcTimeMillis'].max()}")
    
    # Sort both
    gnss = gnss.sort_values('utcTimeMillis').reset_index(drop=True)
    imu = imu.sort_values('utcTimeMillis').reset_index(drop=True)
    
    # Use merge_asof for nearest timestamp matching
    merged = pd.merge_asof(
        gnss,
        imu,
        on='utcTimeMillis',
        direction='nearest',
        tolerance=100,  # 100ms tolerance
        suffixes=('', '_imu')
    )
    
    # Count valid IMU matches
    imu_cols = [col for col in merged.columns if '_imu' in col or col.startswith(('accel_', 'gyro_'))]
    if imu_cols:
        valid_matches = merged[imu_cols[0]].notna().sum()
        print(f"✓ Merged {valid_matches}/{len(gnss)} GNSS measurements with IMU data")
    
    # Keep millisSinceGpsEpoch as primary time column
    merged = merged.drop(columns=['utcTimeMillis'])
    
    return merged


def complete_feature_pipeline(raw_df, status_df, accel_df=None, gyro_df=None, orient_df=None):
    """Complete pipeline: GNSS + IMU features"""
    print("\n" + "="*60)
    print("COMPLETE FEATURE EXTRACTION PIPELINE")
    print("="*60)
    
    # Extract GNSS features
    raw_features = extract_raw_features(raw_df)
    status_features = extract_status_features(status_df)
    raw_features, status_features = sync_raw_to_status_time(raw_features, status_features)
    gnss_features = merge_raw_status_features(raw_features, status_features)
    gnss_features = add_gnss_aggregate_features(gnss_features)
    
    print(f"\n✓ GNSS features: {len(gnss_features)} rows")
    
    # Extract IMU features if available
    imu_features = None
    if accel_df is not None or gyro_df is not None:
        imu_features = extract_imu_features(accel_df, gyro_df, orient_df)
        if len(imu_features) > 0:
            print(f"✓ IMU features: {len(imu_features)} rows")
    
    # Merge GNSS + IMU
    if imu_features is not None and len(imu_features) > 0:
        final_features = merge_gnss_imu_features(gnss_features, imu_features)
    else:
        print("\n⚠️ No IMU data - using GNSS only")
        final_features = gnss_features
    
    print("\n" + "="*60)
    print(f"✓ Complete pipeline done: {len(final_features)} rows, {len(final_features.columns)} features")
    print("="*60 + "\n")
    
    return final_features


# Example usage
if __name__ == "__main__":
    from preprocess import preprocess_pipeline, read_gnss_log
    
    file_path = "C:\\Users\\avnee\\Downloads\\google-smartphone-decimeter-challenge\\train\\2021-04-15-US-MTV-1\\Pixel4\\Pixel4_GnssLog.txt"
    
    print("Step 1: Preprocessing...")
    preprocessed = preprocess_pipeline(file_path)
    
    print("\nStep 2: Reading IMU data...")
    all_data = read_gnss_log(file_path)
    
    accel_df = all_data.get('uncalaccel')
    if accel_df is None:
        accel_df = all_data.get('accel')
    gyro_df = all_data.get('uncalgyro')
    if gyro_df is None:
        gyro_df = all_data.get('gyro')    
    print("\nStep 3: Feature Extraction...")
    features = complete_feature_pipeline(
        preprocessed['raw'],
        preprocessed['status'],
        accel_df,
        gyro_df
    )
    
    # Save
    if len(features) > 0:
        output_file = './complete_features.csv'
        features.to_csv(output_file, index=False)
        print(f"\n✓ Saved to: {output_file}")
        print(f"\nFeature columns ({len(features.columns)}):")
        print(list(features.columns))
        print("\nSample:")
        print(features.head())