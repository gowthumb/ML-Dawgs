import pandas as pd
import numpy as np


def extract_raw_features(raw_df):
    """
    Extract features from Raw GNSS measurements
    FIXED: Keep TimeNanos as-is for now, will sync later
    """
    print("\n=== Extracting Raw Features ===")
    print(f"Input: {len(raw_df)} measurements")
    
    df = raw_df.copy()
    
    # Convert important columns to numeric
    numeric_cols = [
        'Svid', 'Cn0DbHz', 'AccumulatedDeltaRangeMeters',
        'AccumulatedDeltaRangeUncertaintyMeters', 'PseudorangeRateMetersPerSecond',
        'PseudorangeRateUncertaintyMetersPerSecond', 'TimeNanos',
        'ReceivedSvTimeNanos', 'ConstellationType'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # ===== TIME FEATURES - KEEP RELATIVE TIME =====
    # Don't convert yet - we'll sync with Status data later
    if 'TimeNanos' in df.columns:
        # Convert to milliseconds for consistency
        df['rawTimeMillis'] = (df['TimeNanos'] / 1e6).astype(np.int64)
    
    # ===== SIGNAL QUALITY =====
    if 'Cn0DbHz' not in df.columns:
        print("WARNING: Cn0DbHz not found!")
    
    df = df.sort_values(['Svid', 'TimeNanos'])
    df['Cn0DbHz_smooth'] = df.groupby('Svid')['Cn0DbHz'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    df['Cn0DbHz_rate'] = df.groupby('Svid')['Cn0DbHz'].diff()
    df['Cn0DbHz_rate'] = df['Cn0DbHz_rate'].fillna(0)
    
    # ===== CARRIER PHASE =====
    if 'AccumulatedDeltaRangeMeters' in df.columns:
        df['carrier_phase_m'] = df['AccumulatedDeltaRangeMeters']
        df['carrier_phase_rate'] = df.groupby('Svid')['carrier_phase_m'].diff()
        df['carrier_phase_rate'] = df['carrier_phase_rate'].fillna(0)
        
        if 'AccumulatedDeltaRangeUncertaintyMeters' in df.columns:
            df['carrier_phase_uncertainty'] = df['AccumulatedDeltaRangeUncertaintyMeters']
    
    # ===== PSEUDORANGE =====
    if 'ReceivedSvTimeNanos' in df.columns and 'TimeNanos' in df.columns:
        SPEED_OF_LIGHT = 299792458
        df['travel_time_ns'] = df['TimeNanos'] - df['ReceivedSvTimeNanos']
        df['pseudorange_m'] = (df['travel_time_ns'] * SPEED_OF_LIGHT / 1e9)
        df.loc[df['pseudorange_m'] < 0, 'pseudorange_m'] = np.nan
        df.loc[df['pseudorange_m'] > 30000000, 'pseudorange_m'] = np.nan
    
    # ===== DOPPLER =====
    if 'PseudorangeRateMetersPerSecond' in df.columns:
        df['doppler_mps'] = df['PseudorangeRateMetersPerSecond']
        if 'PseudorangeRateUncertaintyMetersPerSecond' in df.columns:
            df['doppler_uncertainty'] = df['PseudorangeRateUncertaintyMetersPerSecond']
    
    # ===== SATELLITE INFO =====
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
    print(f"Raw time range: {features_df['rawTimeMillis'].min()} to {features_df['rawTimeMillis'].max()}")
    return features_df


def extract_status_features(status_df):
    """
    Extract features from Status (satellite geometry) data
    """
    print("\n=== Extracting Status Features ===")
    print(f"Input: {len(status_df)} measurements")
    
    df = status_df.copy()
    
    numeric_cols = ['Svid', 'Cn0DbHz', 'Elevation', 'Azimuth', 
                    'UnixTimeMillis', 'ConstellationType']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # ===== TIME FEATURE =====
    if 'UnixTimeMillis' in df.columns:
        GPS_EPOCH_UNIX_MS = 315964800000  # GPS epoch in Unix time (ms)
        df['millisSinceGpsEpoch'] = (df['UnixTimeMillis'] - GPS_EPOCH_UNIX_MS).astype(np.int64)
        
        # Remove invalid timestamps
        df = df[df['millisSinceGpsEpoch'] > 0].copy()
    
    # ===== GEOMETRY FEATURES =====
    if 'Elevation' in df.columns:
        df['elevation_deg'] = df['Elevation']
    
    if 'Azimuth' in df.columns:
        df['azimuth_deg'] = df['Azimuth']
    
    # ===== SIGNAL QUALITY =====
    if 'Cn0DbHz' in df.columns:
        df['signal_strength'] = df['Cn0DbHz']
    
    # ===== SATELLITE INFO =====
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
    print(f"GPS time range: {features_df['millisSinceGpsEpoch'].min()} to {features_df['millisSinceGpsEpoch'].max()}")
    return features_df


def sync_raw_to_status_time(raw_features, status_features):
    """
    Synchronize Raw data's relative time to Status data's absolute GPS time
    
    Strategy: Find the offset that best aligns Raw and Status times
    """
    print("\n=== Synchronizing Raw to Status Time ===")
    
    raw_df = raw_features.copy()
    status_df = status_features.copy()
    
    # Get time ranges
    raw_time_min = raw_df['rawTimeMillis'].min()
    raw_time_max = raw_df['rawTimeMillis'].max()
    status_time_min = status_df['millisSinceGpsEpoch'].min()
    status_time_max = status_df['millisSinceGpsEpoch'].max()
    
    print(f"Raw relative time: {raw_time_min} to {raw_time_max} ms")
    print(f"Status GPS time: {status_time_min} to {status_time_max} ms")
    
    # Calculate offset assuming Raw starts near Status start
    # Offset = Status_start - Raw_start
    time_offset = status_time_min - raw_time_min
    
    print(f"Calculated time offset: {time_offset} ms")
    
    # Apply offset to Raw data
    raw_df['millisSinceGpsEpoch'] = raw_df['rawTimeMillis'] + time_offset
    
    # Verify alignment
    raw_gps_min = raw_df['millisSinceGpsEpoch'].min()
    raw_gps_max = raw_df['millisSinceGpsEpoch'].max()
    
    print(f"After sync - Raw GPS time: {raw_gps_min} to {raw_gps_max} ms")
    
    # Check overlap
    overlap = not (raw_gps_max < status_time_min or status_time_max < raw_gps_min)
    
    if overlap:
        overlap_start = max(raw_gps_min, status_time_min)
        overlap_end = min(raw_gps_max, status_time_max)
        overlap_duration = overlap_end - overlap_start
        print(f"✓ Time overlap: {overlap_duration} ms ({overlap_duration/1000:.1f} seconds)")
    else:
        print(f"⚠️ WARNING: No overlap detected!")
        print(f"   Gap: {abs(raw_gps_max - status_time_min)} ms")
    
    # Drop the temporary column
    raw_df = raw_df.drop(columns=['rawTimeMillis'])
    
    return raw_df, status_df


def merge_raw_status_features(raw_features, status_features, time_tolerance_ms=500):
    """
    Merge Raw and Status features with nearest timestamp matching
    """
    print("\n=== Merging Raw + Status Features ===")
    print(f"Raw: {len(raw_features)}, Status: {len(status_features)}")
    
    # Clean the data first
    raw_clean = raw_features.dropna(subset=['millisSinceGpsEpoch', 'satellite_id']).copy()
    status_clean = status_features.dropna(subset=['millisSinceGpsEpoch', 'satellite_id']).copy()
    
    if len(raw_clean) == 0 or len(status_clean) == 0:
        print("⚠️ ERROR: Empty data after cleaning!")
        return pd.DataFrame()
    
    # Convert to int
    raw_clean['millisSinceGpsEpoch'] = raw_clean['millisSinceGpsEpoch'].astype(np.int64)
    status_clean['millisSinceGpsEpoch'] = status_clean['millisSinceGpsEpoch'].astype(np.int64)
    raw_clean['satellite_id'] = raw_clean['satellite_id'].astype(int)
    status_clean['satellite_id'] = status_clean['satellite_id'].astype(int)
    
    # Sort data
    raw_sorted = raw_clean.sort_values(['satellite_id', 'millisSinceGpsEpoch']).reset_index(drop=True)
    status_sorted = status_clean.sort_values(['satellite_id', 'millisSinceGpsEpoch']).reset_index(drop=True)
    
    # Get common satellites
    raw_sats = set(raw_sorted['satellite_id'].unique())
    status_sats = set(status_sorted['satellite_id'].unique())
    common_sats = raw_sats & status_sats
    
    print(f"Common satellites: {len(common_sats)}")
    
    if len(common_sats) == 0:
        print("⚠️ No common satellites!")
        return pd.DataFrame()
    
    # Merge satellite by satellite
    merged_list = []
    total_matches = 0
    
    for sat_id in sorted(common_sats):
        raw_sat = raw_sorted[raw_sorted['satellite_id'] == sat_id].copy()
        status_sat = status_sorted[status_sorted['satellite_id'] == sat_id].copy()
        
        # Use merge_asof for nearest match
        merged_sat = pd.merge_asof(
            raw_sat,
            status_sat,
            on='millisSinceGpsEpoch',
            direction='nearest',
            tolerance=time_tolerance_ms,
            suffixes=('_raw', '_status')
        )
        
        # Count valid matches
        status_cols = [col for col in merged_sat.columns if col.endswith('_status') and col != 'satellite_id_status']
        if status_cols:
            valid = merged_sat[status_cols[0]].notna().sum()
            total_matches += valid
        
        merged_list.append(merged_sat)
    
    merged = pd.concat(merged_list, ignore_index=True)
    
    # Remove rows where merge failed
    status_cols = [col for col in merged.columns if col.endswith('_status') and col != 'satellite_id_status']
    if status_cols:
        merged = merged.dropna(subset=[status_cols[0]])
    
    print(f"✓ Merged: {len(merged)} matches (tolerance={time_tolerance_ms}ms)")
    
    # Clean up duplicate columns
    if 'constellation_raw' in merged.columns and 'constellation_status' in merged.columns:
        merged['constellation'] = merged['constellation_raw'].fillna(merged['constellation_status'])
        merged = merged.drop(columns=['constellation_raw', 'constellation_status'])
    elif 'constellation_raw' in merged.columns:
        merged['constellation'] = merged['constellation_raw']
        merged = merged.drop(columns=['constellation_raw'])
    elif 'constellation_status' in merged.columns:
        merged['constellation'] = merged['constellation_status']
        merged = merged.drop(columns=['constellation_status'])
    
    # Keep satellite_id from raw, drop the status duplicate
    if 'satellite_id_status' in merged.columns:
        merged = merged.drop(columns=['satellite_id_status'])
    # Ensure satellite_id exists (from raw data or status data)
    if 'satellite_id' not in merged.columns and 'satellite_id_raw' in merged.columns:
        merged['satellite_id'] = merged['satellite_id_raw']
        merged = merged.drop(columns=['satellite_id_raw'])
    
    # Combine signal strength
    if 'Cn0DbHz' in merged.columns and 'signal_strength' in merged.columns:
        merged['Cn0DbHz_combined'] = (merged['Cn0DbHz'] + merged['signal_strength']) / 2
    
    return merged


def add_aggregate_features(features_df):
    """
    Add aggregate features per timestamp
    """
    print("\n=== Adding Aggregate Features ===")
    
    if len(features_df) == 0:
        print("⚠️ Empty dataframe, skipping aggregate features")
        return features_df
    
    df = features_df.copy()
    
    # Number of satellites visible
    df['num_satellites'] = df.groupby('millisSinceGpsEpoch')['satellite_id'].transform('count')
    
    # Average signal strength
    if 'Cn0DbHz' in df.columns:
        df['mean_cn0'] = df.groupby('millisSinceGpsEpoch')['Cn0DbHz'].transform('mean')
        df['std_cn0'] = df.groupby('millisSinceGpsEpoch')['Cn0DbHz'].transform('std')
        df['std_cn0'] = df['std_cn0'].fillna(0)
    
    # Elevation statistics
    if 'elevation_deg' in df.columns:
        df['mean_elevation'] = df.groupby('millisSinceGpsEpoch')['elevation_deg'].transform('mean')
        df['max_elevation'] = df.groupby('millisSinceGpsEpoch')['elevation_deg'].transform('max')
        df['num_high_elevation'] = df.groupby('millisSinceGpsEpoch')['elevation_deg'].transform(
            lambda x: (x > 30).sum()
        )
    
    print(f"Added aggregate features. Total columns: {len(df.columns)}")
    return df


def feature_extraction_pipeline(raw_df, status_df, time_tolerance_ms=500):
    """
    Complete feature extraction pipeline with time synchronization
    """
    print("\n" + "="*60)
    print("FEATURE EXTRACTION PIPELINE")
    print("="*60)
    
    # Extract features
    raw_features = extract_raw_features(raw_df)
    status_features = extract_status_features(status_df)
    
    # CRITICAL: Sync Raw relative time to Status absolute time
    raw_features, status_features = sync_raw_to_status_time(raw_features, status_features)
    
    # Merge with tolerance
    merged_features = merge_raw_status_features(raw_features, status_features, time_tolerance_ms)
    
    # Add aggregates
    final_features = add_aggregate_features(merged_features)
    
    print("\n" + "="*60)
    print(f"✓ Feature extraction complete: {len(final_features)} rows")
    if len(final_features) > 0:
        print(f"✓ Total features: {len(final_features.columns)}")
        print(f"✓ Unique timestamps: {final_features['millisSinceGpsEpoch'].nunique()}")
        print(f"✓ Time span: {(final_features['millisSinceGpsEpoch'].max() - final_features['millisSinceGpsEpoch'].min())/1000:.1f} seconds")
    print("="*60 + "\n")
    
    return final_features


def validate_features(features_df):
    """
    Validate and print summary statistics
    """
    print("\n=== FEATURE VALIDATION ===\n")
    
    if len(features_df) == 0:
        print("⚠️ ERROR: No features extracted! Check merge issues.")
        return
    
    # Missing values
    print("Missing Values:")
    missing = features_df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing)
    else:
        print("  None - all features complete!")
    
    # Summary statistics
    print("\nFeature Ranges:")
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:10]:
        vals = features_df[col].dropna()
        if len(vals) > 0:
            print(f"  {col}: min={vals.min():.2f}, max={vals.max():.2f}, mean={vals.mean():.2f}")
    
    # Coverage
    print(f"\nData Coverage:")
    print(f"  Total measurements: {len(features_df)}")
    print(f"  Unique timestamps: {features_df['millisSinceGpsEpoch'].nunique()}")
    print(f"  Unique satellites: {features_df['satellite_id'].nunique()}")
    if 'num_satellites' in features_df.columns:
        print(f"  Avg satellites per epoch: {features_df['num_satellites'].mean():.1f}")
    
    # Signal quality
    if 'Cn0DbHz' in features_df.columns:
        cn0_vals = features_df['Cn0DbHz'].dropna()
        if len(cn0_vals) > 0:
            print(f"\nSignal Quality:")
            print(f"  Mean CN0: {cn0_vals.mean():.1f} dBHz")
            strong_pct = (cn0_vals > 30).sum() / len(cn0_vals) * 100
            print(f"  Strong signals (>30): {strong_pct:.1f}%")
    
    # Cycle slips
    if 'cycle_slip_flag' in features_df.columns:
        slip_count = features_df['cycle_slip_flag'].sum()
        slip_pct = slip_count / len(features_df) * 100
        print(f"\nCycle Slips: {slip_count} ({slip_pct:.2f}%)")
    
    print("\n" + "="*60)


# Example usage
if __name__ == "__main__":
    from preprocess import preprocess_pipeline
    
    # UPDATE THIS PATH
    file_path = "C:\\Users\\avnee\\Downloads\\google-smartphone-decimeter-challenge\\train\\2021-04-15-US-MTV-1\\Pixel4\\Pixel4_GnssLog.txt"
    
    print("Step 1: Preprocessing...")
    preprocessed = preprocess_pipeline(file_path)
    
    print("\nStep 2: Feature Extraction...")
    features = feature_extraction_pipeline(
        preprocessed['raw'],
        preprocessed['status'],
        time_tolerance_ms=500  # Increased tolerance
    )
    
    # Validate
    validate_features(features)
    
    # Save
    if len(features) > 0:
        output_file = './features.csv'
        features.to_csv(output_file, index=False)
        print(f"\n✓ Saved features to: {output_file}")
        print("\nSample Features (first 5 rows):")
        print(features.head())
    else:
        print("\n⚠️ No features to save. Please check your data files.")