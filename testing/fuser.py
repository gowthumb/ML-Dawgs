import pandas as pd
import numpy as np
# Dependencies for reading, cleaning, and time preparation
from data_uitiles import read_and_separate_log, apply_sanity_filters, add_time_columns
# Dependencies for feature creation and final aggregation
from feature import extract_raw_features, extract_status_features, extract_imu_features, add_aggregate_features

def synchronize_imu_time(imu_df: pd.DataFrame, gnss_raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the time offset between millisSinceBoot (IMU time) and 
    millisSinceGpsEpoch (GNSS/Absolute time) and applies it to the IMU data.
    
    The synchronization relies on finding a common point in the GNSS Raw data
    where both the absolute GPS time ('millisSinceGpsEpoch') and the 
    device's internal boot time ('millisSinceBoot_raw') are available.
    
    Args:
        imu_df: IMU features dataframe (must contain 'millisSinceBoot').
        gnss_raw_df: Cleaned GNSS Raw dataframe (must contain 'millisSinceGpsEpoch'
                     and 'millisSinceBoot_raw').
                     
    Returns:
        The IMU dataframe with a new 'millisSinceGpsEpoch' column, or an empty
        DataFrame if synchronization fails.
    """
    if imu_df.empty or gnss_raw_df.empty:
        print("Synchronization failed: Empty IMU or GNSS raw data.")
        return pd.DataFrame({})

    # 1. Find the offset from the GNSS Raw data
    # We use the first point where both absolute GPS time and relative boot time are available.
    sync_points = gnss_raw_df.dropna(subset=['millisSinceGpsEpoch', 'millisSinceBoot_raw'])
    
    if sync_points.empty:
        print("Synchronization failed: Cannot find a common time reference in raw data.")
        return pd.DataFrame({})
        
    first_sync_point = sync_points.iloc[0]
    
    # Time Offset = Absolute Time - Relative Time
    gps_time = first_sync_point['millisSinceGpsEpoch']
    boot_time = first_sync_point['millisSinceBoot_raw']
    time_offset = gps_time - boot_time
    
    # 2. Apply the offset to the IMU data (which only has millisSinceBoot)
    # The output column name is 'millisSinceGpsEpoch' for consistency
    imu_df['millisSinceGpsEpoch'] = imu_df['millisSinceBoot'] + time_offset
    
    # Use the nearest 10ms for merging to match the GNSS time resolution
    imu_df['millisSinceGpsEpoch'] = imu_df['millisSinceGpsEpoch'].round(-1).astype(np.int64)
    
    return imu_df.reset_index(drop=True)


def process_full_log_to_features(job: dict) -> pd.DataFrame:
    """
    Reads, preprocesses, synchronizes, extracts features, and merges 
    GNSS and IMU data for a single drive/phone log pair.
    
    Args:
        job (dict): Dictionary containing 'gnss_file', 'imu_file', 'drive_id', and 'phone_id'.
        
    Returns:
        A single merged DataFrame containing all extracted features, 
        or an empty DataFrame on failure.
    """
    gnss_log_path = job['gnss_file']
    imu_log_path = job['imu_file']
    
    print(f"\n--- Processing Log Pair: {job['drive_id']}/{job['phone_id']} ---")
    
    # --- 1. READ AND SEPARATE LOGS ---
    gnss_data = read_and_separate_log(gnss_log_path)
    imu_data = read_and_separate_log(imu_log_path)
    
    if not gnss_data or not imu_data:
        print("Skipping: One or both log files could not be read/separated.")
        return pd.DataFrame({})

    # --- 2. SANITY FILTERS & TIME CALCULATION (Using standardized names) ---
    
    # GNSS RAW: Filter, standardize columns, and calculate absolute time ('millisSinceGpsEpoch')
    raw_df = apply_sanity_filters(gnss_data.get('raw', pd.DataFrame({})), 'raw')
    raw_df = add_time_columns(raw_df) 
    
    # GNSS STATUS: Filter and standardize columns
    status_df = apply_sanity_filters(gnss_data.get('status', pd.DataFrame({})), 'status')
    
    # IMU ACCEL/GYRO: Filter and standardize columns
    imu_accel = apply_sanity_filters(imu_data.get('accel', pd.DataFrame({})), 'accel')
    imu_gyro = apply_sanity_filters(imu_data.get('gyro', pd.DataFrame({})), 'gyro')
    
    # Check for critical data availability after filtering
    if raw_df.empty or status_df.empty or imu_accel.empty:
        missing_parts = []
        if raw_df.empty:
            missing_parts.append("GNSS Raw")
        if status_df.empty:
            missing_parts.append("GNSS Status")
        if imu_accel.empty:
            missing_parts.append("IMU Accel")

        # --- MODIFIED LOGGING HERE ---
        print(f"Skipping: Critical data is missing after filtering/standardization. Missing components: {', '.join(missing_parts)}.")
        return pd.DataFrame({})

    # --- 3. IMU FEATURE EXTRACTION & SYNCHRONIZATION ---
    
    # Extract IMU features before synchronization
    imu_features_unaligned = extract_imu_features(imu_accel, imu_gyro)
    
    # Align IMU time to GNSS absolute time
    imu_features_aligned = synchronize_imu_time(imu_features_unaligned, raw_df)
    
    if imu_features_aligned.empty:
        print("Skipping: IMU synchronization failed.")
        return pd.DataFrame({})

    # Aggregate IMU features to the common GNSS time step (10ms steps) for smoother merging
    imu_agg_features = imu_features_aligned.groupby('millisSinceGpsEpoch').agg({
        'accel_mag': ['mean', 'std'],
        'gyro_mag': ['mean', 'std'],
        'accel_mag_roll_mean': 'mean',
        'gyro_mag_roll_mean': 'mean'
    }).reset_index()
    
    # Flatten multi-level columns
    # This handles the aggregation result (e.g., ('accel_mag', 'mean') -> 'accel_mag_mean')
    imu_agg_features.columns = ['_'.join(col).strip() if col[1] else col[0] for col in imu_agg_features.columns.values]
    imu_agg_features.rename(columns={'millisSinceGpsEpoch_': 'millisSinceGpsEpoch'}, inplace=True)
    
    # --- 4. GNSS FEATURE EXTRACTION ---
    # GNSS features are extracted using the data that already has the 'millisSinceGpsEpoch' column
    raw_features = extract_raw_features(raw_df)
    status_features = extract_status_features(status_df)
    
    # --- 5. MERGING ALL FEATURES ---
    # Merge GNSS features first
    gnss_features = pd.merge(raw_features, status_features, on='millisSinceGpsEpoch', how='outer')
    
    # Then merge with the IMU aggregated features
    final_merged_df = pd.merge(gnss_features, imu_agg_features, on='millisSinceGpsEpoch', how='outer')

    # --- 6. FINAL CLEANUP ---
    final_merged_df = add_aggregate_features(final_merged_df)

    # Add identifying columns back
    final_merged_df['drive_id'] = job['drive_id']
    final_merged_df['phone_id'] = job['phone_id']
    
    print(f"✅ Success: Extracted {len(final_merged_df)} feature rows.")
    return final_merged_df.sort_values('millisSinceGpsEpoch').reset_index(drop=True)
