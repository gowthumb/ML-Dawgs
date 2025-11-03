import pandas as pd
import numpy as np
import os
# Dependencies for reading, cleaning, and time preparation
from data_uitiles import read_and_separate_log, apply_sanity_filters, add_time_columns, read_pos_file
# Dependencies for feature creation and final aggregation
from feature import extract_raw_features, extract_status_features, extract_imu_features, extract_pos_features, add_aggregate_features

# ==========================================
# ----------- CRITICAL FIXES APPLIED ------
# ==========================================
# This file has been updated to work with the smartphone-decimeter-2022 dataset:
#
# 1. STATUS DATA HANDLING: The original code required Status data to be present,
#    but the smartphone-decimeter-2022 dataset contains NO Status messages.
#    The code now gracefully handles empty Status data.
#
# 2. CRITICAL DATA CHECK: Removed the requirement for Status data in the critical
#    data availability check. Only GNSS Raw and IMU Accel are now required.
#
# 3. FEATURE MERGING: Updated the GNSS feature merging logic to handle cases
#    where Status data is empty, using only Raw features when necessary.
#
# 4. SCHEMA COMPATIBILITY: The code now works with the updated schemas in loader.py
#    that include both original and actual CSV column names.

def find_pos_file(drive_id: str, phone_id: str, ppk_output_dir: str = "/Users/yash.rayapaty/Downloads/ppk_output") -> str:
    """
    Find the .pos file path for a given drive_id and phone_id.
    
    The .pos files are stored as: ppk_output/{DATE}-{LOCATION}-{PHONE}/gnss_rinex.pos
    The drive_id format is: {DATE}-{LOCATION}
    The phone_id format is: {PHONE}
    
    Args:
        drive_id: Drive ID (e.g., "2020-06-04-US-MTV-1")
        phone_id: Phone ID (e.g., "GooglePixel4XL")
        ppk_output_dir: Root directory for ppk_output files
        
    Returns:
        Path to the .pos file, or None if not found
    """
    # Construct folder name: {DATE}-{LOCATION}-{PHONE}
    folder_name = f"{drive_id}-{phone_id}"
    pos_file_path = os.path.join(ppk_output_dir, folder_name, "gnss_rinex.pos")
    
    if os.path.exists(pos_file_path):
        return pos_file_path
    else:
        return None


def synchronize_imu_time(imu_df: pd.DataFrame, gnss_raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts IMU timestamps to millisSinceGpsEpoch.
    
    FIXED: Both GNSS and IMU CSV files use 'utcTimeMillis' which maps to:
    - GNSS: 'ElapsedRealtimeMillis' -> converted to 'millisSinceGpsEpoch' 
    - IMU: 'millisSinceBoot' -> needs same conversion
    
    The IMU's 'millisSinceBoot' is actually utcTimeMillis (Unix time in milliseconds),
    so we convert it directly to GPS epoch time using GPS_EPOCH_OFFSET,
    just like we do for GNSS's ElapsedRealtimeMillis.
    
    Args:
        imu_df: IMU features dataframe (must contain 'millisSinceBoot').
        gnss_raw_df: Cleaned GNSS Raw dataframe (used for verification only).
                     
    Returns:
        The IMU dataframe with a new 'millisSinceGpsEpoch' column, or an empty
        DataFrame if conversion fails.
    """
    if imu_df.empty:
        print("Synchronization failed: Empty IMU data.")
        return pd.DataFrame({})
    
    if 'millisSinceBoot' not in imu_df.columns:
        print("Synchronization failed: IMU dataframe missing 'millisSinceBoot' column.")
        return pd.DataFrame({})
        
    # FIXED: IMU's millisSinceBoot is actually utcTimeMillis (Unix time)
    # Convert directly to GPS epoch time using the same offset as GNSS
    # GPS Epoch is LATER than Unix Epoch, so we SUBTRACT the offset
    # GPS Epoch offset: 315964800000ms (1980-01-06T00:00:00Z is 315964800 seconds after Unix epoch)
    GPS_EPOCH_OFFSET_MILLIS = 315964800000
    
    # Convert millisSinceBoot (which is utcTimeMillis in Unix time) to GPS epoch time
    # GPS epoch is LATER, so subtract offset to get GPS epoch time
    imu_df['millisSinceGpsEpoch'] = imu_df['millisSinceBoot'].astype(np.int64) - GPS_EPOCH_OFFSET_MILLIS
    
    # Round to nearest 10ms for consistency with GNSS time resolution
    imu_df['millisSinceGpsEpoch'] = imu_df['millisSinceGpsEpoch'].round(-1).astype(np.int64)
    
    # Drop any rows with invalid timestamps
    imu_df = imu_df.dropna(subset=['millisSinceGpsEpoch'])
    
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
    # CRITICAL FIX: Removed Status data requirement - it doesn't exist in the dataset
    # Only GNSS Raw and IMU Accel are required for successful processing
    if raw_df.empty or imu_accel.empty:
        missing_parts = []
        if raw_df.empty:
            missing_parts.append("GNSS Raw")
        if imu_accel.empty:
            missing_parts.append("IMU Accel")

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
    
    # Handle Status features (may be empty)
    # CRITICAL FIX: Status data doesn't exist in smartphone-decimeter-2022 dataset
    # The code now gracefully handles empty Status data by using only Raw features
    if not status_df.empty:
        status_features = extract_status_features(status_df)
        # Merge GNSS features
        gnss_features = pd.merge(raw_features, status_features, on='millisSinceGpsEpoch', how='outer')
    else:
        # No Status data available, use only Raw features
        # This is the expected case for the smartphone-decimeter-2022 dataset
        gnss_features = raw_features
    
    # Then merge with the IMU aggregated features
    # FIX: Use merge_asof with tolerance to align nearby timestamps instead of exact match
    # This allows IMU data (10ms resolution) to merge with GNSS data (sparse timestamps)
    
    # Start with the data source that has more frequent timestamps (IMU)
    if not imu_agg_features.empty:
        # Sort by timestamp (required for merge_asof)
        final_merged_df = imu_agg_features.sort_values('millisSinceGpsEpoch').copy()
        
        # Merge GNSS features onto IMU timeline if available
        if not gnss_features.empty:
            gnss_sorted = gnss_features.sort_values('millisSinceGpsEpoch')
            
            # Use merge_asof to align GNSS data to IMU timestamps (within 50ms)
            final_merged_df = pd.merge_asof(
                final_merged_df,
                gnss_sorted,
                on='millisSinceGpsEpoch',
                direction='nearest',
                tolerance=50,  # 50ms tolerance for timestamp matching
                suffixes=('', '_gnss')
            )
            
            # Handle duplicate column names from merge_asof
            # If merge_asof creates duplicate columns, prefer the IMU version (first one)
            duplicate_cols = [col for col in final_merged_df.columns if col.endswith('_gnss')]
            for col in duplicate_cols:
                original_col = col.replace('_gnss', '')
                if original_col in final_merged_df.columns:
                    # Use GNSS value where available, else keep IMU value
                    final_merged_df[original_col] = final_merged_df[original_col].fillna(final_merged_df[col])
                    final_merged_df = final_merged_df.drop(columns=[col])
            
            # Also do outer merge to preserve GNSS timestamps that don't have nearby IMU data
            gnss_only_timestamps = set(gnss_features['millisSinceGpsEpoch']) - set(final_merged_df['millisSinceGpsEpoch'])
            if gnss_only_timestamps:
                gnss_only = gnss_features[gnss_features['millisSinceGpsEpoch'].isin(gnss_only_timestamps)]
                final_merged_df = pd.concat([final_merged_df, gnss_only], ignore_index=True)
    elif not gnss_features.empty:
        # Start with GNSS if no IMU data
        final_merged_df = gnss_features.sort_values('millisSinceGpsEpoch').copy()
    else:
        final_merged_df = pd.DataFrame({'millisSinceGpsEpoch': []})

    # --- 5. PPK .POS FILE FEATURE EXTRACTION (if available) ---
    # Try to find and read the corresponding .pos file
    pos_file_path = find_pos_file(job['drive_id'], job['phone_id'])
    
    if pos_file_path:
        print(f"✓ Found .pos file: {pos_file_path}")
        pos_df = read_pos_file(pos_file_path)
        
        if not pos_df.empty:
            # Extract features from .pos data
            pos_features = extract_pos_features(pos_df)
            
            if not pos_features.empty:
                # Merge .pos features with existing features using merge_asof for better alignment
                if not final_merged_df.empty:
                    final_sorted = final_merged_df.sort_values('millisSinceGpsEpoch')
                    pos_sorted = pos_features.sort_values('millisSinceGpsEpoch')
                    
                    # Merge POS onto existing timeline with 500ms tolerance (POS is 1Hz, ~1000ms intervals)
                    # Larger tolerance needed for POS since it's lower frequency
                    final_merged_df = pd.merge_asof(
                        final_sorted,
                        pos_sorted,
                        on='millisSinceGpsEpoch',
                        direction='nearest',
                        tolerance=500,  # 500ms tolerance for POS data (1Hz = 1000ms intervals)
                        suffixes=('', '_pos')
                    )
                    
                    # Handle duplicate column names
                    duplicate_cols = [col for col in final_merged_df.columns if col.endswith('_pos')]
                    for col in duplicate_cols:
                        original_col = col.replace('_pos', '')
                        if original_col in final_merged_df.columns:
                            # Use POS value where available, else keep existing value
                            final_merged_df[original_col] = final_merged_df[original_col].fillna(final_merged_df[col])
                            final_merged_df = final_merged_df.drop(columns=[col])
                    
                    # Also preserve POS timestamps that don't have nearby existing data
                    pos_only_timestamps = set(pos_features['millisSinceGpsEpoch']) - set(final_merged_df['millisSinceGpsEpoch'])
                    if pos_only_timestamps:
                        pos_only = pos_features[pos_features['millisSinceGpsEpoch'].isin(pos_only_timestamps)]
                        final_merged_df = pd.concat([final_merged_df, pos_only], ignore_index=True)
                    
                print(f"✓ Merged .pos features: {len(pos_features)} time epochs")
        else:
            print("⚠️ .pos file found but contained no valid data")
    else:
        print(f"⚠️ No .pos file found for {job['drive_id']}/{job['phone_id']}")

    # --- 6. FINAL CLEANUP ---
    # Prepare a real UTC timestamp index for correct time-based rolling in feature aggregation
    if not final_merged_df.empty and 'millisSinceGpsEpoch' in final_merged_df.columns:
        GPS_EPOCH_START = pd.Timestamp('1980-01-06 00:00:00', tz='UTC')
        final_merged_df['timestamp_utc'] = GPS_EPOCH_START + pd.to_timedelta(
            final_merged_df['millisSinceGpsEpoch'].astype('int64'), unit='ms'
        )
        final_merged_df = final_merged_df.set_index('timestamp_utc').sort_index()

    final_merged_df = add_aggregate_features(final_merged_df)

    # Add identifying columns back
    final_merged_df['drive_id'] = job['drive_id']
    final_merged_df['phone_id'] = job['phone_id']
    
    # --- 6.5 ENSURE CONSISTENT COLUMN ORDER ---
    # Define standard column order for consistent output across all datasets
    # This ensures that columns are in the same order regardless of merge sequence
    STANDARD_COLUMN_ORDER = [
        # Time column first (required)
        'millisSinceGpsEpoch',
        
        # IMU features (base features)
        'accel_mag_mean', 'accel_mag_std', 
        'gyro_mag_mean', 'gyro_mag_std',
        'accel_mag_roll_mean', 'gyro_mag_roll_mean',
        
        # GNSS features (if available)
        'num_sats', 'mean_cn0', 'std_cn0', 'max_cn0', 'min_cn0', 
        'mean_cn0_norm', 'max_sv_time_diff', 'mean_cn0_smooth',
        'std_cn0_rate', 'mean_pseudorange', 'std_pseudorange',
        'mean_doppler', 'num_cycle_slips', 'cn0_trend', 'num_sats_std_5s',
        
        # POS features (if available)
        'mean_latitude', 'mean_longitude', 'mean_height',
        'mean_quality', 'max_quality', 'mean_num_satellites', 
        'max_num_satellites', 'mean_solution_quality_score',
        'mean_horizontal_uncertainty', 'max_horizontal_uncertainty',
        'mean_position_uncertainty_3d', 'mean_sdn', 'mean_sde', 'mean_sdu',
        'mean_age', 'mean_ratio', 'mean_position_velocity',
        'std_position_velocity', 'mean_position_stability',
        
        # Aggregate features (from add_aggregate_features)
        'accel_mag_std_1s_roll', 'hae_std_roll', 'is_stationary', 'high_qual_sat_ratio',
        
        # Identifiers (always last)
        'drive_id', 'phone_id',
    ]
    
    # Reorder columns: existing columns in standard order, then any extras
    existing_cols = [col for col in STANDARD_COLUMN_ORDER if col in final_merged_df.columns]
    extra_cols = [col for col in final_merged_df.columns if col not in STANDARD_COLUMN_ORDER]
    final_merged_df = final_merged_df[existing_cols + extra_cols]
    
    # --- 7. DIAGNOSTICS: Merge coverage and null percentages ---
    try:
        total_rows = len(final_merged_df)
        total_cells = total_rows * len(final_merged_df.columns) if total_rows > 0 else 0

        # Presence heuristics per source
        imu_keys = [c for c in ['accel_mag_mean', 'gyro_mag_mean'] if c in final_merged_df.columns]
        gnss_keys = [c for c in ['num_sats', 'mean_cn0', 'mean_doppler'] if c in final_merged_df.columns]
        pos_keys = [c for c in ['mean_latitude', 'mean_quality', 'mean_position_uncertainty_3d'] if c in final_merged_df.columns]

        imu_rows = final_merged_df[imu_keys].notna().any(axis=1).sum() if imu_keys else 0
        gnss_rows = final_merged_df[gnss_keys].notna().any(axis=1).sum() if gnss_keys else 0
        pos_rows = final_merged_df[pos_keys].notna().any(axis=1).sum() if pos_keys else 0

        imu_pct = (imu_rows / total_rows * 100) if total_rows else 0.0
        gnss_merge_pct = (gnss_rows / total_rows * 100) if total_rows else 0.0
        pos_merge_pct = (pos_rows / total_rows * 100) if total_rows else 0.0

        overall_null_pct = 0.0
        if total_cells:
            overall_null_pct = float(final_merged_df.isna().sum().sum()) / float(total_cells) * 100.0

        print(
            f"Merge coverage → IMU: {imu_pct:.1f}% | GNSS-aligned: {gnss_merge_pct:.1f}% | POS-aligned: {pos_merge_pct:.1f}%"
        )
        print(
            f"Nulls → overall cell null percentage: {overall_null_pct:.1f}% over {total_rows} rows and {len(final_merged_df.columns)} cols"
        )
    except Exception as diag_e:
        print(f"⚠️ Diagnostics failed: {diag_e}")

    print(f"✅ Success: Extracted {len(final_merged_df)} feature rows.")
    return final_merged_df.sort_values('millisSinceGpsEpoch').reset_index(drop=True)
