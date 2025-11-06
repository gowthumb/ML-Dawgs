import pandas as pd
import numpy as np
import os
import tempfile

# Dependencies from your project (must be in the same directory or environment)
try:
    from data_uitiles import read_and_separate_log, apply_sanity_filters, add_time_columns, read_pos_file
    from feature import extract_raw_features, extract_status_features, extract_imu_features, extract_pos_features, add_aggregate_features
except ImportError:
    print("CRITICAL ERROR: Could not import 'data_uitiles.py' or 'feature.py'. Make sure they are in the same directory.")
    # Define dummy functions to avoid crashing the whole script
    def read_and_separate_log(path): return None
    def apply_sanity_filters(df, name): return None
    def add_time_columns(df): return None
    def read_pos_file(path): return None
    def extract_imu_features(accel, gyro): return None
    def extract_raw_features(df): return None
    def extract_pos_features(df): return None
    def add_aggregate_features(df): return None


# ==========================================
# ----------- ALL HELPER FUNCTIONS -----------
# ==========================================

def find_pos_file(drive_id: str, phone_id: str, ppk_output_dir: str = "D:/NTU/Y3S1/SC4000/ML-Dawgs/ppk_output/ppk_output") -> str | None:
    """
    Find the .pos file path for a given drive_id and phone_id.
    """
    folder_name = f"{drive_id}-{phone_id}"
    pos_file_path = os.path.join(ppk_output_dir, folder_name, "gnss_rinex.pos")
    
    if os.path.exists(pos_file_path):
        return pos_file_path
    else:
        return None

def synchronize_imu_time(imu_df: pd.DataFrame, gnss_raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts IMU timestamps to millisSinceGpsEpoch.
    """
    if imu_df.empty:
        return pd.DataFrame({})
    
    if 'millisSinceBoot' not in imu_df.columns:
        return pd.DataFrame({})
        
    GPS_EPOCH_OFFSET_MILLIS = 315964800000
    imu_df['millisSinceGpsEpoch'] = imu_df['millisSinceBoot'].astype(np.int64) - GPS_EPOCH_OFFSET_MILLIS
    # Round to 10ms and then coarsen to 100ms bins for 10 Hz representation
    imu_df['millisSinceGpsEpoch'] = imu_df['millisSinceGpsEpoch'].round(-1).astype(np.int64)
    imu_df['millisSinceGpsEpoch'] = (imu_df['millisSinceGpsEpoch'] // 100) * 100
    imu_df = imu_df.dropna(subset=['millisSinceGpsEpoch'])
    
    return imu_df.reset_index(drop=True)

# ==========================================
# --- MAIN PROCESSING FUNCTION (THE WORKING VERSION) ---
# ==========================================

def process_logs_to_features(job: dict) -> str | None:
    """
    Reads, preprocesses, synchronizes, extracts features, and merges 
    GNSS and IMU data for a single drive/phone log pair.
    
    On success, saves features to a Parquet/CSV file and returns the file path.
    On failure, returns None.
    """
    gnss_log_path = job['gnss_file']
    imu_log_path = job['imu_file']
    log_id = f"{job['drive_id']}/{job['phone_id']}"
    
    print(f"\n--- [Features] Processing Log Pair: {log_id} ---")
    
    # --- 1. READ AND SEPARATE LOGS ---
    gnss_data = read_and_separate_log(gnss_log_path)
    imu_data = read_and_separate_log(imu_log_path)
    
    if gnss_data is None or imu_data is None:
        print(f"Skipping {log_id}: One or both log files could not be read (returned None).")
        return None

    # --- 2. SANITY FILTERS & TIME CALCULATION ---
    raw_df = apply_sanity_filters(gnss_data.get('raw', pd.DataFrame({})), 'raw')
    raw_df = pd.DataFrame({}) if raw_df is None else raw_df
    raw_df = add_time_columns(raw_df) 
    raw_df = pd.DataFrame({}) if raw_df is None else raw_df
    
    status_df = apply_sanity_filters(gnss_data.get('status', pd.DataFrame({})), 'status')
    status_df = pd.DataFrame({}) if status_df is None else status_df
    
    imu_accel = apply_sanity_filters(imu_data.get('accel', pd.DataFrame({})), 'accel')
    imu_accel = pd.DataFrame({}) if imu_accel is None else imu_accel
    
    imu_gyro = apply_sanity_filters(imu_data.get('gyro', pd.DataFrame({})), 'gyro')
    imu_gyro = pd.DataFrame({}) if imu_gyro is None else imu_gyro
    
    if raw_df.empty or imu_accel.empty:
        missing_parts = []
        if raw_df.empty: missing_parts.append("GNSS Raw")
        if imu_accel.empty: missing_parts.append("IMU Accel")
        print(f"Skipping {log_id}: Critical data is missing. Missing: {', '.join(missing_parts)}.")
        return None

    # --- 3. IMU FEATURE EXTRACTION & SYNCHRONIZATION ---
    imu_features_unaligned = extract_imu_features(imu_accel, imu_gyro)
    imu_features_unaligned = pd.DataFrame({}) if imu_features_unaligned is None else imu_features_unaligned
    
    imu_features_aligned = synchronize_imu_time(imu_features_unaligned, raw_df)
    imu_features_aligned = pd.DataFrame({}) if imu_features_aligned is None else imu_features_aligned
    
    if imu_features_aligned.empty:
        print(f"Skipping {log_id}: IMU synchronization failed.")
        return None

    imu_agg_features = imu_features_aligned.groupby('millisSinceGpsEpoch').agg({
        'accel_mag': ['mean', 'std'],
        'gyro_mag': ['mean', 'std'],
        'accel_mag_roll_mean': 'mean',
        'gyro_mag_roll_mean': 'mean'
    }).reset_index()
    
    imu_agg_features.columns = ['_'.join(col).strip() if col[1] else col[0] for col in imu_agg_features.columns.values]
    imu_agg_features.rename(columns={'millisSinceGpsEpoch_': 'millisSinceGpsEpoch'}, inplace=True)
    
    IMU_FEATURE_COLS = [col for col in imu_agg_features.columns if col != 'millisSinceGpsEpoch']
    imu_agg_features[IMU_FEATURE_COLS] = imu_agg_features[IMU_FEATURE_COLS].ffill()
    
    # --- 4. GNSS FEATURE EXTRACTION & MERGING ---
    raw_features = extract_raw_features(raw_df)
    raw_features = pd.DataFrame({}) if raw_features is None else raw_features
    
    status_features = pd.DataFrame({})
    if not status_df.empty:
        status_features = extract_status_features(status_df)
        status_features = pd.DataFrame({}) if status_features is None else status_features

    if not status_features.empty:
        gnss_features = pd.merge(raw_features, status_features, on='millisSinceGpsEpoch', how='outer')
    else:
        gnss_features = raw_features
    
    final_merged_df = imu_agg_features.sort_values('millisSinceGpsEpoch').copy()
    GNSS_FEATURE_COLS = [] # Initialize
    
    if not gnss_features.empty:
        gnss_sorted = gnss_features.sort_values('millisSinceGpsEpoch')
        
        final_merged_df = pd.merge_asof(
            final_merged_df,
            gnss_sorted,
            on='millisSinceGpsEpoch',
            direction='nearest',
            tolerance=100,
            suffixes=('', '_gnss')
        )
        
        duplicate_cols = [col for col in final_merged_df.columns if col.endswith('_gnss')]
        for col in duplicate_cols:
            original_col = col.replace('_gnss', '')
            if original_col in final_merged_df.columns:
                final_merged_df[original_col] = final_merged_df[original_col].fillna(final_merged_df[col])
                final_merged_df = final_merged_df.drop(columns=[col])

        GNSS_FEATURE_COLS = [col for col in gnss_features.columns if col not in IMU_FEATURE_COLS + ['millisSinceGpsEpoch']]
        cols_to_interpolate = [col for col in GNSS_FEATURE_COLS if col in final_merged_df.columns]
        
        if cols_to_interpolate:
            final_merged_df = final_merged_df.set_index('millisSinceGpsEpoch')
            final_merged_df[cols_to_interpolate] = final_merged_df[cols_to_interpolate].interpolate(
                method='linear', limit_direction='both', limit=5
            )
            final_merged_df = final_merged_df.reset_index()
        
        gnss_only_timestamps = set(gnss_features['millisSinceGpsEpoch']) - set(final_merged_df['millisSinceGpsEpoch'])
        if gnss_only_timestamps:
            gnss_only = gnss_features[gnss_features['millisSinceGpsEpoch'].isin(gnss_only_timestamps)]
            final_merged_df = pd.concat([final_merged_df, gnss_only], ignore_index=True)
    
    elif not gnss_features.empty:
        final_merged_df = gnss_features.sort_values('millisSinceGpsEpoch').copy()
    
    # --- 5. PPK .POS FILE FEATURE EXTRACTION (Original Logic) ---
    pos_features = pd.DataFrame({}) # Initialize
    pos_file_path = find_pos_file(job['drive_id'], job['phone_id'])
    
    if pos_file_path and not final_merged_df.empty:
        print(f"✓ [Features] Found .pos file: {pos_file_path}")
        pos_df = read_pos_file(pos_file_path)
        pos_df = pd.DataFrame({}) if pos_df is None else pos_df
        
        if not pos_df.empty:
            pos_features = extract_pos_features(pos_df)
            pos_features = pd.DataFrame({}) if pos_features is None else pos_features
            
            if not pos_features.empty:
                final_sorted = final_merged_df.sort_values('millisSinceGpsEpoch')
                pos_sorted = pos_features.sort_values('millisSinceGpsEpoch')
                
                final_merged_df = pd.merge_asof(
                    final_sorted,
                    pos_sorted,
                    on='millisSinceGpsEpoch',
                    direction='nearest',
                    tolerance=500,
                    suffixes=('', '_pos')
                )
                
                duplicate_cols = [col for col in final_merged_df.columns if col.endswith('_pos')]
                for col in duplicate_cols:
                    original_col = col.replace('_pos', '')
                    if original_col in final_merged_df.columns:
                        final_merged_df[original_col] = final_merged_df[original_col].fillna(final_merged_df[col])
                        final_merged_df = final_merged_df.drop(columns=[col])

                POS_FEATURE_COLS = [col for col in pos_features.columns if col != 'millisSinceGpsEpoch']
                cols_to_interpolate = [col for col in POS_FEATURE_COLS if col in final_merged_df.columns]
                
                if cols_to_interpolate:
                    final_merged_df = final_merged_df.set_index('millisSinceGpsEpoch')
                    final_merged_df[cols_to_interpolate] = final_merged_df[cols_to_interpolate].interpolate(
                        method='linear', limit_direction='both', limit=10
                    )
                    final_merged_df = final_merged_df.reset_index()
                
                pos_only_timestamps = set(pos_features['millisSinceGpsEpoch']) - set(final_merged_df['millisSinceGpsEpoch'])
                if pos_only_timestamps:
                    pos_only = pos_features[pos_features['millisSinceGpsEpoch'].isin(pos_only_timestamps)]
                    final_merged_df = pd.concat([final_merged_df, pos_only], ignore_index=True)
                    
                print(f"✓ [Features] Merged .pos features: {len(pos_features)} time epochs")
        else:
            print("⚠️ [Features] .pos file found but contained no valid data")
    else:
        if not pos_file_path:
            print(f"⚠️ [Features] No .pos file found for {log_id}")
        
    # --- 6. FINAL CLEANUP ---
    if final_merged_df.empty:
        print(f"Skipping {log_id}: No features were merged.")
        return None

    final_merged_df.ffill(inplace=True)
    final_merged_df.bfill(inplace=True)

    # Ensure clean 10 Hz cadence: drop duplicate timestamps after merges
    if 'millisSinceGpsEpoch' in final_merged_df.columns:
        final_merged_df = final_merged_df.sort_values('millisSinceGpsEpoch')
        final_merged_df = final_merged_df.drop_duplicates(subset=['millisSinceGpsEpoch'], keep='first')
        final_merged_df = final_merged_df.reset_index(drop=True)

    final_merged_df = add_aggregate_features(final_merged_df)
    final_merged_df = pd.DataFrame({}) if final_merged_df is None else final_merged_df

    if final_merged_df.empty:
        print(f"Skipping {log_id}: No features after final aggregation.")
        return None

    final_merged_df['drive_id'] = job['drive_id']
    final_merged_df['phone_id'] = job['phone_id']
    
    STANDARD_COLUMN_ORDER = [
        'millisSinceGpsEpoch', 'accel_mag_mean', 'accel_mag_std', 'gyro_mag_mean', 'gyro_mag_std',
        'accel_mag_roll_mean', 'gyro_mag_roll_mean', 'num_sats', 'mean_cn0', 'std_cn0', 'max_cn0', 'min_cn0', 
        'mean_cn0_norm', 'max_sv_time_diff', 'mean_cn0_smooth', 'std_cn0_rate', 'mean_pseudorange', 'std_pseudorange',
        'mean_doppler', 'num_cycle_slips', 'cn0_trend', 'num_sats_std_5s', 'mean_latitude', 'mean_longitude', 'mean_height',
        'mean_quality', 'max_quality', 'mean_num_satellites', 'max_num_satellites', 'mean_solution_quality_score',
        'mean_horizontal_uncertainty', 'max_horizontal_uncertainty', 'mean_position_uncertainty_3d', 'mean_sdn', 'mean_sde', 'mean_sdu',
        'mean_age', 'mean_ratio', 'mean_position_velocity', 'std_position_velocity', 'mean_position_stability',
        'accel_mag_std_1s_roll', 'hae_std_roll', 'is_stationary', 'high_qual_sat_ratio', 'drive_id', 'phone_id',
    ]
    
    existing_cols = [col for col in STANDARD_COLUMN_ORDER if col in final_merged_df.columns]
    extra_cols = [col for col in final_merged_df.columns if col not in STANDARD_COLUMN_ORDER]
    final_merged_df = final_merged_df[existing_cols + extra_cols]
    
    final_merged_df = final_merged_df.sort_values('millisSinceGpsEpoch').reset_index(drop=True)

    # --- 9. SAVE TO PARQUET (PREFERRED) OR CSV (FALLBACK) ---
    # Cross-platform temp directory for cached feature tables
    temp_output_dir = os.path.join(tempfile.gettempdir(), "dask_feature_cache_FEATURES")
    os.makedirs(temp_output_dir, exist_ok=True)
    job_id_str = f"{job['drive_id'].replace('/', '_')}_{job['phone_id']}"
        
    output_path = os.path.join(temp_output_dir, f"{job_id_str}.parquet")

    try:
        final_merged_df.to_parquet(output_path, index=False, engine='auto')
        print(f"✅ [Features] Success: Extracted {len(final_merged_df)} rows. Saved to {output_path}")
        return output_path

    except Exception as e:
        print(f"⚠️ [Features] WARNING: Parquet save failed ({e}). Falling back to CSV.")
        try:
            output_path_csv = os.path.join(temp_output_dir, f"{job_id_str}.csv")
            final_merged_df.to_csv(output_path_csv, index=False)
            print(f"✅ [Features] Success: Extracted {len(final_merged_df)} rows. Saved to {output_path_csv}")
            return output_path_csv
        except Exception as e_csv:
            print(f"❌ [Features] Error during file save (CSV fallback) for {job_id_str}: {e_csv}")
            return None
