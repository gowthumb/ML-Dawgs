import os
import pandas as pd
import numpy as np

# --- Dependencies (make sure these are in the same folder) ---
from data_uitiles import read_and_separate_log, apply_sanity_filters, add_time_columns, read_pos_file
from feature import extract_raw_features, extract_status_features, extract_imu_features, extract_pos_features, add_aggregate_features

# ----------------- FUSER FUNCTIONS -----------------

def find_pos_file(drive_id: str, phone_id: str, ppk_output_dir: str = "/Users/yash.rayapaty/Downloads/ppk_output") -> str:
    """
    Find the .pos file path for a given drive_id and phone_id.
    """
    folder_name = f"{drive_id}-{phone_id}"
    pos_file_path = os.path.join(ppk_output_dir, folder_name, "gnss_rinex.pos")
    return pos_file_path if os.path.exists(pos_file_path) else None

def synchronize_imu_time(imu_df: pd.DataFrame, gnss_raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts IMU timestamps to millisSinceGpsEpoch.
    """
    if imu_df.empty or 'millisSinceBoot' not in imu_df.columns:
        return pd.DataFrame({})

    GPS_EPOCH_OFFSET_MILLIS = 315964800000
    imu_df['millisSinceGpsEpoch'] = imu_df['millisSinceBoot'].astype(np.int64) - GPS_EPOCH_OFFSET_MILLIS
    imu_df['millisSinceGpsEpoch'] = imu_df['millisSinceGpsEpoch'].round(-1).astype(np.int64)
    imu_df = imu_df.dropna(subset=['millisSinceGpsEpoch'])
    return imu_df.reset_index(drop=True)

def process_full_log_to_features(job: dict) -> pd.DataFrame:
    """
    Processes a single drive/phone log using the paths from the job dictionary.
    """
    gnss_log_path = job['gnss_file']
    imu_log_path = job['imu_file']
    
    print(f"\n--- Processing Log Pair: {job['drive_id']}/{job['phone_id']} ---")
    
    # --- 1. Read logs ---
    gnss_data = read_and_separate_log(gnss_log_path)
    imu_data = read_and_separate_log(imu_log_path)
    
    if not gnss_data or not imu_data:
        print("Skipping: One or both log files could not be read/separated.")
        return pd.DataFrame({})

    # --- 2. Sanity Filters & Time ---
    raw_df = apply_sanity_filters(gnss_data.get('raw', pd.DataFrame({})), 'raw')
    raw_df = add_time_columns(raw_df)
    status_df = apply_sanity_filters(gnss_data.get('status', pd.DataFrame({})), 'status')
    imu_accel = apply_sanity_filters(imu_data.get('accel', pd.DataFrame({})), 'accel')
    imu_gyro = apply_sanity_filters(imu_data.get('gyro', pd.DataFrame({})), 'gyro')
    
    if raw_df.empty or imu_accel.empty:
        print(f"Skipping: Missing critical data (GNSS Raw or IMU Accel).")
        return pd.DataFrame({})

    # --- 3. IMU Features ---
    imu_features_unaligned = extract_imu_features(imu_accel, imu_gyro)
    imu_features_aligned = synchronize_imu_time(imu_features_unaligned, raw_df)
    if imu_features_aligned.empty:
        print("Skipping: IMU synchronization failed.")
        return pd.DataFrame({})

    imu_agg_features = imu_features_aligned.groupby('millisSinceGpsEpoch').agg({
        'accel_mag': ['mean', 'std'],
        'gyro_mag': ['mean', 'std'],
        'accel_mag_roll_mean': 'mean',
        'gyro_mag_roll_mean': 'mean'
    }).reset_index()
    imu_agg_features.columns = ['_'.join(col).strip() if col[1] else col[0] for col in imu_agg_features.columns.values]
    imu_agg_features.rename(columns={'millisSinceGpsEpoch_': 'millisSinceGpsEpoch'}, inplace=True)

    # --- 4. GNSS Features ---
    raw_features = extract_raw_features(raw_df)
    gnss_features = raw_features
    if not status_df.empty:
        status_features = extract_status_features(status_df)
        gnss_features = pd.merge(raw_features, status_features, on='millisSinceGpsEpoch', how='outer')

    # Merge IMU and GNSS features
    if not imu_agg_features.empty:
        final_merged_df = imu_agg_features.sort_values('millisSinceGpsEpoch').copy()
        if not gnss_features.empty:
            gnss_sorted = gnss_features.sort_values('millisSinceGpsEpoch')
            final_merged_df = pd.merge_asof(
                final_merged_df, gnss_sorted,
                on='millisSinceGpsEpoch',
                direction='nearest',
                tolerance=50,
                suffixes=('', '_gnss')
            )
            # Resolve duplicates
            for col in [c for c in final_merged_df.columns if c.endswith('_gnss')]:
                original = col.replace('_gnss', '')
                final_merged_df[original] = final_merged_df[original].fillna(final_merged_df[col])
                final_merged_df.drop(columns=[col], inplace=True)
    elif not gnss_features.empty:
        final_merged_df = gnss_features.sort_values('millisSinceGpsEpoch').copy()
    else:
        final_merged_df = pd.DataFrame({'millisSinceGpsEpoch': []})

    # --- 5. Optional PPK POS features ---
    pos_file_path = find_pos_file(job['drive_id'], job['phone_id'])
    if pos_file_path:
        pos_df = read_pos_file(pos_file_path)
        if not pos_df.empty:
            pos_features = extract_pos_features(pos_df)
            if not final_merged_df.empty:
                final_sorted = final_merged_df.sort_values('millisSinceGpsEpoch')
                pos_sorted = pos_features.sort_values('millisSinceGpsEpoch')
                final_merged_df = pd.merge_asof(
                    final_sorted, pos_sorted,
                    on='millisSinceGpsEpoch',
                    direction='nearest',
                    tolerance=500,
                    suffixes=('', '_pos')
                )
                for col in [c for c in final_merged_df.columns if c.endswith('_pos')]:
                    original = col.replace('_pos', '')
                    final_merged_df[original] = final_merged_df[original].fillna(final_merged_df[col])
                    final_merged_df.drop(columns=[col], inplace=True)

    # --- 6. Final Aggregation ---
    if not final_merged_df.empty and 'millisSinceGpsEpoch' in final_merged_df.columns:
        GPS_EPOCH_START = pd.Timestamp('1980-01-06 00:00:00', tz='UTC')
        final_merged_df['timestamp_utc'] = GPS_EPOCH_START + pd.to_timedelta(final_merged_df['millisSinceGpsEpoch'].astype('int64'), unit='ms')
        final_merged_df = final_merged_df.set_index('timestamp_utc').sort_index()

    final_merged_df = add_aggregate_features(final_merged_df)
    final_merged_df['drive_id'] = job['drive_id']
    final_merged_df['phone_id'] = job['phone_id']

    # --- 7. Diagnostics ---
    total_rows = len(final_merged_df)
    if total_rows > 0:
        total_cells = total_rows * len(final_merged_df.columns)
        imu_keys = [c for c in ['accel_mag_mean', 'gyro_mag_mean'] if c in final_merged_df.columns]
        gnss_keys = [c for c in ['num_sats', 'mean_cn0', 'mean_doppler'] if c in final_merged_df.columns]
        pos_keys = [c for c in ['mean_latitude', 'mean_quality', 'mean_position_uncertainty_3d'] if c in final_merged_df.columns]
        imu_pct = (final_merged_df[imu_keys].notna().any(axis=1).sum()/total_rows*100) if imu_keys else 0
        gnss_pct = (final_merged_df[gnss_keys].notna().any(axis=1).sum()/total_rows*100) if gnss_keys else 0
        pos_pct = (final_merged_df[pos_keys].notna().any(axis=1).sum()/total_rows*100) if pos_keys else 0
        overall_null_pct = float(final_merged_df.isna().sum().sum()) / total_cells * 100 if total_cells else 0
        print(f"Merge coverage → IMU: {imu_pct:.1f}% | GNSS-aligned: {gnss_pct:.1f}% | POS-aligned: {pos_pct:.1f}%")
        print(f"Nulls → overall cell null percentage: {overall_null_pct:.1f}% over {total_rows} rows and {len(final_merged_df.columns)} cols")

    print(f"✅ Success: Extracted {len(final_merged_df)} feature rows.")
    return final_merged_df.sort_values('millisSinceGpsEpoch').reset_index(drop=True)