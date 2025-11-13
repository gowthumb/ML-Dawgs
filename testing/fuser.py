import os
import pandas as pd
import numpy as np

# --- Dependencies (make sure these are in the same folder) ---
from data_uitiles import read_and_separate_log, apply_sanity_filters, add_time_columns, read_pos_file
from feature import extract_raw_features, extract_status_features, extract_imu_features, extract_pos_features, add_aggregate_features

# ----------------- FUSER FUNCTIONS -----------------

def find_pos_file(drive_id: str, phone_id: str, ppk_output_dir: str = r"C:\Users\avnee\Downloads\pos_output_test-20251113T151209Z-1-001\pos_output_test") -> str:
    """
    Find the .txt file path for a given drive_id and phone_id.
    New format: ppk_results2/{drive_id}-{phone_id}_pos.txt
    """
    file_name = f"{drive_id}-{phone_id}-pos.txt"
    pos_file_path = os.path.join(ppk_output_dir, file_name)
    return pos_file_path if os.path.exists(pos_file_path) else None

def downsample_imu(imu_df: pd.DataFrame, downsample_factor: int = 10) -> pd.DataFrame:
    """
    Downsamples IMU data by keeping every Nth sample.
    For example, downsample_factor=10 converts 100 Hz to 10 Hz.

    Args:
        imu_df: IMU dataframe with millisSinceBoot time column
        downsample_factor: Factor to downsample by (default 10: 100Hz -> 10Hz)

    Returns:
        Downsampled IMU dataframe
    """
    if imu_df.empty or 'millisSinceBoot' not in imu_df.columns:
        return imu_df

    # Sort by time to ensure proper ordering
    imu_df = imu_df.sort_values('millisSinceBoot').reset_index(drop=True)

    # Keep every Nth sample
    downsampled = imu_df.iloc[::downsample_factor].copy()

    print(f"  IMU downsampled: {len(imu_df)} -> {len(downsampled)} samples (factor: {downsample_factor})")
    return downsampled.reset_index(drop=True)

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

    # --- 2.5. Downsample IMU from 100 Hz to 10 Hz ---
    imu_accel = downsample_imu(imu_accel, downsample_factor=10)
    imu_gyro = downsample_imu(imu_gyro, downsample_factor=10)

    if raw_df.empty or imu_accel.empty:
        print(f"Skipping: Missing critical data (GNSS Raw or IMU Accel).")
        return pd.DataFrame({})

    # --- 3. IMU Features ---
    imu_features_unaligned = extract_imu_features(imu_accel, imu_gyro)
    imu_features_aligned = synchronize_imu_time(imu_features_unaligned, raw_df)
    if imu_features_aligned.empty:
        print("Skipping: IMU synchronization failed.")
        return pd.DataFrame({})

    # Aggregate IMU features including 3D components for EKF
    agg_dict = {
        'accel_mag': ['mean', 'std'],
        'gyro_mag': ['mean', 'std'],
        'accel_mag_roll_mean': 'mean',
        'gyro_mag_roll_mean': 'mean'
    }

    # Add 3D components if available
    if 'accel_x' in imu_features_aligned.columns:
        agg_dict['accel_x'] = 'mean'
        agg_dict['accel_y'] = 'mean'
        agg_dict['accel_z'] = 'mean'
    if 'gyro_x' in imu_features_aligned.columns:
        agg_dict['gyro_x'] = 'mean'
        agg_dict['gyro_y'] = 'mean'
        agg_dict['gyro_z'] = 'mean'

    imu_agg_features = imu_features_aligned.groupby('millisSinceGpsEpoch').agg(agg_dict).reset_index()
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

            # Identify GNSS columns before merge (exclude millisSinceGpsEpoch)
            gnss_cols = [col for col in gnss_sorted.columns if col != 'millisSinceGpsEpoch']

            # Add flag to GNSS data to mark actual measurements
            gnss_sorted['gnss_measurement'] = True

            # Perform merge_asof to align timestamps
            final_merged_df = pd.merge_asof(
                final_merged_df, gnss_sorted,
                on='millisSinceGpsEpoch',
                direction='nearest',
                tolerance=50,
                suffixes=('', '_gnss')
            )

            # Resolve duplicates from merge
            for col in [c for c in final_merged_df.columns if c.endswith('_gnss')]:
                original = col.replace('_gnss', '')
                final_merged_df[original] = final_merged_df[original].fillna(final_merged_df[col])
                final_merged_df.drop(columns=[col], inplace=True)

            # Mark rows without actual GNSS measurements (where gnss_measurement is NaN)
            final_merged_df['gnss_measurement'] = final_merged_df['gnss_measurement'].fillna(False)

            # --- NEW APPROACH: Interpolate SHORT gaps only, flag NaNs ---
            # Define max gap for interpolation: 500ms (GNSS typically updates at 1Hz)
            MAX_GNSS_GAP_MS = 500

            # Identify gap sizes between actual GNSS measurements
            gnss_timestamps = final_merged_df.loc[final_merged_df['gnss_measurement'], 'millisSinceGpsEpoch']

            for col in gnss_cols:
                if col in final_merged_df.columns:
                    # Create mask for which rows have actual data
                    has_data = final_merged_df['gnss_measurement']

                    # Interpolate using limit_area to only fill gaps, not edges
                    # This prevents backfill/forward-fill at the beginning/end
                    final_merged_df[col] = final_merged_df[col].interpolate(
                        method='linear',
                        limit=5,  # Max 5 consecutive NaN values (~500ms at 10Hz)
                        limit_area='inside'  # Only interpolate between data points, not at edges
                    )

            # Count interpolated vs actual vs missing
            actual_gnss = final_merged_df['gnss_measurement'].sum()
            total_rows = len(final_merged_df)
            nan_gnss = final_merged_df[gnss_cols].isna().any(axis=1).sum()
            interpolated_gnss = total_rows - actual_gnss - nan_gnss

            print(f"  GNSS data: {actual_gnss} actual measurements, {interpolated_gnss} interpolated, {nan_gnss} NaN (gap too long)")
    elif not gnss_features.empty:
        final_merged_df = gnss_features.sort_values('millisSinceGpsEpoch').copy()
        # All rows are actual GNSS measurements if we only have GNSS data
        final_merged_df['gnss_measurement'] = True
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

                # Identify POS columns before merge
                pos_cols = [col for col in pos_sorted.columns if col != 'millisSinceGpsEpoch']

                # Add flag to POS data to mark actual measurements
                pos_sorted['pos_measurement'] = True

                # Perform merge_asof to align timestamps
                final_merged_df = pd.merge_asof(
                    final_sorted, pos_sorted,
                    on='millisSinceGpsEpoch',
                    direction='nearest',
                    tolerance=500,
                    suffixes=('', '_pos')
                )

                # Resolve duplicates from merge
                for col in [c for c in final_merged_df.columns if c.endswith('_pos')]:
                    original = col.replace('_pos', '')
                    final_merged_df[original] = final_merged_df[original].fillna(final_merged_df[col])
                    final_merged_df.drop(columns=[col], inplace=True)

                # Mark rows without actual POS measurements
                final_merged_df['pos_measurement'] = final_merged_df['pos_measurement'].fillna(False)

                # --- NEW APPROACH: Interpolate SHORT gaps only, flag NaNs ---
                # Define max gap for interpolation: 2000ms (POS typically updates at 1Hz or slower)
                MAX_POS_GAP_MS = 2000

                # Identify gap sizes between actual POS measurements
                pos_timestamps = final_merged_df.loc[final_merged_df['pos_measurement'], 'millisSinceGpsEpoch']

                for col in pos_cols:
                    if col in final_merged_df.columns:
                        # Create mask for which rows have actual data
                        has_data = final_merged_df['pos_measurement']

                        # Interpolate using limit_area to only fill gaps, not edges
                        # This prevents backfill/forward-fill at the beginning/end
                        final_merged_df[col] = final_merged_df[col].interpolate(
                            method='linear',
                            limit=20,  # Max 20 consecutive NaN values (~2000ms at 10Hz)
                            limit_area='inside'  # Only interpolate between data points, not at edges
                        )

                # Count interpolated vs actual vs missing
                actual_pos = final_merged_df['pos_measurement'].sum()
                total_rows = len(final_merged_df)
                nan_pos = final_merged_df[pos_cols].isna().any(axis=1).sum()
                interpolated_pos = total_rows - actual_pos - nan_pos

                print(f"  POS data: {actual_pos} actual measurements, {interpolated_pos} interpolated, {nan_pos} NaN (gap too long)")

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
        print(f"Merge coverage - IMU: {imu_pct:.1f}% | GNSS-aligned: {gnss_pct:.1f}% | POS-aligned: {pos_pct:.1f}%")
        print(f"Nulls - overall cell null percentage: {overall_null_pct:.1f}% over {total_rows} rows and {len(final_merged_df.columns)} cols")

    print(f"[SUCCESS] Extracted {len(final_merged_df)} feature rows.")
    return final_merged_df.sort_values('millisSinceGpsEpoch').reset_index(drop=True)