import os
import pandas as pd
import numpy as np
from datetime import datetime
# --- DASK IMPORTS ---
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
# --------------------

# NOTE: This requires fuser.py (and its dependencies: data_uitiles.py, feature.py)
# to be in the same directory.
from fuser import process_full_log_to_features
from ekf_filter_improved import apply_ekf_to_features

def find_and_pair_files(root_dir: str) -> list[dict]:
    """
    Scans a directory structure (DRIVE_ID/PHONE_FOLDER/...) to find and pair
    GNSS (device_gnss.csv) and IMU (device_imu.csv) log files.
    
    Args:
        root_dir: The starting directory to scan (e.g., './train').
        
    Returns:
        A list of job dictionaries, each containing file paths and IDs.
    """
    paired_jobs = []
    
    # os.walk traverses the directory tree
    for root, _, files in os.walk(root_dir):
        
        gnss_file = None
        imu_file = None
        
        # Check for the specific required files in the current folder (PHONE_FOLDER level)
        if 'device_gnss.csv' in files:
            gnss_file = os.path.join(root, 'device_gnss.csv')
        
        if 'device_imu.csv' in files:
            imu_file = os.path.join(root, 'device_imu.csv')
        
        # If both files are found, package the job
        if gnss_file and imu_file:
            
            # Extract IDs from the path structure
            phone_id = os.path.basename(root)
            # The drive ID is the directory above the phone ID folder
            drive_id = os.path.basename(os.path.dirname(root)) 
            
            job = {
                'drive_id': drive_id,
                'phone_id': phone_id,
                'gnss_file': gnss_file,
                'imu_file': imu_file
            }
            paired_jobs.append(job)
            
    return paired_jobs

def process_log_safe(job: dict) -> pd.DataFrame:
    """
    Wrapper to safely call the main processing function and handle exceptions.
    Ensures that a log file failure doesn't crash the entire pipeline.
    """
    log_id = f"{job['drive_id']}/{job['phone_id']}"
    print(f"Processing log: {log_id}")
    try:
        # Call the actual feature extraction function from fuser.py
        return process_full_log_to_features(job) 
    except Exception as e:
        print(f"[ERROR] Error processing {log_id}: {e}")
        return pd.DataFrame({})

def run_dask_pipeline(all_jobs: list[dict]) -> pd.DataFrame:
    """
    Initializes a Dask cluster, processes log files in parallel, and returns a merged DataFrame.
    Includes a sequential fallback if Dask initialization fails.
    """
    if not all_jobs:
        print("No paired logs found to process.")
        return pd.DataFrame({})

    # Attempt to initialize Dask
    try:
        # 1. Initialize Dask cluster
        # Using 5 workers for parallel processing, suitable for I/O and CPU tasks
        cluster = LocalCluster(n_workers=5, threads_per_worker=1, processes=True, dashboard_address=':8787', silence_logs=False)
        client = Client(cluster)
        print(f"Dask client initialized with {len(client.scheduler_info()['workers'])} workers.")

        # 2. Build the Dask Graph
        # We use the safe wrapper function (process_log_safe) and delay its execution
        delayed_results = [delayed(process_log_safe)(job) for job in all_jobs]
            
        # 3. Execute the Graph in parallel
        print("Executing tasks in parallel...")
        all_features_raw = compute(*delayed_results)
        
        # 4. Filter, Concatenate, and Cleanup
        all_features = [df for df in all_features_raw if not df.empty]

        client.close()
        cluster.close()
        
        if all_features:
            final_df = pd.concat(all_features, ignore_index=True)
            print("\n--- Dask Parallel Processing Complete ---")
            print(f"Total features extracted: {len(final_df)} rows.")
            return final_df
        else:
            return pd.DataFrame({})
            
    except Exception as e:
        print(f"[WARNING] Dask cluster initialization failed ({e}). Falling back to sequential processing.")
        
        # Safe Fallback (Sequential)
        all_features = []
        for job in all_jobs:
            # Use the safe wrapper directly
            features = process_log_safe(job) 
            if not features.empty:
                all_features.append(features)
        
        if all_features:
            final_df = pd.concat(all_features, ignore_index=True)
            print("\n--- Sequential Processing Complete ---")
            print(f"Total features extracted: {len(final_df)} rows.")
            return final_df
        else:
            return pd.DataFrame({})
        

# --- Main Execution ---

if __name__ == "__main__":
    # Define the root directory where your 'train' data is located
    ROOT_DIR = r"C:\Users\avnee\Downloads\smartphone-decimeter-2022\test"

    # 1. Scan for jobs
    all_jobs = find_and_pair_files(ROOT_DIR)

    print(f"Found {len(all_jobs)} total paired phone logs.")
    print(f"Processing ALL folders (no filtering applied).")

    # Optional: Filter to specific folders (commented out)
    # TARGET_FOLDERS = ['2020-12-10-US-SJC-1', '2020-12-10-US-SJC-2', '2021-01-04-US-SFO-1']
    # filtered_jobs = [job for job in all_jobs if job['drive_id'] in TARGET_FOLDERS]
    # print(f"Filtering to {len(TARGET_FOLDERS)} target folders: {TARGET_FOLDERS}")
    # print(f"Processing {len(filtered_jobs)} phone logs from target folders.")

    # 2. Run the pipeline
    final_feature_set = run_dask_pipeline(all_jobs)

    if not final_feature_set.empty:
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save the raw features (before EKF)
        raw_output_file = f"training_set_all_folders_raw.csv"
        final_feature_set.to_csv(raw_output_file, index=False)
        print(f"\n[SUCCESS] Raw features saved to {raw_output_file}")
        print(f"   Total features: {len(final_feature_set)} rows")
        print(f"   Total columns: {len(final_feature_set.columns)}")

        # Apply Extended Kalman Filter
        print("\n" + "="*60)
        print("Applying Extended Kalman Filter to features...")
        print("="*60)
        ekf_feature_set = apply_ekf_to_features(final_feature_set)

        # Save the EKF-filtered features
        ekf_output_file = f"training_set_all_folders_ekf.csv"
        ekf_feature_set.to_csv(ekf_output_file, index=False)
        print(f"\n[SUCCESS] EKF-filtered features saved to {ekf_output_file}")
        print(f"   Total features: {len(ekf_feature_set)} rows")
        print(f"   Total columns: {len(ekf_feature_set.columns)}")
        print(f"   New EKF columns added: {len(ekf_feature_set.columns) - len(final_feature_set.columns)}")

        # ========================================
        # COMPREHENSIVE STATISTICS REPORT
        # ========================================
        print("\n" + "="*80)
        print("FINAL TRAINING SET STATISTICS")
        print("="*80)

        # Basic info
        print(f"\nFinal training set written: {len(ekf_feature_set):,} rows, {len(ekf_feature_set.columns)} columns")

        # NaN ratio analysis
        print("\n" + "-"*80)
        print("NaN RATIOS FOR KEY FIELDS")
        print("-"*80)

        key_field_groups = {
            'EKF': ['ekf_latitude', 'ekf_longitude', 'ekf_pos_std'],
            'GT': ['mean_latitude', 'mean_longitude', 'mean_height'],
            'IMU': ['accel_x_mean', 'accel_y_mean', 'accel_z_mean'],
            'GNSS': ['num_sats', 'mean_cn0']
        }

        nan_ratios = {}
        for group_name, cols in key_field_groups.items():
            available_cols = [c for c in cols if c in ekf_feature_set.columns]
            if available_cols:
                nan_count = ekf_feature_set[available_cols].isna().any(axis=1).sum()
                nan_ratio = nan_count / len(ekf_feature_set)
                nan_ratios[group_name] = nan_ratio
                print(f"  {group_name:6s}: {nan_ratio:.3f} ({nan_count:,}/{len(ekf_feature_set):,} rows have NaNs)")
            else:
                nan_ratios[group_name] = 1.0
                print(f"  {group_name:6s}: N/A (columns not present)")

        all_nan_ratio = sum(nan_ratios.values()) / len(nan_ratios)
        if all_nan_ratio < 0.1:
            print(f"\n  -> Alignment is SOLID (avg NaN ratio: {all_nan_ratio:.3f})")
        else:
            print(f"\n  -> Alignment needs improvement (avg NaN ratio: {all_nan_ratio:.3f})")

        # Time cadence analysis
        print("\n" + "-"*80)
        print("TIME CADENCE ANALYSIS")
        print("-"*80)

        if 'millisSinceGpsEpoch' in ekf_feature_set.columns:
            ekf_sorted = ekf_feature_set.sort_values('millisSinceGpsEpoch')
            time_diffs = ekf_sorted['millisSinceGpsEpoch'].diff().dropna() / 1000.0  # Convert to seconds

            # Check for duplicates
            duplicate_count = ekf_feature_set.duplicated(subset=['millisSinceGpsEpoch']).sum()

            print(f"  Time step statistics (seconds):")
            print(f"    Median: {time_diffs.median():.3f} s")
            print(f"    Mean:   {time_diffs.mean():.3f} s")
            print(f"    P95:    {time_diffs.quantile(0.95):.3f} s")
            print(f"    Min:    {time_diffs.min():.3f} s")
            print(f"    Max:    {time_diffs.max():.3f} s")
            print(f"\n  Duplicate timestamps: {duplicate_count:,} rows")

            if duplicate_count > 0:
                print(f"    -> You have duplicate timestamps (multiple rows at same time)")
                print(f"    -> Optional: resample to strict 10 Hz for uniform cadence")
            else:
                print(f"    -> No duplicate timestamps")

        # EKF Error Analysis (if ground truth available)
        print("\n" + "-"*80)
        print("EKF ERROR ANALYSIS")
        print("-"*80)

        has_gt = all(c in ekf_feature_set.columns for c in ['mean_latitude', 'mean_longitude', 'mean_height'])
        has_ekf = all(c in ekf_feature_set.columns for c in ['ekf_latitude', 'ekf_longitude', 'ekf_height'])

        if has_gt and has_ekf:
            # Calculate errors in local ENU frame
            valid_mask = (
                ekf_feature_set['mean_latitude'].notna() &
                ekf_feature_set['ekf_latitude'].notna()
            )
            valid_data = ekf_feature_set[valid_mask].copy()

            if len(valid_data) > 0:
                # Convert lat/lon differences to meters (approximate)
                lat_diff = valid_data['ekf_latitude'] - valid_data['mean_latitude']
                lon_diff = valid_data['ekf_longitude'] - valid_data['mean_longitude']
                height_diff = valid_data['ekf_height'] - valid_data['mean_height']

                # Convert to meters
                err_n = lat_diff * 110540  # North error (latitude)
                err_e = lon_diff * 111320 * np.cos(np.radians(valid_data['mean_latitude']))  # East error
                err_u = height_diff  # Up error

                print(f"  Computed on {len(valid_data):,} valid samples:")
                print(f"\n  Mean error (m):")
                print(f"    East:  {err_e.mean():7.1f} m")
                print(f"    North: {err_n.mean():7.1f} m")
                print(f"    Up:    {err_u.mean():7.1f} m")

                print(f"\n  Std deviation (m):")
                print(f"    East:  {err_e.std():7.1f} m")
                print(f"    North: {err_n.std():7.1f} m")
                print(f"    Up:    {err_u.std():7.1f} m")

                print(f"\n  Median error (m):")
                print(f"    East:  {err_e.median():7.1f} m")
                print(f"    North: {err_n.median():7.1f} m")
                print(f"    Up:    {err_u.median():7.1f} m")

                print(f"\n  Min/Max error (m):")
                print(f"    East:  [{err_e.min():7.1f}, {err_e.max():7.1f}]")
                print(f"    North: [{err_n.min():7.1f}, {err_n.max():7.1f}]")
                print(f"    Up:    [{err_u.min():7.1f}, {err_u.max():7.1f}]")

                # Identify outliers
                threshold = 100  # meters
                outliers = (np.abs(err_e) > threshold) | (np.abs(err_n) > threshold) | (np.abs(err_u) > threshold)
                outlier_count = outliers.sum()

                print(f"\n  Outliers (>{threshold}m error): {outlier_count:,} samples ({outlier_count/len(valid_data)*100:.1f}%)")

                if outlier_count > 0:
                    print(f"    -> Tip: Clip outliers or exclude low-quality epochs to stabilize tails")

                # Overall assessment
                medians_near_zero = (abs(err_e.median()) < 5) and (abs(err_n.median()) < 5) and (abs(err_u.median()) < 10)
                if medians_near_zero:
                    print(f"\n  -> Medians close to 0 indicate LOW BIAS in all axes")
                else:
                    print(f"\n  -> Medians show some bias - check calibration")

            else:
                print("  No valid samples with both GT and EKF data")
        else:
            missing = []
            if not has_gt:
                missing.append("Ground Truth (mean_latitude/longitude/height)")
            if not has_ekf:
                missing.append("EKF estimates")
            print(f"  Cannot compute errors - missing: {', '.join(missing)}")

        # Dataset composition
        print("\n" + "-"*80)
        print("DATASET COMPOSITION")
        print("-"*80)

        if 'drive_id' in ekf_feature_set.columns and 'phone_id' in ekf_feature_set.columns:
            composition = ekf_feature_set.groupby(['drive_id', 'phone_id']).size().reset_index(name='count')
            print(f"\n  Datasets processed: {len(composition)}")
            for _, row in composition.iterrows():
                print(f"    {row['drive_id']}/{row['phone_id']}: {row['count']:,} rows")

        print("\n" + "="*80)
        print("STATISTICS REPORT COMPLETE")
        print("="*80 + "\n")

    else:
        print("\n[WARNING] Pipeline finished but no features were extracted.")
