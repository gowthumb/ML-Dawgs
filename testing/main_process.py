import os
import pandas as pd
import numpy as np
# --- DASK IMPORTS ---
from dask import delayed, compute 
from dask.distributed import Client, LocalCluster 
# --------------------

# --- NEW DEPENDENCIES ---
# Make sure to run: pip install numpy pandas scipy pymap3d pyarrow dask distributed
# ------------------------

# 🟨🟨🟨 IMPORTING FROM BOTH FUSER FILES 🟨🟨🟨
try:
    from fuser_features import process_logs_to_features
    from fuser_ekf import process_logs_to_ekf_trajectory
except ImportError:
    print("CRITICAL ERROR: Could not import 'fuser_features.py' or 'fuser_ekf.py'.")
    print("Please make sure all three .py files are in the same directory.")
    exit()


def find_and_pair_files(root_dir: str) -> list[dict]:
    """
    Scans a directory structure (DRIVE_ID/PHONE_FOLDER/...) to find and pair
    GNSS (device_gnss.csv) and IMU (device_imu.csv) log files.
    Also finds ground_truth.csv in the same folders.
    """
    paired_jobs = []
    for root, _, files in os.walk(root_dir):
        gnss_file = None
        imu_file = None
        gt_file = None
        if 'device_gnss.csv' in files:
            gnss_file = os.path.join(root, 'device_gnss.csv')
        if 'device_imu.csv' in files:
            imu_file = os.path.join(root, 'device_imu.csv')
        if 'ground_truth.csv' in files:
            gt_file = os.path.join(root, 'ground_truth.csv')
        
        if gnss_file and imu_file:
            phone_id = os.path.basename(root)
            drive_id = os.path.basename(os.path.dirname(root)) 
            job = {'drive_id': drive_id, 'phone_id': phone_id, 'gnss_file': gnss_file, 'imu_file': imu_file}
            if gt_file:
                job['ground_truth_file'] = gt_file
            paired_jobs.append(job)
            
    return paired_jobs

def process_logs_safe(job: dict, processor_func) -> str | None:
    """
    Generic wrapper to safely call a processing function (either EKF or Features).
    Returns a filepath (str) on success or None on failure.
    """
    log_id = f"{job['drive_id']}/{job['phone_id']}"
    # The print statement is now inside the processor_func
    try:
        return processor_func(job) 
    except Exception as e:
        print(f"❌ CRITICAL DASK ERROR processing {log_id} with {processor_func.__name__}: {e}")
        return None

def read_cache_file(path: str) -> pd.DataFrame:
    """Reads a .parquet or .csv file from the cache."""
    try:
        if path.endswith('.parquet'):
            return pd.read_parquet(path)
        elif path.endswith('.csv'):
            return pd.read_csv(path)
        else:
            print(f"⚠️ Skipping unknown file type: {path}")
            return pd.DataFrame({})
    except Exception as e:
        print(f"⚠️ Error reading {path}: {e}")
        return pd.DataFrame({})


def read_ground_truth(gt_path: str, drive_id: str, phone_id: str) -> pd.DataFrame:
    """Reads and normalizes ground truth CSV from a single phone folder.
    Tries multiple time/position column variants.
    Adds drive_id and phone_id from parameters.
    """
    if not os.path.exists(gt_path):
        return pd.DataFrame({})
    try:
        gt = pd.read_csv(gt_path)
    except Exception as e:
        print(f"⚠️ Failed to read ground truth from {gt_path}: {e}")
        return pd.DataFrame({})

    # Normalize position column names
    rename_map = {}
    # Common alternatives
    if 'lat' in gt.columns and 'latitude' not in gt.columns:
        rename_map['lat'] = 'latitude'
    if 'LatitudeDegrees' in gt.columns:
        rename_map['LatitudeDegrees'] = 'latitude'
    if 'lon' in gt.columns and 'longitude' not in gt.columns:
        rename_map['lon'] = 'longitude'
    if 'LongitudeDegrees' in gt.columns:
        rename_map['LongitudeDegrees'] = 'longitude'
    if 'h' in gt.columns and 'height' not in gt.columns:
        rename_map['h'] = 'height'
    if 'HeightMeters' in gt.columns:
        rename_map['HeightMeters'] = 'height'
    if 'AltitudeMeters' in gt.columns:
        rename_map['AltitudeMeters'] = 'height'
    gt = gt.rename(columns=rename_map)

    # Time normalization: accept several possibilities
    gps_epoch_offset_ms = 315964800000
    if 'millisSinceGpsEpoch' in gt.columns:
        gt['millisSinceGpsEpoch'] = pd.to_numeric(gt['millisSinceGpsEpoch'], errors='coerce').round(-1).astype('Int64')
    elif 'gpst_sec' in gt.columns:
        gt['millisSinceGpsEpoch'] = (pd.to_numeric(gt['gpst_sec'], errors='coerce') * 1000.0).round(-1).astype('Int64')
    elif 'utcTimeMillis' in gt.columns:
        # Convert Unix epoch ms to GPS epoch ms
        gt['millisSinceGpsEpoch'] = (pd.to_numeric(gt['utcTimeMillis'], errors='coerce') - gps_epoch_offset_ms).round(-1).astype('Int64')
    elif 'UnixTimeMillis' in gt.columns:
        # Some GT files use UnixTimeMillis (Unix epoch ms)
        gt['millisSinceGpsEpoch'] = (pd.to_numeric(gt['UnixTimeMillis'], errors='coerce') - gps_epoch_offset_ms).round(-1).astype('Int64')
    elif 'TimeNanos' in gt.columns:
        # Convert ns device clock to ms GPS epoch if absolute; if relative, this may be unusable
        # Heuristic: if values are ~1e18, treat as Unix ns
        tn = pd.to_numeric(gt['TimeNanos'], errors='coerce')
        if tn.notna().any():
            # Assume Unix ns then convert to GPS ms
            ms_unix = (tn / 1_000_000.0)
            gt['millisSinceGpsEpoch'] = (ms_unix - gps_epoch_offset_ms).round(-1).astype('Int64')
    else:
        print(f"⚠️ Ground truth missing recognizable time column in {gt_path}")
        return pd.DataFrame({})

    # Add drive_id and phone_id if not present
    if 'drive_id' not in gt.columns:
        gt['drive_id'] = drive_id
    if 'phone_id' not in gt.columns:
        gt['phone_id'] = phone_id

    # Drop rows with missing essentials
    gt = gt.dropna(subset=['millisSinceGpsEpoch'])

    # Basic required columns check (positions)
    pos_required = ['latitude', 'longitude', 'height']
    if not all(c in gt.columns for c in pos_required):
        print(f"⚠️ Ground truth missing required position columns in {gt_path}. Found: {list(gt.columns)}")
        return pd.DataFrame({})

    # Ensure integer ms
    gt['millisSinceGpsEpoch'] = gt['millisSinceGpsEpoch'].astype(np.int64)

    return gt[['drive_id', 'phone_id', 'millisSinceGpsEpoch', 'latitude', 'longitude', 'height']].copy()

def run_dask_hybrid_pipeline(all_jobs: list[dict]) -> pd.DataFrame:
    """
    Runs the full hybrid EKF + ML pipeline.
    1. Runs EKF pipeline in parallel.
    2. Runs Feature pipeline in parallel.
    3. Loads all results.
    4. Merges EKF trajectories with features.
    5. Calculates EKF error (the ML target).
    6. Returns a final training DataFrame (X_features, y_target).
    """
    if not all_jobs:
        print("No paired logs found to process.")
        return pd.DataFrame({})
        
    final_training_df = pd.DataFrame({})
    
    try:
        cluster = LocalCluster(n_workers=5, threads_per_worker=1, processes=True, dashboard_address=':8787', silence_logs=False)
        client = Client(cluster)
        print(f"Dask client initialized with {len(client.scheduler_info()['workers'])} workers.")

        # --- 1. Build Dask graphs for BOTH pipelines ---
        delayed_ekf_paths = [delayed(process_logs_safe)(job, process_logs_to_ekf_trajectory) for job in all_jobs]
        delayed_feature_paths = [delayed(process_logs_safe)(job, process_logs_to_features) for job in all_jobs]
            
        print("Executing EKF and Feature pipelines in parallel...")
        # --- 2. Compute BOTH pipelines ---
        all_ekf_paths, all_feature_paths = compute(delayed_ekf_paths, delayed_feature_paths)
        
        client.close()
        cluster.close()

        # --- 3. Load and Merge Results ---
        print("\n--- Dask Parallel Processing Complete ---")
        
        success_ekf_paths = [path for path in all_ekf_paths if path is not None]
        success_feature_paths = [path for path in all_feature_paths if path is not None]
        
        print(f"Successfully processed {len(success_ekf_paths)} EKF trajectories.")
        print(f"Successfully processed {len(success_feature_paths)} Feature tables.")

        if not success_ekf_paths or not success_feature_paths:
            print("\n⚠️ Missing results from one or both pipelines. Cannot create training set.")
            return pd.DataFrame({})

        print("Reading and merging all cached files...")
        all_ekf_dfs = [read_cache_file(path) for path in success_ekf_paths]
        all_feature_dfs = [read_cache_file(path) for path in success_feature_paths]
        
        if not all_ekf_dfs or not all_feature_dfs:
            print("\n⚠️ Failed to read cached files. Cannot create training set.")
            return pd.DataFrame({})

        full_ekf_df = pd.concat(all_ekf_dfs, ignore_index=True)
        full_features_df = pd.concat(all_feature_dfs, ignore_index=True)
        
        print(f"Total EKF states: {len(full_ekf_df)}")
        print(f"Total Feature rows: {len(full_features_df)}")
        
        # --- 4. Align EKF, Features, and GT to create training set ---
        
        # Convert feature table time to seconds to match EKF table
        full_features_df['gpst_sec'] = full_features_df['millisSinceGpsEpoch'] / 1000.0
        
        # Merge the EKF results (which includes GT) onto the dense feature timeline
        final_training_df = pd.merge_asof(
            full_features_df.sort_values('gpst_sec'),
            full_ekf_df.sort_values('gpst_sec'),
            on='gpst_sec',
            direction='nearest',
            tolerance=0.05, # 50ms tolerance
            suffixes=('_feat', '_ekf') # Add suffixes to avoid column name collisions
        )
        
        # --- 5. Calculate EKF Error (The ML Target) ---
        # We need the Ground Truth (GT) position, which is now in the merged dataframe
        
        gt_cols = ['gt_e', 'gt_n', 'gt_u']
        ekf_cols = ['ekf_e', 'ekf_n', 'ekf_u']
        
        if not all(col in final_training_df.columns for col in gt_cols + ekf_cols):
            print("\n⚠️ Training set is missing GT or EKF columns. Cannot calculate error.")
            return final_training_df

        print("Calculating EKF error as ML target...")
        final_training_df['err_e'] = final_training_df['gt_e'] - final_training_df['ekf_e']
        final_training_df['err_n'] = final_training_df['gt_n'] - final_training_df['ekf_n']
        final_training_df['err_u'] = final_training_df['gt_u'] - final_training_df['ekf_u']
        
        # Drop rows where GT was not available
        final_training_df.dropna(subset=['err_e', 'err_n', 'err_u'], inplace=True)
        
        print(f"Final training set created with {len(final_training_df)} aligned rows.")
        # Additionally: create POS-vs-GT residual dataset for ML correction model
        try:
            # Collect all ground truth files from jobs
            all_gt_dfs = []
            for job in all_jobs:
                if 'ground_truth_file' in job:
                    gt_df = read_ground_truth(job['ground_truth_file'], job['drive_id'], job['phone_id'])
                    if not gt_df.empty:
                        all_gt_dfs.append(gt_df)
            
            if all_gt_dfs:
                gt_df_combined = pd.concat(all_gt_dfs, ignore_index=True)
                print(f"Loaded ground truth from {len(all_gt_dfs)} files ({len(gt_df_combined)} rows)")
                
                # Merge GT onto features timeline per drive/phone (nearest time, 500ms)
                features_t = full_features_df.copy()
                features_t['millisSinceGpsEpoch'] = features_t['millisSinceGpsEpoch'].round(-1).astype(np.int64)
                gt_df_combined['millisSinceGpsEpoch'] = gt_df_combined['millisSinceGpsEpoch'].round(-1).astype(np.int64)
                # Ensure consistent dtypes and strict sorting for merge_asof
                for df_ in (features_t, gt_df_combined):
                    df_['drive_id'] = df_['drive_id'].astype(str)
                    df_['phone_id'] = df_['phone_id'].astype(str)
                
                # Sort and reset index to ensure clean state for merge_asof
                left_sorted = features_t.sort_values(['drive_id', 'phone_id', 'millisSinceGpsEpoch'], kind='mergesort').reset_index(drop=True)
                right_sorted = gt_df_combined.sort_values(['drive_id', 'phone_id', 'millisSinceGpsEpoch'], kind='mergesort').reset_index(drop=True)
                
                # Verify sorting is correct (within each by-group, time must be monotonic)
                # If merge_asof still fails, we'll do per-group merges
                try:
                    gt_merged = pd.merge_asof(
                        left_sorted,
                        right_sorted,
                        on='millisSinceGpsEpoch',
                        by=['drive_id', 'phone_id'],
                        direction='nearest',
                        tolerance=500,
                        suffixes=('', '_gt')
                    )
                except ValueError as e:
                    # Fallback: merge per group manually
                    print(f"⚠️ merge_asof with 'by' failed, using per-group merge: {e}")
                    gt_merged_list = []
                    for (d_id, p_id), left_group in left_sorted.groupby(['drive_id', 'phone_id']):
                        right_group = right_sorted[(right_sorted['drive_id'] == d_id) & (right_sorted['phone_id'] == p_id)]
                        if not right_group.empty:
                            left_group_sorted = left_group.sort_values('millisSinceGpsEpoch').reset_index(drop=True)
                            right_group_sorted = right_group.sort_values('millisSinceGpsEpoch').reset_index(drop=True)
                            merged_group = pd.merge_asof(
                                left_group_sorted,
                                right_group_sorted,
                                on='millisSinceGpsEpoch',
                                direction='nearest',
                                tolerance=500,
                                suffixes=('', '_gt')
                            )
                            gt_merged_list.append(merged_group)
                    if gt_merged_list:
                        gt_merged = pd.concat(gt_merged_list, ignore_index=True)
                    else:
                        raise ValueError("No groups could be merged")

                # Compute POS -> GT residuals (GT - POS) in deg and meters
                if all(c in gt_merged.columns for c in ['mean_latitude', 'mean_longitude', 'mean_height', 'latitude', 'longitude', 'height']):
                    dlat = gt_merged['latitude'] - gt_merged['mean_latitude']
                    dlon = gt_merged['longitude'] - gt_merged['mean_longitude']
                    dhu = gt_merged['height'] - gt_merged['mean_height']
                    gt_merged['residual_lat_deg'] = dlat
                    gt_merged['residual_lon_deg'] = dlon
                    gt_merged['residual_h_m'] = dhu
                    # Approximate meters using local scale
                    lat_rad = np.radians(gt_merged['mean_latitude'].clip(-89.999, 89.999))
                    gt_merged['residual_n_m'] = dlat * 111000.0
                    gt_merged['residual_e_m'] = dlon * 111000.0 * np.cos(lat_rad)
                    gt_merged['residual_u_m'] = dhu

                    # Save POS residual dataset
                    pos_out = 'pos_residual_training_set.csv'
                    gt_merged.to_csv(pos_out, index=False)
                    print(f"\n✅ POS residual training set saved to {pos_out} ({len(gt_merged)} rows)")
                    print(f"   Residual stats (m): E={gt_merged['residual_e_m'].mean():.2f}±{gt_merged['residual_e_m'].std():.2f}, "
                          f"N={gt_merged['residual_n_m'].mean():.2f}±{gt_merged['residual_n_m'].std():.2f}, "
                          f"U={gt_merged['residual_u_m'].mean():.2f}±{gt_merged['residual_u_m'].std():.2f}")
                else:
                    print("⚠️ Skipping POS residual output: required POS/GT columns missing.")
            else:
                print("⚠️ No ground truth files found in any phone folders.")
        except Exception as e:
            print(f"⚠️ POS residual dataset creation failed: {e}")
            import traceback
            traceback.print_exc()

        return final_training_df
            
    except Exception as e:
        print(f"⚠️ Dask cluster initialization failed ({e}).")
        return pd.DataFrame({})
        

# --- Main Execution ---
if __name__ == "__main__":
    # Define the root directory where your 'train' data is located
    ROOT_DIR = "D:/NTU/Y3S1/SC4000/ML-Dawgs/train"
    
    # 1. Scan for jobs
    all_jobs = find_and_pair_files(ROOT_DIR)
    print(f"Found {len(all_jobs)} paired phone logs to process.")
    
    # 2. Run the new hybrid pipeline
    final_training_set = run_dask_hybrid_pipeline(all_jobs)
    
    if not final_training_set.empty:
        # Save the final training set
        output_file = "hybrid_ml_training_set.csv"
        final_training_set.to_csv(output_file, index=False)
        print(f"\n✅ Full ML training set saved to {output_file}")
        
        print(f"   Total features: {len(final_training_set)} rows")
        print(f"   Total columns: {len(final_training_set.columns)}")

        # --- Basic sanity checks on saved data ---
        try:
            # Time cadence check (seconds)
            dt = final_training_set['gpst_sec'].sort_values().diff().dropna()
            print(f"   gpst_sec dt median: {dt.median():.3f}s, p5: {dt.quantile(0.05):.3f}s, p95: {dt.quantile(0.95):.3f}s")
        except Exception:
            pass

        # NaN ratios for key columns
        key_cols = [c for c in ['ekf_e','ekf_n','ekf_u','gt_e','gt_n','gt_u','accel_mag_mean','gyro_mag_mean','mean_cn0'] if c in final_training_set.columns]
        if key_cols:
            na_ratios = final_training_set[key_cols].isna().mean().round(3)
            print("   NaN ratios:")
            for k, v in na_ratios.to_dict().items():
                print(f"     - {k}: {v}")
        
        # Error stats
        err_cols = [c for c in ['err_e','err_n','err_u'] if c in final_training_set.columns]
        if err_cols:
            desc = final_training_set[err_cols].describe(percentiles=[0.05,0.5,0.95]).round(3)
            print("\n   EKF error summary (m):")
            print(desc.to_string())
    else:
        print("\n⚠️ Pipeline finished but no training data was extracted.")

