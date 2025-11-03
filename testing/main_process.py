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
    """
    paired_jobs = []
    for root, _, files in os.walk(root_dir):
        gnss_file = None
        imu_file = None
        if 'device_gnss.csv' in files:
            gnss_file = os.path.join(root, 'device_gnss.csv')
        if 'device_imu.csv' in files:
            imu_file = os.path.join(root, 'device_imu.csv')
        
        if gnss_file and imu_file:
            phone_id = os.path.basename(root)
            drive_id = os.path.basename(os.path.dirname(root)) 
            job = {'drive_id': drive_id, 'phone_id': phone_id, 'gnss_file': gnss_file, 'imu_file': imu_file}
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
        return final_training_df
            
    except Exception as e:
        print(f"⚠️ Dask cluster initialization failed ({e}).")
        return pd.DataFrame({})
        

# --- Main Execution ---
if __name__ == "__main__":
    # Define the root directory where your 'train' data is located
    ROOT_DIR = "/Users/avantika/Documents/ML-Dawgs/train"
    
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
    else:
        print("\n⚠️ Pipeline finished but no training data was extracted.")

