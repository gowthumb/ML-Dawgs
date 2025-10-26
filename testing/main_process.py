import os
import pandas as pd
# --- DASK IMPORTS ---
from dask import delayed, compute 
from dask.distributed import Client, LocalCluster 
# --------------------

# NOTE: This requires fuser.py (and its dependencies: data_uitiles.py, feature.py) 
# to be in the same directory.
from fuser import process_full_log_to_features

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
        print(f"❌ Error processing {log_id}: {e}")
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
        print(f"⚠️ Dask cluster initialization failed ({e}). Falling back to sequential processing.")
        
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
    ROOT_DIR = "./train" 
    
    # 1. Scan for jobs
    all_jobs = find_and_pair_files(ROOT_DIR)
    print(f"Found {len(all_jobs)} paired phone logs across all drives to process.")
    
    # 2. Run the pipeline
    final_feature_set = run_dask_pipeline(all_jobs)
    
    if not final_feature_set.empty:
        # You can save the final merged features here
        output_file = "all_extracted_features.csv"
        final_feature_set.to_csv(output_file, index=False)
        print(f"\n✅ All features saved to {output_file}")
    else:
        print("\n⚠️ Pipeline finished but no features were extracted.")
