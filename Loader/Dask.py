import pandas as pd
import numpy as np
import os
import glob
from dask import dataframe as dd
from dask import delayed
from dask.distributed import Client, LocalCluster

# --- MOCK / Placeholder Logic (Assume these come from data_io/loader.py and data_io/schema.py) ---
# NOTE: In the actual project, you will import your finalized functions and schema maps here.

# Simplified version of the schema map for testing the pipeline structure
GNSS_SCHEMA_MAP = {
    'ElapsedRealtimeMillis': ('TIME_MILLIS', np.int64),
    'Svid': ('SV_ID', np.int16),
    'Cn0DbHz': ('CNO_DBHZ', np.float32),
    # Including columns for the final Dask output
    'TimeNanos': ('TIME_RX_NANOS', np.float64),
    'PseudorangeRateMetersPerSecond': ('DOPPLER_M_S', np.float32),
    'ConstellationType': ('CONSTELLATION_TYPE', np.int8),
}

def enforce_schema(df: pd.DataFrame, schema_map: dict) -> pd.DataFrame:
    """Applies schema (rename and dtype cast). Actual logic from data_io/schema.py."""
    rename_dict = {orig: new for orig, (new, _) in schema_map.items()}
    
    # 1. Rename columns
    df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})
    
    # 2. Cast types (Optimized for Dask)
    dtype_dict = {new: dtype for _, (new, dtype) in schema_map.items() if new in df.columns}
    df = df.astype(dtype_dict)
    
    # Select only the columns defined in the schema to keep the final output clean
    final_cols = list(dtype_dict.keys())
    return df[final_cols]


def load_gnss_log_single(filepath: str) -> pd.DataFrame:
    """
    Simulates the complex Pandas function that reads the single raw GNSS log file.
    (This is where the logic from data_io/loader.py's load_gnss_log goes)
    """
    if not os.path.exists(filepath):
        # Create a mock DataFrame for testing the Dask pipeline structure
        mock_data = {
            'ElapsedRealtimeMillis': [1000, 1050],
            'Svid': [1, 5],
            'Cn0DbHz': [30.5, 45.1],
            'TimeNanos': [1.23e18, 1.23e18],
            'PseudorangeRateMetersPerSecond': [100.1, 50.2],
            'ConstellationType': [1, 1],
        }
        df_raw = pd.DataFrame(mock_data)
        # Adding a file ID for tracking
        df_raw['FILE_PATH'] = filepath
        print(f"Mocking data for {filepath}. Rows: {len(df_raw)}")
    else:
        # In a real run, this would call your full loading logic:
        # df_raw = load_gnss_log(filepath) 
        pass # placeholder for actual file read

    return enforce_schema(df_raw, GNSS_SCHEMA_MAP)


# --- Core Dask Pipeline Implementation ---

def run_distributed_pipeline(gnss_data_path: str, output_path: str, client: Client):
    """
    Main function to run the Dask distributed data loading and standardization.
    
    Args:
        gnss_data_path: A glob pattern or directory path containing raw GNSS log files.
        output_path: The location to save the standardized Parquet files.
        client: The Dask distributed client object.
    """
    print(f"🚀 Dask Client Dashboard: {client.dashboard_link}")
    print(f"Searching for files in: {gnss_data_path}")
    
    # --- 1. Identify all raw log files (Person 1's output) ---
    # We use a glob pattern to find all log files within the directory structure
    if os.path.isdir(gnss_data_path):
        all_files = glob.glob(os.path.join(gnss_data_path, '**', '*gnss_log.txt'), recursive=True)
    else:
        # Use mock files if the directory doesn't exist for testing the structure
        all_files = [f"MOCK_DRIVE_A/phone{i}/gnss_log.txt" for i in range(5)]
    
    if not all_files:
        print("❌ No GNSS log files found! Check path or file pattern.")
        return

    print(f"✅ Found {len(all_files)} files to process.")

    # --- 2. Create Delayed Tasks for Custom Loading ---
    # We delay the execution of the complex, custom-parsing Pandas function for each file
    delayed_results = []
    for fpath in all_files:
        # Dask uses delayed() to wrap the function, allowing it to execute in parallel later
        delayed_df = delayed(load_gnss_log_single)(fpath)
        delayed_results.append(delayed_df)

    # --- 3. Construct the Dask DataFrame ---
    # We combine all the delayed Pandas DataFrames into a single Dask DataFrame
    # Note: We must specify the metadata (schema) so Dask knows what to expect before computing.
    # We use a temporary empty DataFrame to define the structure based on the schema map.
    temp_metadata = enforce_schema(pd.DataFrame(), GNSS_SCHEMA_MAP)

    ddf = dd.from_delayed(delayed_results, meta=temp_metadata)
    
    print(f"Dask DataFrame created with {ddf.npartitions} partitions.")
    print(f"Metadata (Schema Check):\n{ddf.dtypes}")

    # --- 4. Persist (Save) the Standardized Data ---
    # Dask only performs the computation (loads the data, applies schema) when you call compute() or persist/to_parquet.
    print(f"Writing standardized Parquet data to {output_path}...")
    
    # Saving to Parquet is distributed and highly efficient for large datasets
    ddf.to_parquet(
        output_path,
        engine='pyarrow', # Fast Parquet engine
        write_metadata_file=True, # Recommended for Dask to find the dataset easily
        overwrite=True
    )
    
    print("✅ Standardization complete. Parquet files saved.")


if __name__ == '__main__':
    # --- LOCAL DASK CLUSTER SETUP ---
    # We set up a LocalCluster which uses all available cores on your machine for parallel processing.
    
    # NOTE: You can adjust n_workers (number of parallel processes) and threads_per_worker
    print("✨ Starting Local Dask Cluster...")
    cluster = LocalCluster(n_workers=os.cpu_count(), threads_per_worker=1) 
    client = Client(cluster) 

    # --- RUN THE PIPELINE ---
    MOCK_INPUT_PATH = "MOCK_RAW_LOGS_DIR" # This should be the root folder where Person 1 stored the logs
    MOCK_OUTPUT_PATH = "STANDARDIZED_DATA/gnss_raw"
    
    run_distributed_pipeline(MOCK_INPUT_PATH, MOCK_OUTPUT_PATH, client)
    
    print("\n--- Dask Pipeline Execution Finished ---")
    
    # Close the cluster when done
    client.close()
    cluster.close()
    print("Dask cluster shut down.")
