import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# ==========================================
# ---------- UTILITY FUNCTIONS -------------
# ==========================================

def _safe_read_csv(filepath, **kwargs):
    """Safe CSV reader that prints helpful errors."""
    try:
        return pd.read_csv(filepath, **kwargs)
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return pd.DataFrame()


def _get_raw_column_names(filepath: str, tag: str = "# Raw") -> list:
    """
    Extracts column names for the 'Raw' measurements block in GNSS logs.
    """
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith(tag):
                columns = [c.strip() for c in line[len(tag)+1:].strip().split(',')]
                columns.insert(0, 'MessageType')
                print(f"✅ Found {len(columns)} columns in GNSS header.")
                return columns
    print(f"⚠️ No '{tag}' header found in {filepath}")
    return []

# ==========================================
# ----------- SCHEMA DEFINITIONS -----------
# ==========================================

GNSS_RAW_SCHEMA_MAP = {
    'ElapsedRealtimeMillis': ('TIME_MILLIS', np.int64),
    'TimeNanos': ('TIME_RX_NANOS', np.float64),
    'FullBiasNanos': ('BIAS_FULL_NANOS', np.float64),
    'BiasNanos': ('BIAS_NANO', np.float64),
    'ReceivedSvTimeNanos': ('TIME_SV_TX_NANOS', np.float64),
    'ReceivedSvTimeUncertaintyNanos': ('TIME_SV_UNC_NANOS', np.float32),
    'Svid': ('SV_ID', np.int16),
    'ConstellationType': ('CONSTELLATION_TYPE', np.int8),
    'CarrierFrequencyHz': ('FREQ_HZ', np.float32),
    'PseudorangeRateMetersPerSecond': ('DOPPLER_M_S', np.float32),
    'Cn0DbHz': ('CNO_DBHZ', np.float32),
    'PseudorangeRateUncertaintyMetersPerSecond': ('DOPPLER_UNC_M_S', np.float32),
    'AccumulatedDeltaRangeState': ('ADR_STATE', np.int32),
    'CodeType': ('CODE_TYPE', str),
    'MultipathIndicator': ('MULTIPATH_INDICATOR', np.int8),
}

IMU_SCHEMA_MAP = {
    'TimeSinceGpsEpoch': ('TIME_IMU_SECONDS', np.float64),
    'UncalAccelXMetersPerSecondSquared': ('ACCEL_X_UNCAL', np.float32),
    'UncalAccelYMetersPerSecondSquared': ('ACCEL_Y_UNCAL', np.float32),
    'UncalAccelZMetersPerSecondSquared': ('ACCEL_Z_UNCAL', np.float32),
    'UncalGyroXRadPerSec': ('GYRO_X_UNCAL', np.float32),
    'UncalGyroYRadPerSec': ('GYRO_Y_UNCAL', np.float32),
    'UncalGyroZRadPerSec': ('GYRO_Z_UNCAL', np.float32),
    'UncalMagXMicroT': ('MAG_X_UNCAL', np.float32),
    'UncalMagYMicroT': ('MAG_Y_UNCAL', np.float32),
    'UncalMagZMicroT': ('MAG_Z_UNCAL', np.float32),
}

# ==========================================
# ------- SCHEMA ENFORCEMENT FUNCTION ------
# ==========================================

def enforce_schema(df: pd.DataFrame, schema_map: dict, strict: bool = False) -> pd.DataFrame:
    """
    Renames and type-casts DataFrame columns according to the schema.
    If strict=True, drops columns not in schema.
    """
    rename_dict = {orig: new for orig, (new, _) in schema_map.items()}
    dtype_dict = {new: dtype for _, (new, dtype) in schema_map.items()}

    df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})

    if strict:
        df = df[[v for v in rename_dict.values() if v in df.columns]]

    for col, dtype in dtype_dict.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                print(f"⚠️ Failed to cast {col} to {dtype}: {e}")
    return df

# ==========================================
# ----------- GNSS LOADER ------------------
# ==========================================

def load_gnss(filepath: str) -> pd.DataFrame:
    """
    Loads and standardizes GNSS 'Raw' measurements from either .txt or .csv.
    """
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return pd.DataFrame()

    if filepath.endswith(".txt"):
        columns = _get_raw_column_names(filepath)
        if not columns:
            return pd.DataFrame()

        df_all = _safe_read_csv(filepath, comment="#", header=None)
        df_all.columns = columns
        df_raw = df_all[df_all["MessageType"] == "Raw"].copy()
        df_raw.drop(columns=["MessageType"], inplace=True, errors="ignore")
    else:
        df_raw = _safe_read_csv(filepath)

    if df_raw.empty:
        print("⚠️ No GNSS data found.")
        return pd.DataFrame()

    df_raw = enforce_schema(df_raw, GNSS_RAW_SCHEMA_MAP)
    print(f"✅ Loaded GNSS data with shape {df_raw.shape}")
    return df_raw

# ==========================================
# ----------- IMU LOADER -------------------
# ==========================================

def load_imu(filepath: str) -> pd.DataFrame:
    """
    Loads IMU data (Accel, Gyro, Mag) from CSV or log.
    """
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return pd.DataFrame()

    df_imu = _safe_read_csv(filepath)
    if df_imu.empty:
        print("⚠️ Empty IMU file.")
        return df_imu

    df_imu = enforce_schema(df_imu, IMU_SCHEMA_MAP)
    print(f"✅ Loaded IMU data with shape {df_imu.shape}")
    return df_imu

# ==========================================
# ----------- AUTO-DETECT LOADER -----------
# ==========================================

def load_sensor_data(filepath: str) -> pd.DataFrame:
    """
    Automatically detects and loads GNSS or IMU data.
    """
    filename = os.path.basename(filepath).lower()
    if "gnss" in filename:
        print(f"📡 Detected GNSS file: {filename}")
        return load_gnss(filepath)
    elif "imu" in filename:
        print(f"🧭 Detected IMU file: {filename}")
        return load_imu(filepath)
    else:
        print(f"⚠️ Could not auto-detect file type: {filename}")
        return pd.DataFrame()

# ==========================================
# ----------- DATA MERGING -----------------
# ==========================================

def merge_gnss_imu(gnss_df, imu_df, time_col_gnss='utcTimeMillis', time_col_imu='utcTimeMillis'):
    """
    Merges GNSS and IMU dataframes based on time columns using nearest neighbor matching.
    
    Args:
        gnss_df: GNSS dataframe
        imu_df: IMU dataframe  
        time_col_gnss: Time column name in GNSS dataframe
        time_col_imu: Time column name in IMU dataframe
        
    Returns:
        Merged dataframe with nearest time matches
    """
    merged = pd.merge_asof(
        imu_df.sort_values(time_col_imu),
        gnss_df.sort_values(time_col_gnss),
        left_on=time_col_imu,
        right_on=time_col_gnss,
        direction='nearest',
        tolerance=50  # milliseconds
    )
    return merged

# ==========================================
# --------------- TEST ---------------------
# ==========================================

if __name__ == "__main__":
    print("--- Running Unified Loader Test ---")

    gnss_df = load_sensor_data("Loader/device_gnss.csv")
    imu_df = load_sensor_data("Loader/device_imu.csv")

    print("\nGNSS Data Sample:")
    print(gnss_df.head())
    print(gnss_df.dtypes)

    print("\nIMU Data Sample:")
    print(imu_df.head())
    print(imu_df.dtypes)
    
    # Test the merge function
    print("\n--- Testing GNSS-IMU Merge ---")
    merged_df = merge_gnss_imu(gnss_df, imu_df)
    print(merged_df.head())
    
    # Plot the merged data
    plt.figure()
    plt.title("IMU AccelZ vs GNSS Position Time Trace")
    plt.plot(merged_df["utcTimeMillis"], merged_df["MeasurementZ"], label="Accel Z")
    plt.xlabel("Time (ms)")
    plt.legend()
    plt.show()



