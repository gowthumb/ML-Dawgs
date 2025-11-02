import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from loader import SCHEMA_MAPS
import os # <--- CRITICAL: ADD THIS IMPORT

def read_and_separate_log(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Reads a CSV log file robustly and separates the data based on the MessageType or # Type column.
    """
    try:
        # --- 1. ROBUST HEADER DETECTION ---
        # Count all comment lines (#) to find the start of the data/header.
        skip_rows = 0
        with open(file_path, 'r', errors='ignore') as f:
            for line in f:
                if line.startswith('#'):
                    skip_rows += 1
                else:
                    break
        
        # Read the file, skipping metadata lines. The first non-comment line is the header.
        df = pd.read_csv(file_path, skiprows=skip_rows, skipinitialspace=True, low_memory=False)
        # --- END ROBUST HEADER DETECTION ---

        log_file_name = file_path.split(os.sep)[-1] 

        # --- 2. CHECK FOR EMPTY DATA ---
        if df.empty:
            print(f"ERROR in {log_file_name}: DataFrame is empty after initial read (Skipping {skip_rows} lines).")
            return {}
        
        # --- 3. DETERMINE SEPARATION COLUMN ---
        if 'MessageType' in df.columns:
            type_col = 'MessageType'
        elif '# Type' in df.columns:
            type_col = '# Type'
        else:
            print(f"ERROR in {log_file_name}: Could not find primary type column (MessageType or # Type).")
            return {}

        # --- 4. SEPARATE DATA ---
        separated_data = {}
        all_groups = list(df[type_col].unique())
        print(f"DEBUG {log_file_name}: All unique message types found: {all_groups}") # <--- ADDED DIAGNOSTIC

        for message_type, data in df.groupby(type_col):
            lower_key = message_type.lower()
            
            if 'raw' in lower_key:
                separated_data['raw'] = data.copy().drop(columns=[type_col], errors='ignore')
            elif 'status' in lower_key:
                separated_data['status'] = data.copy().drop(columns=[type_col], errors='ignore')
            elif 'accel' in lower_key:
                separated_data['accel'] = data.copy().drop(columns=[type_col], errors='ignore')
            elif 'gyro' in lower_key:
                separated_data['gyro'] = data.copy().drop(columns=[type_col], errors='ignore')
                
        # --- 5. DIAGNOSTICS (to trace next step) ---
        print(f"DEBUG {log_file_name}: Read {len(df)} total rows. Separated sections:")
        for key, data in separated_data.items():
            if data.empty:
                print(f"  ⚠️ WARNING: '{key}' section is EMPTY after separation.")
            else:
                print(f"  ✓ '{key}' section has {len(data)} rows.")
        
        return separated_data

    except Exception as e:
        # Catch any unexpected file reading error
        print(f"CRITICAL ERROR in read_and_separate_log for {file_path}: {e}")
        return {}





def standardize_columns(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """
    Standardizes column names in a DataFrame based on predefined schema maps,
    handling various naming conventions and converting to the correct dtypes.
    
    Returns an empty DataFrame if critical columns are missing.
    """
    if df.empty or data_type not in SCHEMA_MAPS:
        return df

    schema = SCHEMA_MAPS[data_type]
    df_standardized = pd.DataFrame()
    
    current_columns = set(df.columns)
    
    # Define CRITICAL columns that MUST be present for time sync and feature extraction
    critical_cols = {
        'raw': ['TimeNanos', 'ElapsedRealtimeMillis', 'Svid'],
        'status': ['millisSinceGpsEpoch', 'NumSatellitesUsed'],
        'accel': ['millisSinceBoot', 'UncalAccel.X', 'UncalAccel.Y', 'UncalAccel.Z'],
        'gyro': ['millisSinceBoot', 'UncalGyro.X', 'UncalGyro.Y', 'UncalGyro.Z'],
    }

    # Iterate through the required standard columns
    for standard_name, sources in schema.items():
        
        # Last element in the sources list is the dtype
        dtype = sources[-1]
        possible_sources = sources[:-1] 

        # Find the first matching column name in the current DataFrame
        found_name = None
        for source_name in possible_sources:
            if source_name in current_columns:
                found_name = source_name
                break

        if found_name:
            # Rename and copy the column to the standardized DataFrame with type conversion
            # Use to_numeric with 'coerce' to turn non-numeric strings into NaN before casting
            df_standardized[standard_name] = pd.to_numeric(df[found_name], errors='coerce').astype(dtype, errors='ignore')
        else:
            # Handle missing columns
            if standard_name in critical_cols.get(data_type, []):
                # print(f"Warning: CRITICAL column '{standard_name}' missing for {data_type} data. Returning empty DF.")
                return pd.DataFrame({})
            
            # For non-critical missing columns, create an empty column of the specified dtype
            df_standardized[standard_name] = pd.Series(dtype=dtype)

    return df_standardized.reset_index(drop=True)


def _validate_imu_columns(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """Helper to perform final IMU data validation after initial standardization."""
    if df.empty:
        return pd.DataFrame({})
        
    # Validation step: Ensure required columns are present after standardization
    if data_type == 'accel':
        # These are the standard names defined in SCHEMA_MAPS for the vectors
        required_cols = ['millisSinceBoot', 'UncalAccel.X', 'UncalAccel.Y', 'UncalAccel.Z']
    elif data_type == 'gyro':
        required_cols = ['millisSinceBoot', 'UncalGyro.X', 'UncalGyro.Y', 'UncalGyro.Z']
    else:
        return pd.DataFrame({})

    # Check if all required columns are in the DF after standardization
    missing_cols = list(set(required_cols) - set(df.columns))
    if missing_cols:
        # print(f"Warning: Failed final IMU validation for {data_type}. Missing: {missing_cols}")
        return pd.DataFrame({}) 

    # Drop rows where required IMU vectors or time are missing
    df.dropna(subset=required_cols, inplace=True)
    
    return df.reset_index(drop=True)


def apply_sanity_filters(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """
    Applies column standardization, basic cleaning, and converts necessary columns to numeric types.
    """
    if df.empty: return df
    
    # STEP 1: COLUMN STANDARDIZATION
    df = standardize_columns(df, data_type)
    if df.empty:
        # Standardization failed due to missing critical columns
        return pd.DataFrame({})

    # --- GNSS Raw Cleaning ---
    if data_type == 'raw':
        # Apply C/N0 filter (common practice: filter out weak signals)
        if 'Cn0DbHz' in df.columns and df['Cn0DbHz'].notna().any():
            df = df[df['Cn0DbHz'].between(10, 60)].copy()
        
        # Drop rows with no satellite ID or inconsistent TimeNanos
        df.dropna(subset=['Svid', 'TimeNanos'], inplace=True)
        
    # --- GNSS Status Cleaning ---
    elif data_type == 'status':
        # Drop NaNs on critical columns
        required_cols = ['millisSinceGpsEpoch', 'NumSatellitesUsed']
        df.dropna(subset=required_cols, inplace=True)
        
    # --- IMU Accel/Gyro Cleaning ---
    elif data_type in ['accel', 'gyro']:
        # This helper function performs final validation and cleaning after standardization
        df = _validate_imu_columns(df, data_type)
        
    return df.reset_index(drop=True)


def add_time_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the synchronized time columns needed for merging.
    - 'millisSinceBoot_raw': Device clock time in ms (from TimeNanos).
    - 'millisSinceGpsEpoch': Absolute time in ms (from ElapsedRealtimeMillis + GPS Epoch Offset).
    """
    if raw_df.empty:
        return raw_df

    df = raw_df.copy()
    
    # Check for critical time columns
    if 'TimeNanos' not in df.columns or df['TimeNanos'].isnull().all():
        # print("ERROR: Missing 'TimeNanos' in Raw data after standardization.")
        return pd.DataFrame({})
    if 'ElapsedRealtimeMillis' not in df.columns or df['ElapsedRealtimeMillis'].isnull().all():
        # print("WARNING: 'ElapsedRealtimeMillis' missing. Cannot compute absolute time reliably.")
        return pd.DataFrame({})
    
    # 1. Calculate millisSinceBoot_raw (Relative time from device clock)
    # TimeNanos (ns) -> milliseconds (ms)
    df['millisSinceBoot_raw'] = (df['TimeNanos'] / 1_000_000).astype(np.float64)

    # 2. Calculate millisSinceGpsEpoch (Absolute time from GPS Epoch)
    # FIXED: GPS Epoch is LATER than Unix Epoch (1980-01-06 vs 1970-01-01)
    # So to convert Unix time to GPS epoch time: SUBTRACT the offset
    # GPS Epoch offset: 315964800000ms (1980-01-06T00:00:00Z is 315964800 seconds after Unix epoch)
    GPS_EPOCH_OFFSET_MILLIS = 315964800000 
    # utcTimeMillis (ElapsedRealtimeMillis) is Unix time, so subtract offset to get GPS epoch time
    df['millisSinceGpsEpoch'] = df['ElapsedRealtimeMillis'].astype(np.int64) - GPS_EPOCH_OFFSET_MILLIS
    
    # Final cleanup and rounding for absolute time (Round to nearest 10ms for better merging/aggregation later)
    df.dropna(subset=['millisSinceGpsEpoch'], inplace=True)
    df['millisSinceGpsEpoch'] = df['millisSinceGpsEpoch'].round(-1).astype(np.int64)
    
    return df


def read_pos_file(file_path: str) -> pd.DataFrame:
    """
    Reads and parses an RTKLIB .pos file containing precise positioning solutions.
    
    The .pos file format is:
    % GPST          latitude(deg) longitude(deg)  height(m)   Q  ns   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m) age(s)  ratio
    where:
    - GPST: GPS week and seconds
    - latitude, longitude, height: Position in WGS84/ellipsoidal
    - Q: Quality flag (1=fix, 2=float, 3=sbas, 4=dgps, 5=single, 6=ppp)
    - ns: Number of satellites
    - sdn, sde, sdu: Standard deviations in north, east, up (meters)
    - sdne, sdeu, sdun: Covariances
    - age: Age (seconds)
    - ratio: Ratio
    
    Args:
        file_path: Path to the .pos file
        
    Returns:
        DataFrame with parsed .pos data, including millisSinceGpsEpoch column
    """
    if not os.path.exists(file_path):
        print(f"Warning: .pos file not found: {file_path}")
        return pd.DataFrame({})
    
    # GPS epoch: January 6, 1980 00:00:00 UTC
    # GPS time in seconds = GPS week * 604800 + GPS seconds
    # Convert to milliseconds since GPS epoch
    
    data_lines = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip header lines and empty lines
                if line.startswith('%') or not line:
                    continue
                
                # Parse data line
                parts = line.split()
                if len(parts) < 14:
                    continue
                
                try:
                    gps_week = int(parts[0])
                    gps_seconds = float(parts[1])
                    
                    # Convert GPS week+seconds to milliseconds since GPS epoch
                    # GPS epoch: Jan 6, 1980 00:00:00 UTC
                    # millisSinceGpsEpoch = (gps_week * 604800 + gps_seconds) * 1000
                    millis_since_gps_epoch = int((gps_week * 604800 + gps_seconds) * 1000)
                    
                    # Round to nearest 10ms for consistency with other data
                    millis_since_gps_epoch = int((millis_since_gps_epoch / 10) * 10)
                    
                    latitude = float(parts[2])
                    longitude = float(parts[3])
                    height = float(parts[4])
                    quality = int(parts[5])
                    num_satellites = int(parts[6])
                    sdn = float(parts[7])
                    sde = float(parts[8])
                    sdu = float(parts[9])
                    sdne = float(parts[10])
                    sdeu = float(parts[11])
                    sdun = float(parts[12])
                    age = float(parts[13])
                    ratio = float(parts[14]) if len(parts) > 14 else 0.0
                    
                    data_lines.append({
                        'millisSinceGpsEpoch': millis_since_gps_epoch,
                        'latitude': latitude,
                        'longitude': longitude,
                        'height': height,
                        'quality': quality,
                        'num_satellites': num_satellites,
                        'sdn': sdn,  # Standard deviation north (m)
                        'sde': sde,  # Standard deviation east (m)
                        'sdu': sdu,  # Standard deviation up (m)
                        'sdne': sdne,  # Covariance north-east
                        'sdeu': sdeu,  # Covariance east-up
                        'sdun': sdun,  # Covariance up-north
                        'age': age,
                        'ratio': ratio
                    })
                except (ValueError, IndexError) as e:
                    continue
        
        if not data_lines:
            print(f"Warning: No valid data found in .pos file: {file_path}")
            return pd.DataFrame({})
        
        df = pd.DataFrame(data_lines)
        print(f"✓ Read .pos file: {len(df)} rows from {file_path}")
        return df
        
    except Exception as e:
        print(f"Error reading .pos file {file_path}: {e}")
        return pd.DataFrame({})
