import pandas as pd
import numpy as np
from typing import Dict, List

# ==========================================
# ----------- SCHEMA DEFINITIONS -----------
# ==========================================
# Structure: Standardized Name: [List of possible source column names, final dtype]
#
# SCHEMA UPDATE EXPLANATION:
# =========================
# The original schemas were designed for .txt log files but we're now processing .csv files
# from the smartphone-decimeter-2022 dataset. The actual column names in the CSV files
# are different from what was originally expected. Here's what we discovered and fixed:
#
# 1. TIME COLUMNS: The CSV files use 'utcTimeMillis' instead of 'ElapsedRealtimeMillis'
# 2. IMU MEASUREMENTS: The CSV files use 'MeasurementX/Y/Z' instead of 'UncalAccel.X/Y/Z'
# 3. BIAS COLUMNS: The CSV files use 'BiasX/Y/Z' instead of 'UncalAccel.BiasX/Y/Z'
# 4. STATUS DATA: No Status messages exist in the dataset - only Raw GNSS data
#
# Each schema entry now includes BOTH the original expected names AND the actual CSV names
# to ensure compatibility with the real data structure.

# GNSS Raw Data Schema (device_gnss.csv - Type: Raw)
# ===================================================
# This schema handles the main GNSS satellite measurement data from device_gnss.csv
# Key change: Added 'utcTimeMillis' as the primary time column (it's what actually exists)
RAW_SCHEMA = {
    'MessageType': ['MessageType', '# Type', object],
    # Time Columns - CRITICAL FIX: Added 'utcTimeMillis' as primary time column
    # The CSV files actually use 'utcTimeMillis', not 'ElapsedRealtimeMillis'
    'ElapsedRealtimeMillis': ['utcTimeMillis', 'ElapsedRealtimeMillis', np.int64],
    'TimeNanos': ['TimeNanos', np.int64], # Device clock time in ns
    'ReceivedSvTimeInNanos': ['ReceivedSvTimeInNanos', 'ReceivedSvTimeNanos', np.int64],
    # Derived/Target standardized columns (used in data_uitiles.py)
    'millisSinceGpsEpoch': ['millisSinceGpsEpoch', np.int64], 
    
    # GNSS Measurement Columns - These remain the same as they match the CSV structure
    'Svid': ['Svid', np.int32],
    'ConstellationType': ['ConstellationType', np.int32],
    'Cn0DbHz': ['Cn0DbHz', np.float64],
    'AccumulatedDeltaRangeMeters': ['AccumulatedDeltaRangeMeters', np.float64],
    'AccumulatedDeltaRangeUncertaintyMeters': ['AccumulatedDeltaRangeUncertaintyMeters', np.float64],
    'PseudorangeRateMetersPerSecond': ['PseudorangeRateMetersPerSecond', np.float64],
    'PseudorangeRateUncertaintyMetersPerSecond': ['PseudorangeRateUncertaintyMetersPerSecond', np.float64],
    
    # Other necessary columns
    'CarrierFrequencyHz': ['CarrierFrequencyHz', np.float64],
}

# GNSS Status Data Schema (device_gnss.csv - Type: Status)
# =========================================================
# IMPORTANT: This schema is kept for compatibility but Status messages DO NOT EXIST
# in the smartphone-decimeter-2022 dataset. The fuser.py has been updated to handle
# empty Status data gracefully.
STATUS_SCHEMA = {
    'MessageType': ['MessageType', '# Type', object],
    # Time Column
    'millisSinceGpsEpoch': ['millisSinceGpsEpoch', np.int64], 
    # Status/Position Columns
    'NumSatellitesUsed': ['NumSatellitesUsed', np.int32],
    'WlsPositionVelocity.WlsPosition.WlsPositionStatus': ['WlsPositionVelocity.WlsPosition.WlsPositionStatus', np.int32],
    'WlsPositionVelocity.WlsPosition.HaeMeters': ['WlsPositionVelocity.WlsPosition.HaeMeters', np.float64],
}

# IMU Accel Data Schema (device_imu.csv - Type: Accel/UncalAccel)
# ================================================================
# CRITICAL FIXES for accelerometer data:
# 1. Time column: Added 'utcTimeMillis' as primary (what actually exists in CSV)
# 2. Measurements: Added 'MeasurementX/Y/Z' as primary (actual CSV column names)
# 3. Bias: Added 'BiasX/Y/Z' as primary (actual CSV column names)
# The original 'UncalAccel.X/Y/Z' names are kept for backward compatibility
ACCEL_SCHEMA = {
    'MessageType': ['MessageType', '# Type', object],
    # IMU Time - CRITICAL FIX: Added 'utcTimeMillis' as primary time column
    # The CSV files use 'utcTimeMillis' for both GNSS and IMU data
    'millisSinceBoot': ['utcTimeMillis', 'millisSinceBoot', 'MillisSinceBoot', 'UncalAccel.MillisSinceBoot', np.int64],
    # Vector Components - CRITICAL FIX: Added 'MeasurementX/Y/Z' as primary columns
    # The CSV files actually use 'MeasurementX/Y/Z', not 'UncalAccel.X/Y/Z'
    'UncalAccel.X': ['MeasurementX', 'UncalAccel.X', 'UncalAccel.uncalX', 'X', np.float64],
    'UncalAccel.Y': ['MeasurementY', 'UncalAccel.Y', 'UncalAccel.uncalY', 'Y', np.float64],
    'UncalAccel.Z': ['MeasurementZ', 'UncalAccel.Z', 'UncalAccel.uncalZ', 'Z', np.float64],
    # Bias Estimates - CRITICAL FIX: Added 'BiasX/Y/Z' as primary columns
    # The CSV files use 'BiasX/Y/Z', not 'UncalAccel.BiasX/Y/Z'
    'UncalAccel.BiasX': ['BiasX', 'UncalAccel.BiasX', 'UncalAccel.biasX', np.float64],
    'UncalAccel.BiasY': ['BiasY', 'UncalAccel.BiasY', 'UncalAccel.biasY', np.float64],
    'UncalAccel.BiasZ': ['BiasZ', 'UncalAccel.BiasZ', 'UncalAccel.biasZ', np.float64],
}

# IMU Gyro Data Schema (device_imu.csv - Type: Gyro/UncalGyro)
# =============================================================
# CRITICAL FIXES for gyroscope data (same pattern as accelerometer):
# 1. Time column: Added 'utcTimeMillis' as primary
# 2. Measurements: Added 'MeasurementX/Y/Z' as primary
# 3. Bias: Added 'BiasX/Y/Z' as primary
# The original 'UncalGyro.X/Y/Z' names are kept for backward compatibility
GYRO_SCHEMA = {
    'MessageType': ['MessageType', '# Type', object],
    # IMU Time - CRITICAL FIX: Added 'utcTimeMillis' as primary time column
    'millisSinceBoot': ['utcTimeMillis', 'millisSinceBoot', 'MillisSinceBoot', 'UncalGyro.MillisSinceBoot', np.int64],
    # Vector Components - CRITICAL FIX: Added 'MeasurementX/Y/Z' as primary columns
    'UncalGyro.X': ['MeasurementX', 'UncalGyro.X', 'UncalGyro.uncalX', 'X', np.float64],
    'UncalGyro.Y': ['MeasurementY', 'UncalGyro.Y', 'UncalGyro.uncalY', 'Y', np.float64],
    'UncalGyro.Z': ['MeasurementZ', 'UncalGyro.Z', 'UncalGyro.uncalZ', 'Z', np.float64],
    # Bias Estimates - CRITICAL FIX: Added 'BiasX/Y/Z' as primary columns
    'UncalGyro.BiasX': ['BiasX', 'UncalGyro.BiasX', 'UncalGyro.biasX', np.float64],
    'UncalGyro.BiasY': ['BiasY', 'UncalGyro.BiasY', 'UncalGyro.biasY', np.float64],
    'UncalGyro.BiasZ': ['BiasZ', 'UncalGyro.BiasZ', 'UncalGyro.biasZ', np.float64],
}

# CRITICAL EXPORT: Dictionary used by data_uitiles.py to standardize columns
SCHEMA_MAPS: Dict[str, Dict[str, List]] = {
    'raw': RAW_SCHEMA,
    'status': STATUS_SCHEMA,
    'accel': ACCEL_SCHEMA,
    'gyro': GYRO_SCHEMA,
}


# ==========================================
# ----------- DATA MERGING UTILITY ----------
# ==========================================

def merge_dataframes(gnss_df: pd.DataFrame, imu_df: pd.DataFrame, time_col_gnss: str, time_col_imu: str) -> pd.DataFrame:
    """
    Merges IMU data onto GNSS data based on nearest time using pd.merge_asof.
    The IMU dataframe is used as the 'left' (base) for the merge.
    
    Args:
        gnss_df: GNSS dataframe
        imu_df: IMU dataframe  
        time_col_gnss: Time column name in GNSS dataframe
        time_col_imu: Time column name in IMU dataframe
        
    Returns:
        Merged dataframe with nearest time matches
    """
    if gnss_df.empty or imu_df.empty:
        return pd.DataFrame({})
        
    merged = pd.merge_asof(
        imu_df.sort_values(time_col_imu),
        gnss_df.sort_values(time_col_gnss),
        left_on=time_col_imu,
        right_on=time_col_gnss,
        direction='nearest',
        tolerance=50  # milliseconds tolerance for nearest match
    )
    return merged
