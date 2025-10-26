import pandas as pd
import numpy as np
from typing import Dict, List

# ==========================================
# ----------- SCHEMA DEFINITIONS -----------
# ==========================================
# Structure: Standardized Name: [List of possible source column names, final dtype]

# GNSS Raw Data Schema (device_gnss.csv - Type: Raw)
RAW_SCHEMA = {
    'MessageType': ['MessageType', '# Type', object],
    # Time Columns 
    'ElapsedRealtimeMillis': ['ElapsedRealtimeMillis', np.int64],
    'TimeNanos': ['TimeNanos', np.int64], # Device clock time in ns
    'ReceivedSvTimeInNanos': ['ReceivedSvTimeInNanos', 'ReceivedSvTimeNanos', np.int64],
    # Derived/Target standardized columns (used in data_uitiles.py)
    'millisSinceGpsEpoch': ['millisSinceGpsEpoch', np.int64], 
    
    # GNSS Measurement Columns
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
ACCEL_SCHEMA = {
    'MessageType': ['MessageType', '# Type', object],
    # IMU Time 
    'millisSinceBoot': ['millisSinceBoot', 'MillisSinceBoot', 'UncalAccel.MillisSinceBoot', np.int64],
    # Vector Components
    'UncalAccel.X': ['UncalAccel.X', 'UncalAccel.uncalX', 'X', np.float64],
    'UncalAccel.Y': ['UncalAccel.Y', 'UncalAccel.uncalY', 'Y', np.float64],
    'UncalAccel.Z': ['UncalAccel.Z', 'UncalAccel.uncalZ', 'Z', np.float64],
    # Bias Estimates (may be missing)
    'UncalAccel.BiasX': ['UncalAccel.BiasX', 'UncalAccel.biasX', np.float64],
    'UncalAccel.BiasY': ['UncalAccel.BiasY', 'UncalAccel.biasY', np.float64],
    'UncalAccel.BiasZ': ['UncalAccel.BiasZ', 'UncalAccel.biasZ', np.float64],
}

# IMU Gyro Data Schema (device_imu.csv - Type: Gyro/UncalGyro)
GYRO_SCHEMA = {
    'MessageType': ['MessageType', '# Type', object],
    # IMU Time 
    'millisSinceBoot': ['millisSinceBoot', 'MillisSinceBoot', 'UncalGyro.MillisSinceBoot', np.int64],
    # Vector Components
    'UncalGyro.X': ['UncalGyro.X', 'UncalGyro.uncalX', 'X', np.float64],
    'UncalGyro.Y': ['UncalGyro.Y', 'UncalGyro.uncalY', 'Y', np.float64],
    'UncalGyro.Z': ['UncalGyro.Z', 'UncalGyro.uncalZ', 'Z', np.float64],
    # Bias Estimates (may be missing)
    'UncalGyro.BiasX': ['UncalGyro.BiasX', 'UncalGyro.biasX', np.float64],
    'UncalGyro.BiasY': ['UncalGyro.BiasY', 'UncalGyro.biasY', np.float64],
    'UncalGyro.BiasZ': ['UncalGyro.BiasZ', 'UncalGyro.biasZ', np.float64],
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
