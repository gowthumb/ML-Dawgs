"""
Configuration file for residual learning model.
Contains constants, feature definitions, and model parameters.
"""

import os

# ============================================================
# DATA PATHS
# ============================================================
DATA_PATH = r"C:\Users\avnee\ML\ML-Dawgs\testing\training_set_all_folders_ekf.csv"
TRAIN_DATA_ROOT = r"C:\Users\avnee\Downloads\smartphone-decimeter-2022\train"
OUTPUT_DIR = "model/outputs"

# ============================================================
# BASELINE AND GROUND TRUTH COLUMNS
# ============================================================
# Baseline: POS output from PPK .pos files
BASELINE_LAT = 'mean_latitude'
BASELINE_LON = 'mean_longitude'
BASELINE_HEIGHT = 'mean_height'

# Ground truth from separate reference (loaded from ground_truth.csv)
GT_LAT = 'gt_latitude'
GT_LON = 'gt_longitude'
GT_HEIGHT = 'gt_altitude'

# ============================================================
# FEATURE DEFINITIONS
# ============================================================
# GNSS features
GNSS_FEATURES = [
    'num_sats', 'mean_cn0', 'std_cn0', 'max_cn0', 'min_cn0',
    'mean_cn0_norm', 'mean_cn0_smooth', 'std_cn0_rate',
    'mean_pseudorange', 'std_pseudorange', 'mean_doppler',
    'num_sats_status', 'mean_wls_status', 'max_sats_used',
    'hae_std', 'mean_elevation', 'std_elevation', 'max_elevation', 'min_elevation',
    'azimuth_spread', 'mean_weighted_cn0', 'num_high_elev_sats',
    'cn0_trend', 'num_sats_std_5s', 'high_qual_sat_ratio', 'hae_std_roll',
]

# IMU features
IMU_FEATURES = [
    'accel_mag_mean', 'accel_mag_roll_mean', 'accel_variance', 'accel_jerk',
    'accel_mag_no_gravity', 'gyro_mag_mean', 'gyro_mag_roll_mean',
    'gyro_variance', 'gyro_jerk', 'accel_mag_std_1s_roll', 'is_stationary',
]

# EKF features (optional, can also be used as baseline instead of POS)
EKF_FEATURES = [
    'ekf_latitude', 'ekf_longitude', 'ekf_height',
    'ekf_vel_n', 'ekf_vel_e', 'ekf_vel_u',
    'ekf_pos_std', 'ekf_vel_std',
]

# Missingness indicators
MISSINGNESS_INDICATORS = [
    'gnss_raw_missing',
    'gnss_status_missing',
]

# ============================================================
# MODEL PARAMETERS
# ============================================================
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'use_missing': True,
    'zero_as_missing': False,
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42,
}

NUM_BOOST_ROUNDS = 1000
LOG_EVALUATION_PERIOD = 50

# ============================================================
# TRAINING PARAMETERS
# ============================================================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Time-based merge tolerance (milliseconds)
MERGE_TOLERANCE_MS = 100

# GPS Epoch offset for timestamp conversion
# GPS Epoch: 1980-01-06 00:00:00 UTC
# Unix Epoch: 1970-01-01 00:00:00 UTC
# Difference: 315964800 seconds = 315964800000 milliseconds
GPS_EPOCH_OFFSET_MILLIS = 315964800000

# ============================================================
# CONSTANTS
# ============================================================
# Approximate conversion: 1 degree latitude ≈ 111 km
METERS_PER_DEGREE_LAT = 111000


def get_all_features(df):
    """
    Get all available features from dataframe.

    Args:
        df: pandas DataFrame with featurization data

    Returns:
        list: List of available feature column names
    """
    available_gnss = [f for f in GNSS_FEATURES if f in df.columns]
    available_imu = [f for f in IMU_FEATURES if f in df.columns]
    available_ekf = [f for f in EKF_FEATURES if f in df.columns]
    available_missing = [f for f in MISSINGNESS_INDICATORS if f in df.columns]

    return available_gnss + available_imu + available_ekf + available_missing


def validate_columns(df):
    """
    Validate that required columns exist in the dataframe.

    Args:
        df: pandas DataFrame

    Returns:
        tuple: (bool, list) - (all_present, missing_columns)
    """
    required_cols = [BASELINE_LAT, BASELINE_LON, GT_LAT, GT_LON]
    missing_cols = [col for col in required_cols if col not in df.columns]

    return len(missing_cols) == 0, missing_cols
