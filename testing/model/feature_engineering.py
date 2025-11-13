"""
Feature engineering utilities for residual learning.
Handles computing residuals and preparing training data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

from config import (
    BASELINE_LAT,
    BASELINE_LON,
    BASELINE_HEIGHT,
    GT_LAT,
    GT_LON,
    GT_HEIGHT,
    METERS_PER_DEGREE_LAT,
    get_all_features,
)


def compute_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute position residuals in meters.

    Formula:
        Residual = Ground Truth - Baseline

    Args:
        df: DataFrame with baseline and ground truth positions

    Returns:
        DataFrame with added residual columns:
        - lat_residual_m: North residual in meters
        - lon_residual_m: East residual in meters
        - height_residual_m: Height residual in meters
        - horizontal_residual_m: 2D horizontal residual magnitude
    """
    # Latitude residual (degrees) → meters
    lat_residual_deg = df[GT_LAT] - df[BASELINE_LAT]
    df['lat_residual_m'] = lat_residual_deg * METERS_PER_DEGREE_LAT

    # Longitude residual (degrees) → meters (account for latitude)
    lon_residual_deg = df[GT_LON] - df[BASELINE_LON]
    df['lon_residual_m'] = lon_residual_deg * METERS_PER_DEGREE_LAT * np.cos(np.radians(df[GT_LAT]))

    # Height residual (already in meters)
    if BASELINE_HEIGHT in df.columns and GT_HEIGHT in df.columns:
        df['height_residual_m'] = df[GT_HEIGHT] - df[BASELINE_HEIGHT]
    else:
        df['height_residual_m'] = 0

    # Horizontal error (2D)
    df['horizontal_residual_m'] = np.sqrt(df['lat_residual_m']**2 + df['lon_residual_m']**2)

    return df


def filter_large_residuals(df: pd.DataFrame, max_error_m: float = 50.0) -> pd.DataFrame:
    """
    Filter out samples with horizontal residual > max_error_m.
    
    These are likely outliers (wrong baseline solution, GPS jumps, etc.)
    that can negatively affect model training.
    
    Args:
        df: DataFrame with horizontal_residual_m column
        max_error_m: Maximum allowed horizontal error in meters
        
    Returns:
        Filtered DataFrame
    """
    before_count = len(df)
    
    # Filter
    df = df[df['horizontal_residual_m'] <= max_error_m].copy()
    
    after_count = len(df)
    removed_count = before_count - after_count
    removed_pct = (removed_count / before_count * 100) if before_count > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"FILTERING LARGE RESIDUALS (>{max_error_m}m)")
    print(f"{'='*70}")
    print(f"Before filtering: {before_count:,} samples")
    print(f"After filtering:  {after_count:,} samples")
    print(f"Removed:          {removed_count:,} samples ({removed_pct:.1f}%)")
    print(f"{'='*70}")
    
    return df


def print_residual_statistics(df: pd.DataFrame):
    """
    Print statistics about the residual distribution.

    Args:
        df: DataFrame with residual columns
    """
    print("="*70)
    print("RESIDUAL STATISTICS (Baseline Error)")
    print("="*70)
    print(f"Horizontal residual:")
    print(f"  Mean:   {df['horizontal_residual_m'].mean():.2f} m")
    print(f"  Median: {df['horizontal_residual_m'].median():.2f} m")
    print(f"  Std:    {df['horizontal_residual_m'].std():.2f} m")
    print(f"  Min:    {df['horizontal_residual_m'].min():.2f} m")
    print(f"  Max:    {df['horizontal_residual_m'].max():.2f} m")
    print(f"  95th:   {df['horizontal_residual_m'].quantile(0.95):.2f} m")
    print("="*70)


def prepare_training_data(
    df: pd.DataFrame,
    feature_list: List[str] = None,
    use_component_targets: bool = False,
) -> Tuple:
    """
    Extract features and targets, removing invalid samples.

    Args:
        df: DataFrame with features, baseline, GT, and residuals
        feature_list: List of feature column names. If None, uses all available features.
        use_component_targets: If True, returns separate lat/lon targets. If False, returns magnitude.

    Returns:
        If use_component_targets=False:
            tuple: (X, y, baseline_lat, baseline_lon, gt_lat, gt_lon, lat_residual, lon_residual, valid_mask)
            - X: Feature matrix
            - y: Target (horizontal residual magnitude)
            - baseline_lat/lon: Baseline positions
            - gt_lat/lon: Ground truth positions
            - lat_residual/lon_residual: Residual components for correction
            - valid_mask: Boolean mask indicating which rows were kept

        If use_component_targets=True:
            tuple: (X, y_lat, y_lon, baseline_lat, baseline_lon, gt_lat, gt_lon, valid_mask)
            - X: Feature matrix
            - y_lat: Latitude residual target (meters)
            - y_lon: Longitude residual target (meters)
            - baseline_lat/lon: Baseline positions
            - gt_lat/lon: Ground truth positions
            - valid_mask: Boolean mask indicating which rows were kept
    """
    # Get feature list
    if feature_list is None:
        feature_list = get_all_features(df)

    # Extract features
    X = df[feature_list].copy()

    if use_component_targets:
        # Component targets: separate lat and lon residuals
        y_lat = df['lat_residual_m'].copy()
        y_lon = df['lon_residual_m'].copy()

        # Drop rows with NaN targets
        valid_mask = y_lat.notna() & y_lon.notna() & df[BASELINE_LAT].notna() & df[GT_LAT].notna()
        X = X[valid_mask].reset_index(drop=True)
        y_lat = y_lat[valid_mask].reset_index(drop=True)
        y_lon = y_lon[valid_mask].reset_index(drop=True)

        # Store baseline and GT for later evaluation
        baseline_lat = df.loc[valid_mask, BASELINE_LAT].reset_index(drop=True)
        baseline_lon = df.loc[valid_mask, BASELINE_LON].reset_index(drop=True)
        gt_lat = df.loc[valid_mask, GT_LAT].reset_index(drop=True)
        gt_lon = df.loc[valid_mask, GT_LON].reset_index(drop=True)

        print(f"\nPreparing training data (COMPONENT MODE):")
        print(f"  Final samples: {len(X):,}")
        print(f"  Features: {len(feature_list)}")
        print(f"  Features with NaNs: {X.isna().any().sum()}/{len(feature_list)}")
        print(f"  Target: Separate lat/lon residuals")

        return X, y_lat, y_lon, baseline_lat, baseline_lon, gt_lat, gt_lon, valid_mask

    else:
        # Single target: horizontal residual magnitude (legacy mode)
        y = df['horizontal_residual_m'].copy()

        # Drop rows with NaN targets
        valid_mask = y.notna() & df[BASELINE_LAT].notna() & df[GT_LAT].notna()
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)

        # Store baseline and GT for later evaluation
        baseline_lat = df.loc[valid_mask, BASELINE_LAT].reset_index(drop=True)
        baseline_lon = df.loc[valid_mask, BASELINE_LON].reset_index(drop=True)
        gt_lat = df.loc[valid_mask, GT_LAT].reset_index(drop=True)
        gt_lon = df.loc[valid_mask, GT_LON].reset_index(drop=True)
        lat_residual = df.loc[valid_mask, 'lat_residual_m'].reset_index(drop=True)
        lon_residual = df.loc[valid_mask, 'lon_residual_m'].reset_index(drop=True)

        print(f"\nPreparing training data (MAGNITUDE MODE):")
        print(f"  Final samples: {len(X):,}")
        print(f"  Features: {len(feature_list)}")
        print(f"  Features with NaNs: {X.isna().any().sum()}/{len(feature_list)}")
        print(f"  Target: Horizontal residual magnitude")

        return X, y, baseline_lat, baseline_lon, gt_lat, gt_lon, lat_residual, lon_residual


def compute_position_error(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> np.ndarray:
    """
    Compute horizontal position error in meters between two sets of coordinates.

    Args:
        lat1, lon1: First set of coordinates (degrees)
        lat2, lon2: Second set of coordinates (degrees)

    Returns:
        Array of horizontal errors in meters
    """
    # Latitude error in meters
    lat_err_m = (lat2 - lat1) * METERS_PER_DEGREE_LAT

    # Longitude error in meters (account for latitude)
    lon_err_m = (lon2 - lon1) * METERS_PER_DEGREE_LAT * np.cos(np.radians(lat2))

    # Horizontal error (2D)
    horizontal_err_m = np.sqrt(lat_err_m**2 + lon_err_m**2)

    return horizontal_err_m


def apply_residual_correction(
    baseline_lat: pd.Series,
    baseline_lon: pd.Series,
    lat_residual: pd.Series,
    lon_residual: pd.Series,
    predicted_residual_magnitude: np.ndarray,
    actual_residual_magnitude: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """
    Apply residual correction to baseline positions.

    This uses the predicted residual magnitude with the direction from actual residuals.

    Args:
        baseline_lat, baseline_lon: Baseline positions
        lat_residual, lon_residual: True residual components
        predicted_residual_magnitude: LGBM predicted residual magnitude
        actual_residual_magnitude: True residual magnitude

    Returns:
        tuple: (corrected_lat, corrected_lon)
    """
    # Use the predicted magnitude with actual direction
    # (This is simplified; for production, train separate models for each component)
    correction_scale = predicted_residual_magnitude / (actual_residual_magnitude + 1e-6)  # Avoid division by zero

    # Apply correction
    lat_correction_m = lat_residual * correction_scale
    lon_correction_m = lon_residual * correction_scale

    # Convert corrections back to degrees
    lat_correction_deg = lat_correction_m / METERS_PER_DEGREE_LAT
    lon_correction_deg = lon_correction_m / (METERS_PER_DEGREE_LAT * np.cos(np.radians(baseline_lat)))

    # Apply corrections
    corrected_lat = baseline_lat + lat_correction_deg
    corrected_lon = baseline_lon + lon_correction_deg

    return corrected_lat, corrected_lon