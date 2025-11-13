"""
Data loading utilities for residual learning.
Handles loading featurization data and ground truth, and merging them.
"""

import os
import numpy as np
import pandas as pd
from glob import glob
from typing import Tuple

from config import (
    GPS_EPOCH_OFFSET_MILLIS,
    MERGE_TOLERANCE_MS,
    GT_LAT,
    GT_LON,
    GT_HEIGHT,
)


def load_all_ground_truth(train_root: str) -> pd.DataFrame:
    """
    Load all ground_truth.csv files from the training dataset.

    Args:
        train_root: Path to training data root directory

    Returns:
        DataFrame with columns: drive_id, phone_id, gt_latitude, gt_longitude,
        gt_altitude, millisSinceGpsEpoch
    """
    all_gt = []

    # Find all ground_truth.csv files
    pattern = os.path.join(train_root, "*", "*", "ground_truth.csv")
    gt_files = glob(pattern)

    print(f"Found {len(gt_files)} ground_truth.csv files")

    for gt_file in gt_files:
        # Extract drive_id and phone_id from path
        # Path structure: .../train/DRIVE_ID/PHONE_ID/ground_truth.csv
        parts = gt_file.split(os.sep)
        phone_id = parts[-2]  # GooglePixel4, etc.
        drive_id = parts[-3]  # 2020-12-10-US-SJC-1, etc.

        try:
            # Read ground truth file
            gt_df = pd.read_csv(gt_file)

            # Keep only relevant columns and rename
            gt_df = gt_df[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters', 'UnixTimeMillis']].copy()
            gt_df.columns = ['gt_latitude', 'gt_longitude', 'gt_altitude', 'UnixTimeMillis']

            # Convert UnixTimeMillis to millisSinceGpsEpoch
            gt_df['millisSinceGpsEpoch'] = gt_df['UnixTimeMillis'] - GPS_EPOCH_OFFSET_MILLIS
            gt_df['millisSinceGpsEpoch'] = (gt_df['millisSinceGpsEpoch'] / 10).round() * 10
            gt_df['millisSinceGpsEpoch'] = gt_df['millisSinceGpsEpoch'].astype(np.int64)

            # Add identifiers
            gt_df['drive_id'] = drive_id
            gt_df['phone_id'] = phone_id

            # Drop UnixTimeMillis (no longer needed)
            gt_df.drop(columns=['UnixTimeMillis'], inplace=True)

            all_gt.append(gt_df)

        except Exception as e:
            print(f"  Warning: Failed to load {drive_id}/{phone_id}: {e}")
            continue

    if not all_gt:
        print("ERROR: No ground truth files loaded!")
        return pd.DataFrame()

    # Concatenate all ground truth data
    gt_combined = pd.concat(all_gt, ignore_index=True)

    # CRITICAL: Sort after concatenation to ensure proper order for merge_asof
    gt_combined = gt_combined.sort_values(['drive_id', 'phone_id', 'millisSinceGpsEpoch']).reset_index(drop=True)

    print(f"Loaded {len(gt_combined):,} ground truth samples")
    print(f"Unique drives: {gt_combined['drive_id'].nunique()}")
    print(f"Unique phones: {gt_combined['phone_id'].nunique()}")

    return gt_combined


def merge_ground_truth_with_features(
    df: pd.DataFrame,
    gt_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, float]:
    """
    Merge ground truth with featurization data by timestamp.

    Args:
        df: Featurization dataframe
        gt_data: Ground truth dataframe

    Returns:
        tuple: (merged_df, merge_rate_percent)
    """
    print(f"Before merge: {len(df):,} rows in featurization data")
    print(f"              {len(gt_data):,} rows in ground truth data")

    # Ensure all merge keys are the correct type
    df['drive_id'] = df['drive_id'].astype(str)
    df['phone_id'] = df['phone_id'].astype(str)
    df['millisSinceGpsEpoch'] = df['millisSinceGpsEpoch'].astype(np.int64)

    gt_data['drive_id'] = gt_data['drive_id'].astype(str)
    gt_data['phone_id'] = gt_data['phone_id'].astype(str)
    gt_data['millisSinceGpsEpoch'] = gt_data['millisSinceGpsEpoch'].astype(np.int64)

    # Remove any rows with NaN in merge keys (these cause sorting issues)
    df = df.dropna(subset=['drive_id', 'phone_id', 'millisSinceGpsEpoch']).reset_index(drop=True)
    gt_data = gt_data.dropna(subset=['drive_id', 'phone_id', 'millisSinceGpsEpoch']).reset_index(drop=True)

    # Sort both datasets for merge_asof - must be sorted by 'by' columns first, then 'on' column
    df = df.sort_values(['drive_id', 'phone_id', 'millisSinceGpsEpoch'], ascending=True).reset_index(drop=True)
    gt_data = gt_data.sort_values(['drive_id', 'phone_id', 'millisSinceGpsEpoch'], ascending=True).reset_index(drop=True)

    # WORKAROUND: Perform merge_asof per group to avoid sorting validation issues
    print("\nPerforming group-wise merge_asof to ensure proper sorting...")
    merged_groups = []

    for (drive_id, phone_id), group_df in df.groupby(['drive_id', 'phone_id'], sort=False):
        # Get corresponding ground truth for this group
        gt_group = gt_data[(gt_data['drive_id'] == drive_id) & (gt_data['phone_id'] == phone_id)]

        if gt_group.empty:
            # No ground truth for this group, add NaN columns
            group_df = group_df.copy()
            for col in [GT_LAT, GT_LON, GT_HEIGHT]:
                group_df[col] = np.nan
            merged_groups.append(group_df)
            continue

        # Sort within this group (ensures monotonic increasing)
        group_df = group_df.sort_values('millisSinceGpsEpoch').reset_index(drop=True)
        gt_group = gt_group.sort_values('millisSinceGpsEpoch').reset_index(drop=True)

        # Merge this group
        merged = pd.merge_asof(
            group_df,
            gt_group[['millisSinceGpsEpoch', GT_LAT, GT_LON, GT_HEIGHT]],
            on='millisSinceGpsEpoch',
            direction='nearest',
            tolerance=MERGE_TOLERANCE_MS
        )
        merged_groups.append(merged)

    # Concatenate all merged groups
    df = pd.concat(merged_groups, ignore_index=True)
    print(f"Completed group-wise merge for {len(merged_groups)} groups")

    # Count successful merges
    gt_merged = df[GT_LAT].notna().sum()
    merge_rate = (gt_merged / len(df)) * 100

    print(f"\nAfter merge:  {len(df):,} rows (same as before)")
    print(f"Ground truth matched: {gt_merged:,} rows ({merge_rate:.1f}%)")
    print(f"Ground truth missing: {(len(df) - gt_merged):,} rows ({(100-merge_rate):.1f}%)")

    if merge_rate < 50:
        print("\nWARNING: Low merge rate! Check that drive_id and phone_id match between datasets.")
    elif merge_rate < 80:
        print("\nModerate merge rate. Some samples don't have ground truth - they'll be filtered during training.")
    else:
        print("\nGood merge rate!")

    return df, merge_rate


def load_training_data(data_path: str, train_root: str = None) -> pd.DataFrame:
    """
    Load featurization data and optionally merge with ground truth.

    Args:
        data_path: Path to featurization CSV file
        train_root: Path to training data root (for ground truth). If None, skip GT merge.

    Returns:
        DataFrame with featurization data and ground truth (if available)
    """
    # Load featurization data
    print(f"Loading featurization data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

    # CRITICAL: Sort immediately after loading CSV (CSV files are not guaranteed to be sorted)
    if 'drive_id' in df.columns and 'phone_id' in df.columns and 'millisSinceGpsEpoch' in df.columns:
        df = df.sort_values(['drive_id', 'phone_id', 'millisSinceGpsEpoch']).reset_index(drop=True)
        print("Sorted data by drive_id, phone_id, millisSinceGpsEpoch")

    # Load and merge ground truth if path provided
    if train_root:
        print("\nLoading ground truth data...")
        gt_data = load_all_ground_truth(train_root)

        if not gt_data.empty:
            print("\nMerging ground truth with featurization data...")
            df, merge_rate = merge_ground_truth_with_features(df, gt_data)
        else:
            print("Warning: No ground truth data loaded")
    else:
        print("\nSkipping ground truth merge (no train_root provided)")

    return df


def display_sample_data(df: pd.DataFrame):
    """
    Display sample of the merged data for verification.

    Args:
        df: Merged dataframe
    """
    sample_cols = ['drive_id', 'phone_id', 'mean_latitude', 'mean_longitude', GT_LAT, GT_LON]
    available_sample_cols = [c for c in sample_cols if c in df.columns]

    if available_sample_cols:
        print("\nSample with both POS (baseline) and GT:")
        print(df[available_sample_cols].head())
    else:
        print("\nSample data:")
        print(df.head())
