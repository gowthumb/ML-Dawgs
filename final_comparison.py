#!/usr/bin/env python3
"""
GSDC Ground Truth vs. RTKLIB Comparator (Fixed for GPS Week/TOW Format)

This script handles RTKLIB output in GPS Week + Time of Week format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from io import StringIO

# --- CONFIGURE YOUR PATH HERE ---
ROOT_DIR = Path(r'C:\Users\gauth\Documents\SC4000_proj')

POS_FILENAME_PATTERN = 'supplemental/gnss_log_rtklib.pos'
GT_FILENAME = 'ground_truth.csv'

# GPS epoch: January 6, 1980, 00:00:00 UTC
GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)

# GPS is ahead of UTC by 18 seconds (as of 2020)
GPS_UTC_OFFSET_SECONDS = 18

# Limit number of matched (drive_id, phone_id) pairs processed
MAX_FILES = None 


def gps_week_tow_to_unix_millis(gps_week, tow_seconds):
    """
    Convert GPS Week and Time of Week to Unix milliseconds (UTC).
    
    Args:
        gps_week: GPS week number
        tow_seconds: Time of week in seconds (with fractional part)
    
    Returns:
        Unix time in milliseconds (UTC)
    """
    # Calculate total seconds since GPS epoch
    total_seconds = gps_week * 7 * 24 * 3600 + tow_seconds
    
    # Convert to datetime (still in GPS time)
    gps_datetime = GPS_EPOCH + timedelta(seconds=total_seconds)
    
    # Convert to Unix time (UTC) by subtracting GPS-UTC offset
    utc_datetime = gps_datetime - timedelta(seconds=GPS_UTC_OFFSET_SECONDS)
    
    # Convert to Unix milliseconds
    unix_millis = int((utc_datetime - datetime(1970, 1, 1)).total_seconds() * 1000)
    
    return unix_millis


def load_pos_file(pos_path: Path) -> pd.DataFrame:
    """
    Loads a RTKLIB pos file with GPS Week/TOW format.
    
    Expected format (space-separated):
    GPS_Week TOW Latitude Longitude Height Q NumSats [additional columns...]
    """
    try:
        with open(pos_path, 'r') as f:
            lines = f.readlines()

        # Filter out header lines (starting with '%')
        data_lines = [line for line in lines if not line.strip().startswith('%')]
        if not data_lines:
            print(f"  ✗ {pos_path.name} is empty or all comments.") 
            return pd.DataFrame()

        data_io = StringIO("".join(data_lines))

        # Parse with whitespace separator, no header
        df = pd.read_csv(data_io, sep=r'\s+', header=None, engine='python')

        if df.empty or len(df.columns) < 6:
            print(f"  ✗ {pos_path.name} has insufficient columns (Expected >=6, got {len(df.columns)}).")
            return pd.DataFrame()

        # Extract GPS Week and TOW
        df['gps_week'] = df[0].astype(int)
        df['tow'] = df[1].astype(float)
        
        # Extract coordinates and quality
        df['latitude(deg)'] = df[2]
        df['longitude(deg)'] = df[3]
        df['Q'] = pd.to_numeric(df[5], errors='coerce', downcast='integer')

        # Convert GPS Week/TOW to Unix milliseconds (UTC)
        df['UnixTimeMillis'] = df.apply(
            lambda row: gps_week_tow_to_unix_millis(row['gps_week'], row['tow']),
            axis=1
        )

        return df[['UnixTimeMillis', 'latitude(deg)', 'longitude(deg)', 'Q']]

    except Exception as e:
        print(f"  ✗ ERROR reading {pos_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def load_gt_file(gt_path: Path) -> pd.DataFrame:
    """
    Loads a ground_truth.csv file and exposes UnixTimeMillis, LatitudeDegrees, LongitudeDegrees.
    """
    try:
        df = pd.read_csv(gt_path)

        time_col = 'UnixTimeMillis'
        lat_col = 'LatitudeDegrees'
        lon_col = 'LongitudeDegrees'

        df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
        df.dropna(subset=[time_col], inplace=True)
        df[time_col] = df[time_col].astype(np.int64)

        return df[[time_col, lat_col, lon_col]].rename(
            columns={
                time_col: 'UnixTimeMillis',
                lat_col: 'LatitudeDegrees',
                lon_col: 'LongitudeDegrees',
            }
        )

    except Exception as e:
        print(f"  ✗ ERROR reading {gt_path.name}: {e}")
        return pd.DataFrame()


def haversine_m(lat1, lon1, lat2, lon2):
    """
    Calculate the horizontal distance between two points
    on Earth (decimal degrees) in meters. 
    """
    R = 6371000.0   # Earth radius in meters

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    distance = R * c
    return distance


def main():
    # 1. Setup Directories
    TRAIN_DIR = ROOT_DIR / "train"
    
    # 2. Find all files
    pos_files = list(TRAIN_DIR.rglob(POS_FILENAME_PATTERN))
    gt_files = list(TRAIN_DIR.rglob(GT_FILENAME))
    
    # POS lookup (Key: (drive_id, phone_id) -> Value: pos_path)
    pos_lookup = {}
    for pos_path in pos_files:
        try:
            # pos_path is: .../<drive_id>/<phone_id>/supplemental/gnss_log_rtklib.pos
            phone_id = pos_path.parent.parent.name
            drive_id = pos_path.parent.parent.parent.name
            
            if drive_id and phone_id:
                pos_lookup[(drive_id, phone_id)] = pos_path
        except IndexError:
             continue

    # GT lookup (Key: (drive_id, phone_id) -> Value: gt_path)
    gt_lookup = {}
    for gt_path in gt_files:
        try:
            # gt_path is: .../<drive_id>/<phone_id>/ground_truth.csv
            phone_id = gt_path.parent.name
            drive_id = gt_path.parent.parent.name
            
            if drive_id and phone_id:
                gt_lookup[(drive_id, phone_id)] = gt_path
        except IndexError:
            continue
            
    print(f"Scanning for Ground Truth and RTKLIB POS files in: {TRAIN_DIR}")
    print(f"\nFound {len(gt_files)} Ground Truth files.")
    print(f"Found {len(pos_lookup)} RTKLIB POS file candidates.")

    # 3. Iterate, match, and compare
    all_errors = []
    file_stats = []
    num_processed = 0

    print("\n" + "=" * 80)
    print("STARTING COMPARISON")
    print("=" * 80)

    keys = sorted(pos_lookup.keys()) 

    for drive_id, phone_id in keys:
        if MAX_FILES is not None and num_processed >= MAX_FILES:
            print(f"\nReached MAX_FILES = {MAX_FILES}, stopping.")
            break

        key = (drive_id, phone_id)
        pos_path = pos_lookup[key]

        if key not in gt_lookup:
            print(f"-> Skipping {drive_id}/{phone_id}: GT file missing in expected phone directory.")
            continue
            
        gt_path = gt_lookup[key]

        print(f"-> Processing {drive_id}/{phone_id}")

        df_gt = load_gt_file(gt_path)
        df_pos = load_pos_file(pos_path)

        if df_gt.empty or df_pos.empty:
            print(f"  ! Skipping {drive_id}/{phone_id} (could not load or parse data)")
            print(f"    GT rows: {len(df_gt)}, POS rows: {len(df_pos)}")
            continue

        # Merge the two dataframes on the nearest timestamp (0.5-second tolerance)
        df_merged = pd.merge_asof(
            df_gt.sort_values('UnixTimeMillis'),
            df_pos.sort_values('UnixTimeMillis'),
            on='UnixTimeMillis',
            direction='nearest',
            tolerance=500   # 500 ms
        )

        df_merged.dropna(inplace=True)
        if df_merged.empty:
            print(f"  ! No time overlap for {drive_id}/{phone_id}")
            print(f"    GT time range: {df_gt['UnixTimeMillis'].min()} to {df_gt['UnixTimeMillis'].max()}")
            print(f"    POS time range: {df_pos['UnixTimeMillis'].min()} to {df_pos['UnixTimeMillis'].max()}")
            continue

        df_merged_filtered = df_merged.copy() 
        
        # Calculate horizontal error on the merged data
        df_merged_filtered['error_m'] = haversine_m(
            df_merged_filtered['LatitudeDegrees'], df_merged_filtered['LongitudeDegrees'],
            df_merged_filtered['latitude(deg)'], df_merged_filtered['longitude(deg)']
        )

        errors = df_merged_filtered['error_m']
        
        all_errors.extend(errors.tolist())
        num_processed += 1

        # Per-file stats 
        mean_err = errors.mean()
        median_err = errors.median()
        p95_err = errors.quantile(0.95)
        max_err = errors.max()
        std_err = errors.std()
        count = len(errors)
        avg_q = df_merged_filtered['Q'].mean() 

        print(f"--- RESULTS: {drive_id}/{phone_id} ({count} epochs) ---")
        print(f"  Mean Error:   {mean_err:.2f} m")
        print(f"  Median:       {median_err:.2f} m")
        print(f"  95th pct:     {p95_err:.2f} m")
        print(f"  Avg Q (All):  {avg_q:.2f}")

        file_stats.append({
            'drive_id': drive_id,
            'phone_id': phone_id,
            'mean_err': mean_err,
            'median_err': median_err,
            'p95_err': p95_err,
            'max_err': max_err,
            'std_err': std_err,
            'count': count,
            'avg_q': avg_q,
            'errors': errors 
        })

    # 4. Final Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY (All Solutions Combined)")
    print("=" * 80)
    
    global_upper_fence = np.nan

    if all_errors:
        all_errors = pd.Series(all_errors)
        
        # Global Error Score Calculation (Kaggle Metric)
        median_err_global = all_errors.median()
        p95_err_global = all_errors.quantile(0.95)
        score = (median_err_global + p95_err_global) / 2
        
        print(f"  Total Timestamps Processed: {len(all_errors)}")
        print(f"  Median (50%) Error:     {median_err_global:.3f} m")
        print(f"  95th Pct Error:         {p95_err_global:.3f} m")
        print(f"--------------------------------------------------")
        print(f"  KAGGLE SCORE (Mean of 50th & 95th): {score:.3f} m")
        print(f"--------------------------------------------------")
        
        # Outlier Calculation
        print("  Global Outlier Stats (IQR Method):")
        
        Q1 = all_errors.quantile(0.25)
        Q3 = all_errors.quantile(0.75)
        IQR = Q3 - Q1
        global_upper_fence = Q3 + (1.5 * IQR)
        
        outliers = all_errors[all_errors > global_upper_fence]
        
        print(f"  Outlier Fence >         {global_upper_fence:.2f} m")
        print(f"  Outlier Count:          {len(outliers)} ({(len(outliers) / len(all_errors) * 100):.1f} % of total)")

    else:
        print("No files were successfully compared.")
    
    # 5. Ranking files by error (worst → best)
    if file_stats:
        print("\n" + "=" * 80)
        print("FILES RANKED BY MEAN HORIZONTAL ERROR (worst first)")
        print("=" * 80)

        # Calculate per-file outliers using the global fence
        for stats_dict in file_stats:
            errors_series = stats_dict.pop('errors')
            if pd.notna(global_upper_fence):
                stats_dict['outlier_count'] = (errors_series > global_upper_fence).sum()
            else:
                stats_dict['outlier_count'] = 0
        
        ranking = sorted(file_stats, key=lambda x: x['mean_err'], reverse=True)

        print(f"{'Rank':>4}  {'Drive':<25} {'Phone':<20} {'MeanErr(m)':>10} {'P95(m)':>10} {'Max(m)':>10} {'Outliers':>8} {'Count':>8} {'AvgQ':>6}")
        print("-" * 110)
        for idx, r in enumerate(ranking, start=1):
            print(f"{idx:>4}  {r['drive_id']:<25} {r['phone_id']:<20} "
                  f"{r['mean_err']:>10.2f} {r['p95_err']:>10.2f} {r['max_err']:>10.2f} "
                  f"{r['outlier_count']:>8} {r['count']:>8} {r['avg_q']:>6.2f}")

        # Save ranking to CSV
        ranking_df = pd.DataFrame(ranking)
        ranking_csv = ROOT_DIR / "pos_error_ranking.csv"
        try:
            ranking_df.to_csv(ranking_csv, index=False)
            print(f"\nRanking saved to: {ranking_csv}")
        except Exception as e:
            print(f"\n  ✗ ERROR saving ranking CSV: {e}")


if __name__ == "__main__":
    main()