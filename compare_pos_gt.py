#!/usr/bin/env python3
"""
GSDC Ground Truth vs. .POS File Comparator (Upgraded)

This script implements advanced analysis:
1.  Filters errors by Q-value (Q <= 2 vs. Q > 2)
2.  Segments the final summary by region (Bay Area vs. SoCal)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from io import StringIO
import re

# --- CONFIGURE YOUR PATHS HERE ---
GT_ROOT_DIR = Path(r'C:\Users\gauth\Documents\SC4000_proj\train')
POS_ROOT_DIR = Path(r'C:\Users\gauth\Documents\SC4000_proj\ppk_output_v3')
# ---------------------------------

# This offset is used by the GSDC to convert UnixTimeMillis to millisSinceGpsEpoch
GPS_EPOCH_OFFSET_MILLIS = 315964800000

def load_pos_file(pos_path):
    """
    Loads a .pos file, skipping the header and parsing the data columns.
    """
    try:
        with open(pos_path, 'r') as f:
            lines = f.readlines()
        
        # Filter out header lines (starting with '%')
        data_lines = [line for line in lines if not line.strip().startswith('%')]
        if not data_lines:
            print(f"  X {pos_path.name} is empty or all comments.")
            return pd.DataFrame()
            
        # Re-join data lines and read into pandas
        data_io = StringIO("".join(data_lines))
        
        # Use a regex for one-or-more-spaces as delimiter
        # Read all columns as STRINGS first to prevent pandas from auto-typing
        df = pd.read_csv(data_io, sep=r'\s+', header=None, dtype=str)

        # Handle the two different GPST formats
        # Format 1: 2020/05/15 20:14:57.446 (2 columns)
        # Format 2: 2105 504897.446 (2 columns, GPS Week + TimeOfWeek)
        
        if df.iloc[0, 0].count('/') == 2:
            # This is the old format
            df['datetime_str'] = df[0] + ' ' + df[1]
            df['datetime'] = pd.to_datetime(df['datetime_str'], format='%Y/%m/%d %H:%M:%S.%f', errors='coerce')
            
            # Convert datetime to millisSinceGpsEpoch to match GT
            GPS_EPOCH = datetime(1980, 1, 6)
            df['millisSinceGpsEpoch'] = (df['datetime'] - GPS_EPOCH).dt.total_seconds() * 1000
            
            # Assign columns
            df['latitude(deg)'] = pd.to_numeric(df[2])
            df['longitude(deg)'] = pd.to_numeric(df[3])
            df['Q'] = pd.to_numeric(df[5])

        elif len(df.iloc[0, 0]) == 4: 
            # This is the new (correct) GPS Week format
            df_week = pd.to_numeric(df[0])
            df_tow = pd.to_numeric(df[1])
            
            # Convert GPST week/tow to millisSinceGpsEpoch
            # (Week * 7 days/week * 86400 sec/day + TOW_seconds) * 1000 ms/sec
            df['millisSinceGpsEpoch'] = ((df_week * 7 * 86400) + df_tow) * 1000
            
            # Assign columns
            # [0]Week [1]TOW [2]lat [3]lon [4]height [5]Q
            df['latitude(deg)'] = pd.to_numeric(df[2])
            df['longitude(deg)'] = pd.to_numeric(df[3])
            df['Q'] = pd.to_numeric(df[5])
        
        else:
            print(f"  X {pos_path.name} has unknown time format: {df.iloc[0, 0]}")
            return pd.DataFrame()
            
        df['millisSinceGpsEpoch'] = df['millisSinceGpsEpoch'].astype(np.int64)
        
        return df[['millisSinceGpsEpoch', 'latitude(deg)', 'longitude(deg)', 'Q']]
        
    except Exception as e:
        print(f"  X ERROR reading {pos_path.name}: {e}")
        return pd.DataFrame()

def load_gt_file(gt_path):
    """
    Loads a ground_truth.csv file and converts its time.
    """
    try:
        df = pd.read_csv(gt_path)
        
        # Convert UnixTimeMillis to millisSinceGpsEpoch
        df['millisSinceGpsEpoch'] = df['UnixTimeMillis'] - GPS_EPOCH_OFFSET_MILLIS
        
        return df[['millisSinceGpsEpoch', 'LatitudeDegrees', 'LongitudeDegrees']]
    except Exception as e:
        print(f"  X ERROR reading {gt_path.name}: {e}")
        return pd.DataFrame()

def haversine_m(lat1, lon1, lat2, lon2):
    """
    Calculate the horizontal distance between two points
    on Earth (specified in decimal degrees) in meters.
    """
    R = 6371000  # Earth radius in meters
    
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    return distance

def print_summary(name, errors_list):
    """Helper function to print a formatted summary."""
    if not errors_list:
        print(f"  No data for {name}")
        return

    errors = pd.Series(errors_list)
    print(f"\n  --- {name} ---")
    print(f"  Total Timestamps: {len(errors)}")
    print(f"  Mean Error:       {errors.mean():.2f} m")
    print(f"  Median (50%) Error: {errors.median():.2f} m")
    print(f"  95th Pct Error:   {errors.quantile(0.95):.2f} m")
    print(f"  Max Error:        {errors.max():.2f} m")


def main():
    print(f"Scanning for Ground Truth in: {GT_ROOT_DIR}")
    print(f"Scanning for .pos files in: {POS_ROOT_DIR}")
    
    # 1. Find all files
    gt_files = list(GT_ROOT_DIR.rglob('ground_truth.csv'))
    pos_files = list(POS_ROOT_DIR.rglob('gnss_rinex.pos'))
    
    # 2. Create lookups for matching
    gt_lookup = {}
    for gt_path in gt_files:
        phone_id = gt_path.parent.name
        drive_id = gt_path.parent.parent.name
        gt_lookup[(drive_id, phone_id)] = gt_path
        
    pos_lookup = {}
    for pos_path in pos_files:
        # Folder name is like "2020-05-15-US-MTV-1-GooglePixel4XL"
        folder_name = pos_path.parent.name
        try:
            # Reconstruct the drive_id and phone_id
            parts = folder_name.split('-')
            phone_id = parts[-1]
            drive_id = "-".join(parts[:-1])
            pos_lookup[(drive_id, phone_id)] = pos_path
        except Exception:
            print(f"  ! Skipping folder with unusual name: {folder_name}")
            
    print(f"\nFound {len(gt_lookup)} ground_truth.csv files.")
    print(f"Found {len(pos_lookup)} gnss_rinex.pos files.")
    
    # 3. Iterate, match, and compare
    
    # Create lists to segment results by dataset
    all_errors_bay_area_good_q = []
    all_errors_bay_area_bad_q = []
    all_errors_socal_good_q = []
    all_errors_socal_bad_q = []
    all_errors_other_good_q = []
    all_errors_other_bad_q = []
    
    print("\n" + "="*80)
    print("STARTING COMPARISON")
    print("="*80)
    
    for (drive_id, phone_id), gt_path in sorted(gt_lookup.items()):
        key = (drive_id, phone_id)
        
        if key not in pos_lookup:
            continue
            
        pos_path = pos_lookup[key]
        
        # Load the files
        df_gt = load_gt_file(gt_path)
        df_pos = load_pos_file(pos_path)
        
        if df_gt.empty or df_pos.empty:
            print(f"  ! Skipping {drive_id}/{phone_id} (could not load data)")
            continue
            
        # Merge the two dataframes on the nearest timestamp
        df_merged = pd.merge_asof(
            df_gt.sort_values('millisSinceGpsEpoch'),
            df_pos.sort_values('millisSinceGpsEpoch'),
            on='millisSinceGpsEpoch',
            direction='nearest',
            tolerance=500  # 500ms = 0.5 sec
        )
        
        df_merged.dropna(inplace=True)
        if df_merged.empty:
            print(f"  ! No time overlap for {drive_id}/{phone_id}")
            continue

        # --- IMPLEMENTING FIX (a): Filter by Q <= 2 ---
        df_good_q = df_merged[df_merged['Q'] <= 2].copy()
        df_bad_q = df_merged[df_merged['Q'] > 2].copy()

        good_errors = pd.Series(dtype=float)
        bad_errors = pd.Series(dtype=float)

        print(f"--- {drive_id}/{phone_id} ---")
        
        # Calculate errors for GOOD Q values
        if not df_good_q.empty:
            df_good_q['error_m'] = haversine_m(
                df_good_q['LatitudeDegrees'], df_good_q['LongitudeDegrees'],
                df_good_q['latitude(deg)'], df_good_q['longitude(deg)']
            )
            good_errors = df_good_q['error_m']
            print(f"  GOOD (Q<=2): Median Error: {good_errors.median():.2f} m | Max Error: {good_errors.max():.2f} m | Count: {len(good_errors)}")
        else:
            print("  GOOD (Q<=2): No data points")

        # Calculate errors for BAD Q values
        if not df_bad_q.empty:
            df_bad_q['error_m'] = haversine_m(
                df_bad_q['LatitudeDegrees'], df_bad_q['LongitudeDegrees'],
                df_bad_q['latitude(deg)'], df_bad_q['longitude(deg)']
            )
            bad_errors = df_bad_q['error_m']
            print(f"  BAD  (Q>2): Median Error: {bad_errors.median():.2f} m | Max Error: {bad_errors.max():.2f} m | Count: {len(bad_errors)}")
        else:
            print("  BAD  (Q>2): No data points")

        # --- IMPLEMENTING FIX (b): Segment by baseline ---
        if "MTV" in drive_id or "SJC" in drive_id or "SFO" in drive_id or "SVL" in drive_id:
            all_errors_bay_area_good_q.extend(good_errors)
            all_errors_bay_area_bad_q.extend(bad_errors)
        elif "LAX" in drive_id:
            all_errors_socal_good_q.extend(good_errors)
            all_errors_socal_bad_q.extend(bad_errors)
        else:
            all_errors_other_good_q.extend(good_errors)
            all_errors_other_bad_q.extend(bad_errors)


    # 4. Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY (Segmented by Dataset)")
    print("="*80)
    
    print_summary("Bay Area (MTV, SJC, SFO, SVL) - GOOD Q (<=2)", all_errors_bay_area_good_q)
    print_summary("Bay Area (MTV, SJC, SFO, SVL) - BAD Q (>2)", all_errors_bay_area_bad_q)
    
    print("\n" + "-"*40)
    
    print_summary("SoCal (LAX) - GOOD Q (<=2)", all_errors_socal_good_q)
    print_summary("SoCal (LAX) - BAD Q (>2)", all_errors_socal_bad_q)

    if all_errors_other_good_q:
        print("\n" + "-"*40)
        print_summary("Other Datasets - GOOD Q (<=2)", all_errors_other_good_q)
        print_summary("Other Datasets - BAD Q (>2)", all_errors_other_bad_q)
        
if __name__ == "__main__":
    main()