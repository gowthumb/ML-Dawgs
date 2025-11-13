"""
Quick test to verify timestamp extraction and conversion works correctly
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Test timestamp extraction
DATA_PATH = Path("training_set_all_folders_ekf.csv")

print("Testing timestamp extraction and conversion...")
print("=" * 70)

# Read just the timestamp column to test
df = pd.read_csv(DATA_PATH, nrows=10)

if 'millisSinceGpsEpoch' in df.columns:
    print("[OK] millisSinceGpsEpoch column found")
    timestamps = df['millisSinceGpsEpoch'].values
    print(f"   Sample timestamps: {timestamps[:3]}")

    # Test conversion
    gps_epoch = pd.Timestamp('1980-01-06 00:00:00')  # Must be tz-naive
    timestamps_dt = pd.to_datetime(timestamps, unit='ms', origin=gps_epoch)

    # Convert to Series if DatetimeIndex
    if isinstance(timestamps_dt, pd.DatetimeIndex):
        timestamps_dt = pd.Series(timestamps_dt.values)

    print(f"   Converted to datetime: {timestamps_dt.iloc[0]}")
    print(f"   Type: {type(timestamps_dt)}")
    print(f"   Element type: {type(timestamps_dt.iloc[0])}")

    # Test time difference calculation
    if len(timestamps_dt) > 1:
        time_diff = timestamps_dt.iloc[1] - timestamps_dt.iloc[0]
        print(f"   Time diff between samples: {time_diff}")
        print(f"   Time diff type: {type(time_diff)}")
        if hasattr(time_diff, 'total_seconds'):
            dt = time_diff.total_seconds()
            print(f"   [OK] total_seconds() works: {dt}s")
        else:
            print(f"   [WARNING] total_seconds() not available")

    print("\n[OK] Timestamp integration test PASSED")
else:
    print("[ERROR] millisSinceGpsEpoch column NOT found")
    print(f"   Available columns: {df.columns.tolist()[:10]}")

print("=" * 70)
