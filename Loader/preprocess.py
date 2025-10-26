import pandas as pd
import numpy as np
from io import StringIO


def read_gnss_log(file_path):
    """
    Read GNSS log file including Raw, Status, and IMU data
    """
    print(f"Reading: {file_path}")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Dictionary to store different data types
    data_dict = {}
    headers_dict = {}
    
    for line in lines:
        line = line.strip()
        
        # Find headers (lines starting with #)
        if line.startswith('#'):
            # Extract the data type (e.g., "Raw", "Status", "UncalAccel")
            parts = line.lstrip('#').strip().split(',')
            if len(parts) > 0:
                data_type = parts[0].lower()
                headers_dict[data_type] = line.lstrip('#').strip()
        
        # Collect data lines
        elif ',' in line:
            parts = line.split(',')
            if len(parts) > 0:
                data_type = parts[0].lower()
                if data_type not in data_dict:
                    data_dict[data_type] = []
                data_dict[data_type].append(line)
    
    # Convert to DataFrames
    result = {}
    
    for data_type, data_lines in data_dict.items():
        if data_type in headers_dict:
            header = headers_dict[data_type]
            csv_text = '\n'.join([header] + data_lines)
            try:
                df = pd.read_csv(StringIO(csv_text), low_memory=False)
                result[data_type] = df
                print(f"✓ {data_type}: {df.shape}")
            except Exception as e:
                print(f"⚠️ Error loading {data_type}: {e}")
    
    return result


def apply_sanity_filters(df, data_type='raw'):
    """
    Remove measurements that are clearly wrong
    """
    print(f"\n=== Cleaning {data_type.upper()} data ===")
    print(f"Starting: {len(df)} rows")
    
    df = df.copy()
    
    # Convert important columns to numbers
    numeric_cols = ['Svid', 'Cn0DbHz', 'ConstellationType']
    if data_type == 'status':
        numeric_cols += ['Elevation', 'Azimuth', 'UnixTimeMillis']
    elif data_type == 'raw':
        numeric_cols += ['State', 'TimeNanos']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter 1: Must have satellite ID and signal strength
    df = df.dropna(subset=['Svid', 'Cn0DbHz'])
    print(f"After removing NaNs: {len(df)}")
    
    # Filter 2: Signal strength must be reasonable
    df = df[(df['Cn0DbHz'] > 10) & (df['Cn0DbHz'] < 60)]
    print(f"After CN0 filter (10-60): {len(df)}")
    
    # Filter 3: Only GPS and Galileo (most reliable)
    if 'ConstellationType' in df.columns:
        df = df[df['ConstellationType'].isin([1, 6])]
        print(f"After constellation filter: {len(df)}")
    
    # Filter 4: Type-specific filters
    if data_type == 'raw' and 'State' in df.columns:
        df = df[df['State'] >= 8]
        print(f"After State filter: {len(df)}")
    
    if data_type == 'status':
        if 'Elevation' in df.columns:
            df = df[(df['Elevation'] >= 0) & (df['Elevation'] <= 90)]
            print(f"After elevation filter: {len(df)}")
        if 'Azimuth' in df.columns:
            df = df[(df['Azimuth'] >= 0) & (df['Azimuth'] <= 360)]
            print(f"After azimuth filter: {len(df)}")
    
    print(f"Final: {len(df)} rows\n")
    return df


def convert_to_gps_time(df, data_type='raw'):
    """
    Convert timestamps to GPS time (milliseconds since GPS epoch)
    
    GPS epoch: January 6, 1980 00:00:00 UTC
    """
    print(f"=== Converting {data_type.upper()} to GPS time ===")
    
    df = df.copy()
    
    if data_type == 'raw':
        # TimeNanos is already in GPS time (nanoseconds since GPS epoch)
        # Just convert to milliseconds
        if 'TimeNanos' in df.columns:
            df['millisSinceGpsEpoch'] = (df['TimeNanos'] / 1e6).astype(np.int64)
            print(f"Raw time range: {df['millisSinceGpsEpoch'].min()} to {df['millisSinceGpsEpoch'].max()}")
    
    elif data_type == 'status':
        # UnixTimeMillis is in Unix time (milliseconds since Jan 1, 1970)
        # GPS epoch is 315964800 seconds = 315964800000 milliseconds before Unix epoch
        if 'UnixTimeMillis' in df.columns:
            GPS_TO_UNIX_OFFSET_MS = 315964800000
            df['millisSinceGpsEpoch'] = (df['UnixTimeMillis'] - GPS_TO_UNIX_OFFSET_MS).astype(np.int64)
            print(f"Status time range: {df['millisSinceGpsEpoch'].min()} to {df['millisSinceGpsEpoch'].max()}")
    
    return df


def synchronize_time(df, data_type='raw', max_gap_ms=2000):
    """
    Remove measurements with large time gaps
    """
    print(f"=== Time sync for {data_type.upper()} ===")
    print(f"Starting: {len(df)} rows")
    
    # Use the GPS time column we just created
    time_col = 'millisSinceGpsEpoch'
    
    if time_col not in df.columns:
        print(f"WARNING: {time_col} not found, skipping time sync")
        return df
    
    # Sort by satellite and time
    df = df.sort_values(['Svid', time_col]).reset_index(drop=True)
    
    # Calculate time difference between consecutive measurements per satellite
    df['time_diff'] = df.groupby('Svid')[time_col].diff()
    
    # Remove measurements after large gaps
    initial_len = len(df)
    df = df[(df['time_diff'] < max_gap_ms) | df['time_diff'].isna()]
    df = df.drop(columns=['time_diff'])
    print(f"Removed {initial_len - len(df)} rows with gaps > {max_gap_ms}ms")
    
    # Remove duplicate timestamps for same satellite
    df = df.drop_duplicates(subset=[time_col, 'Svid'], keep='first')
    print(f"Final: {len(df)} rows\n")
    
    return df


def detect_cycle_slips(df):
    """
    Flag potential cycle slips in carrier phase measurements
    """
    print("=== Detecting cycle slips ===")
    
    df = df.copy()
    df['cycle_slip_flag'] = False
    
    # Find the carrier phase column
    adr_col = None
    for col in df.columns:
        if 'AccumulatedDeltaRange' in col and 'Uncertainty' not in col:
            adr_col = col
            break
    
    if adr_col is None:
        print("No carrier phase data found - skipping\n")
        return df
    
    # Convert to numeric
    df[adr_col] = pd.to_numeric(df[adr_col], errors='coerce')
    
    # Sort by satellite and time (use GPS time if available)
    time_col = 'millisSinceGpsEpoch' if 'millisSinceGpsEpoch' in df.columns else 'TimeNanos'
    df = df.sort_values(['Svid', time_col]).reset_index(drop=True)
    
    # Calculate change in carrier phase per satellite
    df['adr_diff'] = df.groupby('Svid')[adr_col].diff()
    
    # Flag jumps larger than 10 meters
    threshold = 10.0
    df.loc[df['adr_diff'].abs() > threshold, 'cycle_slip_flag'] = True
    
    slip_count = df['cycle_slip_flag'].sum()
    print(f"Flagged {slip_count} cycle slips (|Δphase| > {threshold}m)\n")
    
    df = df.drop(columns=['adr_diff'])
    return df


def preprocess_pipeline(file_path, output_path=None):
    """
    Run complete preprocessing pipeline
    
    Steps:
    1. Read GNSS log file (including IMU data)
    2. Clean Raw data (filters + GPS time conversion + time sync + cycle slip detection)
    3. Clean Status data (filters + GPS time conversion + time sync)
    4. Save to CSV if output_path provided
    
    Returns: dict with 'raw', 'status', and IMU DataFrames
    """
    print("\n" + "="*60)
    print("GNSS PREPROCESSING PIPELINE")
    print("="*60 + "\n")
    
    # Step 1: Read ALL data (including IMU)
    data = read_gnss_log(file_path)
    result = {}
    
    # Step 2: Process Raw data
    if 'raw' in data:
        raw = data['raw']
        raw = apply_sanity_filters(raw, data_type='raw')
        raw = convert_to_gps_time(raw, data_type='raw')
        raw = synchronize_time(raw, data_type='raw')
        raw = detect_cycle_slips(raw)
        result['raw'] = raw
        print(f"✓ Raw preprocessing complete: {len(raw)} rows")
    
    # Step 3: Process Status data
    if 'status' in data:
        status = data['status']
        status = apply_sanity_filters(status, data_type='status')
        status = convert_to_gps_time(status, data_type='status')
        status = synchronize_time(status, data_type='status')
        result['status'] = status
        print(f"✓ Status preprocessing complete: {len(status)} rows")
    
    # Step 4: Keep IMU data as-is (no filtering needed)
    imu_types = ['uncalaccel', 'uncalgyro', 'accel', 'gyro', 'uncalmag', 'mag', 'orientdeg']
    for imu_type in imu_types:
        if imu_type in data:
            result[imu_type] = data[imu_type]
            print(f"✓ {imu_type.upper()}: {len(data[imu_type])} rows")
    
    print("="*60)
    
    # Check if times align
    if 'raw' in result and 'status' in result:
        raw_time_range = (result['raw']['millisSinceGpsEpoch'].min(), 
                         result['raw']['millisSinceGpsEpoch'].max())
        status_time_range = (result['status']['millisSinceGpsEpoch'].min(),
                            result['status']['millisSinceGpsEpoch'].max())
        
        print(f"\nTime alignment check:")
        print(f"  Raw:    {raw_time_range[0]} to {raw_time_range[1]}")
        print(f"  Status: {status_time_range[0]} to {status_time_range[1]}")
        
        # Check for overlap
        overlap = not (raw_time_range[1] < status_time_range[0] or 
                      status_time_range[1] < raw_time_range[0])
        
        if overlap:
            print(f"  ✓ Times overlap - ready for merging!")
        else:
            print(f"  ⚠️ WARNING: No time overlap detected!")
    
    # Step 5: Save
    if output_path:
        for data_type, df in result.items():
            output_file = f'{output_path}_{data_type}.csv'
            df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")
    
    print("\n" + "="*60 + "\n")
    return result


# Example usage
if __name__ == "__main__":
    # CRITICAL FIX: Updated from hardcoded Windows path to correct Mac path
    # Original: "C:\\Users\\avnee\\Downloads\\google-smartphone-decimeter-challenge\\train\\2021-04-15-US-MTV-1\\Pixel4\\Pixel4_GnssLog.txt"
    # Fixed: Updated to point to actual smartphone-decimeter-2022 dataset location
    file_path = "/Users/yash.rayapaty/Downloads/smartphone-decimeter-2022/train/2020-12-10-US-SJC-1/GooglePixel4/supplemental/gnss_log.txt"
    
    # Run preprocessing
    result = preprocess_pipeline(file_path, output_path='./processed_data')
    
    # Print summary
    print("SUMMARY:")
    for data_type, df in result.items():
        print(f"\n{data_type.upper()}:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        if 'Svid' in df.columns:
            print(f"  Satellites: {df['Svid'].nunique()}")
        if 'millisSinceGpsEpoch' in df.columns:
            print(f"  Time range: {df['millisSinceGpsEpoch'].min()} to {df['millisSinceGpsEpoch'].max()}")
        elif 'utcTimeMillis' in df.columns:
            print(f"  Time range: {df['utcTimeMillis'].min()} to {df['utcTimeMillis'].max()}")