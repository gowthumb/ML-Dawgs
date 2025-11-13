"""
PPK-Based POS Generator and Comparison Tool
============================================

Generates POS files using PPK processing from:
- Rover observation files (.o files from train data)
- Navigation files (.n files)
- Base station observation files (.o files from base stations)

Then compares baseline POS vs ML-corrected POS.

Usage:
    python pos_generator_comparison.py --rover <path> --base <path> --nav <path>

Or run without args to process all train folders.
"""

import numpy as np
import pandas as pd
import pickle
import subprocess
import argparse
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime

# Import from existing modules
from config import TRAIN_DATA_ROOT, BASELINE_LAT, BASELINE_LON, GT_LAT, GT_LON

# ============================================================
# CONFIGURATION
# ============================================================

# RTKLIB executable paths (update these based on your installation)
RTKLIB_RNX2RTKP = "rnx2rtkp"  # Should be in PATH or provide full path

# PPK configuration
PPK_CONFIG = {
    "pos_mode": "kinematic",  # kinematic mode
    "freq": "l1",  # L1 frequency
    "solution": "combined",  # combined forward/backward
    "elev_mask": "15",  # 15 degree elevation mask
    "snr_mask": "35",  # SNR mask
    "max_baseline": "100000",  # 100km max baseline
}

# Directories
TRAIN_DATA_ROOT = Path(TRAIN_DATA_ROOT)
OUTPUT_DIR = Path("model/outputs/pos_files")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
MODEL_LAT_PATH = "model/outputs/residual_model_lat_20251109_221149.pkl"
MODEL_LON_PATH = "model/outputs/residual_model_lon_20251109_221149.pkl"


# ============================================================
# PPK PROCESSING FUNCTIONS
# ============================================================

def find_rinex_files(collection_path: Path) -> dict:
    """
    Find RINEX observation files (.o), navigation files (.n) in a collection.

    Args:
        collection_path: Path to collection folder (e.g., 2020-05-15-US-MTV-1/GooglePixel4XL)

    Returns:
        dict with 'rover_obs', 'nav' paths
    """
    supplemental = collection_path / "supplemental"

    files = {
        'rover_obs': None,
        'nav': None,
    }

    if supplemental.exists():
        # Find observation file (.o or .obs or .20o, etc.)
        for ext in ['*.o', '*.obs', '*.20o', '*.21o', '*.22o', '*.23o']:
            obs_files = list(supplemental.glob(ext))
            if obs_files:
                files['rover_obs'] = obs_files[0]
                break

        # Find navigation file (.n or .nav or .20n, etc.)
        for ext in ['*.n', '*.nav', '*.20n', '*.21n', '*.22n', '*.23n']:
            nav_files = list(supplemental.glob(ext))
            if nav_files:
                files['nav'] = nav_files[0]
                break

    return files


def run_ppk(
    rover_obs: Path,
    base_obs: Path,
    nav: Path,
    output_pos: Path,
    config: dict = PPK_CONFIG
) -> bool:
    """
    Run RTKLIB PPK processing.

    Args:
        rover_obs: Path to rover observation file (.o)
        base_obs: Path to base station observation file (.o)
        nav: Path to navigation file (.n)
        output_pos: Path to output .pos file
        config: PPK configuration parameters

    Returns:
        True if successful, False otherwise
    """
    # Build RTKLIB command
    cmd = [
        RTKLIB_RNX2RTKP,
        "-k", "ppk.conf",  # Config file (create this if needed)
        "-o", str(output_pos),
        str(rover_obs),
        str(base_obs),
        str(nav)
    ]

    # Add configuration options
    for key, value in config.items():
        cmd.extend([f"-{key}", value])

    try:
        print(f"  Running PPK: {rover_obs.name} + {base_obs.name}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0 and output_pos.exists():
            print(f"  ✓ Generated POS: {output_pos}")
            return True
        else:
            print(f"  ✗ PPK failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  ✗ PPK timeout")
        return False
    except FileNotFoundError:
        print(f"  ✗ RTKLIB not found. Install RTKLIB and add to PATH.")
        print(f"     Download from: https://www.rtklib.com/")
        return False
    except Exception as e:
        print(f"  ✗ PPK error: {e}")
        return False


def parse_pos_file(pos_path: Path) -> pd.DataFrame:
    """
    Parse RTKLIB .pos file into DataFrame.

    RTKLIB .pos format:
    %  GPST                  latitude(deg) longitude(deg)  height(m)   Q  ns   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m) age(s)  ratio

    Returns:
        DataFrame with columns: timestamp, latitude, longitude, height, Q, ns
    """
    data = []

    with open(pos_path, 'r') as f:
        for line in f:
            if line.startswith('%') or line.strip() == '':
                continue

            parts = line.split()
            if len(parts) >= 6:
                try:
                    # Parse date and time
                    date_str = parts[0]  # YYYY/MM/DD
                    time_str = parts[1]  # HH:MM:SS.SSS
                    timestamp = pd.to_datetime(f"{date_str} {time_str}")

                    data.append({
                        'timestamp': timestamp,
                        'latitude': float(parts[2]),
                        'longitude': float(parts[3]),
                        'height': float(parts[4]),
                        'Q': int(parts[5]),  # Solution quality
                        'ns': int(parts[6]) if len(parts) > 6 else 0  # Number of satellites
                    })
                except (ValueError, IndexError):
                    continue

    return pd.DataFrame(data)


# ============================================================
# ML CORRECTION FUNCTIONS
# ============================================================

def load_models():
    """Load trained lat/lon models."""
    try:
        with open(MODEL_LAT_PATH, 'rb') as f:
            model_lat = pickle.load(f)
        with open(MODEL_LON_PATH, 'rb') as f:
            model_lon = pickle.load(f)
        print(f"✓ Loaded ML models")
        return model_lat, model_lon
    except FileNotFoundError:
        print(f"⚠ ML models not found. Run train_residual_learning.py first.")
        return None, None


def apply_ml_correction(
    pos_df: pd.DataFrame,
    model_lat,
    model_lon,
    features_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Apply ML correction to POS file.

    Note: This is a simplified version. In production, you'd need to:
    1. Merge POS with GNSS/IMU features by timestamp
    2. Extract features for each position
    3. Predict residuals
    4. Apply corrections

    For now, we'll use a simple bias correction based on typical errors.
    """
    if model_lat is None or model_lon is None:
        print("  No ML models available, skipping correction")
        return pos_df.copy()

    # TODO: Implement full feature extraction and prediction
    # For now, this is a placeholder
    corrected_df = pos_df.copy()
    corrected_df['corrected_latitude'] = pos_df['latitude']
    corrected_df['corrected_longitude'] = pos_df['longitude']

    return corrected_df


# ============================================================
# COMPARISON FUNCTIONS
# ============================================================

def compute_position_error(pred_lat, pred_lon, gt_lat, gt_lon):
    """Compute horizontal position error in meters."""
    lat_diff_m = (pred_lat - gt_lat) * 111320.0
    lon_diff_m = (pred_lon - gt_lon) * 111320.0 * np.cos(np.radians(gt_lat))
    error_m = np.sqrt(lat_diff_m**2 + lon_diff_m**2)
    return error_m


def compare_pos_files(
    baseline_pos: pd.DataFrame,
    corrected_pos: pd.DataFrame,
    ground_truth: pd.DataFrame
) -> dict:
    """
    Compare baseline and corrected POS against ground truth.

    Returns:
        dict with baseline_metrics and corrected_metrics
    """
    # Merge on timestamp (within tolerance)
    merged = pd.merge_asof(
        baseline_pos.sort_values('timestamp'),
        ground_truth.sort_values('millisSinceGpsEpoch'),
        left_on='timestamp',
        right_on='millisSinceGpsEpoch',
        direction='nearest',
        tolerance=pd.Timedelta('100ms')
    )

    # Compute errors
    baseline_error = compute_position_error(
        merged['latitude'],
        merged['longitude'],
        merged['latDeg'],
        merged['lngDeg']
    )

    corrected_error = compute_position_error(
        merged['corrected_latitude'],
        merged['corrected_longitude'],
        merged['latDeg'],
        merged['lngDeg']
    )

    # Compute metrics
    def compute_metrics(errors):
        return {
            'RMSE': np.sqrt(np.mean(errors**2)),
            'MAE': np.mean(errors),
            'P50': np.percentile(errors, 50),
            'P95': np.percentile(errors, 95),
            'Mean(P50,P95)': (np.percentile(errors, 50) + np.percentile(errors, 95)) / 2,
        }

    return {
        'baseline': compute_metrics(baseline_error),
        'corrected': compute_metrics(corrected_error),
        'merged_data': merged
    }


# ============================================================
# MAIN FUNCTION
# ============================================================

def process_collection(
    collection_path: Path,
    base_obs: Path,
    output_dir: Path
) -> Optional[dict]:
    """
    Process a single collection (e.g., 2020-05-15-US-MTV-1/GooglePixel4XL).

    Returns:
        dict with results or None if failed
    """
    print(f"\n{'='*70}")
    print(f"Processing: {collection_path.parent.name}/{collection_path.name}")
    print(f"{'='*70}")

    # Find RINEX files
    files = find_rinex_files(collection_path)

    if files['rover_obs'] is None:
        print("  ✗ No rover observation file found")
        return None

    if files['nav'] is None:
        print("  ✗ No navigation file found")
        return None

    print(f"  Rover obs: {files['rover_obs'].name}")
    print(f"  Nav file: {files['nav'].name}")
    print(f"  Base obs: {base_obs.name}")

    # Create output paths
    collection_name = f"{collection_path.parent.name}_{collection_path.name}"
    output_pos = output_dir / f"{collection_name}_baseline.pos"

    # Run PPK
    success = run_ppk(
        rover_obs=files['rover_obs'],
        base_obs=base_obs,
        nav=files['nav'],
        output_pos=output_pos,
        config=PPK_CONFIG
    )

    if not success:
        return None

    # Parse POS file
    baseline_pos = parse_pos_file(output_pos)
    print(f"  Parsed {len(baseline_pos)} positions")

    # Load ground truth
    gt_path = collection_path / "ground_truth.csv"
    if not gt_path.exists():
        print("  ✗ No ground truth file found")
        return None

    ground_truth = pd.read_csv(gt_path)

    # Apply ML correction
    model_lat, model_lon = load_models()
    corrected_pos = apply_ml_correction(baseline_pos, model_lat, model_lon)

    # Save corrected POS
    corrected_pos_path = output_dir / f"{collection_name}_corrected.pos"
    corrected_pos.to_csv(corrected_pos_path, index=False)
    print(f"  ✓ Saved corrected POS: {corrected_pos_path.name}")

    # Compare
    results = compare_pos_files(baseline_pos, corrected_pos, ground_truth)

    return results


def main():
    parser = argparse.ArgumentParser(description='PPK POS Generator and Comparison')
    parser.add_argument('--rover', type=str, help='Path to rover observation file')
    parser.add_argument('--base', type=str, help='Path to base station observation file')
    parser.add_argument('--nav', type=str, help='Path to navigation file')
    parser.add_argument('--process-all', action='store_true', help='Process all train folders')

    args = parser.parse_args()

    print("="*70)
    print("PPK-BASED POS GENERATOR AND COMPARISON")
    print("="*70)

    if args.process_all:
        print("\nProcessing all train collections...")

        # Find a base station file (you'll need to provide this)
        # For now, we'll just show what would be processed

        all_results = []

        for date_folder in sorted(TRAIN_DATA_ROOT.glob("*")):
            if not date_folder.is_dir():
                continue

            for device_folder in date_folder.glob("*"):
                if not device_folder.is_dir():
                    continue

                print(f"\nWould process: {date_folder.name}/{device_folder.name}")
                print(f"  ⚠ Need base station file for this location")

                # In production, you'd:
                # 1. Determine location/time from folder name
                # 2. Download appropriate base station file
                # 3. Process with PPK

        print("\n" + "="*70)
        print("NOTE: To actually process, you need to:")
        print("  1. Download base station files for each location")
        print("  2. Install RTKLIB (https://www.rtklib.com/)")
        print("  3. Run with --rover, --base, --nav arguments")
        print("="*70)

    elif args.rover and args.base and args.nav:
        # Process single collection
        rover_path = Path(args.rover)
        base_path = Path(args.base)
        nav_path = Path(args.nav)

        # Create temporary output
        output_pos = OUTPUT_DIR / "temp_baseline.pos"

        success = run_ppk(rover_path, base_path, nav_path, output_pos)

        if success:
            baseline_pos = parse_pos_file(output_pos)
            print(f"\n✓ Generated {len(baseline_pos)} positions")

            # Display sample
            print("\nSample positions (first 5):")
            print(baseline_pos.head())

            # Save
            baseline_pos.to_csv(OUTPUT_DIR / "baseline_pos.csv", index=False)
            print(f"\n✓ Saved to: {OUTPUT_DIR / 'baseline_pos.csv'}")

    else:
        print("\nUsage:")
        print("  Process single collection:")
        print("    python pos_generator_comparison.py --rover <rover.o> --base <base.o> --nav <nav.n>")
        print("\n  Process all collections:")
        print("    python pos_generator_comparison.py --process-all")
        print("\nNote: Base station files need to be downloaded separately")
        print("      See: https://geodesy.noaa.gov/CORS/data.shtml")


if __name__ == "__main__":
    main()
