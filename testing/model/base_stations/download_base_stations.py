"""
CORS Base Station Downloader
=============================

Automatically finds and downloads the closest CORS base station RINEX files
for your train data locations (LAX, MTV, etc.)

This solves the "base station too far away" problem by:
1. Finding the nearest CORS stations to your data collection locations
2. Downloading RINEX observation and navigation files for the correct dates
3. Organizing them for PPK processing

Usage:
    python download_base_stations.py --location LAX --date 2021-12-07
    python download_base_stations.py --auto  # Process all train folders
"""

import sys
import io

# Fix Windows Unicode issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import gzip
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import math

# ============================================================
# CONFIGURATION
# ============================================================

# Known CORS stations near data collection sites
CORS_STATIONS = {
    # Los Angeles Area (LAX)
    'LAX': [
        {'id': 'LONG', 'lat': 33.9313, 'lon': -118.1755, 'name': 'Long Beach'},
        {'id': 'USC1', 'lat': 34.0223, 'lon': -118.2875, 'name': 'USC'},
        {'id': 'OXYC', 'lat': 34.1283, 'lon': -118.2119, 'name': 'Occidental College'},
        {'id': 'CLAR', 'lat': 34.1379, 'lon': -117.5992, 'name': 'Claremont'},
        {'id': 'HOLP', 'lat': 34.0092, 'lon': -118.3362, 'name': 'Hollywood'},
    ],
    # Mountain View Area (MTV) - Bay Area
    'MTV': [
        {'id': 'SUTB', 'lat': 37.3874, 'lon': -122.0527, 'name': 'Sunnyvale'},
        {'id': 'LUTZ', 'lat': 37.2547, 'lon': -121.8633, 'name': 'Santa Clara'},
        {'id': 'MHCB', 'lat': 37.3416, 'lon': -122.0483, 'name': 'Mountain View'},
        {'id': 'P222', 'lat': 37.4536, 'lon': -122.1818, 'name': 'Palo Alto'},
        {'id': 'UCSC', 'lat': 36.9927, 'lon': -122.0591, 'name': 'Santa Cruz'},
    ],
}

# NOAA CORS data server
NOAA_CORS_BASE = "https://noaa-cors-pds.s3.amazonaws.com"

# Output directory
BASE_STATION_DIR = Path("base_stations")
BASE_STATION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points in kilometers using Haversine formula.
    """
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def find_closest_station(target_lat: float, target_lon: float, area: str) -> Dict:
    """
    Find the closest CORS station to target coordinates.

    Args:
        target_lat: Target latitude
        target_lon: Target longitude
        area: Area code ('LAX', 'MTV')

    Returns:
        dict with station info and distance
    """
    stations = CORS_STATIONS.get(area, [])
    if not stations:
        print(f"  ⚠ No known stations for area: {area}")
        return None

    closest = None
    min_distance = float('inf')

    for station in stations:
        distance = haversine_distance(
            target_lat, target_lon,
            station['lat'], station['lon']
        )

        if distance < min_distance:
            min_distance = distance
            closest = station.copy()
            closest['distance_km'] = distance

    return closest


def date_to_gps_info(date_str: str) -> Tuple[str, int, int]:
    """
    Convert date string to GPS week and day of year.

    Args:
        date_str: Date in format 'YYYY-MM-DD'

    Returns:
        (year_short, day_of_year, gps_week)
    """
    date = datetime.strptime(date_str, '%Y-%m-%d')
    year_short = date.strftime('%y')
    day_of_year = date.timetuple().tm_yday

    # GPS week calculation (GPS epoch: Jan 6, 1980)
    gps_epoch = datetime(1980, 1, 6)
    days_since_epoch = (date - gps_epoch).days
    gps_week = days_since_epoch // 7

    return year_short, day_of_year, gps_week


def build_cors_url(station_id: str, date_str: str, file_type: str = 'obs') -> str:
    """
    Build NOAA CORS download URL.

    Args:
        station_id: 4-character station ID (e.g., 'LONG')
        date_str: Date in format 'YYYY-MM-DD'
        file_type: 'obs' for observation, 'nav' for navigation

    Returns:
        Download URL
    """
    year_short, doy, gps_week = date_to_gps_info(date_str)
    year_full = datetime.strptime(date_str, '%Y-%m-%d').year

    station_lower = station_id.lower()

    if file_type == 'obs':
        # Observation file format: rinex/YYYY/DDD/ssss/ssssDDD0.YYd.gz
        filename = f"{station_lower}{doy:03d}0.{year_short}d.gz"
        url = f"{NOAA_CORS_BASE}/rinex/{year_full}/{doy:03d}/{station_lower}/{filename}"
    elif file_type == 'nav':
        # Navigation file format: rinex/YYYY/DDD/ssss/ssssDDD0.YYn.gz
        filename = f"{station_lower}{doy:03d}0.{year_short}n.gz"
        url = f"{NOAA_CORS_BASE}/rinex/{year_full}/{doy:03d}/{station_lower}/{filename}"
    else:
        raise ValueError(f"Unknown file_type: {file_type}")

    return url


def download_file(url: str, output_path: Path, decompress: bool = True) -> bool:
    """
    Download and optionally decompress a file.

    Args:
        url: Download URL
        output_path: Output file path
        decompress: If True, decompress .gz files

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"  Downloading: {url}")
        response = requests.get(url, timeout=60)

        if response.status_code == 200:
            # Save compressed file
            temp_path = output_path.with_suffix(output_path.suffix + '.gz')
            with open(temp_path, 'wb') as f:
                f.write(response.content)

            # Decompress if needed
            if decompress and temp_path.suffix == '.gz':
                with gzip.open(temp_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                temp_path.unlink()  # Delete compressed file
                print(f"  ✓ Saved: {output_path.name}")
            else:
                temp_path.rename(output_path)
                print(f"  ✓ Saved: {output_path.name}")

            return True
        else:
            print(f"  ✗ Download failed: HTTP {response.status_code}")
            return False

    except requests.exceptions.Timeout:
        print(f"  ✗ Download timeout")
        return False
    except Exception as e:
        print(f"  ✗ Download error: {e}")
        return False


# ============================================================
# MAIN FUNCTIONS
# ============================================================

def download_base_station(station_id: str, date_str: str, output_dir: Path) -> Dict:
    """
    Download observation and navigation files for a base station.

    Args:
        station_id: 4-character station ID
        date_str: Date in format 'YYYY-MM-DD'
        output_dir: Output directory

    Returns:
        dict with paths to downloaded files
    """
    print(f"\nDownloading base station: {station_id} for {date_str}")

    year_short, doy, _ = date_to_gps_info(date_str)

    # Create output directory for this station/date
    station_dir = output_dir / station_id.upper() / date_str
    station_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    # Download observation file
    obs_url = build_cors_url(station_id, date_str, 'obs')
    obs_path = station_dir / f"{station_id.lower()}{doy:03d}0.{year_short}o"
    if download_file(obs_url, obs_path):
        files['obs'] = obs_path
    else:
        print(f"  ⚠ Observation file not available")

    # Download navigation file
    nav_url = build_cors_url(station_id, date_str, 'nav')
    nav_path = station_dir / f"{station_id.lower()}{doy:03d}0.{year_short}n"
    if download_file(nav_url, nav_path):
        files['nav'] = nav_path
    else:
        print(f"  ⚠ Navigation file not available")

    return files


def process_location(area: str, date_str: str, target_lat: float = None, target_lon: float = None):
    """
    Find closest station and download base station files for a location.
    Tries multiple stations until successful.

    Args:
        area: Area code ('LAX', 'MTV')
        date_str: Date in format 'YYYY-MM-DD'
        target_lat: Optional target latitude (if None, uses area center)
        target_lon: Optional target longitude (if None, uses area center)
    """
    print("="*70)
    print(f"PROCESSING: {area} on {date_str}")
    print("="*70)

    # If no target specified, use area center
    if target_lat is None or target_lon is None:
        if area == 'LAX':
            target_lat, target_lon = 34.2725, -118.6157
        elif area == 'MTV':
            target_lat, target_lon = 37.3861, -122.0839
        else:
            print(f"Unknown area: {area}")
            return

    # Get all stations for this area, sorted by distance
    stations = CORS_STATIONS.get(area, [])
    if not stations:
        print(f"  ✗ No known stations for area: {area}")
        return

    # Calculate distances and sort
    stations_with_dist = []
    for station in stations:
        distance = haversine_distance(
            target_lat, target_lon,
            station['lat'], station['lon']
        )
        station_copy = station.copy()
        station_copy['distance_km'] = distance
        stations_with_dist.append(station_copy)

    stations_with_dist.sort(key=lambda x: x['distance_km'])

    # Try each station until one succeeds
    for i, station in enumerate(stations_with_dist, 1):
        print(f"\nTrying station {i}/{len(stations_with_dist)}: {station['id']} - {station['name']}")
        print(f"  Distance: {station['distance_km']:.2f} km")
        print(f"  Location: {station['lat']:.4f}°N, {abs(station['lon']):.4f}°W")

        if station['distance_km'] > 50:
            print(f"  ⚠ WARNING: Station is {station['distance_km']:.1f}km away (>50km may affect accuracy)")

        # Download files
        files = download_base_station(station['id'], date_str, BASE_STATION_DIR)

        if files and len(files) > 0:
            print(f"\n✓ Success! Base station files ready:")
            for file_type, path in files.items():
                print(f"  {file_type}: {path}")
            return files

    print(f"\n✗ Failed to download from any station in {area}")


def main():
    parser = argparse.ArgumentParser(description='Download CORS base station files')
    parser.add_argument('--location', type=str, choices=['LAX', 'MTV'], help='Location code')
    parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format')
    parser.add_argument('--lat', type=float, help='Target latitude (optional)')
    parser.add_argument('--lon', type=float, help='Target longitude (optional)')
    parser.add_argument('--list-stations', action='store_true', help='List available stations')

    args = parser.parse_args()

    print("="*70)
    print("CORS BASE STATION DOWNLOADER")
    print("="*70)

    if args.list_stations:
        print("\nAvailable CORS Stations:")
        print("-"*70)
        for area, stations in CORS_STATIONS.items():
            print(f"\n{area}:")
            for station in stations:
                print(f"  {station['id']:4s} - {station['name']:20s} "
                      f"({station['lat']:.4f}°N, {station['lon']:.4f}°W)")
        return

    if args.location and args.date:
        process_location(args.location, args.date, args.lat, args.lon)

    else:
        print("\nQuick Start Examples:")
        print("-"*70)
        print("\n1. List available stations:")
        print("   python download_base_stations.py --list-stations")

        print("\n2. Download for LAX on Dec 7, 2021:")
        print("   python download_base_stations.py --location LAX --date 2021-12-07")

        print("\n3. Download for MTV on May 15, 2020:")
        print("   python download_base_stations.py --location MTV --date 2020-05-15")

        print("\n4. Use custom coordinates:")
        print("   python download_base_stations.py --location LAX --date 2021-12-07 --lat 34.27 --lon -118.62")

        print("\n" + "="*70)
        print("CLOSEST STATIONS FOR YOUR DATA:")
        print("="*70)

        # Show closest stations for LAX
        print("\nLAX Area (Dec 2021):")
        station = find_closest_station(34.2725, -118.6157, 'LAX')
        if station:
            print(f"  Closest: {station['id']} ({station['name']}) - {station['distance_km']:.1f}km away")

        # Show closest stations for MTV
        print("\nMTV Area (May-Jul 2020):")
        station = find_closest_station(37.3861, -122.0839, 'MTV')
        if station:
            print(f"  Closest: {station['id']} ({station['name']}) - {station['distance_km']:.1f}km away")

        print("\n" + "="*70)


if __name__ == "__main__":
    main()
