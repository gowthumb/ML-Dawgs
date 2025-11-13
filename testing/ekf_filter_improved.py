import numpy as np
import pandas as pd
from typing import Tuple, Optional


def lat_lon_to_ecef(lat_deg, lon_deg, height_m):
    """Convert latitude, longitude, height to ECEF coordinates."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    # WGS84 parameters
    a = 6378137.0  # Semi-major axis
    f = 1 / 298.257223563  # Flattening
    e2 = 2 * f - f * f  # Square of eccentricity

    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)

    x = (N + height_m) * np.cos(lat) * np.cos(lon)
    y = (N + height_m) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + height_m) * np.sin(lat)

    return np.array([x, y, z])


def ecef_to_lat_lon(x, y, z):
    """Convert ECEF coordinates to latitude, longitude, height."""
    # WGS84 parameters
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f * f
    b = a * (1 - f)

    # Iterative method
    lon = np.arctan2(y, x)
    p = np.sqrt(x ** 2 + y ** 2)
    lat = np.arctan2(z, p * (1 - e2))

    for _ in range(5):  # Usually converges in 2-3 iterations
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + e2 * N * np.sin(lat), p)

    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    height = p / np.cos(lat) - N

    return np.degrees(lat), np.degrees(lon), height


class SimpleEKF:
    """
    Simplified Extended Kalman Filter for GNSS/IMU fusion.

    State vector (9 dimensions):
    [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, accel_bias_x, accel_bias_y, accel_bias_z]

    Using ECEF coordinates for better numerical stability.
    """

    def __init__(self):
        # State dimension
        self.n = 9

        # State vector: [position(3), velocity(3), accel_bias(3)]
        self.x = np.zeros(self.n)

        # Covariance matrix
        self.P = np.eye(self.n)
        self.P[0:3, 0:3] *= 100.0  # Position uncertainty
        self.P[3:6, 3:6] *= 10.0   # Velocity uncertainty
        self.P[6:9, 6:9] *= 1.0    # Accel bias uncertainty

        # Process noise covariance
        self.Q = np.eye(self.n)
        self.Q[0:3, 0:3] *= 0.01   # Position process noise
        self.Q[3:6, 3:6] *= 0.5    # Velocity process noise
        self.Q[6:9, 6:9] *= 0.001  # Accel bias process noise (slowly varying)

        # Measurement noise
        self.R_gnss = np.eye(3) * 25.0  # GNSS position measurement noise (meters^2)

        self.initialized = False
        self.ref_lat = None  # Reference latitude for local frame
        self.ref_lon = None  # Reference longitude for local frame

    def initialize_state(self, lat_deg, lon_deg, height_m):
        """Initialize the filter state with initial position."""
        # Convert to ECEF
        pos_ecef = lat_lon_to_ecef(lat_deg, lon_deg, height_m)
        self.x[0:3] = pos_ecef
        self.x[3:6] = 0  # Zero initial velocity
        self.x[6:9] = 0  # Zero initial bias

        # Store reference position for local frame
        self.ref_lat = lat_deg
        self.ref_lon = lon_deg

        self.initialized = True

    def predict(self, dt: float, accel_local: np.ndarray):
        """
        Prediction step using IMU acceleration measurement.

        Args:
            dt: Time step in seconds
            accel_local: Acceleration in local NED frame [north, east, down] in m/s^2
        """
        if not self.initialized or dt <= 0:
            return

        # Extract current state
        pos = self.x[0:3]
        vel = self.x[3:6]
        accel_bias = self.x[6:9]

        # Convert acceleration from local NED to ECEF frame
        accel_ecef = self._ned_to_ecef(accel_local, self.ref_lat, self.ref_lon)

        # Correct acceleration with bias
        accel_corrected = accel_ecef - accel_bias

        # State prediction (constant acceleration model)
        pos_new = pos + vel * dt + 0.5 * accel_corrected * dt ** 2
        vel_new = vel + accel_corrected * dt
        bias_new = accel_bias  # Bias remains constant

        # State transition matrix (Jacobian)
        F = np.eye(self.n)
        F[0:3, 3:6] = np.eye(3) * dt  # Position depends on velocity
        F[3:6, 6:9] = -np.eye(3) * dt  # Velocity depends on accel bias

        # Covariance prediction
        Q_scaled = self.Q * dt
        P_pred = F @ self.P @ F.T + Q_scaled

        # Update state and covariance
        self.x[0:3] = pos_new
        self.x[3:6] = vel_new
        self.x[6:9] = bias_new
        self.P = P_pred

    def update_gnss(self, lat_deg, lon_deg, height_m, uncertainty_m=None):
        """
        Update step using GNSS position measurement.

        Args:
            lat_deg: Latitude in degrees
            lon_deg: Longitude in degrees
            height_m: Height in meters
            uncertainty_m: Position uncertainty in meters (optional)
        """
        if not self.initialized:
            return

        # Convert measurement to ECEF
        z = lat_lon_to_ecef(lat_deg, lon_deg, height_m)

        # Measurement matrix (we observe position directly)
        H = np.zeros((3, self.n))
        H[0:3, 0:3] = np.eye(3)

        # Measurement noise
        if uncertainty_m is not None:
            R = np.eye(3) * (uncertainty_m ** 2)
        else:
            R = self.R_gnss

        # Innovation
        y = z - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular matrix, skip update
            return

        K = self.P @ H.T @ S_inv

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # Ensure P remains symmetric and positive definite
        self.P = (self.P + self.P.T) / 2
        eigenvalues = np.linalg.eigvalsh(self.P)
        if np.any(eigenvalues < 0):
            # Add small positive value to diagonal
            self.P += np.eye(self.n) * 1e-6

    def get_state(self) -> dict:
        """Get current state estimate."""
        # Convert position from ECEF to lat/lon
        lat, lon, height = ecef_to_lat_lon(self.x[0], self.x[1], self.x[2])

        return {
            'ekf_pos_x': self.x[0],
            'ekf_pos_y': self.x[1],
            'ekf_pos_z': self.x[2],
            'ekf_vel_x': self.x[3],
            'ekf_vel_y': self.x[4],
            'ekf_vel_z': self.x[5],
            'ekf_accel_bias_x': self.x[6],
            'ekf_accel_bias_y': self.x[7],
            'ekf_accel_bias_z': self.x[8],
            'ekf_latitude': lat,
            'ekf_longitude': lon,
            'ekf_height': height,
            'ekf_pos_std': np.sqrt(np.trace(self.P[0:3, 0:3])),
            'ekf_vel_std': np.sqrt(np.trace(self.P[3:6, 3:6])),
            'ekf_pos_std_x': np.sqrt(self.P[0, 0]),
            'ekf_pos_std_y': np.sqrt(self.P[1, 1]),
            'ekf_pos_std_z': np.sqrt(self.P[2, 2]),
        }

    @staticmethod
    def _ned_to_ecef(ned_vector, lat_deg, lon_deg):
        """Convert vector from local NED frame to ECEF frame."""
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)

        # Rotation matrix from NED to ECEF
        R = np.array([
            [-np.sin(lat) * np.cos(lon), -np.sin(lon), -np.cos(lat) * np.cos(lon)],
            [-np.sin(lat) * np.sin(lon), np.cos(lon), -np.cos(lat) * np.sin(lon)],
            [np.cos(lat), 0, -np.sin(lat)]
        ])

        return R @ ned_vector


def apply_ekf_to_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply EKF to the extracted features dataframe.

    Args:
        features_df: DataFrame with GNSS/IMU features

    Returns:
        DataFrame with EKF-filtered estimates added
    """
    print("\n--- Applying Extended Kalman Filter (Improved) ---")

    # Group by drive_id and phone_id
    grouped = features_df.groupby(['drive_id', 'phone_id'])
    all_filtered = []

    for (drive_id, phone_id), group_df in grouped:
        print(f"Processing EKF for {drive_id}/{phone_id}...")

        # Sort by time
        group_df = group_df.sort_values('millisSinceGpsEpoch').reset_index(drop=True)

        # Initialize EKF
        ekf = SimpleEKF()
        filtered_results = []

        for idx, row in group_df.iterrows():
            # Time step
            if idx > 0:
                dt = (row['millisSinceGpsEpoch'] - group_df.loc[idx - 1, 'millisSinceGpsEpoch']) / 1000.0
                dt = max(0.001, min(dt, 1.0))  # Clamp dt
            else:
                dt = 0.1

            # Initialize on first valid GNSS measurement
            if not ekf.initialized:
                if pd.notna(row.get('mean_latitude')) and pd.notna(row.get('mean_longitude')):
                    lat = row['mean_latitude']
                    lon = row['mean_longitude']
                    height = row.get('mean_height', 0.0) if pd.notna(row.get('mean_height')) else 0.0
                    ekf.initialize_state(lat, lon, height)
                    print(f"  EKF initialized at row {idx}")

            if not ekf.initialized:
                filtered_results.append({})
                continue

            # PREDICTION with IMU
            has_3d_accel = all(pd.notna(row.get(f'accel_{c}_mean')) for c in ['x', 'y', 'z'])

            if has_3d_accel:
                # Use actual 3D measurements in local frame (assume phone frame ~ NED)
                accel_local = np.array([
                    row.get('accel_x_mean', 0.0),
                    row.get('accel_y_mean', 0.0),
                    row.get('accel_z_mean', 0.0) - 9.81  # Remove gravity
                ])
                ekf.predict(dt, accel_local)

            # UPDATE with GNSS
            if pd.notna(row.get('mean_latitude')) and pd.notna(row.get('mean_longitude')):
                lat = row['mean_latitude']
                lon = row['mean_longitude']
                height = row.get('mean_height', 0.0) if pd.notna(row.get('mean_height')) else 0.0
                uncertainty = row.get('mean_position_uncertainty_3d', None)

                ekf.update_gnss(lat, lon, height, uncertainty)

            # Store filtered state
            state = ekf.get_state()
            filtered_results.append(state)

        # Add filtered results to dataframe
        filtered_df = group_df.copy()
        for key in ['ekf_pos_x', 'ekf_pos_y', 'ekf_pos_z',
                    'ekf_vel_x', 'ekf_vel_y', 'ekf_vel_z',
                    'ekf_accel_bias_x', 'ekf_accel_bias_y', 'ekf_accel_bias_z',
                    'ekf_latitude', 'ekf_longitude', 'ekf_height',
                    'ekf_pos_std', 'ekf_vel_std',
                    'ekf_pos_std_x', 'ekf_pos_std_y', 'ekf_pos_std_z']:
            filtered_df[key] = [r.get(key, np.nan) for r in filtered_results]

        all_filtered.append(filtered_df)
        print(f"  [OK] EKF completed for {drive_id}/{phone_id}")

    # Combine results
    final_df = pd.concat(all_filtered, ignore_index=True)

    # Ensure proper sorting for downstream merge_asof operations
    final_df = final_df.sort_values(['drive_id', 'phone_id', 'millisSinceGpsEpoch']).reset_index(drop=True)

    print(f"[SUCCESS] EKF applied to all {len(grouped)} trajectories")
    print(f"   Total rows: {len(final_df)}")

    return final_df


def process_csv_with_ekf(input_csv: str, output_csv: str):
    """Load features CSV, apply EKF, and save to new CSV."""
    print(f"\n{'=' * 60}")
    print(f"EKF Processing Pipeline (Improved)")
    print(f"{'=' * 60}")
    print(f"Input:  {input_csv}")
    print(f"Output: {output_csv}")

    # Load features
    print("\nLoading features...")
    features_df = pd.read_csv(input_csv)
    print(f"[OK] Loaded {len(features_df)} rows")

    # Apply EKF
    filtered_df = apply_ekf_to_features(features_df)

    # Save results
    print(f"\nSaving filtered results to {output_csv}...")
    filtered_df.to_csv(output_csv, index=False)
    print(f"[SUCCESS] EKF-filtered data saved successfully!")
    print(f"   Output columns: {len(filtered_df.columns)}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("EKF Summary Statistics:")
    print(f"{'=' * 60}")
    if 'ekf_pos_std' in filtered_df.columns:
        print(f"Position uncertainty: {filtered_df['ekf_pos_std'].mean():.3f} +/- {filtered_df['ekf_pos_std'].std():.3f} m")
    if 'ekf_vel_std' in filtered_df.columns:
        print(f"Velocity uncertainty: {filtered_df['ekf_vel_std'].mean():.3f} +/- {filtered_df['ekf_vel_std'].std():.3f} m/s")
    if 'ekf_vel_x' in filtered_df.columns:
        vel_mag = np.sqrt(filtered_df['ekf_vel_x'] ** 2 +
                          filtered_df['ekf_vel_y'] ** 2 +
                          filtered_df['ekf_vel_z'] ** 2)
        print(f"Velocity magnitude: {vel_mag.mean():.3f} +/- {vel_mag.std():.3f} m/s")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # Example usage
    input_file = "all_extracted_features_full_170_datasets_with_pos_raw.csv"
    output_file = "all_extracted_features_full_170_datasets_with_pos_ekf.csv"

    process_csv_with_ekf(input_file, output_file)
