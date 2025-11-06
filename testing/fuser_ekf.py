import pandas as pd
import numpy as np
import os
import tempfile
import pymap3d as pm
from scipy.spatial.transform import Rotation

# Dependencies from your project
try:
    from data_uitiles import read_and_separate_log, apply_sanity_filters, add_time_columns, read_pos_file
    from feature import extract_pos_features 
except ImportError:
    print("CRITICAL ERROR: Could not import 'data_uitiles.py' or 'feature.py'.")
    # Define dummy functions to avoid crashing
    def read_and_separate_log(path): return None
    def apply_sanity_filters(df, name): return None
    def add_time_columns(df): return None
    def read_pos_file(path): return None
    def extract_pos_features(df): return None

# ==========================================
# ----------- EKF HELPER FUNCTIONS -----------
# ==========================================

def find_pos_file(drive_id: str, phone_id: str, ppk_output_dir: str = "D:/NTU/Y3S1/SC4000/ML-Dawgs/ppk_output/ppk_output") -> str | None:
    """Find the .pos file path for a given drive_id and phone_id."""
    folder_name = f"{drive_id}-{phone_id}"
    pos_file_path = os.path.join(ppk_output_dir, folder_name, "gnss_rinex.pos")
    return pos_file_path if os.path.exists(pos_file_path) else None

def synchronize_imu_time_to_seconds(imu_df: pd.DataFrame) -> pd.DataFrame:
    """Converts IMU timestamps to GPST seconds."""
    if imu_df.empty or 'millisSinceBoot' not in imu_df.columns:
        print("Synchronization failed: IMU data is empty or missing 'millisSinceBoot'.")
        return pd.DataFrame({})
        
    GPS_EPOCH_OFFSET_MILLIS = 315964800000
    imu_df['gpst_sec'] = (imu_df['millisSinceBoot'].astype(np.int64) - GPS_EPOCH_OFFSET_MILLIS) / 1000.0
    imu_df = imu_df.dropna(subset=['gpst_sec'])
    return imu_df.reset_index(drop=True)

def convert_pos_to_enu_and_seconds(pos_df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Converts a POS DataFrame (from extract_pos_features) to ENU coordinates
    and GPST seconds.
    """
    if pos_df.empty or not all(c in pos_df.columns for c in ['mean_latitude', 'mean_longitude', 'mean_height']):
        print("LLH to ENU failed: DataFrame is empty or missing required columns.")
        return pd.DataFrame({}), {}
    
    pos_df_renamed = pos_df.rename(columns={
        'mean_latitude': 'lat', 'mean_longitude': 'lon', 'mean_height': 'h'
    })
    
    pos_df_renamed['gpst_sec'] = pos_df_renamed['millisSinceGpsEpoch'] / 1000.0
    
    lat0, lon0, h0 = pos_df_renamed.iloc[0][['lat', 'lon', 'h']]
    origin_lla = {'lat': lat0, 'lon': lon0, 'h': h0}
    
    e, n, u = pm.geodetic2enu(pos_df_renamed['lat'], pos_df_renamed['lon'], pos_df_renamed['h'], lat0, lon0, h0)
    
    # Grab all columns from the input, including gt_... and quality
    enu_df = pos_df_renamed.copy()
    enu_df['e'] = e
    enu_df['n'] = n
    enu_df['u'] = u
    
    # Also create the ground truth (GT) columns for error calculation later
    enu_df.rename(columns={'e': 'gt_e', 'n': 'gt_n', 'u': 'gt_u', 
                           'lat': 'gt_lat', 'lon': 'gt_lon', 'h': 'gt_h'}, inplace=True)
    
    return enu_df, origin_lla

def _skew(v):
    """Converts a 3-vector to a 3x3 skew-symmetric matrix."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

# ==========================================
# --- 15-STATE ERROR-STATE KALMAN FILTER ---
# ==========================================

class EKF15State:
    def __init__(self, initial_pos, initial_att_euler, pos_std, vel_std, att_std, accel_bias_std, gyro_bias_std):
        
        # --- 1. Nominal State ---
        self.p_nb = initial_pos.flatten()
        self.v_nb = np.zeros(3)
        self.R_nb = Rotation.from_euler('xyz', initial_att_euler, degrees=True).as_matrix()
        self.b_a = np.zeros(3)
        self.b_g = np.zeros(3)
        self.g_n = np.array([0, 0, -9.80665])

        # --- 2. Error State and Covariance ---
        self.x_error = np.zeros(15)
        self.P = np.diag([
            pos_std**2, pos_std**2, pos_std**2,
            vel_std**2, vel_std**2, vel_std**2,
            np.deg2rad(att_std)**2, np.deg2rad(att_std)**2, np.deg2rad(att_std)**2,
            accel_bias_std**2, accel_bias_std**2, accel_bias_std**2,
            np.deg2rad(gyro_bias_std)**2, np.deg2rad(gyro_bias_std)**2, np.deg2rad(gyro_bias_std)**2
        ]) * 15.0 # Inflate initial covariance

        # --- 3. Noise Parameters ---
        self.Q = np.diag([
            0.01**2, 0.01**2, 0.01**2,        # Position drift
            0.1**2, 0.1**2, 0.1**2,         # Velocity drift
            np.deg2rad(0.01)**2, np.deg2rad(0.01)**2, np.deg2rad(0.01)**2, # Attitude drift
            (1e-3)**2, (1e-3)**2, (1e-3)**2, # Accel bias random walk
            (1e-5)**2, (1e-5)**2, (1e-5)**2  # Gyro bias random walk
        ])
        
        self.I_3 = np.eye(3)
        self.I_15 = np.eye(15)

    def predict(self, accel_data, gyro_data, dt):
        """EKF Predict Step (IMU Integration)"""
        
        # --- 1. Update Nominal State ---
        accel_corrected = accel_data - self.b_a
        gyro_corrected = gyro_data - self.b_g
        
        R_delta = Rotation.from_rotvec(gyro_corrected * dt).as_matrix()
        self.R_nb = self.R_nb @ R_delta
        
        accel_n = self.R_nb @ accel_corrected
        # Smartphone UncalAccel typically includes gravity; do not add gravity again
        self.v_nb = self.v_nb + accel_n * dt
        self.p_nb = self.p_nb + self.v_nb * dt + 0.5 * accel_n * dt**2

        # --- 2. Propagate Error State Covariance ---
        F = np.zeros((15, 15))
        F[0:3, 3:6] = self.I_3
        F[3:6, 6:9] = -_skew(self.R_nb @ accel_corrected)
        F[3:6, 9:12] = -self.R_nb
        F[6:9, 6:9] = -_skew(gyro_corrected)
        F[6:9, 12:15] = -self.I_3
        
        Phi = self.I_15 + F * dt
        self.P = Phi @ self.P @ Phi.T + self.Q * dt

    def update(self, pos_data_enu, pos_quality):
        """EKF Update Step (POS Measurement)"""
        
        H = np.zeros((3, 15))
        H[0:3, 0:3] = self.I_3
        
        if pos_quality == 1:   R_std = 0.5
        elif pos_quality == 2: R_std = 2.0
        else:                  R_std = 10.0
        R = np.diag([R_std**2, R_std**2, (R_std*2)**2])
            
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = pos_data_enu - self.p_nb
        self.x_error = K @ y
        self.P = (self.I_15 - K @ H) @ self.P
        
        # --- Inject Error into Nominal State ---
        self.p_nb = self.p_nb + self.x_error[0:3]
        self.v_nb = self.v_nb + self.x_error[3:6]
        self.R_nb = self.R_nb @ Rotation.from_rotvec(-self.x_error[6:9]).as_matrix()
        self.b_a = self.b_a + self.x_error[9:12]
        self.b_g = self.b_g + self.x_error[12:15]
        
        self.x_error = np.zeros(15)

    def get_state_dict(self):
        """Return the current nominal state as a dictionary."""
        rpy = Rotation.from_matrix(self.R_nb).as_euler('xyz', degrees=True)
        return {
            'ekf_e': self.p_nb[0], 'ekf_n': self.p_nb[1], 'ekf_u': self.p_nb[2],
            'ekf_vE': self.v_nb[0], 'ekf_vN': self.v_nb[1], 'ekf_vU': self.v_nb[2],
            'ekf_roll': rpy[0], 'ekf_pitch': rpy[1], 'ekf_yaw': rpy[2],
        }

# ==========================================
# --- MAIN PROCESSING FUNCTION (NOW AN EKF RUNNER) ---
# ==========================================

def process_logs_to_ekf_trajectory(job: dict) -> str | None:
    """
    Runs a 15-state EKF on the raw IMU and POS data.
    Saves the resulting trajectory to a file and returns the path.
    """
    gnss_log_path = job['gnss_file']
    imu_log_path = job['imu_file']
    log_id = f"{job['drive_id']}/{job['phone_id']}"
    print(f"\n--- [EKF] Processing Log Pair: {log_id} ---")
    
    # --- 1. Load and Prep IMU Data ---
    gnss_data = read_and_separate_log(gnss_log_path)
    imu_data = read_and_separate_log(imu_log_path)
    if gnss_data is None or imu_data is None: return None
    
    raw_df = add_time_columns(apply_sanity_filters(gnss_data.get('raw', pd.DataFrame({})), 'raw'))
    if raw_df is None or raw_df.empty: return None
    
    imu_accel = apply_sanity_filters(imu_data.get('accel', pd.DataFrame({})), 'accel')
    imu_gyro = apply_sanity_filters(imu_data.get('gyro', pd.DataFrame({})), 'gyro')
    if imu_accel is None or imu_gyro is None or imu_accel.empty or imu_gyro.empty: return None

    # 🟨🟨🟨 THIS IS THE FIX 🟨🟨_
    # The column names in the log are 'UncalAccel.X' (with a dot and capital)
    # We rename them to the expected 'accel_x'
    imu_accel = imu_accel.rename(columns={
        'UncalAccel.X': 'accel_x', 'UncalAccel.Y': 'accel_y', 'UncalAccel.Z': 'accel_z'
    })
    imu_gyro = imu_gyro.rename(columns={
        'UncalGyro.X': 'gyro_x', 'UncalGyro.Y': 'gyro_y', 'UncalGyro.Z': 'gyro_z'
    })
    
    # Merge the raw data tables.
    imu_raw_data = pd.merge_asof(
        imu_accel.sort_values('millisSinceBoot'),
        imu_gyro.sort_values('millisSinceBoot'),
        on='millisSinceBoot',
        direction='nearest',
        tolerance=10 # 10ms tolerance
    )
    
    imu_synced = synchronize_imu_time_to_seconds(imu_raw_data)
    if imu_synced.empty: return None
    
    imu_synced['type'] = 'IMU'
    
    # Now, check if the required columns exist AFTER renaming
    required_imu_cols = ['gpst_sec', 'type', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    if not all(col in imu_synced.columns for col in required_imu_cols):
        print(f"❌ Skipping {log_id}: IMU data is missing required x/y/z columns after processing.")
        print(f"   Available columns: {list(imu_synced.columns)}")
        return None
        
    imu_synced = imu_synced[required_imu_cols]
    
    # Interpolate any minor gaps from the asof merge
    imu_synced = imu_synced.set_index('gpst_sec').interpolate(method='linear').reset_index()
    imu_synced.bfill(inplace=True) # Fill any at the start

    # Downsample IMU stream to 10 Hz by binning to 0.1s and averaging
    imu_only = imu_synced.copy()
    imu_only['gpst_sec'] = (imu_only['gpst_sec'] / 0.1).round() * 0.1
    imu_only = imu_only.groupby('gpst_sec', as_index=False).agg({
        'accel_x': 'mean', 'accel_y': 'mean', 'accel_z': 'mean',
        'gyro_x': 'mean', 'gyro_y': 'mean', 'gyro_z': 'mean',
        'type': 'first'
    })

    # --- 2. Load and Prep POS Data ---
    pos_file_path = find_pos_file(job['drive_id'], job['phone_id'])
    if not pos_file_path:
        print(f"⚠️ [EKF] No .pos file found for {log_id}. Skipping.")
        return None
        
    pos_raw = read_pos_file(pos_file_path)
    if pos_raw is None or pos_raw.empty: return None

    pos_features = extract_pos_features(pos_raw)
    if pos_features is None or pos_features.empty: return None
    
    pos_enu, origin_lla = convert_pos_to_enu_and_seconds(pos_features)
    if pos_enu.empty: return None
    
    # Get the ground truth (GT) position columns
    gt_cols = [col for col in pos_enu.columns if col.startswith('gt_')] + ['gpst_sec']
    gt_pos = pos_enu[gt_cols].copy()

    pos_enu['type'] = 'POS'
    # Carry both mean_quality and max_quality if available
    quality_cols = [c for c in ['mean_quality', 'max_quality'] if c in pos_enu.columns]
    pos_enu = pos_enu[['gpst_sec', 'type', 'gt_e', 'gt_n', 'gt_u'] + quality_cols]
    pos_enu.rename(columns={'gt_e': 'e', 'gt_n': 'n', 'gt_u': 'u'}, inplace=True) # EKF expects 'e', 'n', 'u'
    
    # --- 3. Merge Asynchronous Data ---
    full_data = pd.concat([imu_only, pos_enu]).sort_values('gpst_sec').reset_index(drop=True)
    full_data['dt_sec'] = full_data['gpst_sec'].diff().fillna(0.01)
    
    # --- 4. Initialize EKF ---
    first_pos = pos_enu.iloc[0]
    initial_pos = first_pos[['e', 'n', 'u']].values
    initial_attitude_euler = [0, 0, 0] # TODO: Estimate from velocity
    
    ekf = EKF15State(
        initial_pos=initial_pos, initial_att_euler=initial_attitude_euler,
        pos_std=5.0, vel_std=1.0, att_std=10.0,
        accel_bias_std=1e-1, gyro_bias_std=1e-2
    )
    
    # --- 5. Run EKF Loop ---
    results = []
    print(f"[EKF] Running EKF on {len(full_data)} measurements...")
    
    for i, row in full_data.iterrows():
        dt = row['dt_sec']
        if dt <= 0.001: continue # Skip duplicate timestamps or tiny deltas
            
        if row['type'] == 'IMU':
            accel_data = row[['accel_x', 'accel_y', 'accel_z']].values
            gyro_data = row[['gyro_x', 'gyro_y', 'gyro_z']].values
            ekf.predict(accel_data, gyro_data, dt)
        
        elif row['type'] == 'POS':
            pos_data = row[['e', 'n', 'u']].values
            # Prefer max_quality if present; otherwise fall back to mean_quality
            pos_quality_raw = row['max_quality'] if 'max_quality' in row.index else row.get('mean_quality', 5)
            # Discretize to integer quality class in [1..6]
            try:
                pos_quality_disc = int(np.clip(round(float(pos_quality_raw)), 1, 6))
            except Exception:
                pos_quality_disc = 5
            ekf.update(pos_data, pos_quality_disc)
            
        state = ekf.get_state_dict()
        state['gpst_sec'] = row['gpst_sec']
        results.append(state)

    if not results:
        print(f"Skipping {log_id}: EKF produced no results.")
        return None

    # --- 6. Save Trajectory (and merge GT) ---
    final_df = pd.DataFrame(results)
    
    # Merge ground truth data onto the EKF timeline for error calculation
    final_df = pd.merge_asof(
        final_df.sort_values('gpst_sec'),
        gt_pos.sort_values('gpst_sec'),
        on='gpst_sec',
        direction='nearest',
        tolerance=0.5 # 500ms tolerance
    )
    
    final_df['drive_id'] = job['drive_id']
    final_df['phone_id'] = job['phone_id']
    final_df['origin_lat'] = origin_lla['lat']
    final_df['origin_lon'] = origin_lla['lon']
    final_df['origin_h'] = origin_lla['h']

    # Cross-platform temp directory for cached EKF trajectories
    temp_output_dir = os.path.join(tempfile.gettempdir(), "dask_feature_cache_EKF")
    os.makedirs(temp_output_dir, exist_ok=True)
    job_id_str = f"{job['drive_id'].replace('/', '_')}_{job['phone_id']}"
    output_path = os.path.join(temp_output_dir, f"{job_id_str}.parquet")

    try:
        final_df.to_parquet(output_path, index=False, engine='auto')
        print(f"✅ [EKF] Success: Saved EKF trajectory ({len(final_df)} rows) to {output_path}")
        return output_path
    except ImportError:
        print(f"⚠️ [EKF] WARNING: pyarrow not found. Falling back to CSV.")
        output_path_csv = os.path.join(temp_output_dir, f"{job_id_str}.csv")
        final_df.to_csv(output_path_csv, index=False)
        print(f"✅ [EKF] Success: Saved EKF trajectory ({len(final_df)} rows) to {output_path_csv}")
        return output_path_csv
    except Exception as e:
        print(f"❌ [EKF] Error during file save for {job_id_str}: {e}")
        return None

