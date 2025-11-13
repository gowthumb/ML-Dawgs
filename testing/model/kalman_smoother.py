"""
Kalman Filter for smoothing GNSS position estimates.

This implementation provides a 1D Kalman filter that can be applied
independently to latitude and longitude coordinates after residual correction.
Includes comprehensive metrics calculation matching the main training pipeline.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict


class KalmanFilter1D:
    """
    Simple 1D Kalman Filter for position smoothing.
    
    State: [position, velocity]
    Assumes constant velocity model with some process noise.
    """
    
    def __init__(self, process_variance: float = 0.5, 
                 measurement_variance: float = 5.0,
                 initial_position: float = 0.0,
                 initial_velocity: float = 0.0):
        """
        Initialize Kalman Filter.
        
        Args:
            process_variance: Process noise (how much we trust the model)
            measurement_variance: Measurement noise (how much we trust observations)
            initial_position: Starting position estimate
            initial_velocity: Starting velocity estimate
        """
        # State: [position, velocity]
        self.state = np.array([initial_position, initial_velocity])
        
        # State covariance matrix (uncertainty in our estimate)
        self.P = np.eye(2) * 1000  # High initial uncertainty
        
        # Process noise covariance
        self.Q = np.array([
            [process_variance, 0],
            [0, process_variance * 0.1]  # Less noise in velocity
        ])
        
        # Measurement noise covariance
        self.R = np.array([[measurement_variance]])
        
        # State transition matrix (constant velocity model)
        # Will be updated based on actual dt
        self.F = np.eye(2)
        
        # Measurement matrix (we only observe position)
        self.H = np.array([[1, 0]])
    
    def predict(self, dt: float = 1.0):
        """
        Predict next state based on motion model.
        
        Args:
            dt: Time step (seconds)
        """
        # Update state transition matrix with time step
        self.F = np.array([
            [1, dt],
            [0, 1]
        ])
        
        # Predict state
        self.state = self.F @ self.state
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement: float):
        """
        Update state estimate with new measurement.
        
        Args:
            measurement: Observed position value
        """
        # Innovation (measurement residual)
        y = measurement - (self.H @ self.state)[0]
        
        # Innovation covariance
        S = (self.H @ self.P @ self.H.T + self.R)[0, 0]
        
        # Kalman gain
        K = (self.P @ self.H.T) / S
        
        # Update state
        self.state = self.state + K.flatten() * y
        
        # Update covariance
        I_KH = np.eye(2) - K @ self.H
        self.P = I_KH @ self.P
    
    def get_position(self) -> float:
        """Get current position estimate."""
        return self.state[0]

    def get_velocity(self) -> float:
        """Get current velocity estimate."""
        return self.state[1]

    def set_measurement_variance(self, variance: float):
        """
        Dynamically update the measurement noise variance.
        Allows per-timestep adjustment based on prediction confidence.

        Args:
            variance: New measurement variance (minimum 1e-6 to prevent numerical issues)
        """
        self.R[0, 0] = max(variance, 1e-6)


def apply_kalman_smoothing(
    positions: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    process_variance: float = 0.5,
    measurement_variance: Optional[float] = None,
    measurement_variances: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply Kalman filter to smooth position estimates.

    Args:
        positions: Array of position values (lat or lon in meters)
        timestamps: Optional timestamps for computing dt
        process_variance: Process noise parameter (meters²)
        measurement_variance: Fixed measurement noise (meters²) - used if measurement_variances is None
        measurement_variances: Array of per-timestep measurement variances (meters²) from NN uncertainty

    Returns:
        Smoothed position array
    """
    if len(positions) == 0:
        return positions

    # Handle measurement variance parameter
    if measurement_variances is None:
        # Fall back to fixed variance
        if measurement_variance is None:
            measurement_variance = 5.0  # Default
        measurement_variances = np.full_like(positions, measurement_variance)

    # CRITICAL FIX: Without timestamps, DON'T smooth at all
    # The constant velocity model needs timestamps to work correctly
    # Smoothing without time information causes accumulating errors
    if timestamps is None:
        print("  [WARNING] No timestamps - returning positions WITHOUT smoothing (Kalman needs timestamps)")
        return positions.copy()  # Return unsmoothed

    # Initialize filter with first measurement
    kf = KalmanFilter1D(
        process_variance=process_variance,
        measurement_variance=measurement_variances[0],
        initial_position=positions[0]
    )

    smoothed = np.zeros_like(positions)
    smoothed[0] = positions[0]

    for i in range(1, len(positions)):
        # Compute time step - handle both Timestamp and Timedelta
        time_diff = timestamps[i] - timestamps[i-1]

        # If it's a Timedelta, get total_seconds()
        if hasattr(time_diff, 'total_seconds'):
            dt = time_diff.total_seconds()
        else:
            # If it's a numpy timedelta64, convert to seconds
            dt = float(time_diff) / 1e9  # nanoseconds to seconds

        dt = max(dt, 0.01)  # Minimum 0.01s to avoid division by zero

        # Predict
        kf.predict(dt)

        # Update measurement variance for this timestep (dynamic R)
        kf.set_measurement_variance(measurement_variances[i])

        # Update with measurement
        kf.update(positions[i])
        smoothed[i] = kf.get_position()

    return smoothed


def apply_kalman_to_coordinates(
    lat: np.ndarray,
    lon: np.ndarray,
    timestamps: Optional[pd.Series] = None,
    process_variance: float = 0.5,
    measurement_variance: Optional[float] = None,
    lat_variances: Optional[np.ndarray] = None,
    lon_variances: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Kalman filtering to both latitude and longitude.

    CRITICAL: Converts degrees to meters before filtering, then converts back.
    This ensures the variance parameters (in meters) work correctly.

    Args:
        lat: Latitude values (degrees)
        lon: Longitude values (degrees)
        timestamps: Optional timestamps for dt calculation
        process_variance: Process noise parameter (meters²)
        measurement_variance: Fixed measurement noise (meters²) - used if lat/lon_variances is None
        lat_variances: Array of per-timestep lat measurement variances (meters²) from NN
        lon_variances: Array of per-timestep lon measurement variances (meters²) from NN

    Returns:
        Tuple of (smoothed_lat, smoothed_lon)
    """
    if len(lat) == 0:
        return lat, lon

    # Convert degrees to meters for Kalman filtering
    # Use mean latitude for lon conversion
    mean_lat = np.mean(lat)
    METERS_PER_DEGREE_LAT = 111000.0
    METERS_PER_DEGREE_LON = METERS_PER_DEGREE_LAT * np.cos(np.radians(mean_lat))

    # Convert to meters (relative to first position)
    lat_m = (lat - lat[0]) * METERS_PER_DEGREE_LAT
    lon_m = (lon - lon[0]) * METERS_PER_DEGREE_LON

    # Sort by timestamp if provided
    if timestamps is not None:
        sort_idx = np.argsort(timestamps)
        lat_m_sorted = lat_m[sort_idx]
        lon_m_sorted = lon_m[sort_idx]
        ts_sorted = timestamps.iloc[sort_idx].values

        # Sort variances too if provided
        lat_var_sorted = lat_variances[sort_idx] if lat_variances is not None else None
        lon_var_sorted = lon_variances[sort_idx] if lon_variances is not None else None

        # Apply Kalman filtering in meters
        lat_m_smoothed = apply_kalman_smoothing(
            lat_m_sorted, ts_sorted, process_variance,
            measurement_variance=measurement_variance,
            measurement_variances=lat_var_sorted
        )
        lon_m_smoothed = apply_kalman_smoothing(
            lon_m_sorted, ts_sorted, process_variance,
            measurement_variance=measurement_variance,
            measurement_variances=lon_var_sorted
        )

        # Restore original order
        restore_idx = np.argsort(sort_idx)
        lat_m_smoothed = lat_m_smoothed[restore_idx]
        lon_m_smoothed = lon_m_smoothed[restore_idx]
    else:
        # No timestamps, apply sequentially
        lat_m_smoothed = apply_kalman_smoothing(
            lat_m, None, process_variance,
            measurement_variance=measurement_variance,
            measurement_variances=lat_variances
        )
        lon_m_smoothed = apply_kalman_smoothing(
            lon_m, None, process_variance,
            measurement_variance=measurement_variance,
            measurement_variances=lon_variances
        )

    # Convert back to degrees
    lat_smoothed = lat[0] + (lat_m_smoothed / METERS_PER_DEGREE_LAT)
    lon_smoothed = lon[0] + (lon_m_smoothed / METERS_PER_DEGREE_LON)

    return lat_smoothed, lon_smoothed


def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, 
                       lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Calculate haversine distance between two sets of coordinates.
    
    Args:
        lat1, lon1: First set of coordinates (degrees)
        lat2, lon2: Second set of coordinates (degrees)
    
    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def calculate_position_metrics(
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    gt_lat: np.ndarray,
    gt_lon: np.ndarray,
    label: str = ""
) -> Dict[str, float]:
    """
    Calculate comprehensive position error metrics.
    
    Args:
        pred_lat: Predicted latitude
        pred_lon: Predicted longitude
        gt_lat: Ground truth latitude
        gt_lon: Ground truth longitude
        label: Label for the metrics (e.g., "Baseline", "Corrected", "Kalman")
    
    Returns:
        Dictionary with all metrics
    """
    # Calculate 2D distance errors
    errors_2d = haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)
    
    # Calculate component errors (in meters approximately)
    errors_lat_m = haversine_distance(pred_lat, gt_lon, gt_lat, gt_lon)
    errors_lon_m = haversine_distance(gt_lat, pred_lon, gt_lat, gt_lon)
    
    # Sign for lat/lon errors
    errors_lat_signed = np.where(pred_lat > gt_lat, errors_lat_m, -errors_lat_m)
    errors_lon_signed = np.where(pred_lon > gt_lon, errors_lon_m, -errors_lon_m)

    # Return comprehensive metrics
    metrics = {
        # Main 2D Metrics
        f"{label}_2D_Mean": np.mean(errors_2d),
        f"{label}_2D_Median_P50": np.percentile(errors_2d, 50),
        f"{label}_2D_P95": np.percentile(errors_2d, 95),
        f"{label}_2D_P99": np.percentile(errors_2d, 99),
        f"{label}_2D_Max": np.max(errors_2d),
        f"{label}_2D_RMSE": np.sqrt(np.mean(errors_2d**2)),
        f"{label}_2D_Std": np.std(errors_2d),

        # Target Metric: Mean of (Mean, P50, P95)
        f"{label}_Mean_of_Mean_P50_P95": (np.mean(errors_2d) + np.percentile(errors_2d, 50) + np.percentile(errors_2d, 95)) / 3,

        # Alternative Target Metric (Mean of P50 and P95)
        f"{label}_Target_Mean_P50_P95": (np.percentile(errors_2d, 50) + np.percentile(errors_2d, 95)) / 2,
        
        # Latitude Component Metrics
        f"{label}_Lat_MAE": np.mean(np.abs(errors_lat_signed)),
        f"{label}_Lat_ME": np.mean(errors_lat_signed),  # Mean Error (shows bias)
        f"{label}_Lat_Median": np.median(np.abs(errors_lat_signed)),
        f"{label}_Lat_P95": np.percentile(np.abs(errors_lat_signed), 95),
        f"{label}_Lat_RMSE": np.sqrt(np.mean(errors_lat_signed**2)),
        
        # Longitude Component Metrics
        f"{label}_Lon_MAE": np.mean(np.abs(errors_lon_signed)),
        f"{label}_Lon_ME": np.mean(errors_lon_signed),  # Mean Error (shows bias)
        f"{label}_Lon_Median": np.median(np.abs(errors_lon_signed)),
        f"{label}_Lon_P95": np.percentile(np.abs(errors_lon_signed), 95),
        f"{label}_Lon_RMSE": np.sqrt(np.mean(errors_lon_signed**2)),
        
        # Additional Percentiles
        f"{label}_2D_P25": np.percentile(errors_2d, 25),
        f"{label}_2D_P75": np.percentile(errors_2d, 75),
        f"{label}_2D_P90": np.percentile(errors_2d, 90),
        
        # Count metrics
        f"{label}_Count": len(errors_2d),
        f"{label}_Errors_Below_1m": np.sum(errors_2d < 1.0),
        f"{label}_Errors_Below_2m": np.sum(errors_2d < 2.0),
        f"{label}_Errors_Below_5m": np.sum(errors_2d < 5.0),
        f"{label}_Errors_Above_10m": np.sum(errors_2d > 10.0),
    }
    
    return metrics


def print_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]]):
    """
    Print formatted comparison of metrics across different methods.
    
    Args:
        metrics_dict: Dictionary mapping method names to their metrics
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE METRICS COMPARISON")
    print("="*80)
    
    # Main 2D metrics
    print("\n📊 2D POSITION ERROR METRICS:")
    print("-" * 80)
    print(f"{'Metric':<30} " + " ".join([f"{name:>15}" for name in metrics_dict.keys()]))
    print("-" * 80)
    
    key_2d_metrics = [
        ("Mean Error", "2D_Mean"),
        ("Median (P50)", "2D_Median_P50"),
        ("P95", "2D_P95"),
        ("P99", "2D_P99"),
        ("Max Error", "2D_Max"),
        ("RMSE", "2D_RMSE"),
        ("Std Dev", "2D_Std"),
        ("🎯 TARGET (Mean+P50+P95)/3", "Mean_of_Mean_P50_P95"),
        ("Target (P50+P95)/2", "Target_Mean_P50_P95"),
    ]
    
    for metric_name, metric_key in key_2d_metrics:
        values = [metrics[f"{name}_{metric_key}"] for name, metrics in metrics_dict.items()]
        value_str = " ".join([f"{v:>15.4f}" for v in values])
        print(f"{metric_name:<30} {value_str}")
    
    # Latitude component
    print("\n📍 LATITUDE COMPONENT METRICS:")
    print("-" * 80)
    lat_metrics = [
        ("MAE", "Lat_MAE"),
        ("Mean Error (bias)", "Lat_ME"),
        ("Median", "Lat_Median"),
        ("P95", "Lat_P95"),
        ("RMSE", "Lat_RMSE"),
    ]
    
    for metric_name, metric_key in lat_metrics:
        values = [metrics[f"{name}_{metric_key}"] for name, metrics in metrics_dict.items()]
        value_str = " ".join([f"{v:>15.4f}" for v in values])
        print(f"{metric_name:<30} {value_str}")
    
    # Longitude component
    print("\n📍 LONGITUDE COMPONENT METRICS:")
    print("-" * 80)
    lon_metrics = [
        ("MAE", "Lon_MAE"),
        ("Mean Error (bias)", "Lon_ME"),
        ("Median", "Lon_Median"),
        ("P95", "Lon_P95"),
        ("RMSE", "Lon_RMSE"),
    ]
    
    for metric_name, metric_key in lon_metrics:
        values = [metrics[f"{name}_{metric_key}"] for name, metrics in metrics_dict.items()]
        value_str = " ".join([f"{v:>15.4f}" for v in values])
        print(f"{metric_name:<30} {value_str}")
    
    # Distribution metrics
    print("\n📈 ERROR DISTRIBUTION:")
    print("-" * 80)
    dist_metrics = [
        ("P25", "2D_P25"),
        ("P50", "2D_Median_P50"),
        ("P75", "2D_P75"),
        ("P90", "2D_P90"),
        ("P95", "2D_P95"),
    ]
    
    for metric_name, metric_key in dist_metrics:
        values = [metrics[f"{name}_{metric_key}"] for name, metrics in metrics_dict.items()]
        value_str = " ".join([f"{v:>15.4f}" for v in values])
        print(f"{metric_name:<30} {value_str}")
    
    # Count metrics
    print("\n🎯 ACCURACY THRESHOLDS:")
    print("-" * 80)
    count_metrics = [
        ("Errors < 1m", "Errors_Below_1m"),
        ("Errors < 2m", "Errors_Below_2m"),
        ("Errors < 5m", "Errors_Below_5m"),
        ("Errors > 10m", "Errors_Above_10m"),
    ]
    
    for metric_name, metric_key in count_metrics:
        values = [metrics[f"{name}_{metric_key}"] for name, metrics in metrics_dict.items()]
        # Show as percentages
        total = list(metrics_dict.values())[0][f"{list(metrics_dict.keys())[0]}_Count"]
        value_str = " ".join([f"{v/total*100:>14.1f}%" for v in values])
        print(f"{metric_name:<30} {value_str}")
    
    print("="*80)


def evaluate_with_kalman(
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    gt_lat: np.ndarray,
    gt_lon: np.ndarray,
    baseline_lat: np.ndarray,
    baseline_lon: np.ndarray,
    timestamps: Optional[pd.Series] = None,
    process_variance: float = 0.5,
    measurement_variance: Optional[float] = None,
    lat_variances: Optional[np.ndarray] = None,
    lon_variances: Optional[np.ndarray] = None
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Evaluate baseline, corrected, and Kalman-smoothed predictions.

    Args:
        pred_lat, pred_lon: Model predictions (degrees)
        gt_lat, gt_lon: Ground truth (degrees)
        baseline_lat, baseline_lon: Baseline POS/PPK solution (degrees)
        timestamps: Optional timestamps for Kalman filtering
        process_variance: Kalman process noise (meters²)
        measurement_variance: Fixed Kalman measurement noise (meters²) - used if variances not provided
        lat_variances: Per-sample lat uncertainties from NN (meters²)
        lon_variances: Per-sample lon uncertainties from NN (meters²)

    Returns:
        Tuple of (baseline_metrics, corrected_metrics, kalman_metrics)
    """
    # Calculate baseline metrics
    baseline_metrics = calculate_position_metrics(
        baseline_lat, baseline_lon, gt_lat, gt_lon, label="Baseline"
    )

    # Calculate corrected metrics
    corrected_metrics = calculate_position_metrics(
        pred_lat, pred_lon, gt_lat, gt_lon, label="Corrected"
    )

    # Apply Kalman smoothing with dynamic or fixed variances
    lat_kalman, lon_kalman = apply_kalman_to_coordinates(
        pred_lat, pred_lon, timestamps,
        process_variance=process_variance,
        measurement_variance=measurement_variance,
        lat_variances=lat_variances,
        lon_variances=lon_variances
    )
    
    # Calculate Kalman metrics
    kalman_metrics = calculate_position_metrics(
        lat_kalman, lon_kalman, gt_lat, gt_lon, label="Kalman"
    )
    
    # Print comparison
    print_metrics_comparison({
        "Baseline": baseline_metrics,
        "Corrected": corrected_metrics,
        "Kalman": kalman_metrics
    })
    
    return baseline_metrics, corrected_metrics, kalman_metrics


class AdaptiveKalmanFilter1D(KalmanFilter1D):
    """
    Adaptive Kalman Filter that adjusts noise parameters based on residuals.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.innovation_history = []
        self.window_size = 10
    
    def update(self, measurement: float):
        """Update with adaptive noise estimation."""
        # Calculate innovation
        y = measurement - (self.H @ self.state)[0]
        self.innovation_history.append(y)
        
        # Keep only recent innovations
        if len(self.innovation_history) > self.window_size:
            self.innovation_history.pop(0)
        
        # Adapt measurement noise based on recent innovations
        if len(self.innovation_history) >= 3:
            innovation_var = np.var(self.innovation_history)
            # Blend with original R
            alpha = 0.3  # Adaptation rate
            self.R[0, 0] = (1 - alpha) * self.R[0, 0] + alpha * innovation_var
        
        # Standard Kalman update
        super().update(measurement)


def apply_adaptive_kalman(
    positions: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    process_variance: float = 0.5,
    measurement_variance: float = 5.0
) -> np.ndarray:
    """
    Apply adaptive Kalman filter with automatic noise tuning.
    
    Args:
        positions: Array of position values
        timestamps: Optional timestamps
        process_variance: Initial process noise
        measurement_variance: Initial measurement noise
    
    Returns:
        Smoothed position array
    """
    if len(positions) == 0:
        return positions
    
    kf = AdaptiveKalmanFilter1D(
        process_variance=process_variance,
        measurement_variance=measurement_variance,
        initial_position=positions[0]
    )
    
    smoothed = np.zeros_like(positions)
    smoothed[0] = positions[0]
    
    for i in range(1, len(positions)):
        dt = 1.0
        if timestamps is not None:
            dt = (timestamps[i] - timestamps[i-1]).total_seconds()
            dt = max(dt, 0.01)
        
        kf.predict(dt)
        kf.update(positions[i])
        smoothed[i] = kf.get_position()
    
    return smoothed


# Example usage
if __name__ == "__main__":
    # Generate synthetic noisy position data
    np.random.seed(42)
    n_samples = 100
    
    # True trajectory
    true_lat = 37.7749 + np.linspace(0, 0.001, n_samples)  # SF coordinates
    true_lon = -122.4194 + np.linspace(0, 0.001, n_samples)
    
    # Add noise to create baseline
    baseline_lat = true_lat + np.random.normal(0, 0.00005, n_samples)
    baseline_lon = true_lon + np.random.normal(0, 0.00005, n_samples)
    
    # Simulated model correction (improves but still has noise)
    corrected_lat = true_lat + np.random.normal(0, 0.00003, n_samples)
    corrected_lon = true_lon + np.random.normal(0, 0.00003, n_samples)
    
    # Apply Kalman
    lat_kalman, lon_kalman = apply_kalman_to_coordinates(
        corrected_lat, corrected_lon,
        process_variance=0.5,
        measurement_variance=5.0
    )
    
    # Calculate all metrics
    baseline_metrics = calculate_position_metrics(
        baseline_lat, baseline_lon, true_lat, true_lon, "Baseline"
    )
    corrected_metrics = calculate_position_metrics(
        corrected_lat, corrected_lon, true_lat, true_lon, "Corrected"
    )
    kalman_metrics = calculate_position_metrics(
        lat_kalman, lon_kalman, true_lat, true_lon, "Kalman"
    )
    
    # Print comparison
    print_metrics_comparison({
        "Baseline": baseline_metrics,
        "Corrected": corrected_metrics,
        "Kalman": kalman_metrics
    })