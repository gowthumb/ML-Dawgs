"""
Kalman Smoothing for GNSS position corrections.

Applies 1D Kalman filter to smooth noisy predictions and reduce outliers (P95).
"""

import numpy as np
import pandas as pd
from typing import Tuple


class KalmanSmoother1D:
    """
    Simple 1D Kalman Filter for smoothing position corrections.

    This filter assumes a constant velocity model:
    - State: [position, velocity]
    - Observation: position only
    """

    def __init__(
        self,
        process_variance: float = 1.0,
        measurement_variance: float = 1.0,
        initial_position: float = 0.0,
        initial_velocity: float = 0.0,
    ):
        """
        Initialize Kalman filter.

        Args:
            process_variance: Process noise (how much the model can deviate)
            measurement_variance: Measurement noise (how noisy the predictions are)
            initial_position: Initial position estimate
            initial_velocity: Initial velocity estimate
        """
        # State: [position, velocity]
        self.x = np.array([initial_position, initial_velocity])

        # State covariance
        self.P = np.eye(2) * 100  # High initial uncertainty

        # Process noise covariance
        self.Q = np.array([
            [process_variance, 0],
            [0, process_variance * 0.1]  # Velocity has less noise
        ])

        # Measurement noise covariance
        self.R = np.array([[measurement_variance]])

        # State transition matrix (constant velocity model)
        self.dt = 1.0  # Time step (assumed constant)
        self.F = np.array([
            [1, self.dt],
            [0, 1]
        ])

        # Observation matrix (observe position only)
        self.H = np.array([[1, 0]])

    def predict(self):
        """Predict next state."""
        # Predict state
        self.x = self.F @ self.x

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: float):
        """Update state with measurement."""
        # Measurement residual
        y = measurement - (self.H @ self.x)[0]

        # Residual covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K.flatten() * y

        # Update covariance
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P

    def get_position(self) -> float:
        """Get current position estimate."""
        return self.x[0]


def apply_kalman_smoothing(
    predictions: np.ndarray,
    process_variance: float = 1.0,
    measurement_variance: float = 4.0,
) -> np.ndarray:
    """
    Apply Kalman smoothing to a sequence of predictions.

    Args:
        predictions: Array of noisy predictions
        process_variance: Process noise (lower = smoother, higher = follows predictions more)
        measurement_variance: Measurement noise (higher = trust predictions less)

    Returns:
        Array of smoothed predictions
    """
    if len(predictions) == 0:
        return predictions

    # Initialize filter with first measurement
    kf = KalmanSmoother1D(
        process_variance=process_variance,
        measurement_variance=measurement_variance,
        initial_position=predictions[0],
        initial_velocity=0.0,
    )

    smoothed = []

    for measurement in predictions:
        kf.predict()
        kf.update(measurement)
        smoothed.append(kf.get_position())

    return np.array(smoothed)


def apply_kalman_to_corrections(
    pred_lat_m: np.ndarray,
    pred_lon_m: np.ndarray,
    process_variance: float = 1.0,
    measurement_variance: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Kalman smoothing to lat and lon corrections separately.

    This reduces outliers and smooths noisy predictions, which should
    improve P95 while maintaining good P50 performance.

    Args:
        pred_lat_m: Predicted latitude corrections (meters)
        pred_lon_m: Predicted longitude corrections (meters)
        process_variance: Process noise (1.0 = moderate smoothing)
        measurement_variance: Measurement noise (4.0 = trust predictions moderately)

    Returns:
        tuple: (smoothed_lat_m, smoothed_lon_m)
    """
    print("\nApplying Kalman smoothing to predictions...")
    print(f"  Process variance: {process_variance}")
    print(f"  Measurement variance: {measurement_variance}")

    # Smooth latitude
    smoothed_lat = apply_kalman_smoothing(
        pred_lat_m,
        process_variance=process_variance,
        measurement_variance=measurement_variance,
    )

    # Smooth longitude
    smoothed_lon = apply_kalman_smoothing(
        pred_lon_m,
        process_variance=process_variance,
        measurement_variance=measurement_variance,
    )

    # Print statistics
    print(f"\nBefore smoothing:")
    print(f"  Lat - Mean: {pred_lat_m.mean():.2f} m, Std: {pred_lat_m.std():.2f} m")
    print(f"  Lon - Mean: {pred_lon_m.mean():.2f} m, Std: {pred_lon_m.std():.2f} m")
    print(f"\nAfter smoothing:")
    print(f"  Lat - Mean: {smoothed_lat.mean():.2f} m, Std: {smoothed_lat.std():.2f} m")
    print(f"  Lon - Mean: {smoothed_lon.mean():.2f} m, Std: {smoothed_lon.std():.2f} m")

    return smoothed_lat, smoothed_lon


# ============================================================
# USAGE EXAMPLE
# ============================================================
"""
Usage in evaluation:

```python
from kalman_smoother import apply_kalman_to_corrections

# After predicting components
pred_lat_m, pred_lon_m = trainer.predict_components(X_test)

# Apply Kalman smoothing
smoothed_lat_m, smoothed_lon_m = apply_kalman_to_corrections(
    pred_lat_m, pred_lon_m,
    process_variance=1.0,      # Tune this (lower = smoother)
    measurement_variance=4.0,  # Tune this (higher = more smoothing)
)

# Use smoothed predictions for correction
# ... rest of evaluation code
```

## Tuning Parameters

- **process_variance** (default: 1.0)
  - Lower (0.1-0.5): More aggressive smoothing, reduces outliers more
  - Higher (2.0-5.0): Less smoothing, follows predictions more closely

- **measurement_variance** (default: 4.0)
  - Lower (1.0-2.0): Trust predictions more, less smoothing
  - Higher (5.0-10.0): Trust predictions less, more smoothing

For reducing P95 while maintaining P50:
- Start with: process_variance=0.5, measurement_variance=5.0
- If P95 still high: increase measurement_variance to 8.0-10.0
- If P50 gets worse: decrease measurement_variance to 3.0-4.0
"""
