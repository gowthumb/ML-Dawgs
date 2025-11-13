"""
Inference utilities for applying residual learning corrections.
Supports both single magnitude models (legacy) and component models (lat/lon separate).
"""

import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Optional

from config import METERS_PER_DEGREE_LAT


class ComponentResidualCorrector:
    """
    Apply component-based residual corrections at inference time.
    Uses separate lat and lon models with optional Kalman smoothing.
    """

    def __init__(
        self,
        model_lat_path: str,
        model_lon_path: str,
        apply_kalman: bool = True,
        process_variance: float = 0.5,
        measurement_variance: float = 5.0,
    ):
        """
        Initialize corrector with trained component models.

        Args:
            model_lat_path: Path to latitude model pickle file
            model_lon_path: Path to longitude model pickle file
            apply_kalman: Whether to apply Kalman smoothing
            process_variance: Kalman process noise (lower = smoother)
            measurement_variance: Kalman measurement noise (higher = more smoothing)
        """
        with open(model_lat_path, 'rb') as f:
            self.model_lat = pickle.load(f)

        with open(model_lon_path, 'rb') as f:
            self.model_lon = pickle.load(f)

        self.apply_kalman = apply_kalman
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

        print(f"Loaded component models:")
        print(f"  Lat model: {model_lat_path}")
        print(f"  Lon model: {model_lon_path}")
        print(f"  Kalman smoothing: {apply_kalman}")

    def predict_residuals(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict lat and lon residuals for given features.

        Args:
            features: DataFrame with GNSS + IMU features

        Returns:
            tuple: (predicted_lat_residual_m, predicted_lon_residual_m)
        """
        pred_lat = self.model_lat.predict(features)
        pred_lon = self.model_lon.predict(features)

        # Apply Kalman smoothing if enabled
        if self.apply_kalman:
            from kalman_smoother import apply_kalman_to_corrections

            pred_lat, pred_lon = apply_kalman_to_corrections(
                pred_lat, pred_lon,
                process_variance=self.process_variance,
                measurement_variance=self.measurement_variance,
            )

        return pred_lat, pred_lon

    def apply_correction(
        self,
        baseline_lat: pd.Series,
        baseline_lon: pd.Series,
        features: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Apply residual corrections to baseline positions.

        Args:
            baseline_lat, baseline_lon: Baseline positions from POS/PPK
            features: Feature DataFrame

        Returns:
            tuple: (corrected_lat, corrected_lon)
        """
        # Predict residuals
        pred_lat_m, pred_lon_m = self.predict_residuals(features)

        # Convert to degrees
        lat_correction_deg = pred_lat_m / METERS_PER_DEGREE_LAT
        lon_correction_deg = pred_lon_m / (METERS_PER_DEGREE_LAT * np.cos(np.radians(baseline_lat)))

        # Apply corrections
        corrected_lat = baseline_lat + lat_correction_deg
        corrected_lon = baseline_lon + lon_correction_deg

        return corrected_lat, corrected_lon


class ResidualCorrector:
    """
    Apply residual learning corrections at inference time.
    """

    def __init__(self, model_path: str):
        """
        Initialize corrector with trained model.

        Args:
            model_path: Path to saved model pickle file
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        print(f"Loaded model from: {model_path}")

    def predict_residual(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict residual magnitude for given features.

        Args:
            features: DataFrame with GNSS + IMU features

        Returns:
            Array of predicted residual magnitudes in meters
        """
        return self.model.predict(features)

    def apply_correction(
        self,
        baseline_lat: float,
        baseline_lon: float,
        features: pd.DataFrame,
        lat_residual_direction: float = None,
        lon_residual_direction: float = None,
    ) -> Tuple[float, float]:
        """
        Apply residual correction to baseline position.

        This is a simplified correction that assumes you have some estimate
        of the residual direction. In production, you would:
        1. Train separate models for lat and lon residuals, OR
        2. Use a more sophisticated direction estimation

        Args:
            baseline_lat, baseline_lon: Baseline position from POS/PPK
            features: Feature DataFrame (single sample or batch)
            lat_residual_direction: Estimated N/S residual direction (-1 to 1)
            lon_residual_direction: Estimated E/W residual direction (-1 to 1)

        Returns:
            tuple: (corrected_lat, corrected_lon)
        """
        # Predict residual magnitude
        predicted_residual = self.predict_residual(features)

        # If no direction provided, assume no systematic bias
        if lat_residual_direction is None:
            lat_residual_direction = 0
        if lon_residual_direction is None:
            lon_residual_direction = 0

        # Apply correction (simplified)
        # In production, use separate models for each component
        lat_correction_m = predicted_residual * lat_residual_direction
        lon_correction_m = predicted_residual * lon_residual_direction

        # Convert to degrees
        lat_correction_deg = lat_correction_m / METERS_PER_DEGREE_LAT
        lon_correction_deg = lon_correction_m / (
            METERS_PER_DEGREE_LAT * np.cos(np.radians(baseline_lat))
        )

        # Apply
        corrected_lat = baseline_lat + lat_correction_deg
        corrected_lon = baseline_lon + lon_correction_deg

        return corrected_lat, corrected_lon

    def batch_correction(
        self,
        baseline_lat: pd.Series,
        baseline_lon: pd.Series,
        features: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Apply corrections to a batch of positions.

        Note: This simplified version assumes no systematic directional bias.
        For better results, train separate models for lat/lon components.

        Args:
            baseline_lat, baseline_lon: Baseline positions
            features: Feature DataFrame

        Returns:
            tuple: (corrected_lat, corrected_lon)
        """
        # Predict residuals
        predicted_residuals = self.predict_residual(features)

        print(f"\nBatch correction:")
        print(f"  Samples: {len(baseline_lat)}")
        print(f"  Mean predicted residual: {predicted_residuals.mean():.2f} m")
        print(f"  Std predicted residual:  {predicted_residuals.std():.2f} m")

        # For this simplified version, we return baseline positions
        # In production, use the full correction logic from training
        print("\n⚠️  Note: Using simplified correction (no directional component)")
        print("    For production use, train separate models for lat/lon residuals")

        return baseline_lat, baseline_lon


def load_model_and_correct(
    model_path: str,
    baseline_lat: float,
    baseline_lon: float,
    features: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Convenience function to load model and apply correction.

    Args:
        model_path: Path to saved model
        baseline_lat, baseline_lon: Baseline position
        features: Features DataFrame

    Returns:
        tuple: (corrected_lat, corrected_lon)
    """
    corrector = ResidualCorrector(model_path)
    return corrector.apply_correction(baseline_lat, baseline_lon, features)


# ============================================================
# USAGE EXAMPLES
# ============================================================
"""
## Usage 1: Component Models with Kalman Smoothing (RECOMMENDED)

```python
from residual_inference import ComponentResidualCorrector

# Load component models
corrector = ComponentResidualCorrector(
    model_lat_path='model/outputs/residual_model_lat_TIMESTAMP.pkl',
    model_lon_path='model/outputs/residual_model_lon_TIMESTAMP.pkl',
    apply_kalman=True,
    process_variance=0.5,
    measurement_variance=5.0,
)

# Get baseline positions (from POS/PPK)
baseline_lat = pos_data['latitude']
baseline_lon = pos_data['longitude']

# Extract features (GNSS + IMU)
features = extract_features(gnss_data, imu_data)

# Apply corrections (with Kalman smoothing)
corrected_lat, corrected_lon = corrector.apply_correction(
    baseline_lat, baseline_lon, features
)
```

## Usage 2: Legacy Single Model (NOT RECOMMENDED)

```python
from residual_inference import ResidualCorrector

# Load model
corrector = ResidualCorrector('model/outputs/residual_model_TIMESTAMP.pkl')

# Get baseline position (from POS/PPK)
baseline_lat = pos_data['latitude']
baseline_lon = pos_data['longitude']

# Extract features (GNSS + IMU)
features = extract_features(gnss_data, imu_data)

# Predict residual
predicted_residual = corrector.predict_residual(features)

# Apply correction (needs direction estimation)
corrected_lat, corrected_lon = corrector.apply_correction(
    baseline_lat, baseline_lon, features
)
```

## Kalman Tuning for Target Metric (Mean(P50, P95) = 1-2m)

If P95 is too high:
- Increase measurement_variance: 5.0 → 8.0 → 10.0
- Decrease process_variance: 0.5 → 0.3 → 0.1

If P50 gets worse:
- Decrease measurement_variance: 5.0 → 3.0 → 2.0
- Increase process_variance: 0.5 → 1.0 → 2.0

Target: P50 ≈ 0.5-1.5m, P95 ≈ 0.5-3.5m → Mean = 1-2m
"""
