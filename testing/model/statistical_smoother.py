"""
Statistical Adaptive Filter - Kalman-inspired post-processing using statistical metrics
Uses running statistics (mean, std dev, variance) to adaptively smooth predictions
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def statistical_adaptive_filter(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_variance: Optional[np.ndarray] = None,
    lon_variance: Optional[np.ndarray] = None,
    window_size: int = 5,
    alpha_base: float = 0.3,
    use_uncertainty: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Statistical adaptive filter using rolling statistics.

    Key Ideas:
    - Use rolling mean/std to detect anomalies
    - Adjust smoothing based on local variance
    - Weight by NN uncertainty if available
    - More smoothing when predictions are stable
    - Less smoothing when predictions are changing rapidly

    Args:
        lat, lon: Position predictions
        lat_variance, lon_variance: NN uncertainty estimates (optional)
        window_size: Rolling window size for statistics
        alpha_base: Base smoothing factor (0=no smooth, 1=full smooth)
        use_uncertainty: Whether to use NN uncertainties

    Returns:
        Smoothed lat, lon
    """
    n = len(lat)
    lat_smooth = np.zeros_like(lat)
    lon_smooth = np.zeros_like(lon)

    # Initialize with first value
    lat_smooth[0] = lat[0]
    lon_smooth[0] = lon[0]

    for i in range(1, n):
        # Get rolling window
        start_idx = max(0, i - window_size)
        window_lat = lat[start_idx:i]
        window_lon = lon[start_idx:i]

        # Calculate rolling statistics
        lat_mean = np.mean(window_lat)
        lat_std = np.std(window_lat) + 1e-6
        lon_mean = np.mean(window_lon)
        lon_std = np.std(window_lon) + 1e-6

        # Detect if current prediction is anomalous
        lat_z_score = abs((lat[i] - lat_mean) / lat_std)
        lon_z_score = abs((lon[i] - lon_mean) / lon_std)

        # Adaptive smoothing factor based on stability
        # High z-score (anomaly) -> less smoothing (trust new measurement)
        # Low z-score (stable) -> more smoothing
        lat_alpha = alpha_base * np.exp(-lat_z_score / 2.0)
        lon_alpha = alpha_base * np.exp(-lon_z_score / 2.0)

        # Adjust by uncertainty if available
        if use_uncertainty and lat_variance is not None and lon_variance is not None:
            # High uncertainty -> more smoothing
            lat_uncertainty_factor = np.clip(lat_variance[i] / 10.0, 0.5, 2.0)
            lon_uncertainty_factor = np.clip(lon_variance[i] / 10.0, 0.5, 2.0)

            lat_alpha *= lat_uncertainty_factor
            lon_alpha *= lon_uncertainty_factor

        # Clip alpha to valid range
        lat_alpha = np.clip(lat_alpha, 0.0, 0.9)
        lon_alpha = np.clip(lon_alpha, 0.0, 0.9)

        # Exponential moving average with adaptive alpha
        lat_smooth[i] = lat_alpha * lat_smooth[i-1] + (1 - lat_alpha) * lat[i]
        lon_smooth[i] = lon_alpha * lon_smooth[i-1] + (1 - lon_alpha) * lon[i]

    return lat_smooth, lon_smooth


def robust_outlier_filter(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_variance: Optional[np.ndarray] = None,
    lon_variance: Optional[np.ndarray] = None,
    window_size: int = 7,
    z_threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust filter that detects and replaces outliers using statistics.

    Method:
    - Calculate rolling median and MAD (Median Absolute Deviation)
    - Flag points beyond z_threshold * MAD
    - Replace outliers with rolling median
    - Use NN uncertainty as additional weight

    Args:
        lat, lon: Position predictions
        lat_variance, lon_variance: NN uncertainty estimates (optional)
        window_size: Rolling window for statistics
        z_threshold: Z-score threshold for outlier detection

    Returns:
        Filtered lat, lon
    """
    lat_filtered = lat.copy()
    lon_filtered = lon.copy()

    # Convert to pandas for rolling operations
    lat_series = pd.Series(lat)
    lon_series = pd.Series(lon)

    # Calculate rolling median and MAD
    lat_median = lat_series.rolling(window=window_size, center=True, min_periods=1).median()
    lon_median = lon_series.rolling(window=window_size, center=True, min_periods=1).median()

    lat_mad = lat_series.rolling(window=window_size, center=True, min_periods=1).apply(
        lambda x: np.median(np.abs(x - np.median(x)))
    )
    lon_mad = lon_series.rolling(window=window_size, center=True, min_periods=1).apply(
        lambda x: np.median(np.abs(x - np.median(x)))
    )

    # Detect outliers using modified z-score
    lat_z_score = np.abs((lat - lat_median) / (lat_mad + 1e-6))
    lon_z_score = np.abs((lon - lon_median) / (lon_mad + 1e-6))

    # Adjust threshold by uncertainty if available
    if lat_variance is not None and lon_variance is not None:
        # High uncertainty -> lower threshold (more likely to be outlier)
        lat_threshold = z_threshold / np.sqrt(lat_variance + 1.0)
        lon_threshold = z_threshold / np.sqrt(lon_variance + 1.0)
    else:
        lat_threshold = z_threshold
        lon_threshold = z_threshold

    # Replace outliers with rolling median
    lat_outliers = lat_z_score > lat_threshold
    lon_outliers = lon_z_score > lon_threshold

    lat_filtered[lat_outliers] = lat_median[lat_outliers]
    lon_filtered[lon_outliers] = lon_median[lon_outliers]

    print(f"  Detected {lat_outliers.sum()} lat outliers, {lon_outliers.sum()} lon outliers")

    return lat_filtered, lon_filtered


def variance_weighted_smoother(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_variance: np.ndarray,
    lon_variance: np.ndarray,
    window_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted moving average using NN uncertainties as weights.

    Method:
    - Weight each point by inverse variance (precision)
    - Apply weighted moving average
    - More weight to confident predictions

    Args:
        lat, lon: Position predictions
        lat_variance, lon_variance: NN uncertainty estimates (required)
        window_size: Window size for averaging

    Returns:
        Smoothed lat, lon
    """
    n = len(lat)
    lat_smooth = np.zeros_like(lat)
    lon_smooth = np.zeros_like(lon)

    # Convert variance to precision (inverse variance)
    lat_precision = 1.0 / (lat_variance + 1e-6)
    lon_precision = 1.0 / (lon_variance + 1e-6)

    for i in range(n):
        # Get window indices
        start_idx = max(0, i - window_size // 2)
        end_idx = min(n, i + window_size // 2 + 1)

        # Get window values and weights
        lat_window = lat[start_idx:end_idx]
        lon_window = lon[start_idx:end_idx]
        lat_weights = lat_precision[start_idx:end_idx]
        lon_weights = lon_precision[start_idx:end_idx]

        # Weighted average
        lat_smooth[i] = np.average(lat_window, weights=lat_weights)
        lon_smooth[i] = np.average(lon_window, weights=lon_weights)

    return lat_smooth, lon_smooth


def group_wise_statistical_filter(
    df: pd.DataFrame,
    filter_type: str = 'adaptive',
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply statistical filter per drive group.

    Args:
        df: DataFrame with columns:
            - corrected_lat, corrected_lon
            - drive_id
            - timestamp
            - lat_variance, lon_variance (optional)
        filter_type: 'adaptive', 'outlier', or 'weighted'
        **kwargs: Additional arguments for the filter

    Returns:
        Smoothed lat, lon arrays in original order
    """
    smoothed_lat_list = []
    smoothed_lon_list = []

    for drive_id, group in df.groupby('drive_id'):
        # Sort by timestamp
        group = group.sort_values('timestamp')

        # Get arrays
        lat = group['corrected_lat'].values
        lon = group['corrected_lon'].values
        lat_var = group.get('lat_variance', pd.Series([None] * len(group))).values
        lon_var = group.get('lon_variance', pd.Series([None] * len(group))).values

        # Apply selected filter with appropriate parameters
        if filter_type == 'adaptive':
            # Adaptive filter accepts: window_size, alpha_base, use_uncertainty
            lat_smooth, lon_smooth = statistical_adaptive_filter(
                lat, lon, lat_var, lon_var,
                window_size=kwargs.get('window_size', 5),
                alpha_base=kwargs.get('alpha_base', 0.3),
                use_uncertainty=kwargs.get('use_uncertainty', True)
            )
        elif filter_type == 'outlier':
            # Outlier filter accepts: window_size, z_threshold
            lat_smooth, lon_smooth = robust_outlier_filter(
                lat, lon, lat_var, lon_var,
                window_size=kwargs.get('window_size', 7),
                z_threshold=kwargs.get('z_threshold', 3.0)
            )
        elif filter_type == 'weighted':
            if lat_var[0] is not None:
                # Weighted filter only accepts: window_size
                lat_smooth, lon_smooth = variance_weighted_smoother(
                    lat, lon, lat_var, lon_var,
                    window_size=kwargs.get('window_size', 5)
                )
            else:
                print("Warning: 'weighted' filter requires variance, falling back to 'adaptive'")
                lat_smooth, lon_smooth = statistical_adaptive_filter(
                    lat, lon, None, None,
                    window_size=kwargs.get('window_size', 5),
                    alpha_base=kwargs.get('alpha_base', 0.3),
                    use_uncertainty=False
                )
        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")

        # Store with original index
        smoothed_lat_list.append(pd.Series(lat_smooth, index=group.index))
        smoothed_lon_list.append(pd.Series(lon_smooth, index=group.index))

    # Combine and restore original order
    lat_final = pd.concat(smoothed_lat_list).loc[df.index].values
    lon_final = pd.concat(smoothed_lon_list).loc[df.index].values

    return lat_final, lon_final


if __name__ == "__main__":
    # Test the filters
    print("Statistical Smoother - Test")

    # Create test data
    n = 100
    t = np.linspace(0, 10, n)
    true_lat = np.sin(t) * 0.0001 + 37.4
    true_lon = np.cos(t) * 0.0001 - 122.1

    # Add noise
    noise_lat = np.random.normal(0, 0.00002, n)
    noise_lon = np.random.normal(0, 0.00002, n)
    noisy_lat = true_lat + noise_lat
    noisy_lon = true_lon + noise_lon

    # Add outliers
    noisy_lat[20] += 0.0005
    noisy_lon[50] -= 0.0005

    # Create fake variances
    variances = np.random.uniform(1.0, 5.0, n)

    # Test filters
    print("\n1. Adaptive Filter:")
    smooth_lat, smooth_lon = statistical_adaptive_filter(
        noisy_lat, noisy_lon, variances, variances
    )
    print(f"   RMSE before: {np.sqrt(np.mean((noisy_lat - true_lat)**2)):.6f}")
    print(f"   RMSE after:  {np.sqrt(np.mean((smooth_lat - true_lat)**2)):.6f}")

    print("\n2. Outlier Filter:")
    filt_lat, filt_lon = robust_outlier_filter(
        noisy_lat, noisy_lon, variances, variances
    )
    print(f"   RMSE after:  {np.sqrt(np.mean((filt_lat - true_lat)**2)):.6f}")

    print("\n3. Weighted Smoother:")
    weight_lat, weight_lon = variance_weighted_smoother(
        noisy_lat, noisy_lon, variances, variances
    )
    print(f"   RMSE after:  {np.sqrt(np.mean((weight_lat - true_lat)**2)):.6f}")
