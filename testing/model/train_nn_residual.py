"""
Neural Network Residual Learning with Uncertainty Estimation
=============================================================

Stacked GRU model with Gaussian NLL loss for predicting position residuals
with uncertainty estimates.

Architecture:
- Input: (window_size, num_features) temporal sequences
- Stacked GRU layers for temporal pattern learning
- Separate models for lat/lon residuals
- Uncertainty quantification via Gaussian NLL

Usage:
    python train_nn_residual.py
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TensorFlow/Keras imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Import from existing modules
from data_loader import load_training_data
from feature_engineering import compute_residuals, prepare_training_data
from config import BASELINE_LAT, BASELINE_LON, GT_LAT, GT_LON, DATA_PATH, TRAIN_DATA_ROOT

# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SIZE = 50  # Number of time steps to look back
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
VAL_SIZE = 0.2
RANDOM_STATE = 42

# Model architecture
GRU_UNITS_1 = 256
GRU_UNITS_2 = 128
DROPOUT_RATE = 0.3

# Output directory
OUTPUT_DIR = Path("model/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set random seeds
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")


# ============================================================
# SLIDING WINDOW TRANSFORMATION
# ============================================================

def create_sliding_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform data into sliding windows for sequential modeling.

    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Target array of shape (n_samples,)
        window_size: Number of time steps per window
        stride: Step size between windows

    Returns:
        X_windows: Shape (n_windows, window_size, n_features)
        y_windows: Shape (n_windows,) - target at end of each window
    """
    n_samples, n_features = X.shape
    n_windows = (n_samples - window_size) // stride + 1

    X_windows = np.zeros((n_windows, window_size, n_features))
    y_windows = np.zeros(n_windows)

    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        X_windows[i] = X[start_idx:end_idx]
        y_windows[i] = y[end_idx - 1]  # Target is last value in window

    return X_windows, y_windows


# ============================================================
# GAUSSIAN NEGATIVE LOG LIKELIHOOD LOSS
# ============================================================

def gaussian_nll_loss(y_true, y_pred):
    """
    Gaussian Negative Log Likelihood loss.

    Model outputs 2 values per sample:
        y_pred[:, 0]: predicted mean (mu)
        y_pred[:, 1]: predicted log-variance (log(sigma^2))

    NLL = 0.5 * [log(sigma^2) + (y_true - mu)^2 / sigma^2]

    Using log-variance for numerical stability.
    """
    mean = y_pred[:, 0:1]  # Keep dimension
    log_var = y_pred[:, 1:2]

    # Clip log_var for stability
    log_var = tf.clip_by_value(log_var, -10.0, 10.0)

    # Compute variance
    var = tf.exp(log_var)

    # Compute squared error
    squared_error = tf.square(y_true - mean)

    # Compute NLL
    loss = 0.5 * (log_var + squared_error / (var + 1e-6))

    return tf.reduce_mean(loss)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

def build_nll_model(
    window_size: int,
    num_features: int,
    gru_units_1: int = GRU_UNITS_1,
    gru_units_2: int = GRU_UNITS_2,
    dropout_rate: float = DROPOUT_RATE
) -> Model:
    """
    Build Stacked GRU model with Gaussian NLL output.

    Architecture:
        Input -> BatchNorm -> GRU(return_sequences=True) -> Dropout
              -> GRU -> Dropout -> Dense(2: mean, log_var)

    Args:
        window_size: Number of time steps
        num_features: Number of input features
        gru_units_1: Units in first GRU layer
        gru_units_2: Units in second GRU layer
        dropout_rate: Dropout rate

    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=(window_size, num_features), name='input')

    # Batch normalization
    x = layers.BatchNormalization(name='batch_norm')(inputs)

    # First GRU layer (return sequences for stacking)
    x = layers.GRU(
        gru_units_1,
        return_sequences=True,
        name='gru_1'
    )(x)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)

    # Second GRU layer
    x = layers.GRU(
        gru_units_2,
        return_sequences=False,
        name='gru_2'
    )(x)
    x = layers.Dropout(dropout_rate, name='dropout_2')(x)

    # Output: 2 units (mean, log_variance)
    outputs = layers.Dense(2, name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='gru_nll_model')

    return model


# ============================================================
# EVALUATION METRICS
# ============================================================

def compute_position_error(pred_lat, pred_lon, gt_lat, gt_lon):
    """Compute horizontal position error in meters."""
    lat_diff_m = (pred_lat - gt_lat) * 111320.0
    lon_diff_m = (pred_lon - gt_lon) * 111320.0 * np.cos(np.radians(gt_lat))
    error_m = np.sqrt(lat_diff_m**2 + lon_diff_m**2)
    return error_m


def compute_metrics(errors: np.ndarray) -> Dict:
    """Compute comprehensive error metrics."""
    return {
        'RMSE': np.sqrt(np.mean(errors**2)),
        'MAE': np.mean(errors),
        'Median (P50)': np.percentile(errors, 50),
        'P95': np.percentile(errors, 95),
        'Mean(P50,P95)': (np.percentile(errors, 50) + np.percentile(errors, 95)) / 2,
        'Max': np.max(errors)
    }


def display_metrics_comparison(baseline_metrics: Dict, nn_metrics: Dict):
    """Display side-by-side metrics comparison."""
    print("\n" + "="*80)
    print(f"{'Metric':<20} {'Baseline (m)':>15} {'NN Corrected (m)':>18} {'Improvement':>12}")
    print("-"*80)

    for metric in baseline_metrics.keys():
        baseline_val = baseline_metrics[metric]
        nn_val = nn_metrics[metric]
        improvement = (baseline_val - nn_val) / baseline_val * 100
        print(f"{metric:<20} {baseline_val:>15.2f} {nn_val:>18.2f} {improvement:>11.1f}%")

    print("="*80)


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    print("="*70)
    print("NEURAL NETWORK RESIDUAL LEARNING WITH UNCERTAINTY")
    print("="*70)

    # ============================================================
    # STEP 1: LOAD DATA
    # ============================================================
    print("\n" + "="*70)
    print("STEP 1: LOAD DATA")
    print("="*70)

    df = load_training_data(DATA_PATH, TRAIN_DATA_ROOT)
    print(f"Loaded {len(df):,} samples")

    # ============================================================
    # STEP 2: COMPUTE RESIDUALS
    # ============================================================
    print("\n" + "="*70)
    print("STEP 2: COMPUTE RESIDUALS")
    print("="*70)

    df = compute_residuals(df)
    print(f"Computed residuals for {len(df):,} samples")

    # ============================================================
    # STEP 3: PREPARE TRAINING DATA
    # ============================================================
    print("\n" + "="*70)
    print("STEP 3: PREPARE TRAINING DATA")
    print("="*70)

    X, y_lat, y_lon, baseline_lat, baseline_lon, gt_lat, gt_lon, valid_mask = prepare_training_data(
        df, use_component_targets=True
    )

    print(f"Features shape: {X.shape}")
    print(f"Valid samples: {len(X):,}")

    # ============================================================
    # STEP 4: SCALE FEATURES (CRITICAL FOR NN)
    # ============================================================
    print("\n" + "="*70)
    print("STEP 4: SCALE FEATURES (CRITICAL FOR NN)")
    print("="*70)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Scaled features: mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")

    # Save scaler for inference
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    scaler_path = OUTPUT_DIR / f"scaler_nn_{timestamp_str}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")

    # ============================================================
    # STEP 5: CREATE SLIDING WINDOWS
    # ============================================================
    print("\n" + "="*70)
    print("STEP 5: CREATE SLIDING WINDOWS")
    print("="*70)

    print(f"Creating windows with size={WINDOW_SIZE}...")
    X_windows_lat, y_lat_windows = create_sliding_windows(X_scaled, y_lat.values, WINDOW_SIZE)
    X_windows_lon, y_lon_windows = create_sliding_windows(X_scaled, y_lon.values, WINDOW_SIZE)

    print(f"Window data shape: {X_windows_lat.shape}")
    print(f"Target shape: {y_lat_windows.shape}")
    print(f"Lost {len(X_scaled) - len(X_windows_lat)} samples to windowing")

    # Align other data with windows
    window_offset = len(X_scaled) - len(X_windows_lat)
    baseline_lat_windowed = baseline_lat.values[window_offset:]
    baseline_lon_windowed = baseline_lon.values[window_offset:]
    gt_lat_windowed = gt_lat.values[window_offset:]
    gt_lon_windowed = gt_lon.values[window_offset:]

    # ============================================================
    # STEP 6: TRAIN/VAL SPLIT
    # ============================================================
    print("\n" + "="*70)
    print("STEP 6: TRAIN/VALIDATION SPLIT")
    print("="*70)

    # Split for latitude
    X_train_lat, X_val_lat, y_train_lat, y_val_lat, \
    baseline_lat_train, baseline_lat_val, \
    gt_lat_train, gt_lat_val = train_test_split(
        X_windows_lat, y_lat_windows, baseline_lat_windowed, gt_lat_windowed,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=False  # Keep temporal order
    )

    # Split for longitude (same split)
    X_train_lon, X_val_lon, y_train_lon, y_val_lon, \
    baseline_lon_train, baseline_lon_val, \
    gt_lon_train, gt_lon_val = train_test_split(
        X_windows_lon, y_lon_windows, baseline_lon_windowed, gt_lon_windowed,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=False
    )

    print(f"Training samples: {len(X_train_lat):,}")
    print(f"Validation samples: {len(X_val_lat):,}")

    # ============================================================
    # STEP 7: TRAIN LATITUDE MODEL
    # ============================================================
    print("\n" + "="*70)
    print("STEP 7: TRAIN LATITUDE MODEL")
    print("="*70)

    model_lat = build_nll_model(
        window_size=WINDOW_SIZE,
        num_features=X_scaled.shape[1],
        gru_units_1=GRU_UNITS_1,
        gru_units_2=GRU_UNITS_2,
        dropout_rate=DROPOUT_RATE
    )

    model_lat.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=gaussian_nll_loss
    )

    print("\nModel Architecture:")
    model_lat.summary()

    # Callbacks
    model_lat_path = OUTPUT_DIR / f"nn_model_lat_{timestamp_str}.h5"
    callbacks_lat = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            str(model_lat_path),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    print("\nTraining Latitude Model...")
    history_lat = model_lat.fit(
        X_train_lat, y_train_lat,
        validation_data=(X_val_lat, y_val_lat),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks_lat,
        verbose=1
    )

    print(f"\n✓ Best latitude model saved to: {model_lat_path}")

    # ============================================================
    # STEP 8: TRAIN LONGITUDE MODEL
    # ============================================================
    print("\n" + "="*70)
    print("STEP 8: TRAIN LONGITUDE MODEL")
    print("="*70)

    model_lon = build_nll_model(
        window_size=WINDOW_SIZE,
        num_features=X_scaled.shape[1],
        gru_units_1=GRU_UNITS_1,
        gru_units_2=GRU_UNITS_2,
        dropout_rate=DROPOUT_RATE
    )

    model_lon.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=gaussian_nll_loss
    )

    model_lon_path = OUTPUT_DIR / f"nn_model_lon_{timestamp_str}.h5"
    callbacks_lon = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            str(model_lon_path),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    print("\nTraining Longitude Model...")
    history_lon = model_lon.fit(
        X_train_lon, y_train_lon,
        validation_data=(X_val_lon, y_val_lon),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks_lon,
        verbose=1
    )

    print(f"\n✓ Best longitude model saved to: {model_lon_path}")

    # ============================================================
    # STEP 9: PREDICTIONS WITH UNCERTAINTY
    # ============================================================
    print("\n" + "="*70)
    print("STEP 9: PREDICTIONS WITH UNCERTAINTY ESTIMATES")
    print("="*70)

    # Predict on validation set
    pred_lat_full = model_lat.predict(X_val_lat, verbose=0)
    pred_lon_full = model_lon.predict(X_val_lon, verbose=0)

    # Extract mean and uncertainty
    pred_lat_mean = pred_lat_full[:, 0]  # Mean prediction
    pred_lat_log_var = pred_lat_full[:, 1]  # Log variance
    pred_lat_std = np.sqrt(np.exp(pred_lat_log_var))  # Std deviation

    pred_lon_mean = pred_lon_full[:, 0]
    pred_lon_log_var = pred_lon_full[:, 1]
    pred_lon_std = np.sqrt(np.exp(pred_lon_log_var))

    print("\nExample Predictions (first 5 samples):")
    print("-" * 70)
    for i in range(min(5, len(pred_lat_mean))):
        print(f"Sample {i+1}:")
        print(f"  Lat residual: {pred_lat_mean[i]:+.2f}m ± {pred_lat_std[i]:.2f}m (true: {y_val_lat[i]:+.2f}m)")
        print(f"  Lon residual: {pred_lon_mean[i]:+.2f}m ± {pred_lon_std[i]:.2f}m (true: {y_val_lon[i]:+.2f}m)")

    # Uncertainty statistics
    print(f"\nUncertainty Statistics:")
    print(f"  Lat uncertainty: {pred_lat_std.mean():.2f}m ± {pred_lat_std.std():.2f}m")
    print(f"  Lon uncertainty: {pred_lon_std.mean():.2f}m ± {pred_lon_std.std():.2f}m")

    # ============================================================
    # STEP 10: EVALUATE PERFORMANCE
    # ============================================================
    print("\n" + "="*70)
    print("STEP 10: EVALUATE PERFORMANCE")
    print("="*70)

    # Apply corrections
    corrected_lat = baseline_lat_val + pred_lat_mean / 111320.0
    corrected_lon = baseline_lon_val + pred_lon_mean / (111320.0 * np.cos(np.radians(baseline_lat_val)))

    # Compute errors
    baseline_error = compute_position_error(
        baseline_lat_val, baseline_lon_val,
        gt_lat_val, gt_lon_val
    )

    corrected_error = compute_position_error(
        corrected_lat, corrected_lon,
        gt_lat_val, gt_lon_val
    )

    # Compute metrics
    baseline_metrics = compute_metrics(baseline_error)
    nn_metrics = compute_metrics(corrected_error)

    # Display comparison
    display_metrics_comparison(baseline_metrics, nn_metrics)

    # ============================================================
    # STEP 11: SAVE METRICS AND SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("STEP 11: SAVE RESULTS")
    print("="*70)

    # Save metrics
    metrics_df = pd.DataFrame({
        'Metric': list(baseline_metrics.keys()),
        'Baseline': list(baseline_metrics.values()),
        'NN_Corrected': list(nn_metrics.values())
    })
    metrics_path = OUTPUT_DIR / f"nn_metrics_{timestamp_str}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")

    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(len(history_lat.history['loss'])),
        'lat_loss': history_lat.history['loss'],
        'lat_val_loss': history_lat.history['val_loss'],
        'lon_loss': history_lon.history['loss'],
        'lon_val_loss': history_lon.history['val_loss'],
    })
    history_path = OUTPUT_DIR / f"nn_training_history_{timestamp_str}.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to: {history_path}")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nCurrent Mean(P50,P95): {nn_metrics['Mean(P50,P95)']:.2f}m")
    print(f"Target: 1-2m")
    print(f"Gap: {nn_metrics['Mean(P50,P95)'] - 1.5:.2f}m")

    if nn_metrics['Mean(P50,P95)'] <= 2.0:
        print("\n🎉 TARGET ACHIEVED! 🎉")
    elif nn_metrics['Mean(P50,P95)'] <= 4.0:
        print("\n✓ Good progress! Close to target.")
    else:
        print("\n⚠ More work needed to reach target.")

    print(f"\nModels saved:")
    print(f"  Latitude: {model_lat_path}")
    print(f"  Longitude: {model_lon_path}")
    print(f"  Scaler: {scaler_path}")

    print("\nKey advantages of Neural Network approach:")
    print("  ✓ Temporal patterns captured via GRU layers")
    print("  ✓ Uncertainty estimates for each prediction")
    print("  ✓ Separate lat/lon models")
    print("  ✓ Nonlinear feature interactions")

    print("\nNext steps if target not met:")
    print("  - Increase WINDOW_SIZE for longer context")
    print("  - Add more GRU layers or units")
    print("  - Try attention mechanisms")
    print("  - Ensemble with LightGBM model")
    print("  - Engineer velocity/acceleration features")
    print("="*70)


if __name__ == "__main__":
    main()
