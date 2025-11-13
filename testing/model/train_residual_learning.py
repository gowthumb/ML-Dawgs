"""
LightGBM Residual Learning with Sample Weighting
================================================

Trains separate LightGBM models for latitude and longitude residuals
with adaptive sample weighting to focus on hard-to-correct errors.

Weighting Strategy:
- Weights samples by baseline error magnitude (sqrt for stability)
- Forces model to prioritize correcting worst baseline errors
- Improves P95 metric by reducing outliers

Usage:
    python lgbm_residual_weighted.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split

from data_loader import load_training_data
from feature_engineering import (
    compute_residuals,
    filter_large_residuals,
    prepare_training_data,
    compute_position_error,
    print_residual_statistics
)
from config import (
    DATA_PATH,
    TRAIN_DATA_ROOT,
    BASELINE_LAT,
    BASELINE_LON,
    GT_LAT,
    GT_LON,
    RANDOM_STATE,
    validate_columns,
    get_all_features
)

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("model/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_SIZE = 0.2

# LightGBM parameters (optimized for GNSS)
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'random_state': RANDOM_STATE
}

NUM_BOOST_ROUND = 40000
EARLY_STOPPING_ROUNDS = 100


# ============================================================
# SAMPLE WEIGHTING STRATEGIES
# ============================================================

def compute_baseline_error_weights(
    baseline_lat: np.ndarray,
    baseline_lon: np.ndarray,
    gt_lat: np.ndarray,
    gt_lon: np.ndarray,
    strategy: str = 'sqrt'
) -> np.ndarray:
    """
    Compute sample weights based on baseline position errors.
    
    Higher weights for samples with larger baseline errors forces
    the model to focus on correcting the worst cases.
    
    Args:
        baseline_lat, baseline_lon: Baseline position estimates
        gt_lat, gt_lon: Ground truth positions
        strategy: 'sqrt' (default), 'linear', or 'log'
    
    Returns:
        weights: Array of sample weights (normalized to mean=1.0)
    """
    # Compute baseline errors
    errors = compute_position_error(baseline_lat, baseline_lon, gt_lat, gt_lon)
    
    if strategy == 'sqrt':
        # Square root: Moderate emphasis on large errors
        weights = np.sqrt(errors)
    elif strategy == 'linear':
        # Linear: Strong emphasis on large errors
        weights = errors
    elif strategy == 'log':
        # Log: Mild emphasis, more balanced
        weights = np.log1p(errors)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Normalize to mean=1.0
    weights = weights / weights.mean()
    
    # Clip extreme weights to avoid instability
    weights = np.clip(weights, 0.1, 10.0)
    
    return weights


def compute_quality_based_weights(
    df: pd.DataFrame,
    pos_std_col: str = 'ekf_pos_std',
    num_sats_col: str = 'num_sats',
    cn0_col: str = 'max_cn0'
) -> np.ndarray:
    """
    Compute weights based on GNSS quality indicators.
    
    Higher quality measurements get higher weights.
    
    Args:
        df: DataFrame with quality columns
        pos_std_col: Column with position standard deviation
        num_sats_col: Column with number of satellites
        cn0_col: Column with carrier-to-noise ratio
    
    Returns:
        weights: Array of sample weights
    """
    # Lower pos_std = higher quality
    weight_pos_std = 1.0 / (df[pos_std_col].fillna(df[pos_std_col].median()) + 1.0)
    weight_pos_std = weight_pos_std / weight_pos_std.mean()
    
    # More satellites = higher quality
    weight_sats = df[num_sats_col].fillna(df[num_sats_col].median())
    weight_sats = weight_sats / weight_sats.max()
    
    # Higher CN0 = higher quality
    weight_cn0 = df[cn0_col].fillna(df[cn0_col].median())
    weight_cn0 = weight_cn0 / weight_cn0.max()
    
    # Combine (geometric mean for balance)
    weights = np.power(weight_pos_std * weight_sats * weight_cn0, 1/3)
    weights = weights / weights.mean()
    
    # Clip
    weights = np.clip(weights, 0.1, 10.0)
    
    return weights.values


def print_weight_statistics(weights: np.ndarray, name: str = "Weights"):
    """Print statistics about sample weights."""
    print(f"\n{name} Statistics:")
    print(f"  Min:    {weights.min():.3f}")
    print(f"  Max:    {weights.max():.3f}")
    print(f"  Mean:   {weights.mean():.3f}")
    print(f"  Median: {np.median(weights):.3f}")
    print(f"  Std:    {weights.std():.3f}")
    
    # Show distribution
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"  Percentiles:")
    for p in percentiles:
        print(f"    P{p:2d}: {np.percentile(weights, p):.3f}")


# ============================================================
# EVALUATION METRICS
# ============================================================

def compute_metrics(errors: np.ndarray) -> Dict:
    """Compute comprehensive error metrics."""
    return {
        'RMSE': np.sqrt(np.mean(errors**2)),
        'MAE': np.mean(errors),
        'Median (P50)': np.percentile(errors, 50),
        'P95': np.percentile(errors, 95),
        'P99': np.percentile(errors, 99),
        'Mean(P50,P95)': (np.percentile(errors, 50) + np.percentile(errors, 95)) / 2,
        'Max': np.max(errors)
    }


def display_metrics_comparison(baseline_metrics: Dict, lgbm_metrics: Dict):
    """Display side-by-side metrics comparison."""
    print("\n" + "="*80)
    print(f"{'Metric':<20} {'Baseline (m)':>15} {'LGBM Weighted (m)':>20} {'Improvement':>12}")
    print("-"*80)

    for metric in baseline_metrics.keys():
        baseline_val = baseline_metrics[metric]
        lgbm_val = lgbm_metrics[metric]
        improvement = (baseline_val - lgbm_val) / baseline_val * 100 if baseline_val > 0 else 0
        print(f"{metric:<20} {baseline_val:>15.2f} {lgbm_val:>20.2f} {improvement:>11.1f}%")

    print("="*80)


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("="*80)
    print("LIGHTGBM RESIDUAL LEARNING WITH ADAPTIVE SAMPLE WEIGHTING")
    print("="*80)
    
    # ============================================================
    # STEP 1: LOAD DATA
    # ============================================================
    print("\n" + "="*80)
    print("STEP 1: LOAD DATA")
    print("="*80)
    
    df = load_training_data(DATA_PATH, TRAIN_DATA_ROOT)
    
    all_present, missing_cols = validate_columns(df)
    if not all_present:
        print(f"ERROR: Missing columns: {missing_cols}")
        return
    
    print(f"Loaded {len(df):,} samples")
    
    # ============================================================
    # STEP 2: COMPUTE RESIDUALS
    # ============================================================
    print("\n" + "="*80)
    print("STEP 2: COMPUTE RESIDUALS")
    print("="*80)
    
    df = compute_residuals(df)
    print_residual_statistics(df)
    
    # ============================================================
    # STEP 3: FILTER OUTLIERS
    # ============================================================
    print("\n" + "="*80)
    print("STEP 3: FILTER EXTREME OUTLIERS (>200m)")
    print("="*80)
    
    df = filter_large_residuals(df, max_error_m=200.0)
    print(f"After filtering: {len(df):,} samples")
    print_residual_statistics(df)
    
    # ============================================================
    # STEP 4: PREPARE FEATURES
    # ============================================================
    print("\n" + "="*80)
    print("STEP 4: PREPARE FEATURES")
    print("="*80)
    
    X, y_lat, y_lon, baseline_lat, baseline_lon, gt_lat, gt_lon, valid_mask = prepare_training_data(
        df, use_component_targets=True
    )
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Valid samples: {len(X):,}")
    
    # Handle NaN in features (simple median imputation)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    print(f"NaN handling: Imputed with median values")
    
    # ============================================================
    # STEP 5: COMPUTE SAMPLE WEIGHTS
    # ============================================================
    print("\n" + "="*80)
    print("STEP 5: COMPUTE ADAPTIVE SAMPLE WEIGHTS")
    print("="*80)
    
    print("\nWeighting Strategy: Baseline Error (sqrt)")
    print("Rationale: Focus model on correcting worst baseline errors")
    print("          Higher baseline error → Higher sample weight")
    
    # Convert to numpy if needed
    baseline_lat_vals = baseline_lat.values if hasattr(baseline_lat, 'values') else baseline_lat
    baseline_lon_vals = baseline_lon.values if hasattr(baseline_lon, 'values') else baseline_lon
    gt_lat_vals = gt_lat.values if hasattr(gt_lat, 'values') else gt_lat
    gt_lon_vals = gt_lon.values if hasattr(gt_lon, 'values') else gt_lon
    y_lat_vals = y_lat.values if hasattr(y_lat, 'values') else y_lat
    y_lon_vals = y_lon.values if hasattr(y_lon, 'values') else y_lon
    
    # Compute weights based on baseline errors
    weights = compute_baseline_error_weights(
        baseline_lat_vals, baseline_lon_vals,
        gt_lat_vals, gt_lon_vals,
        strategy='sqrt'
    )
    
    print_weight_statistics(weights, "Sample Weights")
    
    # Show correlation between weights and baseline errors
    baseline_errors = compute_position_error(
        baseline_lat_vals, baseline_lon_vals, gt_lat_vals, gt_lon_vals
    )
    print(f"\nCorrelation between weights and baseline errors: {np.corrcoef(weights, baseline_errors)[0,1]:.3f}")
    
    # ============================================================
    # STEP 6: TRAIN/VAL SPLIT
    # ============================================================
    print("\n" + "="*80)
    print("STEP 6: TRAIN/VALIDATION SPLIT")
    print("="*80)
    
    X_train, X_val, \
    y_lat_train, y_lat_val, \
    y_lon_train, y_lon_val, \
    weights_train, weights_val, \
    baseline_lat_train, baseline_lat_val, \
    baseline_lon_train, baseline_lon_val, \
    gt_lat_train, gt_lat_val, \
    gt_lon_train, gt_lon_val = train_test_split(
        X_imputed, y_lat_vals, y_lon_vals, weights,
        baseline_lat_vals, baseline_lon_vals,
        gt_lat_vals, gt_lon_vals,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )
    
    print(f"Training samples:   {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    
    # ============================================================
    # STEP 7: TRAIN LATITUDE MODEL WITH WEIGHTS
    # ============================================================
    print("\n" + "="*80)
    print("STEP 7: TRAIN LATITUDE MODEL (WITH SAMPLE WEIGHTS)")
    print("="*80)
    
    print("\nCreating weighted training dataset...")
    train_data_lat = lgb.Dataset(
        X_train,
        label=y_lat_train,
        weight=weights_train  # ← SAMPLE WEIGHTS APPLIED HERE
    )
    
    val_data_lat = lgb.Dataset(
        X_val,
        label=y_lat_val,
        reference=train_data_lat
    )
    
    print("Training LightGBM (Latitude)...")
    print(f"Parameters: {LGBM_PARAMS}")
    
    model_lat = lgb.train(
        LGBM_PARAMS,
        train_data_lat,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data_lat, val_data_lat],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=100)
        ]
    )
    
    print(f"\n✓ Latitude model trained: {model_lat.num_trees()} trees")
    print(f"  Best iteration: {model_lat.best_iteration}")
    
    # ============================================================
    # STEP 8: TRAIN LONGITUDE MODEL WITH WEIGHTS
    # ============================================================
    print("\n" + "="*80)
    print("STEP 8: TRAIN LONGITUDE MODEL (WITH SAMPLE WEIGHTS)")
    print("="*80)
    
    print("\nCreating weighted training dataset...")
    train_data_lon = lgb.Dataset(
        X_train,
        label=y_lon_train,
        weight=weights_train  # ← SAMPLE WEIGHTS APPLIED HERE
    )
    
    val_data_lon = lgb.Dataset(
        X_val,
        label=y_lon_val,
        reference=train_data_lon
    )
    
    print("Training LightGBM (Longitude)...")
    
    model_lon = lgb.train(
        LGBM_PARAMS,
        train_data_lon,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data_lon, val_data_lon],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=100)
        ]
    )
    
    print(f"\n✓ Longitude model trained: {model_lon.num_trees()} trees")
    print(f"  Best iteration: {model_lon.best_iteration}")
    
    # ============================================================
    # STEP 9: FEATURE IMPORTANCE
    # ============================================================
    print("\n" + "="*80)
    print("STEP 9: FEATURE IMPORTANCE (LAT MODEL)")
    print("="*80)
    
    feature_names = get_all_features(df)
    importance = model_lat.feature_importance(importance_type='gain')
    
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importance)],
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 features that predict baseline errors:")
    print(importance_df.head(15).to_string(index=False))
    
    # ============================================================
    # STEP 10: EVALUATE PERFORMANCE
    # ============================================================
    print("\n" + "="*80)
    print("STEP 10: EVALUATE PERFORMANCE")
    print("="*80)
    
    # Predict on validation set
    pred_lat_m = model_lat.predict(X_val, num_iteration=model_lat.best_iteration)
    pred_lon_m = model_lon.predict(X_val, num_iteration=model_lon.best_iteration)
    
    # Apply corrections
    METERS_PER_DEGREE_LAT = 111000
    lat_correction_deg = pred_lat_m / METERS_PER_DEGREE_LAT
    lon_correction_deg = pred_lon_m / (METERS_PER_DEGREE_LAT * np.cos(np.radians(baseline_lat_val)))
    
    corrected_lat = baseline_lat_val + lat_correction_deg
    corrected_lon = baseline_lon_val + lon_correction_deg
    
    # Compute errors
    baseline_error = compute_position_error(
        baseline_lat_val, baseline_lon_val, gt_lat_val, gt_lon_val
    )
    
    corrected_error = compute_position_error(
        corrected_lat, corrected_lon, gt_lat_val, gt_lon_val
    )
    
    # Compute metrics
    baseline_metrics = compute_metrics(baseline_error)
    lgbm_metrics = compute_metrics(corrected_error)
    
    # Display comparison
    display_metrics_comparison(baseline_metrics, lgbm_metrics)
    
    # ============================================================
    # STEP 11: ANALYZE WEIGHTING IMPACT
    # ============================================================
    print("\n" + "="*80)
    print("STEP 11: WEIGHTING IMPACT ANALYSIS")
    print("="*80)
    
    # Separate samples by weight quartile
    weight_quartiles = np.percentile(weights_val, [25, 50, 75])
    
    print("\nPerformance by sample weight quartile:")
    print("-" * 70)
    print(f"{'Quartile':<15} {'Weight Range':<20} {'Baseline P95':>15} {'LGBM P95':>12} {'Improvement':>12}")
    print("-" * 70)
    
    quartile_ranges = [
        ("Q1 (Low)", 0, weight_quartiles[0]),
        ("Q2", weight_quartiles[0], weight_quartiles[1]),
        ("Q3", weight_quartiles[1], weight_quartiles[2]),
        ("Q4 (High)", weight_quartiles[2], np.inf)
    ]
    
    for q_name, w_min, w_max in quartile_ranges:
        mask = (weights_val >= w_min) & (weights_val < w_max)
        if mask.sum() == 0:
            continue
        
        base_p95 = np.percentile(baseline_error[mask], 95)
        lgbm_p95 = np.percentile(corrected_error[mask], 95)
        improvement = (base_p95 - lgbm_p95) / base_p95 * 100
        
        print(f"{q_name:<15} [{w_min:.2f}, {w_max:.2f})     {base_p95:>15.2f} {lgbm_p95:>12.2f} {improvement:>11.1f}%")
    
    print("-" * 70)
    print("Expectation: Q4 (high weight samples) should show largest improvement")
    
    # ============================================================
    # STEP 12: SAVE MODELS
    # ============================================================
    print("\n" + "="*80)
    print("STEP 12: SAVE OUTPUTS")
    print("="*80)
    
    # Save models
    model_lat_path = OUTPUT_DIR / f"residual_model_lat_weighted_{timestamp_str}.pkl"
    model_lon_path = OUTPUT_DIR / f"residual_model_lon_weighted_{timestamp_str}.pkl"
    
    with open(model_lat_path, 'wb') as f:
        pickle.dump(model_lat, f)
    with open(model_lon_path, 'wb') as f:
        pickle.dump(model_lon, f)
    
    # Save imputer
    imputer_path = OUTPUT_DIR / f"imputer_weighted_{timestamp_str}.pkl"
    with open(imputer_path, 'wb') as f:
        pickle.dump(imputer, f)
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Metric': list(baseline_metrics.keys()),
        'Baseline': list(baseline_metrics.values()),
        'LGBM_Weighted': list(lgbm_metrics.values())
    })
    metrics_path = OUTPUT_DIR / f"residual_metrics_weighted_{timestamp_str}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    
    # Save feature importance
    importance_path = OUTPUT_DIR / f"feature_importance_weighted_{timestamp_str}.csv"
    importance_df.to_csv(importance_path, index=False)
    
    print("\nWeighted models saved:")
    print(f"  Lat model: {model_lat_path}")
    print(f"  Lon model: {model_lon_path}")
    print(f"  Imputer:   {imputer_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Feature importance saved to: {importance_path}")
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    print(f"\n📊 RESULTS:")
    print(f"  Current Mean(P50,P95): {lgbm_metrics['Mean(P50,P95)']:.2f}m")
    print(f"  Target: 1-2m")
    print(f"  Gap: {lgbm_metrics['Mean(P50,P95)'] - 1.5:.2f}m")
    
    if lgbm_metrics['Mean(P50,P95)'] <= 2.0:
        print("\n🎉 TARGET ACHIEVED! 🎉")
    elif lgbm_metrics['Mean(P50,P95)'] <= 4.0:
        print("\n✓ Good progress! Close to target.")
    else:
        print("\n⚠ More work needed to reach target.")
    
    print(f"\n🎯 WEIGHTING BENEFITS:")
    p95_improvement = (baseline_metrics['P95'] - lgbm_metrics['P95']) / baseline_metrics['P95'] * 100
    print(f"  P95 improvement: {p95_improvement:.1f}%")
    print(f"  Model focuses on worst {(weights > weights.mean()).sum()/len(weights)*100:.0f}% of samples")
    
    print("\n💡 KEY ADVANTAGES OF WEIGHTED APPROACH:")
    print("  ✓ Prioritizes correcting large baseline errors")
    print("  ✓ Reduces P95 outliers more effectively")
    print("  ✓ Better handles challenging scenarios")
    print("  ✓ Simple sqrt(error) weighting is stable and effective")
    
    print("\n📈 NEXT STEPS IF TARGET NOT MET:")
    print("  - Try different weight strategies (linear vs sqrt vs log)")
    print("  - Combine with Kalman smoothing post-processing")
    print("  - Ensemble with neural network model")
    print("  - Add temporal features (velocity, acceleration)")
    print("  - Use quality-based weights for GNSS measurements")
    
    print("="*80)


if __name__ == "__main__":
    main()