"""
Two-Stage Residual Learning: LGBM → NN
=======================================

Stage 1: LightGBM corrects PPK baseline
Stage 2: Neural Network corrects remaining LGBM residuals

This approach allows each model to focus on what it does best:
- LGBM: Captures feature interactions and spatial patterns
- NN: Corrects systematic biases left by LGBM

Usage:
    python two_stage_residual_learning.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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

OUTPUT_DIR = Path("model/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_SIZE = 0.2

# LGBM parameters
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

# Simple NN for Stage 2 (smaller than before)
class Stage2NN(nn.Module):
    """
    Smaller NN for correcting LGBM residuals.
    Includes LGBM predictions as input features.
    """
    def __init__(self, num_features):
        super(Stage2NN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1)  # Single output: residual correction
        )
    
    def forward(self, x):
        return self.network(x)


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


def train_stage2_nn(X, y, epochs=50, batch_size=512, lr=0.001):
    """Train Stage 2 NN to correct LGBM residuals."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to torch
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).unsqueeze(1).to(device)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = Stage2NN(num_features=X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Train
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.6f}")
    
    return model


def main():
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("="*80)
    print("TWO-STAGE RESIDUAL LEARNING: LGBM (Stage 1) → NN (Stage 2)")
    print("="*80)
    
    # ============================================================
    # LOAD AND PREPARE DATA
    # ============================================================
    print("\n[1/7] Loading and preparing data...")
    
    df = load_training_data(DATA_PATH, TRAIN_DATA_ROOT)
    all_present, missing_cols = validate_columns(df)
    if not all_present:
        print(f"ERROR: Missing columns: {missing_cols}")
        return
    
    print(f"  Loaded {len(df):,} samples")
    
    df = compute_residuals(df)
    df = filter_large_residuals(df, max_error_m=200.0)
    
    X, y_lat, y_lon, baseline_lat, baseline_lon, gt_lat, gt_lon, valid_mask = prepare_training_data(
        df, use_component_targets=True
    )
    
    # Impute NaN
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Convert to numpy
    baseline_lat_vals = baseline_lat.values if hasattr(baseline_lat, 'values') else baseline_lat
    baseline_lon_vals = baseline_lon.values if hasattr(baseline_lon, 'values') else baseline_lon
    gt_lat_vals = gt_lat.values if hasattr(gt_lat, 'values') else gt_lat
    gt_lon_vals = gt_lon.values if hasattr(gt_lon, 'values') else gt_lon
    y_lat_vals = y_lat.values if hasattr(y_lat, 'values') else y_lat
    y_lon_vals = y_lon.values if hasattr(y_lon, 'values') else y_lon
    
    # Split data
    X_train, X_val, \
    y_lat_train, y_lat_val, \
    y_lon_train, y_lon_val, \
    baseline_lat_train, baseline_lat_val, \
    baseline_lon_train, baseline_lon_val, \
    gt_lat_train, gt_lat_val, \
    gt_lon_train, gt_lon_val = train_test_split(
        X_imputed, y_lat_vals, y_lon_vals,
        baseline_lat_vals, baseline_lon_vals,
        gt_lat_vals, gt_lon_vals,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )
    
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")
    
    # ============================================================
    # STAGE 1: TRAIN LGBM
    # ============================================================
    print("\n[2/7] STAGE 1: Training LightGBM...")
    
    # Latitude LGBM
    print("\n  Training Latitude LGBM...")
    train_data_lat = lgb.Dataset(X_train, label=y_lat_train)
    val_data_lat = lgb.Dataset(X_val, label=y_lat_val, reference=train_data_lat)
    
    lgbm_lat = lgb.train(
        LGBM_PARAMS,
        train_data_lat,
        num_boost_round=2000,
        valid_sets=[val_data_lat],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    # Longitude LGBM
    print("\n  Training Longitude LGBM...")
    train_data_lon = lgb.Dataset(X_train, label=y_lon_train)
    val_data_lon = lgb.Dataset(X_val, label=y_lon_val, reference=train_data_lon)
    
    lgbm_lon = lgb.train(
        LGBM_PARAMS,
        train_data_lon,
        num_boost_round=2000,
        valid_sets=[val_data_lon],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    print(f"\n✓ Stage 1 complete")
    
    # ============================================================
    # GET STAGE 1 PREDICTIONS AND RESIDUALS
    # ============================================================
    print("\n[3/7] Computing Stage 1 residuals...")
    
    # LGBM predictions on training data
    lgbm_lat_pred_train = lgbm_lat.predict(X_train)
    lgbm_lon_pred_train = lgbm_lon.predict(X_train)
    
    # LGBM predictions on validation data
    lgbm_lat_pred_val = lgbm_lat.predict(X_val)
    lgbm_lon_pred_val = lgbm_lon.predict(X_val)
    
    # Compute Stage 1 residuals (what LGBM got wrong)
    stage1_residual_lat_train = y_lat_train - lgbm_lat_pred_train
    stage1_residual_lon_train = y_lon_train - lgbm_lon_pred_train
    
    stage1_residual_lat_val = y_lat_val - lgbm_lat_pred_val
    stage1_residual_lon_val = y_lon_val - lgbm_lon_pred_val
    
    print(f"  Stage 1 residuals (train):")
    print(f"    Lat - mean: {stage1_residual_lat_train.mean():.2f}m, std: {stage1_residual_lat_train.std():.2f}m")
    print(f"    Lon - mean: {stage1_residual_lon_train.mean():.2f}m, std: {stage1_residual_lon_train.std():.2f}m")
    
    # ============================================================
    # STAGE 2: TRAIN NN ON LGBM RESIDUALS
    # ============================================================
    print("\n[4/7] STAGE 2: Training NN to correct LGBM residuals...")
    
    # Augment features with LGBM predictions and original features
    X_stage2_train = np.column_stack([
        X_train,
        lgbm_lat_pred_train,
        lgbm_lon_pred_train,
        np.abs(lgbm_lat_pred_train),  # Magnitude of LGBM prediction
        np.abs(lgbm_lon_pred_train)
    ])
    
    X_stage2_val = np.column_stack([
        X_val,
        lgbm_lat_pred_val,
        lgbm_lon_pred_val,
        np.abs(lgbm_lat_pred_val),
        np.abs(lgbm_lon_pred_val)
    ])
    
    # Scale Stage 2 features
    scaler_stage2 = StandardScaler()
    X_stage2_train_scaled = scaler_stage2.fit_transform(X_stage2_train)
    X_stage2_val_scaled = scaler_stage2.transform(X_stage2_val)
    
    print(f"  Stage 2 features: {X_stage2_train_scaled.shape[1]} (original + LGBM predictions)")
    
    # Train Stage 2 NN for latitude
    print("\n  Training Stage 2 NN (Latitude)...")
    nn_lat = train_stage2_nn(X_stage2_train_scaled, stage1_residual_lat_train, epochs=50)
    
    # Train Stage 2 NN for longitude
    print("\n  Training Stage 2 NN (Longitude)...")
    nn_lon = train_stage2_nn(X_stage2_train_scaled, stage1_residual_lon_train, epochs=50)
    
    print(f"\n✓ Stage 2 complete")
    
    # ============================================================
    # EVALUATE ALL STAGES
    # ============================================================
    print("\n[5/7] Evaluating all stages...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Stage 2 NN predictions
    nn_lat.eval()
    nn_lon.eval()
    
    with torch.no_grad():
        X_stage2_val_tensor = torch.FloatTensor(X_stage2_val_scaled).to(device)
        nn_lat_correction = nn_lat(X_stage2_val_tensor).cpu().numpy().flatten()
        nn_lon_correction = nn_lon(X_stage2_val_tensor).cpu().numpy().flatten()
    
    # Final two-stage predictions
    final_lat_pred = lgbm_lat_pred_val + nn_lat_correction
    final_lon_pred = lgbm_lon_pred_val + nn_lon_correction
    
    # Apply all corrections to baseline
    METERS_PER_DEGREE_LAT = 111000
    
    # Baseline
    baseline_error = compute_position_error(
        baseline_lat_val, baseline_lon_val, gt_lat_val, gt_lon_val
    )
    
    # Stage 1: LGBM only
    stage1_lat = baseline_lat_val + lgbm_lat_pred_val / METERS_PER_DEGREE_LAT
    stage1_lon = baseline_lon_val + lgbm_lon_pred_val / (METERS_PER_DEGREE_LAT * np.cos(np.radians(baseline_lat_val)))
    stage1_error = compute_position_error(stage1_lat, stage1_lon, gt_lat_val, gt_lon_val)
    
    # Stage 2: LGBM + NN
    stage2_lat = baseline_lat_val + final_lat_pred / METERS_PER_DEGREE_LAT
    stage2_lon = baseline_lon_val + final_lon_pred / (METERS_PER_DEGREE_LAT * np.cos(np.radians(baseline_lat_val)))
    stage2_error = compute_position_error(stage2_lat, stage2_lon, gt_lat_val, gt_lon_val)
    
    # Compute metrics
    baseline_metrics = compute_metrics(baseline_error)
    stage1_metrics = compute_metrics(stage1_error)
    stage2_metrics = compute_metrics(stage2_error)
    
    # ============================================================
    # DISPLAY RESULTS
    # ============================================================
    print("\n[6/7] Results...")
    print("\n" + "="*90)
    print(f"{'Metric':<20} {'Baseline (m)':>15} {'Stage 1: LGBM (m)':>20} {'Stage 2: +NN (m)':>18} {'Improvement':>12}")
    print("-"*90)
    
    for metric in baseline_metrics.keys():
        base = baseline_metrics[metric]
        s1 = stage1_metrics[metric]
        s2 = stage2_metrics[metric]
        imp = (base - s2) / base * 100 if base > 0 else 0
        print(f"{metric:<20} {base:>15.2f} {s1:>20.2f} {s2:>18.2f} {imp:>11.1f}%")
    
    print("="*90)
    
    # Check if Stage 2 helped
    if stage2_metrics['Mean(P50,P95)'] < stage1_metrics['Mean(P50,P95)']:
        improvement = (stage1_metrics['Mean(P50,P95)'] - stage2_metrics['Mean(P50,P95)']) / stage1_metrics['Mean(P50,P95)'] * 100
        print(f"\n✅ Stage 2 NN improved results by {improvement:.1f}%!")
        print(f"   Mean(P50,P95): {stage1_metrics['Mean(P50,P95)']:.2f}m → {stage2_metrics['Mean(P50,P95)']:.2f}m")
    else:
        print(f"\n⚠️  Stage 2 NN did not improve (use Stage 1 only)")
        print(f"   Stick with LGBM-only predictions")
    
    # ============================================================
    # SAVE MODELS
    # ============================================================
    print("\n[7/7] Saving models...")
    
    lgbm_lat_path = OUTPUT_DIR / f"two_stage_lgbm_lat_{timestamp_str}.pkl"
    lgbm_lon_path = OUTPUT_DIR / f"two_stage_lgbm_lon_{timestamp_str}.pkl"
    nn_lat_path = OUTPUT_DIR / f"two_stage_nn_lat_{timestamp_str}.pt"
    nn_lon_path = OUTPUT_DIR / f"two_stage_nn_lon_{timestamp_str}.pt"
    scaler_path = OUTPUT_DIR / f"two_stage_scaler_{timestamp_str}.pkl"
    imputer_path = OUTPUT_DIR / f"two_stage_imputer_{timestamp_str}.pkl"
    
    with open(lgbm_lat_path, 'wb') as f:
        pickle.dump(lgbm_lat, f)
    with open(lgbm_lon_path, 'wb') as f:
        pickle.dump(lgbm_lon, f)
    
    torch.save(nn_lat.state_dict(), nn_lat_path)
    torch.save(nn_lon.state_dict(), nn_lon_path)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_stage2, f)
    with open(imputer_path, 'wb') as f:
        pickle.dump(imputer, f)
    
    print(f"  Models saved to: {OUTPUT_DIR}")
    
    print("\n" + "="*80)
    print("TWO-STAGE TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal Mean(P50,P95): {stage2_metrics['Mean(P50,P95)']:.2f}m")
    print(f"Target: 1-2m")
    print(f"Gap: {stage2_metrics['Mean(P50,P95)'] - 1.5:.2f}m")
    
    if stage2_metrics['Mean(P50,P95)'] <= 2.0:
        print("\n🎉 TARGET ACHIEVED! 🎉")
    elif stage2_metrics['Mean(P50,P95)'] <= 4.0:
        print("\n✓ Good progress! Close to target.")
    else:
        print("\n⚠ More work needed to reach target.")
    
    print("="*80)


if __name__ == "__main__":
    main()