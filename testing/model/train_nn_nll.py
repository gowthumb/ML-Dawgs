"""
Improved Neural Network with Gaussian NLL Loss for Residual Learning
- Filters outliers >10m
- Deeper architecture with residual connections
- Better regularization and learning rate scheduling
- Uses NaN indicator features to preserve all data
- Includes Kalman smoothing evaluation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import warnings
import pickle
from pathlib import Path
from datetime import datetime

from config import (
    DATA_PATH,
    TRAIN_DATA_ROOT,
    BASELINE_LAT, BASELINE_LON,
    GT_LAT, GT_LON,
    validate_columns,
    get_all_features,
    RANDOM_STATE,
    TEST_SIZE,
)
from data_loader import load_training_data, display_sample_data
from feature_engineering import (
    compute_residuals,
    filter_large_residuals,
    print_residual_statistics,
    prepare_training_data,
    compute_position_error,
)

warnings.filterwarnings('ignore')

# ============================================================
# HYPERPARAMETERS (IMPROVED)
# ============================================================
BATCH_SIZE = 512
LEARNING_RATE = 0.0005
EPOCHS = 50
HIDDEN_DIM_1 = 512
HIDDEN_DIM_2 = 256
HIDDEN_DIM_3 = 128
WEIGHT_DECAY = 1e-4
VAL_SIZE = 0.2
DROPOUT_RATE = 0.3

OUTPUT_DIR = Path("model/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# IMPROVED MODEL WITH RESIDUAL CONNECTIONS
# ============================================================
class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(x + self.block(x))

class ImprovedNLLModel(nn.Module):
    """
    Improved architecture with:
    - Residual connections for better gradient flow
    - Deeper network for more capacity
    - Separate heads for lat and lon (more parameters)
    """
    def __init__(self, num_features):
        super(ImprovedNLLModel, self).__init__()
        
        # Shared feature extraction
        self.input_layer = nn.Sequential(
            nn.Linear(num_features, HIDDEN_DIM_1),
            nn.BatchNorm1d(HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )
        
        # Residual blocks for deep feature learning
        self.res_block1 = ResidualBlock(HIDDEN_DIM_1)
        self.res_block2 = ResidualBlock(HIDDEN_DIM_1)
        
        # Transition to smaller dimensions
        self.middle_layer = nn.Sequential(
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.BatchNorm1d(HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )
        
        # Separate heads for latitude and longitude (more capacity)
        self.lat_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM_2, HIDDEN_DIM_3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM_3, 2)  # [mu_lat, log_var_lat]
        )
        
        self.lon_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM_2, HIDDEN_DIM_3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM_3, 2)  # [mu_lon, log_var_lon]
        )
    
    def forward(self, x):
        # Shared feature extraction
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.middle_layer(x)
        
        # Separate predictions
        lat_out = self.lat_head(x)  # [mu_lat, log_var_lat]
        lon_out = self.lon_head(x)  # [mu_lon, log_var_lon]
        
        # Concatenate: [mu_lat, log_var_lat, mu_lon, log_var_lon]
        return torch.cat([lat_out, lon_out], dim=1)

# ============================================================
# IMPROVED LOSS FUNCTION
# ============================================================
def gaussian_nll_loss(y_true, mu, log_var):
    """
    Gaussian Negative Log-Likelihood loss with better numerical stability
    """
    # Clamp log_var to reasonable range for GNSS corrections
    # exp(-2) = 0.135, exp(4) = 54.6 → std dev range
    
    squared_error = (y_true - mu) ** 2
    loss = 0.5 * (log_var + squared_error / var)
    return loss.mean()

def combined_loss(y_lat, y_lon, mu_lat, log_var_lat, mu_lon, log_var_lon, alpha=0.5):
    """
    Combined loss with optional weighting
    alpha controls the balance between lat and lon losses
    """
    loss_lat = gaussian_nll_loss(y_lat, mu_lat, log_var_lat)
    loss_lon = gaussian_nll_loss(y_lon, mu_lon, log_var_lon)
    return alpha * loss_lat + (1 - alpha) * loss_lon

# ============================================================
# DATASET
# ============================================================
class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# IMPROVED TRAINING WITH BETTER SCHEDULING
# ============================================================
def train_model(model, train_loader, val_loader, epochs, lr, save_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 7
    min_delta = 0.01

    print(f"Early stopping config: patience={patience}, min_delta={min_delta}")

    train_history = []
    val_history = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            
            # Unpack: [mu_lat, log_var_lat, mu_lon, log_var_lon]
            mu_lat = output[:, 0]
            log_var_lat = output[:, 1]
            mu_lon = output[:, 2]
            log_var_lon = output[:, 3]
            
            y_lat = y_batch[:, 0]
            y_lon = y_batch[:, 1]
            
            # Combined loss
            loss = combined_loss(y_lat, y_lon, mu_lat, log_var_lat, mu_lon, log_var_lon)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                output = model(X_batch)
                
                mu_lat = output[:, 0]
                log_var_lat = output[:, 1]
                mu_lon = output[:, 2]
                log_var_lon = output[:, 3]
                
                y_lat = y_batch[:, 0]
                y_lon = y_batch[:, 1]
                
                loss = combined_loss(y_lat, y_lon, mu_lat, log_var_lat, mu_lon, log_var_lon)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        train_history.append(train_loss)
        val_history.append(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - train: {train_loss:.4f} - val: {val_loss:.4f} - lr: {current_lr:.6f}")
        
        scheduler.step()
        
        # Early stopping with min_delta
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }, OUTPUT_DIR / f"{save_name}_best.pt")
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement > {min_delta} for {patience} epochs)")
                break
    
    # Load best model
    checkpoint = torch.load(OUTPUT_DIR / f"{save_name}_best.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
    
    return model, train_history, val_history

# ============================================================
# MAIN
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("Using device:", device)
    print("="*70)
    print("IMPROVED NEURAL NETWORK WITH GAUSSIAN NLL LOSS + KALMAN")
    print("="*70)
    
    print("\n[1/9] Loading data...")
    df = load_training_data(DATA_PATH, TRAIN_DATA_ROOT)
    
    all_present, missing_cols = validate_columns(df)
    if not all_present:
        print(f"ERROR: Missing columns: {missing_cols}")
        return
    
    print(f"  {len(df):,} samples")
    
    print("\n[2/9] Computing residuals...")
    df = compute_residuals(df)
    print_residual_statistics(df)
    
    print("\n[3/9] Filtering outliers (>200m)...")
    df = filter_large_residuals(df, max_error_m=200.0)
    print_residual_statistics(df)
    
    print("\n[4/9] Preparing features...")
    feature_list = get_all_features(df)
    
    X, y_lat, y_lon, baseline_lat, baseline_lon, gt_lat, gt_lon, valid_mask = prepare_training_data(
        df, use_component_targets=True
    )
    print(f"  Features: {X.shape}")
    
    print("\n[5/9] Handling NaN with indicator features...")
    
    # Convert to numpy
    y_lat_vals = y_lat.values if hasattr(y_lat, 'values') else y_lat
    y_lon_vals = y_lon.values if hasattr(y_lon, 'values') else y_lon
    baseline_lat_vals = baseline_lat.values if hasattr(baseline_lat, 'values') else baseline_lat
    baseline_lon_vals = baseline_lon.values if hasattr(baseline_lon, 'values') else baseline_lon
    gt_lat_vals = gt_lat.values if hasattr(gt_lat, 'values') else gt_lat
    gt_lon_vals = gt_lon.values if hasattr(gt_lon, 'values') else gt_lon
    
    print(f"  Sample count: {len(X):,}")
    
    # NaN Indicator Method (preserves all data)
    nan_indicator = np.isnan(X).astype(np.float32)
    num_features_with_nan = (np.isnan(X).any(axis=0)).sum()
    print(f"  Features with NaN: {num_features_with_nan}/{X.shape[1]}")
    
    # Fill NaN with 0
    X_filled = np.where(np.isnan(X), 0, X)
    
    print(f"  Target ranges:")
    print(f"    Lat residual: [{y_lat_vals.min():.2f}, {y_lat_vals.max():.2f}] m")
    print(f"    Lon residual: [{y_lon_vals.min():.2f}, {y_lon_vals.max():.2f}] m")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled_original = scaler.fit_transform(X_filled)
    
    # Combine scaled features + NaN indicators
    X_scaled = np.concatenate([X_scaled_original, nan_indicator], axis=1)
    print(f"  Final features: {X_scaled.shape[1]} (original {X.shape[1]} + {nan_indicator.shape[1]} indicators)")
    
    # Save scaler
    scaler_path = OUTPUT_DIR / f"scaler_improved_nll_{timestamp_str}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved: {scaler_path}")
    
    print("\n[6/9] Splitting data...")
    
    # Combine targets into [N, 2]
    y_combined = np.stack([y_lat_vals, y_lon_vals], axis=1)
    
    X_train, X_val, y_train, y_val, \
    baseline_lat_train, baseline_lat_val, baseline_lon_train, baseline_lon_val, \
    gt_lat_train, gt_lat_val, gt_lon_train, gt_lon_val = train_test_split(
        X_scaled, y_combined,
        baseline_lat_vals, baseline_lon_vals,
        gt_lat_vals, gt_lon_vals,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")
    
    # DataLoaders
    train_loader = DataLoader(
        SimpleDataset(X_train, y_train), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        drop_last=True,
        num_workers=0
    )
    val_loader = DataLoader(
        SimpleDataset(X_val, y_val), 
        batch_size=BATCH_SIZE, 
        drop_last=False,
        num_workers=0
    )
    
    print("\n[7/9] Training improved model...")
    model = ImprovedNLLModel(num_features=X_scaled.shape[1])
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    print(f"  Architecture: Input({X_scaled.shape[1]}) → {HIDDEN_DIM_1} → ResBlock → ResBlock → {HIDDEN_DIM_2} → Separate Heads({HIDDEN_DIM_3}) → Output(4)")
    
    model, train_history, val_history = train_model(
        model, train_loader, val_loader, EPOCHS, LEARNING_RATE, f"nn_improved_nll_{timestamp_str}"
    )
    
    print("\n[8/9] Evaluating...")
    model.eval()
    
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        output = model(X_val_tensor).cpu().numpy()
        
        # Extract mean predictions (mu) - these are in METERS
        pred_lat_m = output[:, 0]  # mu_lat in meters
        pred_lon_m = output[:, 2]  # mu_lon in meters
        
        # Extract uncertainties (optional, for analysis)
        uncertainty_lat = np.exp(output[:, 1] / 2)  # std_lat
        uncertainty_lon = np.exp(output[:, 3] / 2)  # std_lon
    
    # Apply corrections WITHOUT Kalman first
    METERS_PER_DEGREE_LAT = 111000
    lat_correction_deg = pred_lat_m / METERS_PER_DEGREE_LAT
    lon_correction_deg = pred_lon_m / (METERS_PER_DEGREE_LAT * np.cos(np.radians(baseline_lat_val)))
    
    corrected_lat = baseline_lat_val + lat_correction_deg
    corrected_lon = baseline_lon_val + lon_correction_deg
    
    print("\n[9/9] Results with Kalman Smoothing...")
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION: BASELINE → NN → KALMAN")
    print("="*70)
    
    # Import the evaluation function from kalman_smoother
    from kalman_smoother import evaluate_with_kalman
    
    # This will calculate and print all three sets of metrics
    baseline_metrics, nn_metrics, kalman_metrics = evaluate_with_kalman(
        pred_lat=corrected_lat,
        pred_lon=corrected_lon,
        gt_lat=gt_lat_val,
        gt_lon=gt_lon_val,
        baseline_lat=baseline_lat_val,
        baseline_lon=baseline_lon_val,
        timestamps=None,  # Add if you have timestamps in your data
        process_variance=0.5,      # Tune: lower = more smoothing
        measurement_variance=5.0   # Tune: higher = more smoothing
    )
    
    # Additional uncertainty analysis
    print(f"\n" + "="*70)
    print("UNCERTAINTY ANALYSIS")
    print("="*70)
    print(f"Mean uncertainty (lat): {uncertainty_lat.mean():.2f}m")
    print(f"Mean uncertainty (lon): {uncertainty_lon.mean():.2f}m")
    print(f"P95 uncertainty (lat): {np.percentile(uncertainty_lat, 95):.2f}m")
    print(f"P95 uncertainty (lon): {np.percentile(uncertainty_lon, 95):.2f}m")
    
    print(f"\nModels saved to: {OUTPUT_DIR}")
    print("="*70)
    
    # Feature importance analysis using permutation importance
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    from sklearn.inspection import permutation_importance
    
    # Get feature names (assuming you have them from config)
    feature_names = get_all_features(df)
    # Add NaN indicator names
    feature_names_with_indicators = feature_names + [f"{name}_nan" for name in feature_names]
    
    print("\nComputing permutation importance (this may take a minute)...")
    print("Method: Shuffling each feature and measuring impact on validation error")
    
    # Create a simple wrapper to convert torch model output to scalar predictions
    def predict_wrapper(X_input):
        """Convert model output to position error"""
        X_tensor = torch.FloatTensor(X_input).to(device)
        with torch.no_grad():
            output = model(X_tensor).cpu().numpy()
        
        # Extract means
        pred_lat_m = output[:, 0]
        pred_lon_m = output[:, 2]
        
        # Get corresponding baseline and GT for this subset
        # Note: This uses validation set indices
        subset_baseline_lat = baseline_lat_val[:len(pred_lat_m)]
        subset_baseline_lon = baseline_lon_val[:len(pred_lat_m)]
        subset_gt_lat = gt_lat_val[:len(pred_lat_m)]
        subset_gt_lon = gt_lon_val[:len(pred_lat_m)]
        
        # Apply corrections
        METERS_PER_DEGREE_LAT = 111000
        lat_correction_deg = pred_lat_m / METERS_PER_DEGREE_LAT
        lon_correction_deg = pred_lon_m / (METERS_PER_DEGREE_LAT * np.cos(np.radians(subset_baseline_lat)))
        
        corr_lat = subset_baseline_lat + lat_correction_deg
        corr_lon = subset_baseline_lon + lon_correction_deg
        
        # Compute position error
        errors = compute_position_error(corr_lat, corr_lon, subset_gt_lat, subset_gt_lon)
        return errors
    
    # Compute permutation importance (use subset for speed)
    n_repeats = 5
    sample_size = min(5000, len(X_val))
    X_val_sample = X_val[:sample_size]
    
    perm_importance = permutation_importance(
        estimator=lambda X: predict_wrapper(X),
        X=X_val_sample,
        y=np.zeros(sample_size),  # Dummy (error computed in predict_wrapper)
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    
    # Sort by importance
    importance_df = pd.DataFrame({
        'feature': feature_names_with_indicators[:X_val.shape[1]],
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    print("\n" + "-"*70)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("-"*70)
    print(f"{'Rank':<6} {'Feature':<40} {'Importance':>12} {'Std':>10}")
    print("-"*70)
    
    for idx, row in importance_df.head(20).iterrows():
        print(f"{idx+1:<6} {row['feature']:<40} {row['importance_mean']:>12.6f} {row['importance_std']:>10.6f}")
    
    print("-"*70)
    print("Note: Higher importance = bigger impact on prediction error when shuffled")
    
    # Save feature importance
    importance_path = OUTPUT_DIR / f"feature_importance_{timestamp_str}.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"\nFull feature importance saved to: {importance_path}")
    
    # Print tuning suggestions
    print("\n💡 KALMAN TUNING TIPS:")
    print("-" * 70)
    print("If P95 is still high:")
    print("  - Decrease process_variance (0.1-0.3) for more aggressive smoothing")
    print("  - Increase measurement_variance (8.0-10.0) to trust predictions less")
    print("\nIf P50 gets worse:")
    print("  - Increase process_variance (1.0-2.0) to follow predictions more")
    print("  - Decrease measurement_variance (2.0-3.0) to trust predictions more")
    print("="*70)

if __name__ == "__main__":
    main()