# LightGBM Model for GNSS/IMU Sensor Fusion

This directory contains a complete LightGBM training pipeline for predicting positioning quality metrics from GNSS and IMU sensor data.

## 🎯 Key Features

- **Native Missing Value Handling**: Keeps NaNs in GNSS features (LightGBM handles them automatically with `use_missing=True`)
- **Missingness Indicators**: Uses indicator columns to capture patterns in when GNSS data is unavailable
- **No Imputation Required**: LightGBM learns optimal splits for missing values during training
- **Comprehensive Evaluation**: Feature importance analysis, metrics tracking, and visualizations

## 📁 Directory Structure

```
model/
├── README.md                    # This file
├── config.py                    # Model hyperparameters and feature definitions
├── data_loader.py              # Data loading and preprocessing utilities
├── evaluate.py                 # Model evaluation and metrics computation
├── feature_importance.py       # Feature importance analysis tools
├── train_lgbm.py              # Command-line training script
├── train_lightgbm.ipynb       # Interactive Jupyter notebook (RECOMMENDED)
└── outputs/                    # Training outputs (created automatically)
    ├── lgbm_model_*.txt        # Saved models (LightGBM format)
    ├── lgbm_model_*.pkl        # Saved models (pickle format)
    ├── feature_importance_*.csv # Feature importance tables
    ├── metrics_*.csv           # Training/test metrics
    ├── test_predictions_*.csv  # Model predictions
    └── *.png                   # Visualization plots
```

## 🚀 Quick Start

### Option 1: Interactive Jupyter Notebook (Recommended)

The easiest way to get started is with the Jupyter notebook:

```bash
jupyter notebook model/train_lightgbm.ipynb
```

The notebook includes:
- Step-by-step explanations
- Inline visualizations
- Interactive exploration
- Complete training pipeline

### Option 2: Command-Line Script

For automated training:

```bash
python model/train_lgbm.py \
    --data training_set_all_folders_ekf_20250101_120000.csv \
    --target position_uncertainty \
    --output_dir model/outputs \
    --test_size 0.2
```

**Arguments:**
- `--data`: Path to training data CSV (output from `main_process.py`)
- `--target`: What to predict (see Target Options below)
- `--output_dir`: Where to save outputs (default: `model/outputs`)
- `--test_size`: Test set fraction (default: 0.2)
- `--feature_groups`: Which feature groups to use (default: all)
- `--no_plots`: Disable plot generation

## 🎯 Target Options

Choose what you want to predict by setting the `--target` parameter:

| Target | Description | Use Case |
|--------|-------------|----------|
| `position_uncertainty` | Horizontal uncertainty from PPK solution | Predict positioning quality |
| `position_error` | Computed error between EKF and ground truth | Predict positioning error |
| `solution_quality` | Quality score from PPK (1=best, 6=worst) | Classify solution quality |
| `latitude` | Direct latitude prediction | Position estimation |
| `longitude` | Direct longitude prediction | Position estimation |

**Default:** `position_uncertainty`

## 📊 Feature Groups

The model uses four main feature groups:

### 1. GNSS Features (~50% missing when satellites unavailable)
- Signal quality (CN0, SNR)
- Satellite geometry (elevation, azimuth)
- Pseudorange and carrier phase
- Doppler measurements
- Satellite counts and usage

### 2. IMU Features (minimal missing values)
- Accelerometer magnitude and statistics
- Gyroscope magnitude and statistics
- Motion state indicators
- Jerk and variance metrics

### 3. EKF Features (if using Extended Kalman Filter)
- Position estimates (lat/lon/height)
- Velocity estimates
- Uncertainty estimates

### 4. Missingness Indicators
- `gnss_raw_missing`: Binary flag for missing raw GNSS data
- `gnss_status_missing`: Binary flag for missing GNSS status data

## 🔧 Configuration

Edit `config.py` to customize:

### Model Hyperparameters

```python
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'use_missing': True,  # CRITICAL: Enable native NaN handling
    'zero_as_missing': False,
    # ... more parameters
}
```

### Training Parameters

```python
TRAINING_PARAMS = {
    'num_boost_round': 1000,
    'early_stopping_rounds': 50,
    'verbose_eval': 50,
}
```

## 📈 Workflow

### 1. Data Preparation

Run your feature extraction pipeline first:

```bash
python main_process.py
```

This creates: `training_set_all_folders_ekf_TIMESTAMP.csv`

### 2. Model Training

**Option A: Jupyter Notebook**
```bash
jupyter notebook model/train_lightgbm.ipynb
```

**Option B: Command Line**
```bash
python model/train_lgbm.py --data training_set_all_folders_ekf_TIMESTAMP.csv
```

### 3. Review Results

Check the outputs in `model/outputs/`:
- **Metrics**: Training and test performance
- **Feature Importance**: Which features matter most
- **Predictions**: Model outputs on test set
- **Plots**: Visualizations of performance and importance

## 🎓 Understanding Missing Value Handling

### Why Keep NaNs?

LightGBM has native missing value support. When a feature is missing:
1. The tree learns to send samples down a "default direction"
2. This is optimized during training to minimize loss
3. No information is lost (unlike mean imputation)

### Example

```python
# ❌ DON'T DO THIS:
X['gnss_feature'].fillna(X['gnss_feature'].mean())  # Loses information

# ✅ DO THIS:
X['gnss_feature']  # Keep NaNs, LightGBM handles them
```

### Missingness Indicators

Additional signal from *when* data is missing:

```python
# These columns are already created in your pipeline
'gnss_raw_missing'     # 1 if GNSS raw data unavailable
'gnss_status_missing'  # 1 if GNSS status unavailable
```

## 📊 Evaluation Metrics

The model computes comprehensive metrics:

### Regression Metrics
- **RMSE**: Root Mean Squared Error (penalizes large errors)
- **MAE**: Mean Absolute Error (average error magnitude)
- **Median AE**: Robust to outliers
- **R²**: Explained variance (1.0 = perfect)
- **MAPE**: Mean Absolute Percentage Error

### Error Statistics
- Mean error (bias)
- Standard deviation
- Percentiles (50th, 90th, 95th, 99th)

## 🔍 Feature Importance Analysis

After training, the model provides:

### By Gain (Most Important)
Total information gain from splits using each feature

### By Split Frequency
How often each feature is used in tree splits

### By Group
Aggregate importance for GNSS, IMU, EKF, and missingness indicators

### Example Output

```
TOP 20 MOST IMPORTANT FEATURES (by gain)
========================================================================
Rank   Feature                                  Importance        %  Cumul%
------------------------------------------------------------------------
1      ekf_pos_std                                 1234.56    12.3%   12.3%
2      mean_cn0                                     987.65     9.9%   22.2%
3      num_sats                                     654.32     6.5%   28.7%
...
```

## 🛠️ Advanced Usage

### Custom Feature Selection

```python
from model.data_loader import prepare_features_and_target

# Use only GNSS and missingness indicators
X, y, features = prepare_features_and_target(
    df,
    target_name='position_uncertainty',
    feature_groups=['gnss', 'missingness'],
)
```

### Time-Based Split

For temporal validation:

```python
from model.config import VALIDATION_CONFIG

VALIDATION_CONFIG['use_time_split'] = True
VALIDATION_CONFIG['time_col'] = 'millisSinceGpsEpoch'
```

### Drive-Based Split

For leave-one-drive-out validation:

```python
VALIDATION_CONFIG['use_drive_split'] = True
VALIDATION_CONFIG['drive_col'] = 'drive_id'
```

## 🔄 Model Loading and Inference

### Load Trained Model

```python
import pickle

# Load model
with open('model/outputs/lgbm_model_TIMESTAMP.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions (X can contain NaNs)
predictions = model.predict(X_new)
```

### Batch Inference

```python
import pandas as pd
import lightgbm as lgb

# Load model
model = lgb.Booster(model_file='model/outputs/lgbm_model_TIMESTAMP.txt')

# Load new data
df_new = pd.read_csv('new_data.csv')
X_new = df_new[feature_names]  # Same features used in training

# Predict
y_pred = model.predict(X_new, num_iteration=model.best_iteration)
```

## 🐛 Troubleshooting

### Issue: "Target column not found"

**Solution:** Your data file doesn't have the target column. Check available columns:

```python
df = pd.read_csv('your_data.csv')
print(df.columns.tolist())
```

Then choose an available target or compute it (see `data_loader.py` for examples).

### Issue: "All features are NaN"

**Solution:** Some features weren't created in the pipeline. Edit `config.py` to only include features that exist in your data:

```python
# Check which features exist
available_features = [f for f in ALL_FEATURES if f in df.columns]
```

### Issue: "Poor model performance"

**Solutions:**
1. **Check data quality**: Inspect missing value ratios and outliers
2. **Tune hyperparameters**: Adjust learning rate, num_leaves, regularization
3. **Feature engineering**: Create new features based on importance analysis
4. **Remove noise**: Filter out low-quality samples or outliers

## 📚 References

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [LightGBM Missing Value Handling](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#missing-value-handle)
- [Best Practices for Missing Values](https://stats.stackexchange.com/questions/235489/how-does-xgboost-learn-what-are-the-inputs-for-missing-values)

## 🤝 Contributing

To add new features or modify the pipeline:

1. **Add features** in `feature.py` (parent directory)
2. **Update feature lists** in `config.py`
3. **Test** with the Jupyter notebook
4. **Document** changes in this README

## ⚠️ Important Notes

1. **Don't drop or impute GNSS features**: LightGBM handles NaNs natively
2. **Always drop NaN targets**: Model cannot train on missing targets
3. **Use missingness indicators**: Additional signal from missing patterns
4. **Monitor overfitting**: Check train/test metric ratios
5. **Analyze feature importance**: Remove redundant features for efficiency

## 📝 License

This code is part of the ML-Dawgs GNSS/IMU fusion project.
