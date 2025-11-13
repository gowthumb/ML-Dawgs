# Residual Learning: Improving Position Estimates with Machine Learning

## 🎯 The Core Idea

Instead of predicting position directly, you **predict and correct the errors** of an existing positioning system.

```
┌──────────────────────────────────────────────────────────────┐
│  Traditional Approach                                        │
│  ─────────────────────────────────────────────────────────  │
│  Features (GNSS + IMU) → ML Model → Position Prediction     │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  Residual Learning Approach (Better!)                        │
│  ─────────────────────────────────────────────────────────  │
│  1. POS/PPK → Baseline Position (physics-based)             │
│  2. Features → ML Model → Predicted Error                    │
│  3. Final = Baseline + ML Correction                         │
└──────────────────────────────────────────────────────────────┘
```

---

## 📐 Mathematical Formula

### Training Phase

Given for each sample:
- **Ground Truth (GT)**: True position from high-quality reference
- **Baseline**: Position from POS/PPK solution
- **Features**: GNSS signal quality, IMU motion, satellite geometry, etc.

**Compute the residual (error):**
```
Residual = Ground Truth - Baseline
```

**Train LightGBM:**
```
LGBM(Features) → Predicted Residual
```

### Inference Phase

Given new data:
- **Baseline**: POS/PPK position (available in real-time)
- **Features**: Current GNSS/IMU measurements

**Apply correction:**
```
Final Position = Baseline + LGBM(Features)
```

---

## 🔢 Concrete Example

### Training Data

| Time | CN0 (dB) | Sats | IMU Accel | **Baseline Lat** | **GT Lat** | **Residual** |
|------|----------|------|-----------|------------------|------------|--------------|
| t1   | 42       | 12   | 9.8       | 37.5000°        | 37.5000°   | 0.0000°      |
| t2   | 35       | 8    | 12.3      | 37.5002°        | 37.5000°   | **-0.0002°** |
| t3   | 28       | 6    | 15.1      | 37.5005°        | 37.5000°   | **-0.0005°** |

**Pattern:** Low CN0 + Few satellites → Baseline drifts positive → Negative residual needed

**LGBM learns:**
```
When CN0 < 30 and Sats < 8:
  → Predict residual ≈ -0.0004°
```

### Inference (New Data)

| Time | CN0 (dB) | Sats | IMU Accel | **Baseline Lat** |
|------|----------|------|-----------|------------------|
| t_new| 29       | 7    | 14.5      | 37.5004°        |

**Step 1: Get baseline position**
```
Baseline = 37.5004° (from POS/PPK)
```

**Step 2: Predict residual**
```
LGBM(CN0=29, Sats=7, Accel=14.5) → Predicted Residual = -0.0003°
```

**Step 3: Apply correction**
```
Final = 37.5004° + (-0.0003°) = 37.5001°
```

If ground truth was 37.5000°:
- **Baseline error:** 0.0004° (4 meters)
- **Corrected error:** 0.0001° (1 meter)
- **Improvement:** 75% error reduction! ✓

---

## 🤔 Why Does This Work Better?

### 1. Leverages Existing Physics-Based Solution

Your POS/PPK solution already contains:
- ✅ Satellite geometry calculations
- ✅ Atmospheric corrections
- ✅ Pseudorange measurements
- ✅ Carrier phase processing

**Don't throw this away!** Use it as a starting point.

### 2. ML Learns Systematic Error Patterns

The baseline has errors that correlate with:

| Condition | Error Pattern | Features That Capture It |
|-----------|---------------|--------------------------|
| Urban canyon | Multipath errors | Low CN0, high elevation variance |
| Low satellites | Poor geometry | num_sats < 8, high GDOP |
| Vehicle turning | Centrifugal effects | High gyro_mag, accel_variance |
| Signal blockage | Loss of lock | Cycle slips, missing_indicators |

LGBM learns: **"When I see condition X, baseline tends to be wrong by Y"**

### 3. Combines Physics + Data-Driven

| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| POS/PPK Only | Physics-based, interpretable | Fixed model, misses patterns |
| ML Only | Learns patterns | Needs huge data, black box |
| **Residual Learning** | **Best of both!** | Requires baseline system |

---

## 📊 Expected Performance Gains

### Typical Improvements (from literature)

| Baseline System | Before (RMSE) | After (RMSE) | Improvement |
|-----------------|---------------|--------------|-------------|
| RTK Float       | 0.5 m         | 0.2 m        | 60% ↓       |
| PPK Single      | 5.0 m         | 2.0 m        | 60% ↓       |
| SPP (Standalone)| 10.0 m        | 4.0 m        | 60% ↓       |

### When It Works Best

✅ **High Improvement Scenarios:**
- Baseline has systematic errors (not random noise)
- Features correlate with error patterns
- Sufficient training data covering different conditions

❌ **Low Improvement Scenarios:**
- Baseline is already very accurate (< 10 cm)
- Errors are purely random (no pattern to learn)
- Features don't capture error causes

---

## 🛠️ Implementation Details

### Option 1: Single Model (Magnitude)

**Simple but less accurate:**

Train one model to predict error magnitude:
```python
residual_magnitude = sqrt(lat_error² + lon_error²)
model.predict(features) → predicted_magnitude
```

Then apply correction proportionally to original error direction.

**Pros:** Simple, one model
**Cons:** Loses directional information

### Option 2: Two Models (Components) ⭐ RECOMMENDED

**More accurate:**

Train separate models for each component:
```python
model_north.predict(features) → predicted_north_error
model_east.predict(features) → predicted_east_error
```

Apply corrections independently:
```python
corrected_lat = baseline_lat + predicted_north_error / 111000
corrected_lon = baseline_lon + predicted_east_error / (111000 * cos(lat))
```

**Pros:** More accurate, preserves directional patterns
**Cons:** Need to train multiple models

### Option 3: Three Models (3D) ⭐⭐ BEST

**Most accurate for 3D positioning:**

```python
model_north.predict(features) → north_error
model_east.predict(features) → east_error
model_up.predict(features) → up_error
```

**Pros:** Full 3D correction
**Cons:** Most complex

---

## 📝 Step-by-Step Implementation

### Step 1: Check Your Data

**Required columns:**
- Baseline position: `mean_latitude`, `mean_longitude`, `mean_height` (from POS/PPK)
- Ground truth: Either the same columns (if POS is GT) or separate reference
- Features: GNSS, IMU, EKF features

**Important:** If your `.pos` file **is** the ground truth, use `ekf_latitude`, `ekf_longitude` as baseline instead!

### Step 2: Compute Residuals

```python
# Latitude residual (degrees → meters)
lat_residual_deg = ground_truth_lat - baseline_lat
lat_residual_m = lat_residual_deg * 111000  # 1° lat ≈ 111 km

# Longitude residual (degrees → meters, latitude-dependent)
lon_residual_deg = ground_truth_lon - baseline_lon
lon_residual_m = lon_residual_deg * 111000 * cos(radians(ground_truth_lat))

# Horizontal error magnitude
horizontal_residual_m = sqrt(lat_residual_m² + lon_residual_m²)
```

### Step 3: Train Model

```python
# Target: the residual
y = horizontal_residual_m  # or lat_residual_m, lon_residual_m separately

# Features: same as before
X = df[ALL_FEATURES]

# Train
model = lgb.train(params, lgb.Dataset(X, label=y))
```

### Step 4: Apply Correction at Inference

```python
# Get baseline position (from POS/PPK in real-time)
baseline_lat = pos_solution['latitude']
baseline_lon = pos_solution['longitude']

# Extract features
features = extract_features(gnss_data, imu_data)

# Predict residual
predicted_residual_magnitude = model.predict(features)

# Convert to lat/lon corrections (simplified)
# (In production, use separate models for each component)
lat_correction = predicted_residual_magnitude * direction_lat / 111000
lon_correction = predicted_residual_magnitude * direction_lon / (111000 * cos(lat))

# Apply correction
corrected_lat = baseline_lat + lat_correction
corrected_lon = baseline_lon + lon_correction
```

---

## 🎓 Advanced Techniques

### 1. Uncertainty-Weighted Correction

Don't always trust the ML correction equally:

```python
# POS provides uncertainty estimate
pos_uncertainty = pos_solution['horizontal_uncertainty']

# Trust baseline more when uncertainty is low
weight = 1.0 / (1.0 + pos_uncertainty)

# Blend correction
final = baseline + weight * ml_correction
```

### 2. Conditional Correction

Only apply correction when ML is confident:

```python
if predicted_residual > threshold:
    final = baseline + ml_correction
else:
    final = baseline  # Keep baseline unchanged
```

### 3. Ensemble of Corrections

Train multiple models and average:

```python
correction = mean([model1.predict(), model2.predict(), model3.predict()])
final = baseline + correction
```

---

## 📈 Validation Strategy

### 1. Check Baseline Quality First

```python
baseline_rmse = compute_rmse(ground_truth, baseline)
print(f"Baseline RMSE: {baseline_rmse:.2f} m")
```

If baseline RMSE < 0.5m, improvement may be limited.

### 2. Compare Before/After

```python
baseline_error = ground_truth - baseline
corrected_error = ground_truth - (baseline + ml_correction)

print(f"Baseline RMSE:  {rmse(baseline_error):.2f} m")
print(f"Corrected RMSE: {rmse(corrected_error):.2f} m")
print(f"Improvement:    {improvement_pct:.1f}%")
```

### 3. Analyze When It Helps/Hurts

```python
# Find samples where correction helped
improved = abs(corrected_error) < abs(baseline_error)

# Analyze features for these samples
print(f"Improvement rate: {improved.mean()*100:.1f}%")
print(f"Average CN0 when improved: {X[improved]['mean_cn0'].mean():.1f}")
```

---

## ⚠️ Common Pitfalls

### 1. Data Leakage

❌ **Wrong:** Using EKF position as feature AND as baseline
```python
# This creates circular dependency!
features = [... 'ekf_latitude', 'ekf_longitude']
baseline = ekf_position
```

✅ **Right:** Use EKF uncertainty, not position
```python
features = [... 'ekf_pos_std', 'ekf_vel_std']  # Uncertainty only
baseline = pos_solution  # Different source
```

### 2. Training on Same Data

❌ **Wrong:** GT = Baseline (no residuals to learn)
```python
residual = pos_lat - pos_lat  # Always zero!
```

✅ **Right:** GT must be independent reference
```python
residual = reference_lat - pos_lat  # Real errors
```

### 3. Forgetting Direction

❌ **Wrong:** Predict magnitude only, lose direction
```python
correction = predicted_magnitude  # Which way to correct?
```

✅ **Right:** Predict components separately
```python
correction_north = model_north.predict()
correction_east = model_east.predict()
```

---

## 🚀 Quick Start

### Use the Provided Notebook

1. Open `train_residual_learning.ipynb`
2. Update data path and column names
3. Run all cells
4. Check improvement metrics

### Expected Output

```
PERFORMANCE COMPARISON
======================================================================
Metric          Baseline (m)    Corrected (m)     Improvement
----------------------------------------------------------------------
RMSE                    5.23             2.45          53.2%
MAE                     3.87             1.82          53.0%
Median                  2.95             1.31          55.6%
P95                    12.45             6.78          45.5%
======================================================================

✓ EXCELLENT: 53.2% RMSE improvement!
```

---

## 📚 References

- **Google Smartphone Decimeter Challenge:** Winning solutions used residual learning
- **Gradient Boosting Theory:** Residual learning is the foundation of boosting
- **Sensor Fusion:** Combining physics-based and data-driven models

---

## 💡 Key Takeaways

1. ✅ **Don't replace POS/PPK** - use it as baseline
2. ✅ **Train on residuals** - learn systematic errors
3. ✅ **Apply corrections** - Final = Baseline + ML
4. ✅ **Validate carefully** - check when it helps vs hurts
5. ✅ **Use separate models** - for lat/lon/height components

**Bottom line:** Residual learning combines the best of physics-based positioning with machine learning pattern recognition for superior results.
