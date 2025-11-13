"""
Training and evaluation utilities for residual learning model.
"""

import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple

from config import (
    LGBM_PARAMS,
    NUM_BOOST_ROUNDS,
    LOG_EVALUATION_PERIOD,
    TEST_SIZE,
    RANDOM_STATE,
    OUTPUT_DIR,
)
from feature_engineering import (
    compute_position_error,
    apply_residual_correction,
)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def custom_eval_metric(y_pred, train_data):
    """
    Custom evaluation metric for LightGBM.
    Calculates the mean of 50th percentile and 95th percentile of horizontal errors.

    Args:
        y_pred: Predicted values (residuals)
        train_data: LightGBM Dataset object

    Returns:
        tuple: (metric_name, metric_value, is_higher_better)
    """
    y_true = train_data.get_label()

    # Calculate absolute errors (horizontal errors)
    errors = np.abs(y_true - y_pred)

    # Calculate 50th percentile (median) and 95th percentile
    p50 = np.percentile(errors, 50)
    p95 = np.percentile(errors, 95)

    # Mean of both percentiles
    custom_metric = (p50 + p95) / 2.0

    # Return (name, value, is_higher_better)
    # Lower is better for error metrics
    return 'mean_p50_p95', custom_metric, False


class ResidualLearningTrainer:
    """
    Trainer class for residual learning model.
    """

    def __init__(self, lgbm_params: Dict = None, use_component_models: bool = True):
        """
        Initialize trainer.

        Args:
            lgbm_params: LightGBM parameters. If None, uses default from config.
            use_component_models: If True, train separate lat/lon models. If False, train single magnitude model.
        """
        self.lgbm_params = lgbm_params or LGBM_PARAMS
        self.use_component_models = use_component_models

        # Component models (lat/lon separate)
        self.model_lat = None
        self.model_lon = None
        self.best_iteration_lat = None
        self.best_iteration_lon = None

        # Single magnitude model (legacy)
        self.model = None
        self.best_iteration = None

    def train_test_split_data(
        self,
        X, y, baseline_lat, baseline_lon, gt_lat, gt_lon, lat_residual, lon_residual,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ):
        """
        Split data into train and test sets.

        Args:
            X: Features
            y: Target
            baseline_lat, baseline_lon: Baseline positions
            gt_lat, gt_lon: Ground truth positions
            lat_residual, lon_residual: Residual components
            test_size: Fraction of data for test set
            random_state: Random seed

        Returns:
            tuple: All split data (X_train, X_test, y_train, y_test, ...)
        """
        split_data = train_test_split(
            X, y, baseline_lat, baseline_lon, gt_lat, gt_lon, lat_residual, lon_residual,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        print(f"\nTrain/Test Split:")
        print(f"  Train samples: {len(split_data[0]):,}")
        print(f"  Test samples:  {len(split_data[1]):,}")

        return split_data

    def train_components(self, X_train, y_lat_train, y_lon_train, X_test, y_lat_test, y_lon_test):
        """
        Train separate LightGBM models for lat and lon residuals.

        Args:
            X_train: Training features
            y_lat_train: Latitude residuals (training)
            y_lon_train: Longitude residuals (training)
            X_test: Test features
            y_lat_test: Latitude residuals (test)
            y_lon_test: Longitude residuals (test)

        Returns:
            tuple: (model_lat, model_lon)
        """
        print("\nTraining COMPONENT MODELS for lat and lon residuals separately...\n")
        print("Using custom evaluation metric: mean(P50, P95) of horizontal errors")
        print("(Lower is better)\n")

        # Train latitude model
        print("=" * 70)
        print("Training LATITUDE residual model...")
        print("=" * 70)
        train_data_lat = lgb.Dataset(X_train, label=y_lat_train)
        valid_data_lat = lgb.Dataset(X_test, label=y_lat_test, reference=train_data_lat)

        callbacks = [lgb.log_evaluation(period=LOG_EVALUATION_PERIOD)]
        self.model_lat = lgb.train(
            self.lgbm_params,
            train_data_lat,
            num_boost_round=NUM_BOOST_ROUNDS,
            valid_sets=[train_data_lat, valid_data_lat],
            valid_names=['train', 'valid'],
            feval=custom_eval_metric,
            callbacks=callbacks,
        )
        self.best_iteration_lat = self.model_lat.best_iteration
        print(f"\nBest iteration (lat): {self.best_iteration_lat}")

        # Train longitude model
        print("\n" + "=" * 70)
        print("Training LONGITUDE residual model...")
        print("=" * 70)
        train_data_lon = lgb.Dataset(X_train, label=y_lon_train)
        valid_data_lon = lgb.Dataset(X_test, label=y_lon_test, reference=train_data_lon)

        self.model_lon = lgb.train(
            self.lgbm_params,
            train_data_lon,
            num_boost_round=NUM_BOOST_ROUNDS,
            valid_sets=[train_data_lon, valid_data_lon],
            valid_names=['train', 'valid'],
            feval=custom_eval_metric,
            callbacks=callbacks,
        )
        self.best_iteration_lon = self.model_lon.best_iteration
        print(f"\nBest iteration (lon): {self.best_iteration_lon}")

        return self.model_lat, self.model_lon

    def train(self, X_train, y_train, X_test, y_test):
        """
        Train LightGBM model to predict residuals.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Validation data

        Returns:
            lgb.Booster: Trained model
        """
        print("\nTraining LightGBM to predict residuals...\n")
        print("Using custom evaluation metric: mean(P50, P95) of horizontal errors")
        print("(Lower is better)\n")

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # Train with custom evaluation metric
        callbacks = [lgb.log_evaluation(period=LOG_EVALUATION_PERIOD)]
        self.model = lgb.train(
            self.lgbm_params,
            train_data,
            num_boost_round=NUM_BOOST_ROUNDS,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            feval=custom_eval_metric,  # Add custom evaluation function
            callbacks=callbacks,
        )

        self.best_iteration = self.model.best_iteration
        print(f"\nBest iteration: {self.best_iteration}")

        return self.model

    def predict_components(self, X):
        """
        Predict lat and lon residuals separately using component models.

        Args:
            X: Features

        Returns:
            tuple: (predicted_lat_residual, predicted_lon_residual)
        """
        if self.model_lat is None or self.model_lon is None:
            raise ValueError("Component models not trained yet!")

        pred_lat = self.model_lat.predict(X, num_iteration=self.best_iteration_lat)
        pred_lon = self.model_lon.predict(X, num_iteration=self.best_iteration_lon)

        return pred_lat, pred_lon

    def predict(self, X):
        """
        Predict residuals for given features.

        Args:
            X: Features

        Returns:
            Array of predicted residuals
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        return self.model.predict(X, num_iteration=self.best_iteration)

    def evaluate(
        self,
        X_test,
        y_test,
        baseline_lat_test,
        baseline_lon_test,
        gt_lat_test,
        gt_lon_test,
        lat_res_test,
        lon_res_test,
    ) -> Tuple[Dict, Dict]:
        """
        Evaluate model performance by comparing baseline vs corrected errors.

        Args:
            X_test: Test features
            y_test: Test targets (true residual magnitude)
            baseline_lat_test, baseline_lon_test: Baseline positions
            gt_lat_test, gt_lon_test: Ground truth positions
            lat_res_test, lon_res_test: True residual components

        Returns:
            tuple: (baseline_metrics, corrected_metrics)
        """
        # Predict residuals
        y_pred_residual = self.predict(X_test)

        print("\nPredicted residuals (test set):")
        print(f"  Mean: {y_pred_residual.mean():.2f} m")
        print(f"  Std:  {y_pred_residual.std():.2f} m")

        # Apply corrections
        corrected_lat, corrected_lon = apply_residual_correction(
            baseline_lat_test,
            baseline_lon_test,
            lat_res_test,
            lon_res_test,
            y_pred_residual,
            y_test,
        )

        # Compute errors
        baseline_error_m = compute_position_error(
            baseline_lat_test, baseline_lon_test,
            gt_lat_test, gt_lon_test,
        )

        corrected_error_m = compute_position_error(
            corrected_lat, corrected_lon,
            gt_lat_test, gt_lon_test,
        )

        # Compute metrics
        baseline_metrics = self._compute_metrics(baseline_error_m)
        corrected_metrics = self._compute_metrics(corrected_error_m)

        # Display comparison
        self._display_metrics_comparison(baseline_metrics, corrected_metrics)

        return baseline_metrics, corrected_metrics

    def evaluate_components(
        self,
        X_test,
        baseline_lat_test,
        baseline_lon_test,
        gt_lat_test,
        gt_lon_test,
        timestamp_test=None,
        apply_kalman: bool = True,
        process_variance: float = 0.5,
        measurement_variance: float = 5.0,
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Evaluate component models by directly applying lat/lon corrections.

        Args:
            X_test: Test features
            baseline_lat_test, baseline_lon_test: Baseline positions
            gt_lat_test, gt_lon_test: Ground truth positions
            timestamp_test: Timestamps for sorting before Kalman (optional)
            apply_kalman: Whether to apply Kalman smoothing
            process_variance: Kalman process noise (lower = smoother)
            measurement_variance: Kalman measurement noise (higher = more smoothing)

        Returns:
            tuple: (baseline_metrics, corrected_metrics, corrected_kalman_metrics)
        """
        from config import METERS_PER_DEGREE_LAT

        # Predict lat and lon residuals
        pred_lat_m, pred_lon_m = self.predict_components(X_test)

        print("\nPredicted residuals (test set):")
        print(f"  Lat residual - Mean: {pred_lat_m.mean():.2f} m, Std: {pred_lat_m.std():.2f} m")
        print(f"  Lon residual - Mean: {pred_lon_m.mean():.2f} m, Std: {pred_lon_m.std():.2f} m")

        # Apply corrections directly (without Kalman)
        lat_correction_deg = pred_lat_m / METERS_PER_DEGREE_LAT
        lon_correction_deg = pred_lon_m / (METERS_PER_DEGREE_LAT * np.cos(np.radians(baseline_lat_test)))

        corrected_lat = baseline_lat_test + lat_correction_deg
        corrected_lon = baseline_lon_test + lon_correction_deg

        # Compute baseline errors
        baseline_error_m = compute_position_error(
            baseline_lat_test, baseline_lon_test,
            gt_lat_test, gt_lon_test,
        )

        # Compute corrected errors (without Kalman)
        corrected_error_m = compute_position_error(
            corrected_lat, corrected_lon,
            gt_lat_test, gt_lon_test,
        )

        # Compute metrics
        baseline_metrics = self._compute_metrics(baseline_error_m)
        corrected_metrics = self._compute_metrics(corrected_error_m)

        # Apply Kalman smoothing if requested
        corrected_kalman_metrics = None
        if apply_kalman:
            from kalman_smoother import apply_kalman_to_corrections

            # Sort by timestamp if available for proper temporal smoothing
            if timestamp_test is not None:
                print("\nSorting by timestamp for Kalman smoothing...")
                # Convert to numpy array and get sort indices
                if isinstance(timestamp_test, pd.Series):
                    timestamp_arr = timestamp_test.values
                else:
                    timestamp_arr = np.array(timestamp_test)

                sort_idx = np.argsort(timestamp_arr)
                pred_lat_sorted = pred_lat_m[sort_idx]
                pred_lon_sorted = pred_lon_m[sort_idx]

                # Apply Kalman smoothing to sorted data
                smoothed_lat_sorted, smoothed_lon_sorted = apply_kalman_to_corrections(
                    pred_lat_sorted, pred_lon_sorted,
                    process_variance=process_variance,
                    measurement_variance=measurement_variance,
                )

                # Restore original order
                unsort_idx = np.argsort(sort_idx)
                smoothed_lat_m = smoothed_lat_sorted[unsort_idx]
                smoothed_lon_m = smoothed_lon_sorted[unsort_idx]
            else:
                print("\nWARNING: No timestamp available, applying Kalman without sorting")
                smoothed_lat_m, smoothed_lon_m = apply_kalman_to_corrections(
                    pred_lat_m, pred_lon_m,
                    process_variance=process_variance,
                    measurement_variance=measurement_variance,
                )

            # Apply smoothed corrections
            lat_correction_deg_smooth = smoothed_lat_m / METERS_PER_DEGREE_LAT
            lon_correction_deg_smooth = smoothed_lon_m / (METERS_PER_DEGREE_LAT * np.cos(np.radians(baseline_lat_test)))

            corrected_lat_smooth = baseline_lat_test + lat_correction_deg_smooth
            corrected_lon_smooth = baseline_lon_test + lon_correction_deg_smooth

            # Compute smoothed errors
            corrected_error_m_smooth = compute_position_error(
                corrected_lat_smooth, corrected_lon_smooth,
                gt_lat_test, gt_lon_test,
            )

            corrected_kalman_metrics = self._compute_metrics(corrected_error_m_smooth)

            # Display comparison with Kalman
            print("\n" + "="*80)
            print("PERFORMANCE COMPARISON (WITH KALMAN SMOOTHING)")
            print("="*80)
            print(f"{'Metric':<18} {'Baseline (m)':>15} {'Corrected (m)':>15} {'+ Kalman (m)':>15} {'Kalman Improve':>15}")
            print("-"*80)

            for metric in ['RMSE', 'MAE', 'Median (P50)', 'P95', 'Mean(P50,P95)', 'Max']:
                base_val = baseline_metrics[metric]
                corr_val = corrected_metrics[metric]
                kalman_val = corrected_kalman_metrics[metric]

                improvement = ((base_val - kalman_val) / base_val) * 100
                kalman_improve = ((corr_val - kalman_val) / corr_val) * 100

                print(f"{metric:<18} {base_val:>15.2f} {corr_val:>15.2f} {kalman_val:>15.2f} {kalman_improve:>14.1f}%")

            print("="*80)

            # Summary
            kalman_custom_improvement = ((baseline_metrics['Mean(P50,P95)'] - corrected_kalman_metrics['Mean(P50,P95)'])
                                         / baseline_metrics['Mean(P50,P95)']) * 100
            print(f"\nCustom Metric [Mean(P50,P95)] with Kalman: {kalman_custom_improvement:.1f}% improvement")
            print(f"Target: 1-2m | Current: {corrected_kalman_metrics['Mean(P50,P95)']:.2f}m")

            if corrected_kalman_metrics['Mean(P50,P95)'] <= 2.0:
                print("EXCELLENT: Target achieved!")
            elif corrected_kalman_metrics['Mean(P50,P95)'] <= 3.0:
                print("GOOD: Close to target, tune Kalman parameters")
            else:
                print("NEEDS WORK: Try higher measurement_variance or lower process_variance")

        else:
            # Display comparison without Kalman
            self._display_metrics_comparison(baseline_metrics, corrected_metrics)

        return baseline_metrics, corrected_metrics, corrected_kalman_metrics

    def _compute_metrics(self, errors: np.ndarray) -> Dict:
        """Compute evaluation metrics including custom mean(P50, P95) metric."""
        abs_errors = np.abs(errors)
        p50 = np.percentile(abs_errors, 50)
        p95 = np.percentile(abs_errors, 95)
        mean_p50_p95 = (p50 + p95) / 2.0

        return {
            'RMSE': np.sqrt(np.mean(errors**2)),
            'MAE': np.mean(abs_errors),
            'Median (P50)': p50,
            'P95': p95,
            'Mean(P50,P95)': mean_p50_p95,  # Custom metric
            'Max': np.max(abs_errors),
        }

    def _display_metrics_comparison(self, baseline_metrics: Dict, corrected_metrics: Dict):
        """Display metrics comparison table."""
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON")
        print("="*80)
        print(f"{'Metric':<18} {'Baseline (m)':>15} {'Corrected (m)':>15} {'Improvement':>12}")
        print("-"*80)

        for metric in ['RMSE', 'MAE', 'Median (P50)', 'P95', 'Mean(P50,P95)', 'Max']:
            base_val = baseline_metrics[metric]
            corr_val = corrected_metrics[metric]
            improvement = ((base_val - corr_val) / base_val) * 100

            print(f"{metric:<18} {base_val:>15.2f} {corr_val:>15.2f} {improvement:>11.1f}%")

        print("="*80)

        # Summary - highlight custom metric
        custom_metric_improvement = ((baseline_metrics['Mean(P50,P95)'] - corrected_metrics['Mean(P50,P95)'])
                                     / baseline_metrics['Mean(P50,P95)']) * 100
        rmse_improvement = ((baseline_metrics['RMSE'] - corrected_metrics['RMSE']) / baseline_metrics['RMSE']) * 100

        print(f"\n📊 Custom Metric [Mean(P50,P95)]: {custom_metric_improvement:.1f}% improvement")

        if rmse_improvement > 10:
            print(f"✓ EXCELLENT: {rmse_improvement:.1f}% RMSE improvement!")
        elif rmse_improvement > 5:
            print(f"✓ GOOD: {rmse_improvement:.1f}% RMSE improvement")
        elif rmse_improvement > 0:
            print(f"~ MODEST: {rmse_improvement:.1f}% RMSE improvement")
        else:
            print(f"✗ WARNING: No improvement ({rmse_improvement:.1f}%)")
            print("  → Check if baseline is already very good or features lack signal")

    def plot_feature_importance(self, top_n: int = 20):
        """
        Plot feature importance.

        Args:
            top_n: Number of top features to display
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Get feature importance
        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.model.feature_name()

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        # Plot
        plot_df = importance_df.head(top_n).sort_values('importance', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(plot_df['feature'], plot_df['importance'])
        colors = plt.cm.viridis(plot_df['importance'] / plot_df['importance'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel('Importance (Gain)')
        ax.set_title(f'Top {top_n} Features Predicting Position Error')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"\nTop 10 features that predict baseline errors:")
        print(importance_df.head(10))

        return importance_df

    def save_component_models(self, suffix: str = "") -> Tuple[str, str]:
        """
        Save trained component models to disk.

        Args:
            suffix: Optional suffix for filename

        Returns:
            tuple: (path_to_lat_model, path_to_lon_model)
        """
        if self.model_lat is None or self.model_lon is None:
            raise ValueError("Component models not trained yet!")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save latitude model
        model_lat_filename = f"residual_model_lat_{timestamp}"
        if suffix:
            model_lat_filename += f"_{suffix}"
        model_lat_filename += ".pkl"
        model_lat_path = os.path.join(OUTPUT_DIR, model_lat_filename)

        with open(model_lat_path, 'wb') as f:
            pickle.dump(self.model_lat, f)

        # Save longitude model
        model_lon_filename = f"residual_model_lon_{timestamp}"
        if suffix:
            model_lon_filename += f"_{suffix}"
        model_lon_filename += ".pkl"
        model_lon_path = os.path.join(OUTPUT_DIR, model_lon_filename)

        with open(model_lon_path, 'wb') as f:
            pickle.dump(self.model_lon, f)

        print(f"\nComponent models saved:")
        print(f"  Lat model: {model_lat_path}")
        print(f"  Lon model: {model_lon_path}")

        return model_lat_path, model_lon_path

    def save_model(self, suffix: str = "") -> str:
        """
        Save trained model to disk.

        Args:
            suffix: Optional suffix for filename

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"residual_model_{timestamp}"
        if suffix:
            model_filename += f"_{suffix}"
        model_filename += ".pkl"

        model_path = os.path.join(OUTPUT_DIR, model_filename)

        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        print(f"\nModel saved to: {model_path}")
        return model_path

    def save_metrics(self, baseline_metrics: Dict, corrected_metrics: Dict, suffix: str = "") -> str:
        """
        Save metrics comparison to CSV.

        Args:
            baseline_metrics: Baseline performance metrics
            corrected_metrics: Corrected performance metrics
            suffix: Optional suffix for filename

        Returns:
            Path to saved metrics
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_filename = f"residual_metrics_{timestamp}"
        if suffix:
            metrics_filename += f"_{suffix}"
        metrics_filename += ".csv"

        metrics_path = os.path.join(OUTPUT_DIR, metrics_filename)

        metrics_df = pd.DataFrame({
            'metric': list(baseline_metrics.keys()),
            'baseline': list(baseline_metrics.values()),
            'corrected': list(corrected_metrics.values()),
        })
        metrics_df['improvement_%'] = ((metrics_df['baseline'] - metrics_df['corrected']) / metrics_df['baseline']) * 100

        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to: {metrics_path}")

        return metrics_path
