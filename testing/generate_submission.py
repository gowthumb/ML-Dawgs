"""
Submission Generation Script for Google Smartphone Decimeter Challenge
=======================================================================

This script generates a submission file in the required format:
    tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees

Usage:
    python generate_submission.py --model model/outputs/residual_model_TIMESTAMP.pkl \
                                   --test_data test_features.csv \
                                   --output submission.csv
"""

import pandas as pd
import numpy as np
import pickle
import argparse
from pathlib import Path
from typing import Optional


class SubmissionGenerator:
    """Generates competition submissions from trained models."""

    def __init__(self, model_path: str):
        """
        Initialize the submission generator.

        Args:
            model_path: Path to the trained model pickle file
        """
        print(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print("✓ Model loaded successfully")

    def load_test_data(self, test_data_path: str) -> pd.DataFrame:
        """
        Load test data (featurization CSV).

        Args:
            test_data_path: Path to test featurization CSV

        Returns:
            DataFrame with features and baseline positions
        """
        print(f"\nLoading test data from: {test_data_path}")
        df = pd.read_csv(test_data_path)
        print(f"✓ Loaded {len(df):,} rows with {len(df.columns)} columns")

        # Verify required columns exist
        required_cols = ['drive_id', 'phone_id', 'millisSinceGpsEpoch',
                        'mean_latitude', 'mean_longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features used by the model.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with only feature columns
        """
        # Define all possible features (same as training)
        GNSS_FEATURES = [
            'num_sats', 'mean_cn0', 'std_cn0', 'max_cn0', 'min_cn0',
            'mean_cn0_norm', 'mean_cn0_smooth', 'std_cn0_rate',
            'mean_pseudorange', 'std_pseudorange', 'mean_doppler',
            'num_sats_status', 'mean_wls_status', 'max_sats_used',
            'hae_std', 'mean_elevation', 'std_elevation', 'max_elevation', 'min_elevation',
            'azimuth_spread', 'mean_weighted_cn0', 'num_high_elev_sats',
            'cn0_trend', 'num_sats_std_5s', 'high_qual_sat_ratio', 'hae_std_roll',
        ]

        IMU_FEATURES = [
            'accel_mag_mean', 'accel_mag_roll_mean', 'accel_variance', 'accel_jerk',
            'accel_mag_no_gravity', 'gyro_mag_mean', 'gyro_mag_roll_mean',
            'gyro_variance', 'gyro_jerk', 'accel_mag_std_1s_roll', 'is_stationary',
        ]

        EKF_FEATURES = [
            'ekf_latitude', 'ekf_longitude', 'ekf_height',
            'ekf_vel_n', 'ekf_vel_e', 'ekf_vel_u',
            'ekf_pos_std', 'ekf_vel_std',
        ]

        MISSINGNESS_INDICATORS = [
            'gnss_raw_missing',
            'gnss_status_missing',
        ]

        # Check which features are available
        available_features = []
        for feature_list in [GNSS_FEATURES, IMU_FEATURES, EKF_FEATURES, MISSINGNESS_INDICATORS]:
            available_features.extend([f for f in feature_list if f in df.columns])

        print(f"\nExtracted {len(available_features)} features")

        return df[available_features].copy()

    def predict_residuals(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict residuals using the trained model.

        Args:
            features: Feature DataFrame

        Returns:
            Array of predicted horizontal residuals (meters)
        """
        print("\nPredicting residuals...")
        predictions = self.model.predict(features)

        print(f"✓ Predicted {len(predictions):,} residuals")
        print(f"  Mean residual: {predictions.mean():.2f} m")
        print(f"  Std residual:  {predictions.std():.2f} m")

        return predictions

    def apply_corrections(self, df: pd.DataFrame, predicted_residuals: np.ndarray) -> pd.DataFrame:
        """
        Apply predicted corrections to baseline positions.

        For simplicity, this applies corrections uniformly in all directions.
        For better accuracy, train separate models for lat/lon residuals.

        Args:
            df: DataFrame with baseline positions
            predicted_residuals: Predicted horizontal residuals (meters)

        Returns:
            DataFrame with corrected positions
        """
        print("\nApplying corrections to baseline positions...")

        result = df[['drive_id', 'phone_id', 'millisSinceGpsEpoch',
                     'mean_latitude', 'mean_longitude']].copy()

        # For residual learning models that predict magnitude only,
        # we need to estimate direction.
        #
        # SIMPLIFIED APPROACH (used here):
        # Apply a small uniform correction based on average error patterns
        # This assumes errors are roughly isotropic (same in all directions)
        #
        # BETTER APPROACH (recommended for production):
        # Train separate models for lat_residual and lon_residual

        # Convert residual magnitude to approximate lat/lon corrections
        # Assume errors are distributed evenly between lat and lon
        error_per_axis = predicted_residuals / np.sqrt(2)

        # Convert meters to degrees
        lat_correction_deg = error_per_axis / 111000  # 1° lat ≈ 111 km
        lon_correction_deg = error_per_axis / (111000 * np.cos(np.radians(result['mean_latitude'])))

        # Apply corrections (note: this is a simplified approximation)
        result['corrected_latitude'] = result['mean_latitude'] + lat_correction_deg
        result['corrected_longitude'] = result['mean_longitude'] + lon_correction_deg

        print(f"✓ Applied corrections to {len(result):,} positions")

        return result

    def convert_to_submission_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert to competition submission format.

        Required format:
            tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees

        Args:
            df: DataFrame with corrected positions

        Returns:
            DataFrame in submission format
        """
        print("\nConverting to submission format...")

        submission = pd.DataFrame()

        # 1. Create tripId: DRIVE_ID/PHONE_ID
        submission['tripId'] = df['drive_id'] + '/' + df['phone_id']

        # 2. Convert millisSinceGpsEpoch to UnixTimeMillis
        GPS_EPOCH_OFFSET_MILLIS = 315964800000
        submission['UnixTimeMillis'] = df['millisSinceGpsEpoch'] + GPS_EPOCH_OFFSET_MILLIS

        # 3. Use corrected positions
        submission['LatitudeDegrees'] = df['corrected_latitude']
        submission['LongitudeDegrees'] = df['corrected_longitude']

        # Sort by tripId and time
        submission = submission.sort_values(['tripId', 'UnixTimeMillis']).reset_index(drop=True)

        print(f"✓ Generated submission with {len(submission):,} predictions")
        print(f"  Unique trips: {submission['tripId'].nunique()}")
        print(f"  Time range: {submission['UnixTimeMillis'].min()} to {submission['UnixTimeMillis'].max()}")

        return submission

    def generate(self, test_data_path: str, output_path: str = "submission.csv"):
        """
        Full pipeline: load data, predict, convert format, save.

        Args:
            test_data_path: Path to test featurization CSV
            output_path: Path to save submission CSV
        """
        print("="*70)
        print("GENERATING SUBMISSION")
        print("="*70)

        # Load test data
        df = self.load_test_data(test_data_path)

        # Extract features
        features = self.extract_features(df)

        # Predict residuals
        predicted_residuals = self.predict_residuals(features)

        # Apply corrections
        corrected_df = self.apply_corrections(df, predicted_residuals)

        # Convert to submission format
        submission = self.convert_to_submission_format(corrected_df)

        # Save submission
        print(f"\nSaving submission to: {output_path}")
        submission.to_csv(output_path, index=False)
        print("✓ Submission saved successfully!")

        # Show sample
        print("\nSubmission sample (first 10 rows):")
        print(submission.head(10).to_string(index=False))

        print("\n" + "="*70)
        print("SUBMISSION GENERATION COMPLETE")
        print("="*70)

        return submission


def main():
    parser = argparse.ArgumentParser(description='Generate competition submission from trained model')
    parser.add_argument('--model', required=True, help='Path to trained model (.pkl)')
    parser.add_argument('--test_data', required=True, help='Path to test featurization CSV')
    parser.add_argument('--output', default='submission.csv', help='Output submission file path')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    if not Path(args.test_data).exists():
        raise FileNotFoundError(f"Test data file not found: {args.test_data}")

    # Generate submission
    generator = SubmissionGenerator(args.model)
    submission = generator.generate(args.test_data, args.output)

    print(f"\n✓ Done! Submission saved to: {args.output}")
    print(f"  Total predictions: {len(submission):,}")
    print(f"  Ready to submit to Kaggle!")


if __name__ == "__main__":
    main()
