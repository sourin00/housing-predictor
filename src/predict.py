"""
Prediction script for container verification
Loads the saved model and runs predictions on test set
"""

import numpy as np
import joblib
import os
from sklearn.metrics import r2_score, mean_squared_error


def load_model_and_data():
    """Load the saved model and test data"""
    print("Loading saved model...")

    model_path = "models/linear_model.joblib"
    test_data_path = "models/test_data.joblib"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")

    model = joblib.load(model_path)
    X_test, y_test = joblib.load(test_data_path)

    print(f"Model loaded successfully")
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

    return model, X_test, y_test


def run_predictions(model, X_test, y_test):
    """Run predictions and calculate metrics"""
    print("Running predictions...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Test R² Score: {r2:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    # Print some sample predictions
    print("\nSample predictions:")
    print("Actual\tPredicted")
    for i in range(5):
        print(f"{y_test[i]:.2f}\t{y_pred[i]:.2f}")

    return r2, rmse


def main():
    """Main prediction pipeline"""
    print("=== Container Verification - Prediction Script ===")

    try:
        # Load model and data
        model, X_test, y_test = load_model_and_data()

        # Run predictions
        r2, rmse = run_predictions(model, X_test, y_test)

        # Verification criteria
        if r2 > 0.5:  # Reasonable R² threshold
            print(f"\nContainer verification PASSED (R² = {r2:.4f})")
            exit_code = 0
        else:
            print(f"\nContainer verification FAILED (R² = {r2:.4f} < 0.5)")
            exit_code = 1

    except Exception as e:
        print(f"\nContainer verification FAILED with error: {str(e)}")
        exit_code = 1

    print("=== Verification Complete ===")
    exit(exit_code)


if __name__ == "__main__":
    main()