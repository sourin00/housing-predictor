"""
Training script for California Housing dataset
Creates and saves a LinearRegression model
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os


def load_and_prepare_data():
    """Load and prepare California Housing dataset"""
    print("Loading California Housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    print(f"Dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features: {housing.feature_names}")

    return X, y


def train_model(X, y):
    """Train LinearRegression model"""
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training LinearRegression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Testing R² Score: {test_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Testing RMSE: {test_rmse:.4f}")

    return model, (X_test, y_test)


def save_model(model, test_data):
    """Save the trained model and test data"""
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save the model
    joblib.dump(model, "models/linear_model.joblib")
    print("Model saved as models/linear_model.joblib")

    # Save test data for prediction verification
    joblib.dump(test_data, "models/test_data.joblib")
    print("Test data saved as models/test_data.joblib")

    # Print model parameters for reference
    print(f"\nModel parameters:")
    print(f"Coefficients shape: {model.coef_.shape}")
    print(f"Intercept: {model.intercept_}")
    print(f"Coefficients: {model.coef_}")


def main():
    """Main training pipeline"""
    print("=== California Housing Model Training ===")

    # Load data
    X, y = load_and_prepare_data()

    # Train model
    model, test_data = train_model(X, y)

    # Save model
    save_model(model, test_data)

    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()