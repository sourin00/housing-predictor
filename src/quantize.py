"""
Manual quantization script for model optimization
Converts sklearn model parameters to quantized PyTorch format
"""

import numpy as np
import joblib
import torch
import os
from sklearn.metrics import r2_score, mean_squared_error


def load_sklearn_model():
    """Load the trained sklearn model"""
    print("Loading sklearn model...")

    model_path = "models/linear_model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    print(f"Model loaded: {type(model)}")
    print(f"Coefficients shape: {model.coef_.shape}")
    print(f"Intercept: {model.intercept_}")

    return model


def extract_parameters(model):
    """Extract parameters from sklearn model"""
    print("Extracting model parameters...")

    # Get weights and bias
    weights = model.coef_  # Shape: (n_features,)
    bias = model.intercept_  # Scalar

    print(f"Weights shape: {weights.shape}")
    print(f"Bias value: {bias}")
    print(f"Weights range: [{np.min(weights):.4f}, {np.max(weights):.4f}]")

    return weights, bias


def manual_quantization_uint8(params, param_name):
    """Manually quantize parameters to uint8"""
    print(f"\nQuantizing {param_name}...")

    # Find min and max values
    min_val = float(np.min(params))
    max_val = float(np.max(params))

    print(f"Original range: [{min_val:.6f}, {max_val:.6f}]")

    # Calculate scale and zero_point for uint8 quantization
    # uint8 range: [0, 255]
    scale = (max_val - min_val) / 255.0
    zero_point = int(np.round(-min_val / scale))

    # Clamp zero_point to valid range
    zero_point = max(0, min(255, zero_point))

    print(f"Scale: {scale:.8f}")
    print(f"Zero point: {zero_point}")

    # Quantize
    quantized = np.round(params / scale + zero_point).astype(np.uint8)

    print(f"Quantized range: [{np.min(quantized)}, {np.max(quantized)}]")

    # Verify quantization
    dequantized = (quantized.astype(np.float32) - zero_point) * scale
    quantization_error = np.mean(np.abs(params - dequantized))
    print(f"Quantization error (MAE): {quantization_error:.8f}")

    return {
        'quantized_data': quantized,
        'scale': scale,
        'zero_point': zero_point,
        'shape': params.shape,
        'dtype': 'uint8'
    }


def dequantize_params(quant_info):
    """Dequantize parameters back to float32"""
    quantized = quant_info['quantized_data']
    scale = quant_info['scale']
    zero_point = quant_info['zero_point']

    # Dequantize
    dequantized = (quantized.astype(np.float32) - zero_point) * scale

    return dequantized


def create_pytorch_model(weights, bias):
    """Create a simple PyTorch linear layer"""
    print("Creating PyTorch model...")

    # Create a simple linear layer
    input_dim = weights.shape[0]
    model = torch.nn.Linear(input_dim, 1, bias=True)

    # Set the weights and bias
    with torch.no_grad():
        model.weight.data = torch.tensor(weights, dtype=torch.float32).reshape(1, -1)
        model.bias.data = torch.tensor([bias], dtype=torch.float32)

    return model


def evaluate_models(original_weights, original_bias, quant_weights, quant_bias):
    """Evaluate both original and quantized models"""
    print("Evaluating models...")

    # Load test data
    X_test, y_test = joblib.load("models/test_data.joblib")

    # Original model predictions
    original_pred = X_test @ original_weights + original_bias
    original_r2 = r2_score(y_test, original_pred)

    # Quantized model predictions
    quant_pred = X_test @ quant_weights + quant_bias
    quant_r2 = r2_score(y_test, quant_pred)

    print(f"Original model R² Score: {original_r2:.6f}")
    print(f"Quantized model R² Score: {quant_r2:.6f}")
    print(f"R² Score difference: {abs(original_r2 - quant_r2):.6f}")

    return original_r2, quant_r2


def get_file_size_kb(filepath):
    """Get file size in KB"""
    if os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)
        size_kb = size_bytes / 1024
        return size_kb
    return 0


def save_parameters_and_analysis():
    """Main quantization pipeline"""
    print("=== Model Quantization Pipeline ===")

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Load sklearn model
    model = load_sklearn_model()

    # Extract parameters
    weights, bias = extract_parameters(model)

    # Save original (unquantized) parameters
    unquant_params = {
        'weights': weights,
        'bias': bias,
        'dtype': 'float64',
        'description': 'Original sklearn LinearRegression parameters'
    }

    unquant_path = "models/unquant_params.joblib"
    joblib.dump(unquant_params, unquant_path)
    print(f"\nUnquantized parameters saved to: {unquant_path}")

    # Quantize weights and bias
    quant_weights_info = manual_quantization_uint8(weights, "weights")
    quant_bias_info = manual_quantization_uint8(np.array([bias]), "bias")

    # Save quantized parameters
    quant_params = {
        'weights': quant_weights_info,
        'bias': quant_bias_info,
        'description': 'Quantized (uint8) parameters with scale and zero_point'
    }

    quant_path = "models/quant_params.joblib"
    joblib.dump(quant_params, quant_path)
    print(f"\nQuantized parameters saved to: {quant_path}")

    # Dequantize for evaluation
    dequant_weights = dequantize_params(quant_weights_info)
    dequant_bias = dequantize_params(quant_bias_info)[0]

    # Evaluate models
    print("\n" + "=" * 50)
    original_r2, quant_r2 = evaluate_models(weights, bias, dequant_weights, dequant_bias)

    # Calculate file sizes
    unquant_size = get_file_size_kb(unquant_path)
    quant_size = get_file_size_kb(quant_path)

    print("\n" + "=" * 50)
    print("FINAL COMPARISON TABLE")
    print("=" * 50)
    print(f"{'Metric':<20} {'Original Model':<20} {'Quantized Model':<20}")
    print("-" * 60)
    print(f"{'R² Score':<20} {original_r2:<20.6f} {quant_r2:<20.6f}")
    print(f"{'Model Size (KB)':<20} {unquant_size:<20.2f} {quant_size:<20.2f}")
    print(f"{'Compression Ratio':<20} {'':<20} {(unquant_size / quant_size):<20.2f}x")
    print("=" * 50)

    # Save comparison results
    comparison_results = {
        'original_r2': original_r2,
        'quantized_r2': quant_r2,
        'original_size_kb': unquant_size,
        'quantized_size_kb': quant_size,
        'compression_ratio': unquant_size / quant_size if quant_size > 0 else 0
    }

    joblib.dump(comparison_results, "models/comparison_results.joblib")
    print(f"\nComparison results saved to: models/comparison_results.joblib")

    return comparison_results


def main():
    """Main function"""
    try:
        results = save_parameters_and_analysis()
        print("\nQuantization pipeline completed successfully!")
        return results
    except Exception as e:
        print(f"\nQuantization pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()