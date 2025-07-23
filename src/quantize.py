import numpy as np
import joblib
import os
from sklearn.metrics import r2_score, mean_squared_error

def load_sklearn_model():
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
    print("Extracting model parameters...")

    # Get weights and bias
    weights = model.coef_.astype(np.float32)  # Use float32 for fair comparison
    bias = np.float32(model.intercept_)  # Convert to float32

    print(f"Weights shape: {weights.shape}")
    print(f"Bias value: {bias}")
    print(f"Weights range: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
    print(f"Weights: {weights}")

    return weights, bias

def manual_quantization_uint8(params, param_name):
    print(f"\nQuantizing {param_name}...")

    # Ensure params is a numpy array
    params = np.array(params, dtype=np.float32)

    # Find min and max values
    min_val = float(np.min(params))
    max_val = float(np.max(params))

    print(f"Original range: [{min_val:.6f}, {max_val:.6f}]")
    print(f"Original values: {params}")

    # Handle edge case where min_val == max_val (constant value)
    if np.abs(max_val - min_val) < 1e-8:
        print(f"Warning: {param_name} has constant value, storing as-is")
        # For constants, just store the value directly (no quantization needed)
        return {
            'quantized_data': np.array([128], dtype=np.uint8),  # Arbitrary value
            'scale': np.float32(1.0),
            'zero_point': np.uint8(128),
            'min_val': np.float32(min_val),
            'max_val': np.float32(max_val),
            'is_constant': True,
            'original_value': np.float32(min_val)
        }


    # Map [min_val, max_val] to [0, 255]
    scale = np.float32((max_val - min_val) / 255.0)

    # Quantize: normalize to [0,255] range
    quantized = np.clip(
        np.round((params - min_val) / scale),
        0, 255
    ).astype(np.uint8)

    print(f"Scale: {scale:.8f}")
    print(f"Quantized values: {quantized}")
    print(f"Quantized range: [{np.min(quantized)}, {np.max(quantized)}]")

    # Verify dequantization immediately
    dequantized = (quantized.astype(np.float32) * scale) + min_val
    quantization_error = np.mean(np.abs(params - dequantized))
    print(f"Dequantized values: {dequantized}")
    print(f"Quantization error (MAE): {quantization_error:.8f}")

    return {
        'quantized_data': quantized,
        'scale': scale,
        'zero_point': np.uint8(0),  # Always 0 with our method
        'min_val': np.float32(min_val),
        'max_val': np.float32(max_val),
        'is_constant': False
    }

def dequantize_params_corrected(quant_info):
    if quant_info['is_constant']:
        # Return the original constant value
        original_value = quant_info['original_value']
        size = len(quant_info['quantized_data'])
        return np.full(size, original_value, dtype=np.float32)

    # Normal dequantization: map [0,255] back to [min_val, max_val]
    quantized = quant_info['quantized_data']
    scale = quant_info['scale']
    min_val = quant_info['min_val']

    dequantized = (quantized.astype(np.float32) * scale) + min_val
    return dequantized

def evaluate_models(original_weights, original_bias, quant_weights, quant_bias):
    print("\nEvaluating models...")

    # Load test data
    X_test, y_test = joblib.load("models/test_data.joblib")

    # Original model predictions
    original_pred = X_test @ original_weights + original_bias
    original_r2 = r2_score(y_test, original_pred)
    original_rmse = np.sqrt(mean_squared_error(y_test, original_pred))

    # Quantized model predictions
    quant_pred = X_test @ quant_weights + quant_bias
    quant_r2 = r2_score(y_test, quant_pred)
    quant_rmse = np.sqrt(mean_squared_error(y_test, quant_pred))

    print(f"Original model R² Score: {original_r2:.6f}")
    print(f"Original model RMSE: {original_rmse:.6f}")
    print(f"Quantized model R² Score: {quant_r2:.6f}")
    print(f"Quantized model RMSE: {quant_rmse:.6f}")
    print(f"R² Score difference: {abs(original_r2 - quant_r2):.6f}")
    print(f"RMSE difference: {abs(original_rmse - quant_rmse):.6f}")

    return original_r2, quant_r2

def get_file_size_kb(filepath):
    """Get file size in KB"""
    if os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)
        size_kb = size_bytes / 1024
        return size_kb
    return 0

def calculate_theoretical_sizes(weights, bias, quant_weights_info, quant_bias_info):
    """Calculate theoretical data sizes (without file overhead)"""
    # Original sizes
    orig_weights_bytes = weights.nbytes
    orig_bias_bytes = bias.nbytes if hasattr(bias, 'nbytes') else 4
    orig_total = orig_weights_bytes + orig_bias_bytes

    # Quantized sizes (essential data only)
    quant_weights_bytes = quant_weights_info['quantized_data'].nbytes
    quant_scale_bytes = 4  # float32
    quant_min_bytes = 4    # float32
    quant_bias_bytes = 4   # float32 (unchanged)

    quant_total = quant_weights_bytes + quant_scale_bytes + quant_min_bytes + quant_bias_bytes

    return orig_total, quant_total

def main():
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
        'description': 'Original sklearn LinearRegression parameters (float32)'
    }

    unquant_path = "models/unquant_params.joblib"
    joblib.dump(unquant_params, unquant_path)
    print(f"\nUnquantized parameters saved to: {unquant_path}")

    quant_weights_info = manual_quantization_uint8(weights, "weights")
    quant_bias_info = manual_quantization_uint8([bias], "bias")

    # Save quantized parameters
    quant_params = {
        'weights': quant_weights_info,
        'bias': quant_bias_info,
        'description': 'Quantized (uint8) parameters'
    }

    quant_path = "models/quant_params.joblib"
    joblib.dump(quant_params, quant_path)
    print(f"\nQuantized parameters saved to: {quant_path}")

    # Dequantize for evaluation
    dequant_weights = dequantize_params_corrected(quant_weights_info)
    dequant_bias = dequantize_params_corrected(quant_bias_info)[0]

    print(f"\nDequantized weights: {dequant_weights}")
    print(f"Dequantized bias: {dequant_bias}")
    print(f"Original weights: {weights}")
    print(f"Original bias: {bias}")

    # Evaluate models
    print("\n" + "="*50)
    original_r2, quant_r2 = evaluate_models(weights, bias, dequant_weights, dequant_bias)

    # Calculate file sizes
    unquant_size_kb = get_file_size_kb(unquant_path)
    quant_size_kb = get_file_size_kb(quant_path)

    # Calculate theoretical sizes
    theoretical_orig, theoretical_quant = calculate_theoretical_sizes(
        weights, bias, quant_weights_info, quant_bias_info
    )

    print("\n" + "="*60)
    print("FINAL COMPARISON TABLE")
    print("="*60)
    print(f"{'Metric':<25} {'Original Model':<20} {'Quantized Model':<20}")
    print("-" * 65)
    print(f"{'R² Score':<25} {original_r2:<20.6f} {quant_r2:<20.6f}")
    print(f"{'File Size (KB)':<25} {unquant_size_kb:<20.3f} {quant_size_kb:<20.3f}")
    print(f"{'Theoretical Size (bytes)':<25} {theoretical_orig:<20} {theoretical_quant:<20}")
    print(f"{'Theoretical Size (KB)':<25} {theoretical_orig/1024:<20.3f} {theoretical_quant/1024:<20.3f}")

    # Compression ratios
    file_compression = unquant_size_kb / quant_size_kb if quant_size_kb > 0 else 0
    theoretical_compression = theoretical_orig / theoretical_quant if theoretical_quant > 0 else 0

    print(f"{'File Compression Ratio':<25} {'':<20} {file_compression:<20.2f}x")
    print(f"{'Theoretical Compression':<25} {'':<20} {theoretical_compression:<20.2f}x")
    print("="*60)


    print("| Metric | Original Sklearn Model | Quantized Model |")
    print("|--------|------------------------|-----------------|")
    print(f"| R² Score | {original_r2:.6f} | {quant_r2:.6f} |")
    print(f"| Model Size | {theoretical_orig/1024:.3f} KB | {theoretical_quant/1024:.3f} KB |")

    # Summary
    print(f"\nSUMMARY:")
    print(f"Theoretical compression: {theoretical_compression:.2f}x ({(1-1/theoretical_compression)*100:.1f}% reduction)")
    print(f"R² preserved within: {abs(original_r2 - quant_r2):.6f}")

    if file_compression > 1:
        print(f"File compression: {file_compression:.2f}x")
    else:
        print(f"File overhead dominated (tiny model size)")

    # Save comparison results
    comparison_results = {
        'original_r2': original_r2,
        'quantized_r2': quant_r2,
        'r2_difference': abs(original_r2 - quant_r2),
        'file_size_orig_kb': unquant_size_kb,
        'file_size_quant_kb': quant_size_kb,
        'file_compression_ratio': file_compression,
        'theoretical_size_orig_bytes': theoretical_orig,
        'theoretical_size_quant_bytes': theoretical_quant,
        'theoretical_compression_ratio': theoretical_compression
    }

    joblib.dump(comparison_results, "models/comparison_results.joblib")
    print(f"\nComparison results saved to: models/comparison_results.joblib")

    return comparison_results

if __name__ == "__main__":
    try:
        results = main()
        print("\nQuantization pipeline completed successfully!")
    except Exception as e:
        print(f"\nPipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()