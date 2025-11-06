#!/usr/bin/env python3
"""Verify Direct Depth ONNX model output range and behavior"""

import onnxruntime as ort
import numpy as np
import onnx

def verify_onnx_model(onnx_path):
    """Verify ONNX model structure and output"""
    
    print(f"🔍 Verifying ONNX model: {onnx_path}")
    print("=" * 80)
    
    # 1. Load and inspect ONNX model
    model = onnx.load(onnx_path)
    
    print("\n📊 Model Structure:")
    print(f"   IR version: {model.ir_version}")
    print(f"   Producer: {model.producer_name} {model.producer_version}")
    print(f"   Opset version: {model.opset_import[0].version}")
    
    # Input/Output info
    print("\n📥 Input:")
    for input_tensor in model.graph.input:
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"   Name: {input_tensor.name}")
        print(f"   Shape: {shape}")
        print(f"   Type: {input_tensor.type.tensor_type.elem_type}")
    
    print("\n📤 Output:")
    for output_tensor in model.graph.output:
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"   Name: {output_tensor.name}")
        print(f"   Shape: {shape}")
        print(f"   Type: {output_tensor.type.tensor_type.elem_type}")
    
    # 2. Create ONNX Runtime session
    print("\n🚀 Running inference test...")
    session = ort.InferenceSession(onnx_path)
    
    # Create dummy input
    dummy_input = np.random.randn(1, 3, 384, 640).astype(np.float32)
    
    # Run inference
    output = session.run(None, {"rgb": dummy_input})[0]
    
    print(f"\n✅ Inference successful!")
    print(f"   Output shape: {output.shape}")
    print(f"   Output dtype: {output.dtype}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]m")
    print(f"   Output mean: {output.mean():.3f}m")
    print(f"   Output std: {output.std():.3f}m")
    
    # 3. Verify depth range
    min_depth = 0.5
    max_depth = 15.0
    
    print(f"\n🎯 Expected range: [{min_depth}, {max_depth}]m")
    
    if output.min() >= min_depth and output.max() <= max_depth:
        print(f"   ✅ Output within expected range!")
    else:
        print(f"   ⚠️ Output outside expected range!")
        if output.min() < min_depth:
            print(f"      Min value {output.min():.3f} < {min_depth}")
        if output.max() > max_depth:
            print(f"      Max value {output.max():.3f} > {max_depth}")
    
    # 4. INT8 quantization parameters
    print(f"\n📊 INT8 Quantization Parameters:")
    scale = (max_depth - min_depth) / 255
    zero_point = -int(min_depth / scale)
    
    print(f"   Scale: {scale:.6f}")
    print(f"   Zero point: {zero_point}")
    print(f"   Quantization error: ±{scale/2 * 1000:.1f}mm")
    
    # 5. Quantize and dequantize test
    quantized = np.clip(np.round(output / scale) + zero_point, 0, 255).astype(np.uint8)
    dequantized = (quantized.astype(np.float32) - zero_point) * scale
    
    quantization_error = np.abs(output - dequantized)
    
    print(f"\n🔢 Quantization Test (INT8):")
    print(f"   Original: [{output.min():.3f}, {output.max():.3f}]m")
    print(f"   Quantized range: [{quantized.min()}, {quantized.max()}] (uint8)")
    print(f"   Dequantized: [{dequantized.min():.3f}, {dequantized.max():.3f}]m")
    print(f"   Quantization error:")
    print(f"      Mean: {quantization_error.mean() * 1000:.2f}mm")
    print(f"      Max: {quantization_error.max() * 1000:.2f}mm")
    print(f"      Std: {quantization_error.std() * 1000:.2f}mm")
    
    # 6. Comparison with Bounded Inverse (theoretical)
    print(f"\n📊 Comparison with Bounded Inverse:")
    print(f"   Direct Linear INT8 error: ±{scale/2 * 1000:.1f}mm (uniform)")
    print(f"   Bounded Inverse INT8 error @ 15m: ~853mm")
    print(f"   Improvement: ~{853 / (scale/2 * 1000):.0f}x better at far range!")
    
    print("\n" + "=" * 80)
    print("✅ ONNX model verification complete!")

if __name__ == "__main__":
    onnx_path = "onnx/resnetsan_direct_depth_640x384.onnx"
    verify_onnx_model(onnx_path)
