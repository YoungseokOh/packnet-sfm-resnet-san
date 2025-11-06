"""
ONNX INT8 Quantization and Inference

ONNX Runtime의 quantization API를 사용하여 INT8 모델 생성 및 inference
"""

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm
import onnx

def load_image(image_path):
    """Load and preprocess image for ONNX model"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((640, 384), Image.BILINEAR)
    
    # Convert to array and normalize
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # ImageNet normalization (same as training)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # Transpose to CHW and add batch dimension
    img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dim
    
    return img_array

def quantize_onnx_model(input_model_path, output_model_path):
    """Quantize ONNX model to INT8"""
    print("\n" + "=" * 80)
    print("🔧 Quantizing ONNX model to INT8")
    print("=" * 80)
    
    print(f"\n📂 Input model: {input_model_path}")
    print(f"📂 Output model: {output_model_path}")
    
    # Get original model size
    input_size = Path(input_model_path).stat().st_size / (1024 * 1024)
    print(f"\n📊 Original model size: {input_size:.2f} MB")
    
    # Quantize using dynamic quantization
    print("\n🔄 Running dynamic quantization...")
    quantize_dynamic(
        input_model_path,
        output_model_path,
        weight_type=QuantType.QInt8
    )
    
    # Get quantized model size
    output_size = Path(output_model_path).stat().st_size / (1024 * 1024)
    print(f"✅ Quantized model size: {output_size:.2f} MB")
    print(f"📉 Compression ratio: {input_size / output_size:.2f}x")
    
    return output_model_path

def run_inference(onnx_model_path, test_json_path, output_dir, model_type='FP32'):
    """Run inference with ONNX model"""
    print("\n" + "=" * 80)
    print(f"🚀 ONNX {model_type} Direct Depth Inference")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test JSON
    print(f"\n📂 Loading test split from: {test_json_path}")
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    print(f"   Total test samples: {len(test_data)}")
    
    # Initialize ONNX Runtime session
    print(f"\n🔧 Loading ONNX model: {onnx_model_path}")
    
    # Check available providers
    available_providers = ort.get_available_providers()
    
    # Use CUDA if available, else CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
    print(f"   Using providers: {providers}")
    
    session = ort.InferenceSession(str(onnx_model_path), providers=providers)
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"   Input name: {input_name}")
    print(f"   Output name: {output_name}")
    
    # Run inference on all test images
    print(f"\n🔄 Running inference on {len(test_data)} images...")
    
    for i, entry in enumerate(tqdm(test_data, desc=f"{model_type} Inference")):
        new_filename = entry['new_filename']
        image_path = entry['image_path']
        
        # Load and preprocess image
        img_array = load_image(image_path)
        
        # Run inference
        outputs = session.run([output_name], {input_name: img_array})
        depth_pred = outputs[0][0, 0]  # Remove batch and channel dims
        
        # Save prediction
        output_file = output_dir / f'{new_filename}.npy'
        np.save(output_file, depth_pred)
    
    print(f"\n✅ Inference complete!")
    print(f"   Output directory: {output_dir}")
    print(f"   Total files: {len(list(output_dir.glob('*.npy')))}")
    
    # Summary
    sample_file = list(output_dir.glob('*.npy'))[0]
    sample_depth = np.load(sample_file)
    print(f"\n📊 Sample output stats:")
    print(f"   Shape: {sample_depth.shape}")
    print(f"   Min depth: {sample_depth.min():.3f}m")
    print(f"   Max depth: {sample_depth.max():.3f}m")
    print(f"   Mean depth: {sample_depth.mean():.3f}m")

def main():
    # Paths
    fp32_model_path = 'onnx/resnetsan_direct_depth_640x384.onnx'
    int8_model_path = 'onnx/resnetsan_direct_depth_640x384_int8.onnx'
    test_json_path = '/workspace/data/ncdb-cls-640x384/splits/combined_test.json'
    
    fp32_output_dir = 'outputs/onnx_fp32_direct_depth_inference'
    int8_output_dir = 'outputs/onnx_int8_direct_depth_inference'
    
    # Step 1: Quantize model to INT8
    if not Path(int8_model_path).exists():
        quantize_onnx_model(fp32_model_path, int8_model_path)
    else:
        print(f"\n✅ INT8 model already exists: {int8_model_path}")
        int8_size = Path(int8_model_path).stat().st_size / (1024 * 1024)
        fp32_size = Path(fp32_model_path).stat().st_size / (1024 * 1024)
        print(f"   FP32 size: {fp32_size:.2f} MB")
        print(f"   INT8 size: {int8_size:.2f} MB")
        print(f"   Compression: {fp32_size / int8_size:.2f}x")
    
    # Step 2: Run FP32 inference
    if not Path(fp32_output_dir).exists() or len(list(Path(fp32_output_dir).glob('*.npy'))) == 0:
        run_inference(fp32_model_path, test_json_path, fp32_output_dir, model_type='FP32')
    else:
        print(f"\n✅ FP32 inference results already exist: {fp32_output_dir}")
        print(f"   Total files: {len(list(Path(fp32_output_dir).glob('*.npy')))}")
    
    # Step 3: Run INT8 inference
    if not Path(int8_output_dir).exists() or len(list(Path(int8_output_dir).glob('*.npy'))) == 0:
        run_inference(int8_model_path, test_json_path, int8_output_dir, model_type='INT8')
    else:
        print(f"\n✅ INT8 inference results already exist: {int8_output_dir}")
        print(f"   Total files: {len(list(Path(int8_output_dir).glob('*.npy')))}")
    
    print("\n" + "=" * 80)
    print("✅ All tasks complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()
