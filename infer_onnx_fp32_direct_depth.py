"""
ONNX FP32 Direct Depth Inference

onnx/resnetsan_direct_depth_640x384.onnx 모델로 테스트 세트 inference
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm

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

def main():
    print("=" * 80)
    print("🚀 ONNX FP32 Direct Depth Inference")
    print("=" * 80)
    
    # Paths
    onnx_model_path = 'onnx/resnetsan_direct_depth_640x384.onnx'
    test_json_path = '/workspace/data/ncdb-cls-640x384/splits/combined_test.json'
    output_dir = Path('outputs/onnx_fp32_direct_depth_inference')
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
    print(f"   Available providers: {available_providers}")
    
    # Use CUDA if available, else CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
    print(f"   Using providers: {providers}")
    
    session = ort.InferenceSession(onnx_model_path, providers=providers)
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"   Input name: {input_name}")
    print(f"   Output name: {output_name}")
    print(f"   Input shape: {session.get_inputs()[0].shape}")
    print(f"   Output shape: {session.get_outputs()[0].shape}")
    
    # Run inference on all test images
    print(f"\n🔄 Running inference on {len(test_data)} images...")
    
    for i, entry in enumerate(tqdm(test_data, desc="Inference")):
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
        
        if (i + 1) % 20 == 0:
            print(f"   Processed {i + 1}/{len(test_data)} images...")
    
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
    
    print("\n" + "=" * 80)
    print("✅ Done!")
    print("=" * 80)

if __name__ == '__main__':
    main()
