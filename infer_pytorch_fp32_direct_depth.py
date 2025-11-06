"""
PyTorch Direct Depth Inference using checkpoint

checkpoints/resnetsan01_direct_depth_05_15_640x384/epoch_29.ckpt 사용
"""

import numpy as np
import torch
from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm
import sys

# Add packnet_sfm to path
sys.path.insert(0, str(Path(__file__).parent))

from packnet_sfm.models.SfmModel import SfmModel

def load_model_from_checkpoint(ckpt_path, device):
    """Load model from checkpoint"""
    print(f"📂 Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Clean state dict (remove 'model.' prefix if exists)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            cleaned_state_dict[k[6:]] = v
        else:
            cleaned_state_dict[k] = v
    
    # Create model
    print("🔧 Creating SfmModel...")
    model = SfmModel()
    
    # Load state dict
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model

def load_image(image_path, device):
    """Load and preprocess image for PyTorch model"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((640, 384), Image.BILINEAR)
    
    # Convert to tensor
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # To tensor: HWC -> CHW -> BCHW
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    return img_tensor

def main():
    print("=" * 80)
    print("🚀 PyTorch Direct Depth Inference (from checkpoint)")
    print("=" * 80)
    
    # Paths
    ckpt_path = 'checkpoints/resnetsan_direct_depth_05_15.ckpt'
    test_json_path = '/workspace/data/ncdb-cls-640x384/splits/combined_test.json'
    output_dir = Path('outputs/pytorch_fp32_direct_depth_inference')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    
    # Load model
    model = load_model_from_checkpoint(ckpt_path, device)
    
    print(f"   Model type: {type(model).__name__}")
    print(f"   Depth output mode: {getattr(model.depth_net, 'depth_output_mode', 'unknown')}")
    
    # Load test JSON
    print(f"\n📂 Loading test split from: {test_json_path}")
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    print(f"   Total test samples: {len(test_data)}")
    
    # Run inference
    print(f"\n🔄 Running inference on {len(test_data)} images...")
    
    with torch.no_grad():
        for i, entry in enumerate(tqdm(test_data, desc="PyTorch Inference")):
            new_filename = entry['new_filename']
            image_path = entry['image_path']
            
            # Load and preprocess image
            img_tensor = load_image(image_path, device)
            
            # Create batch dict
            batch = {'rgb': img_tensor}
            
            # Run inference
            output = model(batch)
            
            # Extract depth prediction
            # Output format depends on model, try different keys
            if isinstance(output, dict):
                if 'depth' in output:
                    depth_pred = output['depth'][0, 0].cpu().numpy()  # [B, 1, H, W] -> [H, W]
                elif 'inv_depths' in output:
                    # If inverse depth, convert to depth
                    inv_depth = output['inv_depths'][0][0, 0].cpu().numpy()
                    depth_pred = 1.0 / (inv_depth + 1e-6)
                else:
                    print(f"\n⚠️  Unknown output format. Keys: {output.keys()}")
                    depth_pred = list(output.values())[0][0, 0].cpu().numpy()
            else:
                depth_pred = output[0, 0].cpu().numpy()
            
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
    
    print("\n" + "=" * 80)
    print("✅ Done!")
    print("=" * 80)

if __name__ == '__main__':
    main()
