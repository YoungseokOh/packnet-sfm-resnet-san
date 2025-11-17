"""
Dual-Head 모델의 Integer와 Fractional Head 출력을 개별적으로 저장하는 스크립트

목적:
1. Dual-Head 체크포인트를 로드
2. Test 또는 Validation set에 대해 inference 수행
3. Integer head, Fractional head, 합성 depth를 각각 NPY/NPZ로 저장

사용법:
python scripts/save_dual_head_outputs.py \
    --checkpoint checkpoints/.../epoch=28_....ckpt \
    --output_dir outputs/dual_head_outputs_npy \
    --split test \
    --num_samples 91
    --save_separate_dirs --model_name resnetsan01_fp32 --precision fp32
    --save_separate_dirs --model_name resnetsan01_fp32 --precision fp32
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from packnet_sfm.utils.config import parse_train_file
from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.networks.layers.resnet.layers import dual_head_to_depth


def parse_args():
    parser = argparse.ArgumentParser(description='Save Dual-Head outputs separately')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Dual-Head checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs/dual_head_outputs_npy',
                        help='Output directory for NPY files')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to process')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to process (default: all)')
    parser.add_argument('--save_format', type=str, default='npz', choices=['npy', 'npz'],
                        help='Save format: npy (separate files) or npz (single file per sample)')
    parser.add_argument('--save_separate_dirs', action='store_true', help='Save integer/fractional into separate folders per model (integer_fp32, fractional_fp32)')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'int8'], help='Precision tag to use for folder names (default: fp32)')
    parser.add_argument('--model_name', type=str, default=None, help='Model name used in output folder naming (defaults to checkpoint base name)')
    return parser.parse_args()


def load_model(checkpoint_path):
    """체크포인트에서 모델 로드"""
    print(f"\n{'='*80}")
    print("📦 Loading Dual-Head Model")
    print(f"{'='*80}")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config from checkpoint
    config_path = ckpt.get('config_path', None)
    if config_path is None:
        # Try to infer from checkpoint path
        ckpt_dir = Path(checkpoint_path).parent
        config_name = ckpt_dir.parent.name
        if 'dual_head' in config_name:
            config_path = 'configs/train_resnet_san_ncdb_dual_head_640x384.yaml'
        else:
            raise ValueError("Cannot infer config path from checkpoint")
    
    # Parse config
    config, _ = parse_train_file(config_path)
    
    # Create model wrapper
    model_wrapper = ModelWrapper(config)
    model_wrapper.load_state_dict(ckpt['state_dict'], strict=False)
    model_wrapper.eval()
    model_wrapper.cuda()
    
    print(f"✅ Model loaded from: {checkpoint_path}")
    print(f"   - Config: {config_path}")
    print(f"   - Min depth: {config.model.params.min_depth}m")
    print(f"   - Max depth: {config.model.params.max_depth}m")
    
    # Check if model is Dual-Head
    is_dual_head = hasattr(model_wrapper.model.depth_net, 'is_dual_head') and \
                   model_wrapper.model.depth_net.is_dual_head
    print(f"   - Dual-Head: {is_dual_head}")
    
    if not is_dual_head:
        raise ValueError("❌ Model is not Dual-Head!")
    
    return model_wrapper, config


def get_dataset(model_wrapper, split):
    """데이터셋 가져오기"""
    if split == 'train':
        return model_wrapper.train_dataset[0] if hasattr(model_wrapper, 'train_dataset') else None
    elif split == 'val':
        return model_wrapper.val_dataset[0] if hasattr(model_wrapper, 'val_dataset') else None
    elif split == 'test':
        return model_wrapper.test_dataset[0] if hasattr(model_wrapper, 'test_dataset') else None
    else:
        raise ValueError(f"Unknown split: {split}")


def save_dual_head_outputs_npy(output_dir, sample_idx, filename, 
                                integer_sig, fractional_sig, depth_composed,
                                intrinsics=None):
    """NPY 형식으로 개별 파일 저장"""
    sample_dir = Path(output_dir) / filename.replace('.npz', '')
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each component
    np.save(sample_dir / "integer_sigmoid.npy", integer_sig)
    np.save(sample_dir / "fractional_sigmoid.npy", fractional_sig)
    np.save(sample_dir / "depth_composed.npy", depth_composed)
    
    if intrinsics is not None:
        np.save(sample_dir / "intrinsics.npy", intrinsics)


def save_dual_head_outputs_npz(output_dir, sample_idx, filename,
                                integer_sig, fractional_sig, depth_composed,
                                intrinsics=None):
    """NPZ 형식으로 단일 파일에 모두 저장"""
    output_path = Path(output_dir) / filename
    
    save_dict = {
        'integer_sigmoid': integer_sig,
        'fractional_sigmoid': fractional_sig,
        'depth_composed': depth_composed,
    }
    
    if intrinsics is not None:
        save_dict['intrinsics'] = intrinsics
    
    np.savez_compressed(output_path, **save_dict)


def save_dual_head_outputs_to_dirs(output_dir, model_name, precision, filename, integer_sig, fractional_sig, depth_composed):
    """Save integer/fractional as separate NPY files under per-model directories:
    output_dir/model_name/integer_precision/<filename>.npy and fractional_precision
    """
    base_dir = Path(output_dir) / (model_name if model_name else 'model')
    integer_dir = base_dir / f"integer_{precision}"
    fractional_dir = base_dir / f"fractional_{precision}"
    depth_dir = base_dir / f"depth_{precision}"
    integer_dir.mkdir(parents=True, exist_ok=True)
    fractional_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Use filename without extension
    fname = Path(filename).stem
    np.save(integer_dir / f"{fname}.npy", integer_sig)
    np.save(fractional_dir / f"{fname}.npy", fractional_sig)
    # Store depth composed as well for convenience
    np.save(depth_dir / f"{fname}.npy", depth_composed)


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model_wrapper, config = load_model(args.checkpoint)
    max_depth = float(config.model.params.max_depth)
    
    # Get dataset
    print(f"\n{'='*80}")
    print(f"📊 Loading {args.split.upper()} Dataset")
    print(f"{'='*80}")
    
    dataset = get_dataset(model_wrapper, args.split)
    if dataset is None:
        raise ValueError(f"Cannot load {args.split} dataset")
    
    print(f"   Total samples: {len(dataset)}")
    num_samples = args.num_samples if args.num_samples else len(dataset)
    num_samples = min(num_samples, len(dataset))
    print(f"   Processing: {num_samples} samples")
    print(f"   Save format: {args.save_format.upper()}")
    
    # Process samples
    print(f"\n{'='*80}")
    print("💾 Saving Dual-Head Outputs")
    print(f"{'='*80}")
    
    save_func = save_dual_head_outputs_npz if args.save_format == 'npz' else save_dual_head_outputs_npy
    
    for idx in tqdm(range(num_samples), desc="Processing"):
        try:
            # Get sample
            sample = dataset[idx]
            
            # Prepare batch
            batch = {
                'rgb': sample['rgb'].unsqueeze(0).cuda(),
                'rgb_original': sample['rgb'].unsqueeze(0).cuda(),
            }
            
            # Get filename from sample
            if 'filename' in sample:
                filename = sample['filename']
                if isinstance(filename, list):
                    filename = filename[0]
            else:
                filename = f"{idx:010d}"
            
            # Forward pass
            with torch.no_grad():
                # depth_net.forward() expects rgb tensor, not batch dict
                rgb = batch['rgb']
                outputs = model_wrapper.model.depth_net(rgb)
            
            # Extract Dual-Head outputs
            if ('integer', 0) not in outputs or ('fractional', 0) not in outputs:
                print(f"   ⚠️ Sample {idx}: Missing Dual-Head outputs")
                continue
            
            integer_sigmoid = outputs[('integer', 0)][0, 0].cpu().numpy()  # [H, W]
            fractional_sigmoid = outputs[('fractional', 0)][0, 0].cpu().numpy()  # [H, W]
            
            # Compute composed depth
            depth_composed = dual_head_to_depth(
                outputs[('integer', 0)],
                outputs[('fractional', 0)],
                max_depth
            )[0, 0].cpu().numpy()  # [H, W]
            
            # Get intrinsics if available
            intrinsics = None
            if 'intrinsics' in sample:
                intrinsics = sample['intrinsics'].cpu().numpy()
            
            # Save in the configured format
            stem = Path(filename).stem
            out_filename = f"{stem}_dual_head.npz" if args.save_format == 'npz' else f"{stem}"
            save_func(
                output_dir,
                idx,
                out_filename,
                integer_sigmoid,
                fractional_sigmoid,
                depth_composed,
                intrinsics
            )

            # Additionally save to separate per-model directories if requested
            if args.save_separate_dirs:
                # Determine model_name default from checkpoint path if not provided
                model_name = args.model_name
                if model_name is None:
                    # derive model_name from checkpoint parent directory or file stem
                    model_name = Path(args.checkpoint).stem
                save_dual_head_outputs_to_dirs(output_dir, model_name, args.precision, stem, integer_sigmoid, fractional_sigmoid, depth_composed)
        
        except Exception as e:
            import traceback
            print(f"\n   ⚠️ Error processing sample {idx}:")
            print(f"      {type(e).__name__}: {e}")
            if idx < 3:  # Print full traceback for first 3 errors
                traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("✅ Saving Complete")
    print(f"{'='*80}")
    print(f"   Output directory: {output_dir}")
    print(f"   Samples saved: {num_samples}")
    print(f"   Format: {args.save_format.upper()}")
    
    # Print example of how to load
    print(f"\n{'='*80}")
    print("📖 How to Load Saved Outputs")
    print(f"{'='*80}")
    
    if args.save_format == 'npz':
        print("""
import numpy as np

# Load NPZ file
data = np.load('path/to/file_dual_head.npz')

# Access components
integer_sigmoid = data['integer_sigmoid']    # [H, W], range [0, 1]
fractional_sigmoid = data['fractional_sigmoid']  # [H, W], range [0, 1]
depth_composed = data['depth_composed']      # [H, W], in meters
intrinsics = data['intrinsics']              # Camera intrinsics (if available)

# Verify reconstruction
depth_manual = integer_sigmoid * max_depth + fractional_sigmoid
print(f"Max diff: {np.abs(depth_composed - depth_manual).max()}")  # Should be ~0
""")
    else:
        print("""
import numpy as np

# Load NPY files
integer_sigmoid = np.load('path/to/sample_dir/integer_sigmoid.npy')
fractional_sigmoid = np.load('path/to/sample_dir/fractional_sigmoid.npy')
depth_composed = np.load('path/to/sample_dir/depth_composed.npy')
intrinsics = np.load('path/to/sample_dir/intrinsics.npy')
""")
        if args.save_separate_dirs:
            model_name = args.model_name if args.model_name is not None else Path(args.checkpoint).stem
            print(f"\n📁 Separate directories created under: {output_dir}/{model_name}/")
            print(f"   - Integer: {output_dir}/{model_name}/integer_{args.precision}/")
            print(f"   - Fractional: {output_dir}/{model_name}/fractional_{args.precision}/")
            print(f"   - Depth: {output_dir}/{model_name}/depth_{args.precision}/")


if __name__ == '__main__':
    main()
