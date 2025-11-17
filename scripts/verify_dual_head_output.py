"""
Dual-Head 모델의 Integer + Fractional 합성 결과 검증 스크립트

목적:
1. Dual-Head 체크포인트를 로드
2. Integer head와 Fractional head 출력을 각각 저장
3. 두 출력을 합쳐서 최종 depth 계산
4. 합성된 depth로 평가하여 epoch_28_results.json과 동일한 결과가 나오는지 확인

사용법:
python scripts/verify_dual_head_output.py \
    --checkpoint checkpoints/resnetsan01_dual_head_ncdb_640x384/.../epoch=28_....ckpt \
    --config configs/train_resnet_san_ncdb_dual_head_640x384.yaml \
    --output_dir outputs/dual_head_verification \
    --num_samples 20
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from collections import OrderedDict

from packnet_sfm.utils.config import parse_train_file
from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.utils.load import set_debug
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.depth import depth2inv, inv2depth, compute_depth_metrics
from packnet_sfm.networks.layers.resnet.layers import dual_head_to_depth


def parse_args():
    parser = argparse.ArgumentParser(description='Verify Dual-Head output composition')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Dual-Head checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='outputs/dual_head_verification',
                        help='Output directory for verification results')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to verify')
    return parser.parse_args()


def load_model(checkpoint_path, config_path):
    """체크포인트와 설정에서 모델 로드"""
    print(f"\n{'='*80}")
    print("📦 Loading Dual-Head Model")
    print(f"{'='*80}")
    
    # Parse config
    from packnet_sfm.utils.config import parse_train_file
    config, _ = parse_train_file(config_path)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model wrapper
    model_wrapper = ModelWrapper(config)
    model_wrapper.load_state_dict(ckpt['state_dict'], strict=False)
    model_wrapper.eval()
    model_wrapper.cuda()
    
    print(f"✅ Model loaded from: {checkpoint_path}")
    print(f"   - Min depth: {config.model.params.min_depth}m")
    print(f"   - Max depth: {config.model.params.max_depth}m")
    
    # Check if model is Dual-Head
    is_dual_head = hasattr(model_wrapper.model.depth_net, 'is_dual_head') and \
                   model_wrapper.model.depth_net.is_dual_head
    print(f"   - Dual-Head: {is_dual_head}")
    
    if not is_dual_head:
        raise ValueError("❌ Model is not Dual-Head!")
    
    return model_wrapper, config


def prepare_batch(image_path, depth_path, config):
    """이미지와 depth를 배치로 준비"""
    # Load image
    rgb = load_image(image_path)
    
    # Load depth if available
    depth_gt = None
    if depth_path and os.path.exists(depth_path):
        depth_gt = np.load(depth_path)['velodyne_depth']
    
    # Resize
    image_shape = config.datasets.augmentation.image_shape
    rgb = resize_image(rgb, image_shape, interpolation='bilinear')
    if depth_gt is not None:
        depth_gt = resize_image(depth_gt, image_shape, interpolation='nearest')
    
    # To tensor
    rgb = to_tensor(rgb).unsqueeze(0)
    if depth_gt is not None:
        depth_gt = torch.from_numpy(depth_gt).unsqueeze(0).unsqueeze(0).float()
    
    # Create batch
    batch = {
        'rgb': rgb.cuda(),
        'rgb_original': rgb.cuda(),
    }
    if depth_gt is not None:
        batch['depth'] = depth_gt.cuda()
    
    return batch


def extract_dual_head_outputs(model_wrapper, batch):
    """모델에서 Dual-Head 출력 추출"""
    with torch.no_grad():
        # Forward pass
        output = model_wrapper.model.depth_net(batch)
        
        # Extract integer and fractional heads
        if ('integer', 0) in output and ('fractional', 0) in output:
            integer_sigmoid = output[('integer', 0)]  # [B, 1, H, W], sigmoid [0, 1]
            fractional_sigmoid = output[('fractional', 0)]  # [B, 1, H, W], sigmoid [0, 1]
            
            return {
                'integer_sigmoid': integer_sigmoid,
                'fractional_sigmoid': fractional_sigmoid,
                'has_dual_head': True
            }
        else:
            raise ValueError("❌ Model output doesn't have Dual-Head keys!")


def compute_composed_depth(integer_sigmoid, fractional_sigmoid, max_depth):
    """Integer와 Fractional을 합성하여 최종 depth 계산"""
    # Use dual_head_to_depth helper
    depth_composed = dual_head_to_depth(integer_sigmoid, fractional_sigmoid, max_depth)
    return depth_composed


def save_outputs(output_dir, sample_idx, integer_sigmoid, fractional_sigmoid, depth_composed, depth_gt=None):
    """출력 결과 저장"""
    sample_dir = Path(output_dir) / f"sample_{sample_idx:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Save integer head (as depth for visualization)
    integer_depth = integer_sigmoid.squeeze().cpu().numpy() * 15.0  # Assuming max_depth = 15
    np.save(sample_dir / "integer_depth.npy", integer_depth)
    
    # Save fractional head (as depth for visualization)
    fractional_depth = fractional_sigmoid.squeeze().cpu().numpy()  # [0, 1]m range
    np.save(sample_dir / "fractional_depth.npy", fractional_depth)
    
    # Save composed depth
    depth_composed_np = depth_composed.squeeze().cpu().numpy()
    np.save(sample_dir / "depth_composed.npy", depth_composed_np)
    
    # Save GT if available
    if depth_gt is not None:
        depth_gt_np = depth_gt.squeeze().cpu().numpy()
        np.save(sample_dir / "depth_gt.npy", depth_gt_np)
    
    return {
        'integer_depth': integer_depth,
        'fractional_depth': fractional_depth,
        'depth_composed': depth_composed_np,
        'depth_gt': depth_gt_np if depth_gt is not None else None
    }


def evaluate_composed_depth(depth_composed, depth_gt, config):
    """합성된 depth로 메트릭 계산"""
    if depth_gt is None:
        return None
    
    try:
        # Compute metrics without GT scale
        metrics_no_scale = compute_depth_metrics(
            config.model.params,
            gt=depth_gt,
            pred=depth_composed,
            use_gt_scale=False
        )
        
        # Compute metrics with GT scale
        metrics_with_scale = compute_depth_metrics(
            config.model.params,
            gt=depth_gt,
            pred=depth_composed,
            use_gt_scale=True
        )
        
        return {
            'no_scale': metrics_no_scale,
            'with_scale': metrics_with_scale
        }
    except Exception as e:
        print(f"   ⚠️ Error computing metrics: {e}")
        return None


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model_wrapper, config = load_model(args.checkpoint, args.config)
    max_depth = float(config.model.params.max_depth)
    
    # Get validation dataset
    print(f"\n{'='*80}")
    print("📊 Loading Validation Dataset")
    print(f"{'='*80}")
    
    # Use model_wrapper's validation dataloader
    val_dataset = model_wrapper.val_dataset[0]  # First validation dataset
    
    print(f"   Dataset: {config.datasets.validation.path[0]}")
    print(f"   Split: {config.datasets.validation.split[0]}")
    print(f"   Total samples: {len(val_dataset)}")
    print(f"   Verifying: {min(args.num_samples, len(val_dataset))} samples")
    
    # Process samples
    print(f"\n{'='*80}")
    print("🔍 Verifying Dual-Head Composition")
    print(f"{'='*80}")
    
    all_metrics = []
    
    for idx in tqdm(range(min(args.num_samples, len(val_dataset))), desc="Processing"):
        try:
            # Get sample
            sample = val_dataset[idx]
            
            # Prepare batch
            batch = {
                'rgb': sample['rgb'].unsqueeze(0).cuda(),
                'rgb_original': sample['rgb'].unsqueeze(0).cuda(),
            }
            if 'depth' in sample:
                batch['depth'] = sample['depth'].unsqueeze(0).cuda()
            
            # Extract Dual-Head outputs
            outputs = extract_dual_head_outputs(model_wrapper, batch)
            
            # Compute composed depth
            depth_composed = compute_composed_depth(
                outputs['integer_sigmoid'],
                outputs['fractional_sigmoid'],
                max_depth
            )
            
            # Save outputs
            depth_gt = batch.get('depth', None)
            saved_data = save_outputs(
                output_dir,
                idx,
                outputs['integer_sigmoid'],
                outputs['fractional_sigmoid'],
                depth_composed,
                depth_gt
            )
            
            # Evaluate
            metrics = evaluate_composed_depth(depth_composed, depth_gt, config)
            if metrics:
                all_metrics.append(metrics)
        
        except Exception as e:
            print(f"\n   ⚠️ Error processing sample {idx}: {e}")
            continue
    
    # Aggregate metrics
    print(f"\n{'='*80}")
    print("📈 Aggregated Metrics")
    print(f"{'='*80}")
    
    if all_metrics:
        # Average metrics
        avg_metrics_no_scale = {}
        avg_metrics_with_scale = {}
        
        for metric_name in ['abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']:
            values_no_scale = [m['no_scale'][metric_name] for m in all_metrics if m['no_scale'] is not None]
            values_with_scale = [m['with_scale'][metric_name] for m in all_metrics if m['with_scale'] is not None]
            
            if values_no_scale:
                avg_metrics_no_scale[metric_name] = np.mean(values_no_scale)
            if values_with_scale:
                avg_metrics_with_scale[metric_name] = np.mean(values_with_scale)
        
        print("\n📊 Without GT Scale:")
        for k, v in avg_metrics_no_scale.items():
            print(f"   {k:12s}: {v:.6f}")
        
        print("\n📊 With GT Scale:")
        for k, v in avg_metrics_with_scale.items():
            print(f"   {k:12s}: {v:.6f}")
        
        # Save results
        results = {
            'checkpoint': args.checkpoint,
            'num_samples': len(all_metrics),
            'metrics_no_scale': avg_metrics_no_scale,
            'metrics_with_scale': avg_metrics_with_scale
        }
        
        results_path = output_dir / "verification_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {results_path}")
        
        # Compare with epoch_28_results.json
        epoch_results_path = Path(args.checkpoint).parent / "evaluation_results" / "epoch_28_results.json"
        if epoch_results_path.exists():
            with open(epoch_results_path, 'r') as f:
                epoch_results = json.load(f)
            
            print(f"\n{'='*80}")
            print("🔍 Comparison with Epoch 28 Results")
            print(f"{'='*80}")
            
            # Compare abs_rel_lin_gt (most important metric)
            epoch_abs_rel = epoch_results.get('ncdb-cls-640x384-combined_val-abs_rel_lin_gt', None)
            our_abs_rel = avg_metrics_with_scale.get('abs_rel', None)
            
            if epoch_abs_rel and our_abs_rel:
                print(f"\n📊 abs_rel (with GT scale):")
                print(f"   Epoch 28 result: {epoch_abs_rel:.6f}")
                print(f"   Our verification: {our_abs_rel:.6f}")
                print(f"   Difference:      {abs(epoch_abs_rel - our_abs_rel):.6f}")
                
                if abs(epoch_abs_rel - our_abs_rel) < 0.01:
                    print(f"   ✅ MATCH! (within 1% tolerance)")
                else:
                    print(f"   ⚠️ MISMATCH! (difference > 1%)")
    
    print(f"\n{'='*80}")
    print("✅ Verification Complete")
    print(f"{'='*80}")
    print(f"   Output directory: {output_dir}")
    print(f"   Samples processed: {len(all_metrics)}")


if __name__ == '__main__':
    main()
