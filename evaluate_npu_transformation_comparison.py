#!/usr/bin/env python3
"""
NPU Sigmoid 출력을 Linear 변환으로 재평가

기존: Sigmoid → Bounded Inverse Depth (853mm @ 15m)
새로운: Sigmoid → Direct Linear Depth (28mm @ 15m)

목적: INT8 양자화에서 어떤 변환이 더 나은지 실증적으로 비교
"""

import numpy as np
import json
import torch
from pathlib import Path
from PIL import Image
import sys

# Import from packnet_sfm utilities
sys.path.insert(0, '/workspace/packnet-sfm')
from packnet_sfm.utils.post_process_depth import sigmoid_to_depth_linear, sigmoid_to_depth_direct


def load_gt_depth(new_filename, test_json_path):
    """GT depth 로드"""
    with open(test_json_path, 'r') as f:
        data = json.load(f)
    
    for entry in data:
        if entry['new_filename'] == new_filename:
            dataset_root = entry['dataset_root']
            depth_path = Path(dataset_root) / 'newest_depth_maps' / f'{new_filename}.png'
            
            if not depth_path.exists():
                raise FileNotFoundError(f"GT depth not found: {depth_path}")
            
            depth_img = Image.open(depth_path)
            depth = np.array(depth_img, dtype=np.float32) / 256.0
            
            return depth
    
    raise ValueError(f"new_filename {new_filename} not found in {test_json_path}")


def compute_depth_metrics(gt, pred, min_depth=0.5, max_depth=15.0):
    """Depth metrics 계산 (NO GT SCALE)"""
    valid_mask = (gt > min_depth) & (gt < max_depth)
    
    if valid_mask.sum() == 0:
        return None
    
    gt_valid = gt[valid_mask]
    pred_valid = pred[valid_mask]
    
    # Metrics
    thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    
    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    sq_rel = np.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    
    rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt_valid) - np.log(pred_valid)) ** 2))
    
    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3,
        'valid_pixels': int(valid_mask.sum())
    }


def evaluate_transformation_comparison():
    """
    NPU sigmoid 출력을 2가지 변환으로 평가:
    1. Bounded Inverse (기존): sigmoid → inv_depth → depth
    2. Direct Linear (새로운): sigmoid → depth (linear)
    """
    # Configuration
    npu_output_dir = Path('outputs/raw_sigmoid_npu_outputs')  # 91 images
    test_json = '/workspace/data/ncdb-cls-640x384/splits/combined_test.json'
    min_depth = 0.5
    max_depth = 15.0
    
    npu_files = sorted(npu_output_dir.glob('*.npy'))
    npu_files = [f for f in npu_files if f.stem != 'evaluation_results']
    
    print("="*80)
    print("NPU Sigmoid → Depth 변환 방법 비교")
    print("="*80)
    print(f"Depth range: [{min_depth}, {max_depth}]m")
    print(f"NPU files: {len(npu_files)}")
    print()
    
    print("🔍 변환 방법:")
    print("   1️⃣  Bounded Inverse (기존):")
    print("       sigmoid → inv_depth → depth")
    print("       INT8 error @ 15m: ~853mm ❌")
    print()
    print("   2️⃣  Direct Linear (새로운):")
    print("       sigmoid → depth (linear)")
    print("       INT8 error @ 15m: ~28mm ✅")
    print()
    print("="*80)
    print()
    
    # Results storage
    results_bounded_inverse = []
    results_direct_linear = []
    
    for npu_file in npu_files:
        new_filename = npu_file.stem
        
        # Load NPU sigmoid
        sigmoid = np.load(npu_file)
        while sigmoid.ndim > 2:
            sigmoid = sigmoid.squeeze(0)
        
        # Convert to torch for transformation functions
        sigmoid_torch = torch.from_numpy(sigmoid).float()
        
        # Method 1: Bounded Inverse (기존 방법)
        # sigmoid_to_depth_linear()은 사실상 Bounded Inverse!
        depth_bounded_inverse = sigmoid_to_depth_linear(
            sigmoid_torch,
            min_depth=min_depth,
            max_depth=max_depth
        ).numpy()
        
        # Method 2: Direct Linear (새로운 방법)
        # ✅ True linear transformation
        depth_direct_linear = sigmoid_to_depth_direct(
            sigmoid_torch, 
            min_depth=min_depth, 
            max_depth=max_depth
        ).numpy()
        
        # Load GT
        try:
            gt_depth = load_gt_depth(new_filename, test_json)
        except Exception as e:
            print(f"⚠️  SKIP: {new_filename} - {e}")
            continue
        
        # Compute metrics
        metrics_bounded = compute_depth_metrics(gt_depth, depth_bounded_inverse, min_depth, max_depth)
        metrics_linear = compute_depth_metrics(gt_depth, depth_direct_linear, min_depth, max_depth)
        
        if metrics_bounded is None or metrics_linear is None:
            print(f"⚠️  SKIP: {new_filename} - No valid pixels")
            continue
        
        results_bounded_inverse.append(metrics_bounded)
        results_direct_linear.append(metrics_linear)
        
        print(f"✅ {new_filename}")
        print(f"   Bounded Inverse → abs_rel: {metrics_bounded['abs_rel']:.4f}, "
              f"rmse: {metrics_bounded['rmse']:.4f}m, δ<1.25: {metrics_bounded['a1']:.4f}")
        print(f"   Direct Linear   → abs_rel: {metrics_linear['abs_rel']:.4f}, "
              f"rmse: {metrics_linear['rmse']:.4f}m, δ<1.25: {metrics_linear['a1']:.4f}")
        
        # Improvement
        improvement_abs_rel = (metrics_bounded['abs_rel'] - metrics_linear['abs_rel']) / metrics_bounded['abs_rel'] * 100
        print(f"   Improvement: abs_rel {improvement_abs_rel:+.1f}%")
        print()
    
    if not results_bounded_inverse:
        print("❌ No valid results")
        return
    
    # Average metrics
    print("="*80)
    print("📊 AVERAGE METRICS COMPARISON")
    print("="*80)
    print()
    
    metric_names = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
    
    print(f"{'Metric':<12} {'Bounded Inverse':<18} {'Direct Linear':<18} {'Improvement':<15}")
    print("-"*80)
    
    for key in metric_names:
        avg_bounded = np.mean([m[key] for m in results_bounded_inverse])
        avg_linear = np.mean([m[key] for m in results_direct_linear])
        
        # Calculate improvement
        if key in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log']:
            # Lower is better
            improvement = (avg_bounded - avg_linear) / avg_bounded * 100
            arrow = "✅" if improvement > 0 else "❌"
        else:
            # Higher is better (a1, a2, a3)
            improvement = (avg_linear - avg_bounded) / avg_bounded * 100
            arrow = "✅" if improvement > 0 else "❌"
        
        print(f"{key:<12} {avg_bounded:<18.4f} {avg_linear:<18.4f} {improvement:+6.1f}% {arrow}")
    
    print()
    print("="*80)
    print("🎯 KEY FINDINGS")
    print("="*80)
    print()
    
    avg_abs_rel_bounded = np.mean([m['abs_rel'] for m in results_bounded_inverse])
    avg_abs_rel_linear = np.mean([m['abs_rel'] for m in results_direct_linear])
    improvement_pct = (avg_abs_rel_bounded - avg_abs_rel_linear) / avg_abs_rel_bounded * 100
    
    print(f"Bounded Inverse (기존):")
    print(f"   abs_rel: {avg_abs_rel_bounded:.4f}")
    print(f"   INT8 error @ 15m: ~853mm")
    print()
    print(f"Direct Linear (새로운):")
    print(f"   abs_rel: {avg_abs_rel_linear:.4f}")
    print(f"   INT8 error @ 15m: ~28mm")
    print()
    print(f"Overall Improvement: {improvement_pct:+.1f}%")
    print()
    
    if improvement_pct > 0:
        print("✅ Direct Linear 변환이 더 우수합니다!")
        print("   → NPU INT8 양자화에 Linear 변환 사용 추천")
    else:
        print("⚠️  Bounded Inverse가 더 나은 결과를 보입니다.")
        print("   → 추가 분석 필요")
    
    print()
    print("="*80)
    print(f"Total evaluated: {len(results_bounded_inverse)} images")
    print("="*80)
    
    # Save results
    output_dir = Path('outputs/transformation_comparison')
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'bounded_inverse': {
            'metrics': {k: float(np.mean([m[k] for m in results_bounded_inverse])) for k in metric_names},
            'method': 'sigmoid → inv_depth → depth',
            'int8_error_15m': '853mm'
        },
        'direct_linear': {
            'metrics': {k: float(np.mean([m[k] for m in results_direct_linear])) for k in metric_names},
            'method': 'sigmoid → depth (linear)',
            'int8_error_15m': '28mm'
        },
        'improvement_pct': {k: float((
            (np.mean([m[k] for m in results_bounded_inverse]) - 
             np.mean([m[k] for m in results_direct_linear])) /
            np.mean([m[k] for m in results_bounded_inverse]) * 100
        )) for k in metric_names}
    }
    
    with open(output_dir / 'transformation_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_dir / 'transformation_comparison.json'}")


if __name__ == '__main__':
    evaluate_transformation_comparison()
