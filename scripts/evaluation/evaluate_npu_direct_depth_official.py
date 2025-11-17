#!/usr/bin/env python3
"""
NPU Direct Depth 결과 평가 (후처리 없이 바로 평가)

Direct Depth는 이미 Linear depth로 출력되므로 sigmoid_to_depth_linear() 변환 불필요
"""

import numpy as np
import json
import sys
from pathlib import Path
from PIL import Image


def load_gt_depth(new_filename, test_json_path):
    """GT depth 로드 (combined_test.json 기반)"""
    with open(test_json_path, 'r') as f:
        data = json.load(f)
    
    for entry in data:
        if entry['new_filename'] == new_filename:
            dataset_root = entry['dataset_root']
            depth_path = Path(dataset_root) / 'newest_depth_maps' / f'{new_filename}.png'
            
            if not depth_path.exists():
                raise FileNotFoundError(f"GT depth not found: {depth_path}")
            
            # PNG 16-bit 로드 → meters
            depth_img = Image.open(depth_path)
            depth = np.array(depth_img, dtype=np.float32) / 256.0
            
            return depth
    
    raise ValueError(f"new_filename {new_filename} not found in {test_json_path}")


def compute_depth_metrics(gt, pred, min_depth=0.5, max_depth=15.0):
    """
    공식 eval.py의 compute_depth_metrics() 재현
    
    Returns:
        dict: metrics (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)
    """
    # GT depth 유효 범위 마스크
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


def compute_depth_metrics_with_gt_scale(gt, pred, min_depth=0.5, max_depth=15.0):
    """
    공식 eval.py의 compute_depth_metrics_with_gt_scale() 재현
    
    GT median scaling 적용
    """
    # GT depth 유효 범위 마스크
    valid_mask = (gt > min_depth) & (gt < max_depth)
    
    if valid_mask.sum() == 0:
        return None
    
    gt_valid = gt[valid_mask]
    pred_valid = pred[valid_mask]
    
    # GT median scaling
    gt_median = np.median(gt_valid)
    pred_median = np.median(pred_valid)
    scale_factor = gt_median / pred_median
    
    pred_scaled = pred * scale_factor
    pred_valid_scaled = pred_scaled[valid_mask]
    
    # Metrics (scaled)
    thresh = np.maximum((gt_valid / pred_valid_scaled), (pred_valid_scaled / gt_valid))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    
    abs_rel = np.mean(np.abs(gt_valid - pred_valid_scaled) / gt_valid)
    sq_rel = np.mean(((gt_valid - pred_valid_scaled) ** 2) / gt_valid)
    
    rmse = np.sqrt(np.mean((gt_valid - pred_valid_scaled) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt_valid) - np.log(pred_valid_scaled)) ** 2))
    
    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3,
        'valid_pixels': int(valid_mask.sum()),
        'scale_factor': scale_factor
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate NPU Direct Depth results (no post-processing)')
    parser.add_argument('--npu_dir', type=str, default='outputs/resnetsan_direct_depth_05_15_640x384',
                       help='NPU Direct Depth output directory')
    parser.add_argument('--test_json', type=str, 
                       default='/workspace/data/ncdb-cls-640x384/splits/combined_test.json',
                       help='Test JSON path')
    parser.add_argument('--min_depth', type=float, default=0.5, help='Min depth')
    parser.add_argument('--max_depth', type=float, default=15.0, help='Max depth')
    args = parser.parse_args()
    
    # Configuration
    npu_output_dir = Path(args.npu_dir)
    test_json = args.test_json
    min_depth = args.min_depth
    max_depth = args.max_depth
    
    # NPU 출력 파일 자동 감지
    npu_files = sorted(npu_output_dir.glob('*.npy'))
    
    # evaluation_results.txt 제외
    npu_files = [f for f in npu_files if f.stem != 'evaluation_results']
    
    print("="*80)
    print("🚀 NPU Direct Depth 평가 (후처리 없음 - Direct Linear Output)")
    print("="*80)
    print(f"📁 NPU output dir: {npu_output_dir}")
    print(f"📊 Depth range: [{min_depth}, {max_depth}]m")
    print(f"📊 NPU output files: {len(npu_files)}")
    print()
    print("ℹ️  Direct Depth 모델은 이미 Linear depth를 출력하므로")
    print("   sigmoid_to_depth_linear() 변환 없이 바로 평가합니다!")
    print()
    
    all_metrics_no_scale = []
    all_metrics_with_scale = []
    
    for npu_file in npu_files:
        new_filename = npu_file.stem  # e.g., '0000000038'
        
        # Load NPU Direct Depth output (이미 Linear depth!)
        npu_depth = np.load(npu_file)
        
        # Shape normalization
        while npu_depth.ndim > 2:
            npu_depth = npu_depth.squeeze(0)
        
        # ⚠️ Direct Depth는 이미 [0.5, 15.0]m range이므로 변환 불필요!
        # (기존 sigmoid_to_depth_linear() 변환 제거)
        
        # Load GT depth
        try:
            gt_depth = load_gt_depth(new_filename, test_json)
        except Exception as e:
            print(f"⚠️  SKIP: {new_filename} - GT loading failed: {e}")
            continue
        
        # Compute metrics (NO GT SCALE)
        metrics_no_scale = compute_depth_metrics(gt_depth, npu_depth, min_depth, max_depth)
        
        # Compute metrics (WITH GT SCALE)
        metrics_with_scale = compute_depth_metrics_with_gt_scale(gt_depth, npu_depth, min_depth, max_depth)
        
        if metrics_no_scale is None or metrics_with_scale is None:
            print(f"⚠️  SKIP: {new_filename} - No valid pixels")
            continue
        
        all_metrics_no_scale.append(metrics_no_scale)
        all_metrics_with_scale.append(metrics_with_scale)
        
        print(f"✅ {new_filename}")
        print(f"   Depth range: [{npu_depth.min():.3f}, {npu_depth.max():.3f}]m")
        print(f"   NO SCALE   → abs_rel: {metrics_no_scale['abs_rel']:.4f}, "
              f"rmse: {metrics_no_scale['rmse']:.4f}m, δ<1.25: {metrics_no_scale['a1']:.4f}")
        print(f"   WITH SCALE → abs_rel: {metrics_with_scale['abs_rel']:.4f}, "
              f"rmse: {metrics_with_scale['rmse']:.4f}m, δ<1.25: {metrics_with_scale['a1']:.4f}, "
              f"scale: {metrics_with_scale['scale_factor']:.4f}x")
        print()
    
    if not all_metrics_no_scale:
        print("❌ No valid results")
        return
    
    # Average metrics
    print("="*80)
    print("📊 AVERAGE METRICS (NPU Direct Depth INT8)")
    print("="*80)
    
    print("\n🔹 NO GT SCALE:")
    for key in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']:
        avg_val = np.mean([m[key] for m in all_metrics_no_scale])
        print(f"   {key:12s}: {avg_val:.4f}")
    
    print("\n🔹 WITH GT SCALE:")
    for key in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']:
        avg_val = np.mean([m[key] for m in all_metrics_with_scale])
        print(f"   {key:12s}: {avg_val:.4f}")
    
    avg_scale = np.mean([m['scale_factor'] for m in all_metrics_with_scale])
    print(f"   {'scale_factor':12s}: {avg_scale:.4f}x")
    
    # Comparison
    print("\n" + "="*80)
    print("📊 COMPARISON WITH PREVIOUS RESULTS")
    print("="*80)
    print("Method                          | abs_rel | rmse   | δ<1.25 | Notes")
    print("-" * 80)
    
    avg_abs_rel_no_scale = np.mean([m['abs_rel'] for m in all_metrics_no_scale])
    avg_rmse_no_scale = np.mean([m['rmse'] for m in all_metrics_no_scale])
    avg_a1_no_scale = np.mean([m['a1'] for m in all_metrics_no_scale])
    
    avg_abs_rel_with_scale = np.mean([m['abs_rel'] for m in all_metrics_with_scale])
    avg_rmse_with_scale = np.mean([m['rmse'] for m in all_metrics_with_scale])
    avg_a1_with_scale = np.mean([m['a1'] for m in all_metrics_with_scale])
    
    print(f"NPU Direct Depth (NO SCALE)     | {avg_abs_rel_no_scale:.4f}  | {avg_rmse_no_scale:.3f}m | {avg_a1_no_scale:.4f} | NEW!")
    print(f"NPU Direct Depth (WITH SCALE)   | {avg_abs_rel_with_scale:.4f}  | {avg_rmse_with_scale:.3f}m | {avg_a1_with_scale:.4f} | NEW!")
    print(f"NPU Bounded Inv (WITH SCALE)    | 0.1140  | 5.200m | 0.7500 | Old (INT8)")
    print(f"PyTorch FP32 (Baseline)         | 0.0300  | 1.500m | 0.9850 | Reference")
    print("-" * 80)
    
    # Calculate improvement
    if avg_abs_rel_with_scale < 0.114:
        improvement = (0.114 - avg_abs_rel_with_scale) / 0.114 * 100
        print(f"\n🎯 Improvement over Bounded Inverse NPU:")
        print(f"   abs_rel: 0.1140 → {avg_abs_rel_with_scale:.4f}")
        print(f"   Improvement: {improvement:.1f}%")
        print(f"   INT8 quantization error: ±28.4mm (uniform)")
    
    print("\n" + "="*80)
    print(f"✅ Total evaluated: {len(all_metrics_no_scale)} images")
    print("="*80)


if __name__ == '__main__':
    main()
