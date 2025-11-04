#!/usr/bin/env python3
"""
NPU, GPU(ONNX), GT 3개 비교
GT의 유효한 픽셀 영역만 사용
"""

import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image


def load_depth_png(depth_path):
    """PNG 깊이 맵 로드 (KITTI 스타일)"""
    depth_png = Image.open(depth_path)
    depth_arr = np.asarray(depth_png, dtype=np.uint16)
    depth = depth_arr.astype(np.float32)
    
    # KITTI 스타일: 256으로 나누기
    if depth.max() > 255:
        depth /= 256.0
    
    return depth


def sigmoid_to_depth(sigmoid, min_depth=0.05, max_depth=80.0):
    """Sigmoid [0, 1]을 depth로 변환"""
    min_inv = 1.0 / max_depth
    max_inv = 1.0 / min_depth
    
    inv_depth = min_inv + (max_inv - min_inv) * sigmoid
    depth = 1.0 / inv_depth
    
    return depth


def compute_metrics_with_mask(pred, gt, valid_mask):
    """유효한 픽셀에 대해서만 메트릭 계산"""
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    
    # 절대 오차
    abs_diff = np.abs(pred_valid - gt_valid)
    
    # 상대 오차
    rel_diff = abs_diff / (gt_valid + 1e-8)
    
    metrics = {
        'mae': abs_diff.mean(),
        'rmse': np.sqrt((abs_diff ** 2).mean()),
        'abs_rel': rel_diff.mean(),
        'sq_rel': ((abs_diff ** 2) / (gt_valid ** 2 + 1e-8)).mean(),
        'max_error': abs_diff.max(),
        'median_error': np.median(abs_diff),
        'valid_pixels': valid_mask.sum(),
        'total_pixels': valid_mask.size,
        'valid_ratio': valid_mask.sum() / valid_mask.size * 100,
    }
    
    # Delta thresholds
    for threshold in [1.25, 1.25**2, 1.25**3]:
        ratio = np.maximum(pred_valid / (gt_valid + 1e-8), gt_valid / (pred_valid + 1e-8))
        metrics[f'delta_{threshold:.3f}'] = (ratio < threshold).mean() * 100
    
    return metrics


def compare_three_sources(npu_folder, gpu_folder, gt_folder, output_folder, 
                         min_depth_npu=0.5, max_depth_npu=50.0,
                         min_depth_gpu=0.05, max_depth_gpu=80.0):
    """NPU, GPU, GT 3개 비교"""
    
    print("=" * 90)
    print("🔍 NPU vs GPU vs GT Comparison (Valid Pixels Only)")
    print("=" * 90)
    print(f"\n📁 Folders:")
    print(f"   NPU : {npu_folder}")
    print(f"   GPU : {gpu_folder}")
    print(f"   GT  : {gt_folder}")
    print(f"   Output: {output_folder}")
    print(f"\n⚙️  Depth Range:")
    print(f"   NPU (New): {min_depth_npu}m - {max_depth_npu}m")
    print(f"   GPU (ONNX): {min_depth_gpu}m - {max_depth_gpu}m\n")
    
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    
    # 공통 파일 찾기
    npu_files = set([f.stem for f in Path(npu_folder).glob('*.npy')])
    gpu_files = set([f.stem for f in Path(gpu_folder).glob('*.npy')])
    gt_files = set([f.stem for f in Path(gt_folder).glob('*.png')])
    
    common_files = sorted(npu_files & gpu_files & gt_files)
    
    print(f"📊 File Statistics:")
    print(f"   • NPU files: {len(npu_files)}")
    print(f"   • GPU files: {len(gpu_files)}")
    print(f"   • GT files: {len(gt_files)}")
    print(f"   • Common files: {len(common_files)}\n")
    
    if len(common_files) == 0:
        print("❌ No common files found!")
        return
    
    print(f"📋 Common files to compare:")
    for i, filename in enumerate(common_files, 1):
        print(f"   {i:2d}. {filename}")
    print()
    
    # 비교 결과 저장
    npu_results = []
    gpu_results = []
    
    print("🔄 Comparing NPU, GPU, GT...")
    print("-" * 90)
    
    for filename in common_files:
        npu_path = os.path.join(npu_folder, f"{filename}.npy")
        gpu_path = os.path.join(gpu_folder, f"{filename}.npy")
        gt_path = os.path.join(gt_folder, f"{filename}.png")
        
        # 데이터 로드
        npu_data = np.load(npu_path)
        gpu_data = np.load(gpu_path)
        gt_data = load_depth_png(gt_path)
        
        # NPU/GPU 데이터 squeeze
        if npu_data.ndim == 3 and npu_data.shape[0] == 1:
            npu_data = npu_data.squeeze(0)
        if gpu_data.ndim == 3 and gpu_data.shape[0] == 1:
            gpu_data = gpu_data.squeeze(0)
        
        # NPU가 이미 depth인지 확인
        if npu_data.max() > 1.0:
            npu_depth = npu_data
        else:
            npu_depth = sigmoid_to_depth(npu_data, min_depth_npu, max_depth_npu)
        
        # GPU sigmoid → depth
        gpu_depth = sigmoid_to_depth(gpu_data, min_depth_gpu, max_depth_gpu)
        
        # GT 유효 마스크 생성 (depth > 0)
        valid_mask = gt_data > 0
        
        # NPU vs GT
        npu_metrics = compute_metrics_with_mask(npu_depth, gt_data, valid_mask)
        npu_metrics['filename'] = filename
        npu_metrics['npu_depth'] = npu_depth
        npu_metrics['gt_depth'] = gt_data
        npu_metrics['valid_mask'] = valid_mask
        npu_results.append(npu_metrics)
        
        # GPU vs GT
        gpu_metrics = compute_metrics_with_mask(gpu_depth, gt_data, valid_mask)
        gpu_metrics['filename'] = filename
        gpu_metrics['gpu_depth'] = gpu_depth
        gpu_metrics['gt_depth'] = gt_data
        gpu_metrics['valid_mask'] = valid_mask
        gpu_results.append(gpu_metrics)
        
        print(f"✓ {filename}")
        print(f"  Valid pixels: {valid_mask.sum()} / {valid_mask.size} ({npu_metrics['valid_ratio']:.1f}%)")
        print(f"  NPU vs GT : MAE={npu_metrics['mae']:6.2f}m, RMSE={npu_metrics['rmse']:6.2f}m, AbsRel={npu_metrics['abs_rel']:.3f}")
        print(f"  GPU vs GT : MAE={gpu_metrics['mae']:6.2f}m, RMSE={gpu_metrics['rmse']:6.2f}m, AbsRel={gpu_metrics['abs_rel']:.3f}")
        print()
    
    print("-" * 90)
    
    # 전체 통계
    print_overall_statistics(npu_results, gpu_results)
    
    # 시각화
    print("\n📈 Generating comparison plots...")
    visualize_three_way_comparison(npu_results, gpu_results, output_folder)
    
    # 리포트 저장
    save_three_way_report(npu_results, gpu_results, output_folder)
    
    print("=" * 90)
    print("✅ Three-way comparison complete!")
    print(f"📁 Results saved to: {output_folder}")
    print("=" * 90)


def print_overall_statistics(npu_results, gpu_results):
    """전체 통계 출력"""
    
    print(f"\n📊 Overall Statistics ({len(npu_results)} files):")
    print("-" * 90)
    
    # NPU 통계
    npu_mae = [r['mae'] for r in npu_results]
    npu_rmse = [r['rmse'] for r in npu_results]
    npu_abs_rel = [r['abs_rel'] for r in npu_results]
    
    print(f"\n🔵 NPU vs GT:")
    print(f"   MAE     : {np.mean(npu_mae):6.2f}m ± {np.std(npu_mae):6.2f}m  (min={np.min(npu_mae):6.2f}m, max={np.max(npu_mae):6.2f}m)")
    print(f"   RMSE    : {np.mean(npu_rmse):6.2f}m ± {np.std(npu_rmse):6.2f}m  (min={np.min(npu_rmse):6.2f}m, max={np.max(npu_rmse):6.2f}m)")
    print(f"   Abs Rel : {np.mean(npu_abs_rel):.3f} ± {np.std(npu_abs_rel):.3f}")
    
    # GPU 통계
    gpu_mae = [r['mae'] for r in gpu_results]
    gpu_rmse = [r['rmse'] for r in gpu_results]
    gpu_abs_rel = [r['abs_rel'] for r in gpu_results]
    
    print(f"\n🟢 GPU vs GT:")
    print(f"   MAE     : {np.mean(gpu_mae):6.2f}m ± {np.std(gpu_mae):6.2f}m  (min={np.min(gpu_mae):6.2f}m, max={np.max(gpu_mae):6.2f}m)")
    print(f"   RMSE    : {np.mean(gpu_rmse):6.2f}m ± {np.std(gpu_rmse):6.2f}m  (min={np.min(gpu_rmse):6.2f}m, max={np.max(gpu_rmse):6.2f}m)")
    print(f"   Abs Rel : {np.mean(gpu_abs_rel):.3f} ± {np.std(gpu_abs_rel):.3f}")
    
    # 비교
    print(f"\n⚖️  Comparison (NPU vs GPU):")
    print(f"   MAE diff     : {np.mean(npu_mae) - np.mean(gpu_mae):+6.2f}m")
    print(f"   RMSE diff    : {np.mean(npu_rmse) - np.mean(gpu_rmse):+6.2f}m")
    print(f"   Abs Rel diff : {np.mean(npu_abs_rel) - np.mean(gpu_abs_rel):+.3f}")
    
    # 평균 유효 픽셀
    avg_valid_ratio = np.mean([r['valid_ratio'] for r in npu_results])
    print(f"\n📏 Average valid pixel ratio: {avg_valid_ratio:.1f}%")


def visualize_three_way_comparison(npu_results, gpu_results, output_folder):
    """3-way 비교 시각화"""
    
    # 1. 전체 비교 그래프
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    filenames = [r['filename'] for r in npu_results]
    x_pos = np.arange(len(filenames))
    
    # MAE 비교
    npu_mae = [r['mae'] for r in npu_results]
    gpu_mae = [r['mae'] for r in gpu_results]
    
    axes[0, 0].bar(x_pos - 0.2, npu_mae, 0.4, label='NPU', alpha=0.8, color='blue')
    axes[0, 0].bar(x_pos + 0.2, gpu_mae, 0.4, label='GPU', alpha=0.8, color='green')
    axes[0, 0].set_title('Mean Absolute Error (MAE)', fontweight='bold')
    axes[0, 0].set_ylabel('MAE (m)')
    axes[0, 0].set_xlabel('File Index')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # RMSE 비교
    npu_rmse = [r['rmse'] for r in npu_results]
    gpu_rmse = [r['rmse'] for r in gpu_results]
    
    axes[0, 1].bar(x_pos - 0.2, npu_rmse, 0.4, label='NPU', alpha=0.8, color='blue')
    axes[0, 1].bar(x_pos + 0.2, gpu_rmse, 0.4, label='GPU', alpha=0.8, color='green')
    axes[0, 1].set_title('Root Mean Squared Error (RMSE)', fontweight='bold')
    axes[0, 1].set_ylabel('RMSE (m)')
    axes[0, 1].set_xlabel('File Index')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Abs Rel 비교
    npu_abs_rel = [r['abs_rel'] for r in npu_results]
    gpu_abs_rel = [r['abs_rel'] for r in gpu_results]
    
    axes[0, 2].bar(x_pos - 0.2, npu_abs_rel, 0.4, label='NPU', alpha=0.8, color='blue')
    axes[0, 2].bar(x_pos + 0.2, gpu_abs_rel, 0.4, label='GPU', alpha=0.8, color='green')
    axes[0, 2].set_title('Absolute Relative Error', fontweight='bold')
    axes[0, 2].set_ylabel('Abs Rel')
    axes[0, 2].set_xlabel('File Index')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # MAE 차이 (NPU - GPU)
    mae_diff = np.array(npu_mae) - np.array(gpu_mae)
    colors = ['red' if d > 0 else 'blue' for d in mae_diff]
    
    axes[1, 0].bar(x_pos, mae_diff, color=colors, alpha=0.7)
    axes[1, 0].axhline(0, color='black', linestyle='-', linewidth=0.8)
    axes[1, 0].set_title('MAE Difference (NPU - GPU)', fontweight='bold')
    axes[1, 0].set_ylabel('MAE Diff (m)')
    axes[1, 0].set_xlabel('File Index')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Valid pixel ratio
    valid_ratios = [r['valid_ratio'] for r in npu_results]
    
    axes[1, 1].bar(x_pos, valid_ratios, color='purple', alpha=0.7)
    axes[1, 1].set_title('Valid Pixel Ratio', fontweight='bold')
    axes[1, 1].set_ylabel('Valid Ratio (%)')
    axes[1, 1].set_xlabel('File Index')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Winner count
    npu_wins = sum(1 for i in range(len(npu_mae)) if npu_mae[i] < gpu_mae[i])
    gpu_wins = sum(1 for i in range(len(gpu_mae)) if gpu_mae[i] < npu_mae[i])
    
    axes[1, 2].bar(['NPU Better', 'GPU Better'], [npu_wins, gpu_wins], 
                   color=['blue', 'green'], alpha=0.7)
    axes[1, 2].set_title('Winner Count (Lower MAE)', fontweight='bold')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('NPU vs GPU vs GT Comparison (Valid Pixels Only)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_folder, 'three_way_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path}")
    plt.close()
    
    # 2. 개별 샘플 시각화 (MAE 기준 상위 3개)
    visualize_individual_samples(npu_results, gpu_results, output_folder)


def visualize_individual_samples(npu_results, gpu_results, output_folder):
    """개별 샘플 상세 비교"""
    
    # MAE 기준으로 정렬
    combined = list(zip(npu_results, gpu_results))
    combined_sorted = sorted(combined, key=lambda x: x[0]['mae'] + x[1]['mae'], reverse=True)
    
    # 상위 3개만
    for i, (npu_res, gpu_res) in enumerate(combined_sorted[:3]):
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        filename = npu_res['filename']
        npu_depth = npu_res['npu_depth']
        gpu_depth = gpu_res['gpu_depth']
        gt_depth = npu_res['gt_depth']
        valid_mask = npu_res['valid_mask']
        
        # GT
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(gt_depth, cmap='viridis', vmin=0, vmax=80)
        ax1.set_title(f'GT Depth\nValid: {valid_mask.sum()} pixels', fontsize=10)
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Depth (m)')
        
        # NPU
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(npu_depth, cmap='viridis', vmin=0, vmax=80)
        ax2.set_title(f'NPU Depth\nMAE={npu_res["mae"]:.2f}m', fontsize=10)
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Depth (m)')
        
        # GPU
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(gpu_depth, cmap='viridis', vmin=0, vmax=80)
        ax3.set_title(f'GPU Depth\nMAE={gpu_res["mae"]:.2f}m', fontsize=10)
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Depth (m)')
        
        # Valid mask
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(valid_mask, cmap='gray')
        ax4.set_title(f'Valid Mask\n{npu_res["valid_ratio"]:.1f}%', fontsize=10)
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        
        # NPU error
        npu_error = np.abs(npu_depth - gt_depth)
        npu_error[~valid_mask] = 0
        
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(npu_error, cmap='hot', vmin=0, vmax=10)
        ax5.set_title(f'NPU Error\nMax={npu_res["max_error"]:.2f}m', fontsize=10)
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, label='Error (m)')
        
        # GPU error
        gpu_error = np.abs(gpu_depth - gt_depth)
        gpu_error[~valid_mask] = 0
        
        ax6 = fig.add_subplot(gs[1, 2])
        im6 = ax6.imshow(gpu_error, cmap='hot', vmin=0, vmax=10)
        ax6.set_title(f'GPU Error\nMax={gpu_res["max_error"]:.2f}m', fontsize=10)
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04, label='Error (m)')
        
        # Error comparison
        error_diff = npu_error - gpu_error
        
        ax7 = fig.add_subplot(gs[1, 3])
        im7 = ax7.imshow(error_diff, cmap='RdBu_r', vmin=-5, vmax=5)
        ax7.set_title('Error Diff (NPU - GPU)\nRed=NPU worse', fontsize=10)
        ax7.axis('off')
        plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04, label='Error Diff (m)')
        
        plt.suptitle(f'Detailed Comparison: {filename}', fontsize=14, fontweight='bold')
        
        output_path = os.path.join(output_folder, f'detail_{filename}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {output_path}")
        plt.close()


def save_three_way_report(npu_results, gpu_results, output_folder):
    """리포트 저장"""
    report_path = os.path.join(output_folder, 'three_way_comparison_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("NPU vs GPU vs GT Comparison Report (Valid Pixels Only)\n")
        f.write("=" * 90 + "\n\n")
        
        f.write(f"Total files compared: {len(npu_results)}\n\n")
        
        # 전체 통계
        npu_mae = [r['mae'] for r in npu_results]
        npu_rmse = [r['rmse'] for r in npu_results]
        npu_abs_rel = [r['abs_rel'] for r in npu_results]
        
        gpu_mae = [r['mae'] for r in gpu_results]
        gpu_rmse = [r['rmse'] for r in gpu_results]
        gpu_abs_rel = [r['abs_rel'] for r in gpu_results]
        
        f.write("NPU vs GT:\n")
        f.write("-" * 90 + "\n")
        f.write(f"MAE     : {np.mean(npu_mae):6.2f}m ± {np.std(npu_mae):6.2f}m\n")
        f.write(f"RMSE    : {np.mean(npu_rmse):6.2f}m ± {np.std(npu_rmse):6.2f}m\n")
        f.write(f"Abs Rel : {np.mean(npu_abs_rel):.3f} ± {np.std(npu_abs_rel):.3f}\n\n")
        
        f.write("GPU vs GT:\n")
        f.write("-" * 90 + "\n")
        f.write(f"MAE     : {np.mean(gpu_mae):6.2f}m ± {np.std(gpu_mae):6.2f}m\n")
        f.write(f"RMSE    : {np.mean(gpu_rmse):6.2f}m ± {np.std(gpu_rmse):6.2f}m\n")
        f.write(f"Abs Rel : {np.mean(gpu_abs_rel):.3f} ± {np.std(gpu_abs_rel):.3f}\n\n")
        
        f.write("Individual Results:\n")
        f.write("-" * 90 + "\n")
        
        for npu_res, gpu_res in zip(npu_results, gpu_results):
            f.write(f"\n{npu_res['filename']}\n")
            f.write(f"  Valid pixels: {npu_res['valid_pixels']} / {npu_res['total_pixels']} ({npu_res['valid_ratio']:.1f}%)\n")
            f.write(f"  NPU vs GT: MAE={npu_res['mae']:6.2f}m, RMSE={npu_res['rmse']:6.2f}m, AbsRel={npu_res['abs_rel']:.3f}\n")
            f.write(f"  GPU vs GT: MAE={gpu_res['mae']:6.2f}m, RMSE={gpu_res['rmse']:6.2f}m, AbsRel={gpu_res['abs_rel']:.3f}\n")
    
    print(f"   ✓ Saved: {report_path}")


if __name__ == '__main__':
    # 새 결과 vs ONNX GPU 비교 (GT 기준)
    npu_folder = '/workspace/packnet-sfm/outputs/sigmoid_prediction_linear_05_100_e4_a6es'  # 새 결과
    gpu_folder = '/workspace/packnet-sfm/outputs/sigmoid_predictions_from_onnx_gpu'  # ONNX GPU
    gt_folder = '/workspace/packnet-sfm/outputs/sigmoid_prediction_GT'  # GT
    output_folder = '/workspace/packnet-sfm/outputs/new_vs_onnx_with_gt'
    
    # 각 결과의 올바른 깊이 범위 설정
    compare_three_sources(npu_folder, gpu_folder, gt_folder, output_folder,
                         min_depth_npu=0.5, max_depth_npu=50.0,    # 새 결과: 0.5~50m
                         min_depth_gpu=0.05, max_depth_gpu=80.0)   # ONNX GPU: 0.05~80m
