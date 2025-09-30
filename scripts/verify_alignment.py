#!/usr/bin/env python3
"""RGB 이미지, GT depth, 마스크의 정렬 검증 스크립트"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from packnet_sfm.utils.depth import load_depth


def verify_sample(dataset_root: Path, stem: str, class_name: str = "car"):
    """단일 샘플의 RGB, GT, 마스크 정렬 확인"""
    
    print(f"\n{'='*80}")
    print(f"검증 샘플: {stem}")
    print(f"{'='*80}")
    
    # 1. 경로 찾기
    # RGB 이미지
    rgb_candidates = list(dataset_root.rglob(f"**/image_a6/{stem}.png"))
    if not rgb_candidates:
        print(f"❌ RGB 이미지를 찾을 수 없습니다: {stem}")
        return False
    
    rgb_path = rgb_candidates[0]
    sequence_root = rgb_path.parent.parent
    
    print(f"\nSequence root: {sequence_root}")
    print(f"RGB path: {rgb_path}")
    
    # GT depth
    gt_candidates = [
        sequence_root / "newest_depth_maps" / f"{stem}.png",
        sequence_root / "newest_depth_maps" / f"{stem}.npz",
    ]
    gt_path = next((p for p in gt_candidates if p.exists()), None)
    
    if not gt_path:
        print(f"❌ GT depth를 찾을 수 없습니다")
        return False
    
    print(f"GT path: {gt_path}")
    
    # 마스크
    mask_dir = sequence_root / "segmentation_results" / "class_masks" / class_name
    if not mask_dir.exists():
        print(f"❌ 마스크 디렉토리가 없습니다: {mask_dir}")
        return False
    
    mask_files = sorted(mask_dir.glob(f"{stem}*.png"))
    if not mask_files:
        print(f"❌ 마스크를 찾을 수 없습니다")
        return False
    
    print(f"마스크 개수: {len(mask_files)}")
    print(f"첫 마스크: {mask_files[0]}")
    
    # 2. 데이터 로드
    rgb_img = Image.open(rgb_path).convert("RGB")
    gt_depth = load_depth(str(gt_path))
    
    print(f"\n해상도:")
    print(f"  RGB: {rgb_img.size} (W x H)")
    print(f"  GT depth: {gt_depth.shape} (H x W)")
    
    # 3. 각 마스크 검증
    for i, mask_path in enumerate(mask_files[:3]):  # 처음 3개만
        print(f"\n--- 마스크 {i}: {mask_path.name} ---")
        
        mask_img = Image.open(mask_path).convert("L")
        mask_array = np.array(mask_img)
        
        print(f"  마스크 해상도: {mask_img.size} (W x H)")
        print(f"  마스크 픽셀 수: {(mask_array > 0).sum()}")
        print(f"  마스크 unique values: {np.unique(mask_array)}")
        
        # GT depth 해상도로 마스크 resize
        if mask_img.size != (gt_depth.shape[1], gt_depth.shape[0]):
            mask_resized = mask_img.resize((gt_depth.shape[1], gt_depth.shape[0]), Image.NEAREST)
            mask_array_resized = np.array(mask_resized)
            print(f"  마스크 resize 후: {mask_resized.size}, 픽셀 수: {(mask_array_resized > 0).sum()}")
        else:
            mask_array_resized = mask_array
        
        # GT depth에서 마스크 영역의 값 확인
        valid_mask = (mask_array_resized > 0) & (gt_depth > 0)
        if valid_mask.any():
            gt_in_mask = gt_depth[valid_mask]
            print(f"  GT depth in mask: min={gt_in_mask.min():.2f}, max={gt_in_mask.max():.2f}, mean={gt_in_mask.mean():.2f}")
            print(f"  유효 픽셀: {valid_mask.sum()}")
        else:
            print(f"  ❌ 마스크 영역에 유효한 GT가 없습니다!")
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # RGB
        axes[0, 0].imshow(rgb_img)
        axes[0, 0].set_title(f"RGB ({rgb_img.size[0]}x{rgb_img.size[1]})")
        axes[0, 0].axis('off')
        
        # GT depth
        gt_vis = gt_depth.copy()
        gt_vis[gt_vis == 0] = np.nan
        axes[0, 1].imshow(gt_vis, cmap='viridis')
        axes[0, 1].set_title(f"GT Depth ({gt_depth.shape[1]}x{gt_depth.shape[0]})")
        axes[0, 1].axis('off')
        
        # 마스크 (GT 해상도)
        axes[1, 0].imshow(mask_array_resized, cmap='gray')
        axes[1, 0].set_title(f"Mask resized ({mask_array_resized.shape[1]}x{mask_array_resized.shape[0]})")
        axes[1, 0].axis('off')
        
        # RGB에 마스크 오버레이 (해상도 맞춤)
        rgb_array = np.array(rgb_img)
        # 마스크를 RGB 해상도로 resize
        mask_for_rgb = np.array(mask_img.resize(rgb_img.size, Image.NEAREST))
        overlay = rgb_array.copy()
        overlay[mask_for_rgb > 0] = overlay[mask_for_rgb > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        
        axes[1, 1].imshow(overlay.astype(np.uint8))
        axes[1, 1].set_title(f"RGB + Mask Overlay\nMask pixels: {(mask_for_rgb > 0).sum()}")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        output_dir = Path("/workspace/packnet-sfm/outputs/alignment_verification")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{stem}_mask{i}_verification.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  저장: {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--split-file", type=str, default="splits/combined_test.json")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--class-name", type=str, default="car")
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    split_path = dataset_root / args.split_file
    
    # split 로드
    with open(split_path, 'r') as f:
        data = json.load(f)
    
    print(f"Split: {split_path}")
    print(f"Total samples: {len(data)}")
    
    # 처음 몇 개 샘플 검증
    for i, entry in enumerate(data[:args.num_samples]):
        stem = entry['new_filename']
        success = verify_sample(dataset_root, stem, args.class_name)
        if not success:
            print(f"⚠️  샘플 {i} 검증 실패")
    
    print(f"\n✅ 검증 이미지 저장 위치: /workspace/packnet-sfm/outputs/alignment_verification/")


if __name__ == "__main__":
    main()
