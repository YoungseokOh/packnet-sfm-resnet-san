#!/usr/bin/env python3
"""
통합 스플릿 생성 스크립트
- 여러 폴더에서 image_a6/*.png 또는 *.jpg 파일들을 스캔하여 스플릿 생성
- 간단한 JSON 포맷: dataset_root + new_filename만 포함
- 80/10/10 기본 비율
"""

import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm


def scan_dataset_folder(dataset_root):
    """데이터셋 폴더를 스캔하여 유효한 샘플 목록 생성"""
    dataset_root = Path(dataset_root)
    image_dir = dataset_root / 'image_a6'
    
    if not image_dir.exists():
        print(f"⚠️  image_a6 폴더 없음: {dataset_root}")
        return []
    
    # 이미지 파일 스캔 (.png, .jpg)
    image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
    
    samples = []
    for img_path in image_files:
        stem = img_path.stem
        samples.append({
            "dataset_root": str(dataset_root),
            "new_filename": stem
        })
    
    return samples


def create_combined_splits(dataset_roots, output_dir,
                           train_ratio=0.80, val_ratio=0.10, test_ratio=0.10,
                           seed=42):
    """여러 데이터셋 폴더를 통합하여 train/val/test 스플릿을 생성"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 비율 검증
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"비율 합이 1.0이어야 합니다. 현재: {total_ratio}")
    
    print(f"\n{'='*60}")
    print("📂 데이터셋 스캔 중...")
    print(f"{'='*60}")
    
    all_samples = []
    for root in tqdm(dataset_roots, desc="폴더 스캔"):
        samples = scan_dataset_folder(root)
        print(f"  �� {Path(root).name}: {len(samples):,}개 샘플")
        all_samples.extend(samples)
    
    print(f"\n총 샘플 수: {len(all_samples):,}개")
    
    # 랜덤 셔플
    random.seed(seed)
    random.shuffle(all_samples)
    
    # 스플릿 계산
    total = len(all_samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_data = all_samples[:train_end]
    val_data = all_samples[train_end:val_end]
    test_data = all_samples[val_end:]
    
    # 저장
    print(f"\n{'='*60}")
    print("💾 스플릿 파일 저장 중...")
    print(f"{'='*60}")
    
    splits = {
        'combined_train.json': train_data,
        'combined_val.json': val_data,
        'combined_test.json': test_data
    }
    
    for filename, data in splits.items():
        filepath = output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  ✅ {filename}: {len(data):,}개 ({len(data)/total*100:.1f}%)")
    
    print(f"\n{'='*60}")
    print(f"📊 스플릿 완료!")
    print(f"{'='*60}")
    print(f"  • 출력 폴더: {output_dir}")
    print(f"  • Train: {len(train_data):,}개 ({train_ratio*100:.0f}%)")
    print(f"  • Val:   {len(val_data):,}개 ({val_ratio*100:.0f}%)")
    print(f"  • Test:  {len(test_data):,}개 ({test_ratio*100:.0f}%)")
    print(f"  • 랜덤 시드: {seed}")
    
    return train_data, val_data, test_data


def main():
    parser = argparse.ArgumentParser(description='통합 train/val/test 스플릿 생성')
    parser.add_argument('--datasets', '-d', nargs='+', required=True, help='데이터셋 루트 경로들')
    parser.add_argument('--output', '-o', required=True, help='출력 디렉토리')
    parser.add_argument('--ratio', '-r', nargs=3, type=float, default=[0.80, 0.10, 0.10], help='train/val/test 비율')
    parser.add_argument('--seed', '-s', type=int, default=42, help='랜덤 시드')
    
    args = parser.parse_args()
    
    create_combined_splits(
        dataset_roots=args.datasets,
        output_dir=args.output,
        train_ratio=args.ratio[0],
        val_ratio=args.ratio[1],
        test_ratio=args.ratio[2],
        seed=args.seed
    )


if __name__ == '__main__':
    main()
