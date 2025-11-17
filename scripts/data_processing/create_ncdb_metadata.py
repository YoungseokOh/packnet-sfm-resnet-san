#!/usr/bin/env python3
# create_ncdb_metadata.py
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm

class NcdbMetadataGenerator:
    """NCDB 데이터셋 메타데이터 생성기"""
    
    # ncdb_dataset.py와 동일한 우선순위
    DEFAULT_DEPTH_VARIANTS = [
        'newest_depth_maps',
        'newest_synthetic_depth_maps',
        'new_depth_maps',
        'depth_maps',
    ]
    
    def __init__(self, dataset_root, depth_variants=None):
        self.dataset_root = Path(dataset_root)
        self.depth_variants = depth_variants or self.DEFAULT_DEPTH_VARIANTS
        
    def _load_depth_png(self, depth_path):
        """ncdb_dataset.py의 _load_depth_png와 동일한 로직"""
        try:
            depth_png = Image.open(depth_path)
            arr16 = np.asarray(depth_png, dtype=np.uint16)
            depth = arr16.astype(np.float32)
            
            # KITTI 스타일로 256으로 나누기
            if depth.max() > 255:
                depth /= 256.0
            
            # 유효하지 않은 픽셀을 0으로 마스킹
            depth[arr16 == 0] = 0
            
            return depth
        except (FileNotFoundError, OSError) as e:
            print(f"Depth load failed: {depth_path} ({e})")
            return None
    
    def _resolve_depth_path(self, base_dir, stem):
        """variant 우선순위에 따라 존재하는 depth 경로 반환"""
        for variant in self.depth_variants:
            depth_path = base_dir / variant / f"{stem}.png"
            if depth_path.exists():
                return depth_path, variant
        return None, None
    
    def analyze_split(self, split_file):
        """
        JSON split 파일을 읽어서 각 샘플의 메타데이터를 생성합니다.
        
        Args:
            split_file: JSON split 파일 경로 (예: 'train_split.json')
        
        Returns:
            DataFrame: 각 샘플의 메타데이터
        """
        # Split 파일 로드
        split_path = self.dataset_root / split_file
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        
        with open(split_path, 'r') as f:
            split_data = json.load(f)
        
        print(f"총 {len(split_data)}개의 샘플 분석 중...")
        
        metadata = []
        skipped = 0
        
        for entry in tqdm(split_data):
            dataset_root = entry.get('dataset_root', '')
            stem = entry.get('new_filename', '')
            
            if not stem:
                skipped += 1
                continue
            
            # 경로 구성
            base_dir = self.dataset_root / dataset_root
            image_path = base_dir / 'image_a6' / f"{stem}.png"
            
            # 이미지 파일 존재 여부 확인
            if not image_path.exists():
                skipped += 1
                continue
            
            # Depth 파일 탐색 (우선순위 순서)
            depth_path, depth_variant = self._resolve_depth_path(base_dir, stem)
            
            if depth_path is None:
                # Depth가 없는 샘플은 스킵
                skipped += 1
                continue
            
            # 깊이 데이터 로드 및 분석
            try:
                depth = self._load_depth_png(depth_path)
                
                if depth is None:
                    skipped += 1
                    continue
                
                # 유효한 깊이 값만 선택
                valid_depth = depth[depth > 0]
                
                if len(valid_depth) == 0:
                    skipped += 1
                    continue
                
                # 이미지 크기
                img = Image.open(image_path)
                width, height = img.size
                
                # 깊이 통계 계산
                mean_depth = float(np.mean(valid_depth))
                median_depth = float(np.median(valid_depth))
                min_depth = float(np.min(valid_depth))
                max_depth = float(np.max(valid_depth))
                std_depth = float(np.std(valid_depth))
                p50 = float(np.percentile(valid_depth, 50))
                p90 = float(np.percentile(valid_depth, 90))
                p95 = float(np.percentile(valid_depth, 95))
                
                # Scene 타입 추정 (평균 깊이 기반)
                if mean_depth < 5.0:
                    scene_type = 'indoor'
                elif mean_depth < 15.0:
                    scene_type = 'outdoor_near'
                else:
                    scene_type = 'outdoor_far'
                
                metadata.append({
                    'dataset_root': dataset_root,
                    'filename': stem,
                    'image_path': str(image_path.relative_to(self.dataset_root)),
                    'depth_path': str(depth_path.relative_to(self.dataset_root)),
                    'depth_variant': depth_variant,
                    'mean_depth': mean_depth,
                    'median_depth': median_depth,
                    'min_depth': min_depth,
                    'max_depth': max_depth,
                    'std_depth': std_depth,
                    'p50': p50,
                    'p90': p90,
                    'p95': p95,
                    'width': width,
                    'height': height,
                    'scene_type': scene_type,
                    'valid_pixels': len(valid_depth),
                    'total_pixels': depth.size,
                })
                
            except Exception as e:
                print(f"Error processing {stem}: {e}")
                skipped += 1
                continue
        
        # DataFrame 생성
        df = pd.DataFrame(metadata)
        
        # 통계 출력
        print("\n" + "="*60)
        print("NCDB 데이터셋 깊이 분포 통계")
        print("="*60)
        print(f"총 샘플 수: {len(df)}")
        print(f"스킵된 샘플: {skipped}")
        print(f"\n평균 깊이 통계:")
        print(f"  Mean: {df['mean_depth'].mean():.2f}m (std: {df['mean_depth'].std():.2f}m)")
        print(f"  Median: {df['median_depth'].median():.2f}m")
        print(f"  Range: [{df['min_depth'].min():.2f}m, {df['max_depth'].max():.2f}m]")
        print(f"\nScene 타입 분포:")
        print(df['scene_type'].value_counts())
        print(f"\nDepth Variant 사용 분포:")
        print(df['depth_variant'].value_counts())
        print("="*60)
        
        return df

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate NCDB metadata for calibration')
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='NCDB 데이터셋 루트 경로 (예: /workspace/data/ncdb-cls-640x384)')
    parser.add_argument('--split_file', type=str, default='splits/combined_train.json',
                        help='Split 파일명 (기본: splits/combined_train.json)')
    parser.add_argument('--output', type=str, default='outputs/calibration/ncdb_metadata.csv',
                        help='출력 CSV 파일명 (기본: outputs/calibration/ncdb_metadata.csv)')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 메타데이터 생성기 초기화
    generator = NcdbMetadataGenerator(args.dataset_root)
    
    # 메타데이터 생성
    metadata_df = generator.analyze_split(args.split_file)
    
    # CSV 저장
    metadata_df.to_csv(args.output, index=False)
    print(f"\n✅ 메타데이터가 '{args.output}'에 저장되었습니다.")
