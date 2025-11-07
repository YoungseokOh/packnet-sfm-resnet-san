# ST1: Advanced PTQ Calibration - 실행 계획

**목표**: NCDB 데이터셋에서 Representative Calibration 이미지 300개를 선별하여 NPU PTQ에 사용  
**예상 소요 시간**: 2-3시간  
**예상 성능 개선**: abs_rel 0.1133 → 0.085 (25% 개선)

---

## 📋 전체 프로세스 요약

```
Step 1: 메타데이터 생성 (1-2시간)
   ↓
Step 2: Stratified Sampling (30분)
   ↓
Step 3: 이미지 복사 (10분)
   ↓
Step 3.5: 분석 및 시각화 (선택, 5분)
   ↓
Step 4: NPU PTQ 실행 (30분-1시간)
   ↓
Step 5: 성능 평가 (30분)
```

---

## Step 1: 메타데이터 생성 스크립트 작성 및 실행

### 1.1. 스크립트 작성

`create_ncdb_metadata.py` 파일을 생성합니다:

```python
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
                        help='NCDB 데이터셋 루트 경로 (예: /data/ncdb)')
    parser.add_argument('--split_file', type=str, default='train_split.json',
                        help='Split 파일명 (기본: train_split.json)')
    parser.add_argument('--output', type=str, default='ncdb_metadata.csv',
                        help='출력 CSV 파일명 (기본: ncdb_metadata.csv)')
    
    args = parser.parse_args()
    
    # 메타데이터 생성기 초기화
    generator = NcdbMetadataGenerator(args.dataset_root)
    
    # 메타데이터 생성
    metadata_df = generator.analyze_split(args.split_file)
    
    # CSV 저장
    metadata_df.to_csv(args.output, index=False)
    print(f"\n✅ 메타데이터가 '{args.output}'에 저장되었습니다.")
```

### 1.2. 실행

```bash
# NCDB 데이터셋 경로를 실제 경로로 변경
python create_ncdb_metadata.py \
    --dataset_root /data/ncdb \
    --split_file train_split.json \
    --output ncdb_train_metadata.csv
```

### 1.3. 결과 확인

```bash
# 생성된 CSV 파일 확인
head -n 5 ncdb_train_metadata.csv
wc -l ncdb_train_metadata.csv
```

---

## Step 2: Calibration Dataset 선별 (Stratified Sampling)

### 2.1. 스크립트 작성

`create_calibration_split.py` 파일을 생성합니다:

```python
# create_calibration_split.py
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class CalibrationDatasetCreator:
    """Representative Calibration Dataset 생성기"""
    
    def __init__(self, metadata_path):
        self.df = pd.read_csv(metadata_path)
        print(f"총 {len(self.df)}개의 샘플이 메타데이터에 있습니다.")
    
    def create_stratified_split(self, target_size=300, output_file='calibration_split.json',
                                depth_bins=None, sampling_ratios=None):
        """
        Depth 분포에 기반하여 계층화된 샘플링을 수행합니다.
        """
        # 기본값 설정
        if depth_bins is None:
            depth_bins = [0, 3, 8, 15, 100]
        
        if sampling_ratios is None:
            # 근거리(25%), 중거리(40%), 원거리(25%), 초원거리(10%)
            sampling_ratios = [0.25, 0.40, 0.25, 0.10]
        
        # 구간 라벨
        labels = ['near', 'mid', 'far', 'very_far'][:len(depth_bins)-1]
        
        # Depth 범위별로 분류
        self.df['depth_range'] = pd.cut(
            self.df['mean_depth'], 
            bins=depth_bins, 
            labels=labels, 
            right=True
        )
        
        # 각 범위별 데이터 개수 확인
        print("\n" + "="*60)
        print("Depth Range 분포")
        print("="*60)
        range_counts = self.df['depth_range'].value_counts(sort=False)
        print(range_counts)
        print("\n비율:")
        print(self.df['depth_range'].value_counts(normalize=True, sort=False))
        
        # 각 구간별 샘플링 크기 결정
        sampled_dfs = []
        total_sampled = 0
        
        print("\n" + "="*60)
        print("샘플링 계획")
        print("="*60)
        
        for i, label in enumerate(labels):
            available = range_counts.get(label, 0)
            target = int(target_size * sampling_ratios[i])
            actual = min(target, available)
            
            if actual > 0:
                samples = self.df[self.df['depth_range'] == label].sample(
                    n=actual, replace=False, random_state=42
                )
                sampled_dfs.append(samples)
                total_sampled += actual
                print(f"{label:10s} ({depth_bins[i]:>5.1f}-{depth_bins[i+1]:>5.1f}m): "
                      f"목표 {target:3d}, 실제 {actual:3d} (가용 {available:3d})")
        
        # 목표 크기에 미달하면 가장 많은 범위에서 추가 샘플링
        if total_sampled < target_size:
            shortage = target_size - total_sampled
            mid_available = range_counts.get('mid', 0) - int(target_size * sampling_ratios[1])
            if mid_available > 0:
                additional = min(shortage, mid_available)
                already_sampled = sampled_dfs[1] if len(sampled_dfs) > 1 else pd.DataFrame()
                mid_pool = self.df[self.df['depth_range'] == 'mid']
                mid_pool = mid_pool[~mid_pool.index.isin(already_sampled.index)]
                
                if len(mid_pool) >= additional:
                    extra_samples = mid_pool.sample(n=additional, replace=False, random_state=42)
                    sampled_dfs.append(extra_samples)
                    total_sampled += additional
                    print(f"\n중거리에서 {additional}개 추가 샘플링")
        
        print(f"\n총 샘플링: {total_sampled}개")
        print("="*60)
        
        # 최종 데이터셋 병합
        representative_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # JSON 형식으로 변환
        calibration_data = []
        for _, row in representative_df.iterrows():
            calibration_data.append({
                'dataset_root': row['dataset_root'],
                'new_filename': row['filename']
            })
        
        # JSON 저장
        with open(output_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"\n✅ '{output_file}' 생성 완료 ({len(calibration_data)}개 샘플)")
        
        # 시각화
        self.visualize_distribution(self.df, representative_df, output_file)
        
        return representative_df
    
    def visualize_distribution(self, original_df, sampled_df, output_file):
        """원본과 샘플링된 데이터셋의 분포 비교"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 원본 분포 - 히스토그램
        axes[0, 0].hist(original_df['mean_depth'], bins=50, alpha=0.7, 
                        color='blue', edgecolor='black')
        axes[0, 0].set_title(f'Original Dataset (n={len(original_df)})')
        axes[0, 0].set_xlabel('Mean Depth (m)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 샘플링된 분포 - 히스토그램
        axes[0, 1].hist(sampled_df['mean_depth'], bins=50, alpha=0.7, 
                        color='green', edgecolor='black')
        axes[0, 1].set_title(f'Calibration Dataset (n={len(sampled_df)})')
        axes[0, 1].set_xlabel('Mean Depth (m)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Depth Range별 비교
        range_labels = ['near\n(0-3m)', 'mid\n(3-8m)', 'far\n(8-15m)', 'very_far\n(15m+)']
        original_counts = original_df['depth_range'].value_counts(sort=False)
        sampled_counts = sampled_df['depth_range'].value_counts(sort=False)
        
        x = np.arange(len(range_labels))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, original_counts.values, width, 
                       label='Original', alpha=0.7, color='blue')
        axes[1, 0].bar(x + width/2, sampled_counts.values, width, 
                       label='Calibration', alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Depth Range')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Depth Range Distribution Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(range_labels)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Scene Type별 비교
        if 'scene_type' in original_df.columns and 'scene_type' in sampled_df.columns:
            scene_orig = original_df['scene_type'].value_counts()
            scene_samp = sampled_df['scene_type'].value_counts()
            
            scene_labels = list(set(scene_orig.index) | set(scene_samp.index))
            x_scene = np.arange(len(scene_labels))
            
            orig_vals = [scene_orig.get(label, 0) for label in scene_labels]
            samp_vals = [scene_samp.get(label, 0) for label in scene_labels]
            
            axes[1, 1].bar(x_scene - width/2, orig_vals, width, 
                           label='Original', alpha=0.7, color='blue')
            axes[1, 1].bar(x_scene + width/2, samp_vals, width, 
                           label='Calibration', alpha=0.7, color='green')
            axes[1, 1].set_xlabel('Scene Type')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Scene Type Distribution Comparison')
            axes[1, 1].set_xticks(x_scene)
            axes[1, 1].set_xticklabels(scene_labels, rotation=15)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_file = output_file.replace('.json', '_distribution.png')
        plt.savefig(plot_file, dpi=150)
        print(f"✅ 분포 비교 그래프가 '{plot_file}'에 저장되었습니다.")
        plt.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create calibration split from metadata')
    parser.add_argument('--metadata', type=str, required=True,
                        help='메타데이터 CSV 파일 경로')
    parser.add_argument('--target_size', type=int, default=300,
                        help='목표 샘플 개수 (기본: 300)')
    parser.add_argument('--output', type=str, default='calibration_split.json',
                        help='출력 JSON 파일명')
    
    args = parser.parse_args()
    
    # Calibration Dataset 생성기 초기화
    creator = CalibrationDatasetCreator(args.metadata)
    
    # Stratified Sampling 수행
    creator.create_stratified_split(
        target_size=args.target_size,
        output_file=args.output
    )
```

### 2.2. 실행

```bash
python create_calibration_split.py \
    --metadata ncdb_train_metadata.csv \
    --target_size 300 \
    --output calibration_split.json
```

### 2.3. 결과 확인

```bash
# JSON 파일 확인
python -c "import json; data=json.load(open('calibration_split.json')); print(f'총 {len(data)}개 샘플')"

# 분포 그래프 확인
ls -lh calibration_split_distribution.png
```

---

## Step 3: ⭐ Calibration 이미지 복사 (핵심!)

### 3.1. 이미지 복사 스크립트 작성

`copy_calibration_images.py` 파일을 생성합니다:

```python
# copy_calibration_images.py
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def copy_calibration_images(
    dataset_root,
    calibration_split_json,
    output_dir='calibration_images'
):
    """
    calibration_split.json에 지정된 이미지들을 output_dir로 복사합니다.
    
    Args:
        dataset_root: NCDB 데이터셋 루트 경로
        calibration_split_json: Calibration split JSON 파일 경로
        output_dir: 이미지를 복사할 출력 디렉토리
    
    Returns:
        복사된 이미지 경로 리스트
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # JSON 파일 로드
    with open(calibration_split_json, 'r') as f:
        split_data = json.load(f)
    
    print("\n" + "="*60)
    print(f"Calibration 이미지 복사")
    print("="*60)
    print(f"출력 디렉토리: {output_dir.absolute()}")
    print(f"총 {len(split_data)}개 이미지 복사 중...")
    
    image_list = []
    copied = 0
    failed = 0
    
    for entry in tqdm(split_data):
        base_dir = dataset_root / entry['dataset_root']
        stem = entry['new_filename']
        
        src_path = base_dir / 'image_a6' / f"{stem}.png"
        
        # 파일명 충돌 방지: dataset_root 경로를 파일명에 포함
        # 예: synced_data/scene_001/frame_0001 -> scene_001_frame_0001.png
        safe_name = entry['dataset_root'].replace('/', '_').replace('synced_data_', '')
        dst_filename = f"{safe_name}_{stem}.png"
        dst_path = output_dir / dst_filename
        
        if src_path.exists():
            try:
                shutil.copy2(src_path, dst_path)
                image_list.append({
                    'original_path': str(src_path),
                    'copied_path': str(dst_path),
                    'filename': dst_filename
                })
                copied += 1
            except Exception as e:
                print(f"\n⚠️ 복사 실패: {src_path} -> {e}")
                failed += 1
        else:
            print(f"\n⚠️ 파일 없음: {src_path}")
            failed += 1
    
    # 이미지 경로 리스트 저장 (절대 경로)
    list_file = output_dir / 'image_list.txt'
    with open(list_file, 'w') as f:
        for img in image_list:
            f.write(f"{Path(img['copied_path']).absolute()}\n")
    
    # 간단한 파일명 리스트도 저장 (상대 경로)
    simple_list_file = output_dir / 'image_filenames.txt'
    with open(simple_list_file, 'w') as f:
        for img in image_list:
            f.write(f"{img['filename']}\n")
    
    # 메타데이터 JSON 저장
    meta_file = output_dir / 'calibration_metadata.json'
    with open(meta_file, 'w') as f:
        json.dump(image_list, f, indent=2)
    
    print("\n" + "="*60)
    print("복사 완료!")
    print("="*60)
    print(f"✅ 성공: {copied}개")
    print(f"❌ 실패: {failed}개")
    print(f"\n생성된 파일:")
    print(f"  - 이미지 디렉토리: {output_dir.absolute()}")
    print(f"  - 절대 경로 리스트: {list_file}")
    print(f"  - 파일명 리스트: {simple_list_file}")
    print(f"  - 메타데이터: {meta_file}")
    print("="*60)
    
    return image_list

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Copy calibration images to a separate folder'
    )
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='NCDB 데이터셋 루트 경로')
    parser.add_argument('--calibration_split', type=str, 
                        default='calibration_split.json',
                        help='Calibration split JSON 파일')
    parser.add_argument('--output_dir', type=str, 
                        default='calibration_images',
                        help='이미지를 복사할 출력 디렉토리')
    
    args = parser.parse_args()
    
    # 이미지 복사 실행
    copy_calibration_images(
        dataset_root=args.dataset_root,
        calibration_split_json=args.calibration_split,
        output_dir=args.output_dir
    )
```

### 3.2. 실행

```bash
# Calibration 이미지를 별도 폴더로 복사
python copy_calibration_images.py \
    --dataset_root /data/ncdb \
    --calibration_split calibration_split.json \
    --output_dir calibration_images
```

### 3.3. 결과 확인

```bash
# 복사된 이미지 개수 확인
ls calibration_images/*.png | wc -l

# 처음 5개 파일 확인
ls calibration_images/ | head -n 5

# 디렉토리 크기 확인
du -sh calibration_images/

# 이미지 리스트 파일 확인
head -n 5 calibration_images/image_list.txt
```

**생성되는 파일 구조**:
```
calibration_images/
├── scene_001_frame_0001.png
├── scene_001_frame_0145.png
├── scene_003_frame_0032.png
├── ...
├── image_list.txt              # 절대 경로 리스트
├── image_filenames.txt         # 파일명만 리스트
└── calibration_metadata.json   # 상세 메타데이터
```

---

## Step 3.5: 📊 Calibration Dataset 분석 및 시각화 (선택적)

### 3.5.1. 분석 시각화 스크립트 작성

복사된 Calibration 이미지들의 통계를 분석하고 시각화하는 스크립트입니다.

`analyze_calibration_dataset.py` 파일을 생성합니다:

```python
# analyze_calibration_dataset.py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
from collections import Counter

def analyze_calibration_dataset(
    calibration_metadata_json='calibration_images/calibration_metadata.json',
    ncdb_metadata_csv='ncdb_train_metadata.csv',
    output_dir='calibration_analysis'
):
    """
    Calibration 데이터셋의 상세 분석 및 시각화
    
    Args:
        calibration_metadata_json: 복사된 이미지의 메타데이터 JSON
        ncdb_metadata_csv: 전체 NCDB 메타데이터 CSV
        output_dir: 분석 결과를 저장할 디렉토리
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. 메타데이터 로드
    print("\n" + "="*70)
    print("Calibration Dataset 분석")
    print("="*70)
    
    with open(calibration_metadata_json, 'r') as f:
        calib_meta = json.load(f)
    
    full_meta = pd.read_csv(ncdb_metadata_csv)
    
    # 복사된 이미지들의 filename 추출
    calib_filenames = set()
    for item in calib_meta:
        # original_path에서 filename 추출
        # 예: /data/ncdb/synced_data/scene_001/image_a6/frame_0001.png
        original_path = Path(item['original_path'])
        stem = original_path.stem
        calib_filenames.add(stem)
    
    # 전체 메타데이터에서 calibration에 사용된 샘플만 필터링
    calib_df = full_meta[full_meta['filename'].isin(calib_filenames)].copy()
    
    print(f"\n✅ 로드 완료:")
    print(f"  - 전체 NCDB 샘플: {len(full_meta)}개")
    print(f"  - Calibration 샘플: {len(calib_df)}개")
    print(f"  - 복사된 이미지: {len(calib_meta)}개")
    
    # 2. 기본 통계 계산
    print("\n" + "="*70)
    print("깊이 분포 통계")
    print("="*70)
    
    stats = {
        'mean': calib_df['mean_depth'].mean(),
        'median': calib_df['median_depth'].median(),
        'std': calib_df['mean_depth'].std(),
        'min': calib_df['min_depth'].min(),
        'max': calib_df['max_depth'].max(),
        'p25': calib_df['mean_depth'].quantile(0.25),
        'p50': calib_df['mean_depth'].quantile(0.50),
        'p75': calib_df['mean_depth'].quantile(0.75),
        'p90': calib_df['mean_depth'].quantile(0.90),
        'p95': calib_df['mean_depth'].quantile(0.95),
    }
    
    print(f"평균 깊이: {stats['mean']:.2f}m (± {stats['std']:.2f}m)")
    print(f"중앙값: {stats['median']:.2f}m")
    print(f"범위: [{stats['min']:.2f}m, {stats['max']:.2f}m]")
    print(f"백분위: p25={stats['p25']:.2f}m, p50={stats['p50']:.2f}m, "
          f"p75={stats['p75']:.2f}m, p90={stats['p90']:.2f}m, p95={stats['p95']:.2f}m")
    
    # Scene type 분포
    print(f"\nScene 타입 분포:")
    scene_counts = calib_df['scene_type'].value_counts()
    for scene, count in scene_counts.items():
        pct = count / len(calib_df) * 100
        print(f"  {scene:15s}: {count:3d}개 ({pct:5.1f}%)")
    
    # Depth variant 분포
    print(f"\nDepth Variant 분포:")
    variant_counts = calib_df['depth_variant'].value_counts()
    for variant, count in variant_counts.items():
        pct = count / len(calib_df) * 100
        print(f"  {variant:30s}: {count:3d}개 ({pct:5.1f}%)")
    
    # 3. 상세 시각화
    print("\n" + "="*70)
    print("시각화 생성 중...")
    print("="*70)
    
    create_comprehensive_visualization(calib_df, full_meta, output_dir)
    create_depth_analysis_visualization(calib_df, output_dir)
    create_image_samples_grid(calib_meta, output_dir)
    
    # 4. 통계 리포트 저장
    report_path = output_dir / 'calibration_statistics.json'
    report = {
        'total_samples': len(calib_df),
        'depth_statistics': stats,
        'scene_type_distribution': scene_counts.to_dict(),
        'depth_variant_distribution': variant_counts.to_dict(),
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ 분석 완료! 결과가 '{output_dir}' 디렉토리에 저장되었습니다.")
    print("="*70)

def create_comprehensive_visualization(calib_df, full_meta, output_dir):
    """종합 분석 시각화 (6개 서브플롯)"""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. 깊이 분포 히스토그램 (전체 vs calibration)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(full_meta['mean_depth'], bins=60, alpha=0.5, 
             label=f'Full Dataset (n={len(full_meta)})', color='blue', edgecolor='black')
    ax1.hist(calib_df['mean_depth'], bins=60, alpha=0.7, 
             label=f'Calibration (n={len(calib_df)})', color='green', edgecolor='black')
    ax1.set_xlabel('Mean Depth (m)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Depth Distribution Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 깊이 박스플롯
    ax2 = fig.add_subplot(gs[0, 2])
    box_data = [full_meta['mean_depth'], calib_df['mean_depth']]
    bp = ax2.boxplot(box_data, labels=['Full', 'Calib'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax2.set_ylabel('Mean Depth (m)', fontsize=11)
    ax2.set_title('Depth Boxplot', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Scene Type 분포
    ax3 = fig.add_subplot(gs[1, 0])
    scene_counts = calib_df['scene_type'].value_counts()
    colors_scene = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    wedges, texts, autotexts = ax3.pie(
        scene_counts.values, 
        labels=scene_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors_scene[:len(scene_counts)]
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    ax3.set_title('Scene Type Distribution', fontsize=13, fontweight='bold')
    
    # 4. Depth Variant 분포
    ax4 = fig.add_subplot(gs[1, 1])
    variant_counts = calib_df['depth_variant'].value_counts()
    variant_labels = [v.replace('_', '\n') for v in variant_counts.index]
    bars = ax4.bar(range(len(variant_counts)), variant_counts.values, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3'][:len(variant_counts)])
    ax4.set_xticks(range(len(variant_counts)))
    ax4.set_xticklabels(variant_labels, fontsize=9, rotation=0)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Depth Variant Distribution', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 각 바 위에 개수 표시
    for i, (bar, count) in enumerate(zip(bars, variant_counts.values)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 5. Depth Range 분포 (4개 구간)
    ax5 = fig.add_subplot(gs[1, 2])
    bins = [0, 3, 8, 15, 100]
    labels = ['Near\n(0-3m)', 'Mid\n(3-8m)', 'Far\n(8-15m)', 'Very Far\n(15m+)']
    calib_df['depth_range'] = pd.cut(calib_df['mean_depth'], bins=bins, labels=labels)
    range_counts = calib_df['depth_range'].value_counts(sort=False)
    
    colors_range = ['#FF6B6B', '#FFD93D', '#6BCB77', '#4D96FF']
    bars2 = ax5.bar(range(len(range_counts)), range_counts.values, color=colors_range)
    ax5.set_xticks(range(len(range_counts)))
    ax5.set_xticklabels(labels, fontsize=9)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('Depth Range Distribution', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars2, range_counts.values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 6. 깊이 통계 누적 분포 (CDF)
    ax6 = fig.add_subplot(gs[2, :])
    full_sorted = np.sort(full_meta['mean_depth'])
    calib_sorted = np.sort(calib_df['mean_depth'])
    full_cdf = np.arange(1, len(full_sorted) + 1) / len(full_sorted)
    calib_cdf = np.arange(1, len(calib_sorted) + 1) / len(calib_sorted)
    
    ax6.plot(full_sorted, full_cdf, label='Full Dataset', color='blue', linewidth=2, alpha=0.6)
    ax6.plot(calib_sorted, calib_cdf, label='Calibration', color='green', linewidth=2)
    ax6.set_xlabel('Mean Depth (m)', fontsize=11)
    ax6.set_ylabel('Cumulative Probability', fontsize=11)
    ax6.set_title('Cumulative Distribution Function (CDF)', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 30)  # 30m까지만 표시
    
    plt.savefig(output_dir / 'calibration_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ 종합 분석: calibration_comprehensive_analysis.png")
    plt.close()

def create_depth_analysis_visualization(calib_df, output_dir):
    """깊이 분석 상세 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Mean vs Median Depth
    axes[0, 0].scatter(calib_df['mean_depth'], calib_df['median_depth'], 
                       alpha=0.5, s=30, c='green', edgecolors='black', linewidth=0.5)
    axes[0, 0].plot([0, 30], [0, 30], 'r--', linewidth=2, label='y=x')
    axes[0, 0].set_xlabel('Mean Depth (m)', fontsize=11)
    axes[0, 0].set_ylabel('Median Depth (m)', fontsize=11)
    axes[0, 0].set_title('Mean vs Median Depth', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Depth Standard Deviation
    axes[0, 1].hist(calib_df['std_depth'], bins=30, color='orange', 
                    edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Depth Std Dev (m)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Depth Variation Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Valid Pixels Ratio
    calib_df['valid_ratio'] = calib_df['valid_pixels'] / calib_df['total_pixels']
    axes[1, 0].hist(calib_df['valid_ratio'] * 100, bins=30, color='skyblue',
                    edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Valid Depth Pixels (%)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Depth Coverage Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Scene Type별 평균 깊이
    scene_depth = calib_df.groupby('scene_type')['mean_depth'].agg(['mean', 'std'])
    x_pos = np.arange(len(scene_depth))
    bars = axes[1, 1].bar(x_pos, scene_depth['mean'], 
                          yerr=scene_depth['std'], capsize=5,
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(scene_depth)],
                          alpha=0.7, edgecolor='black')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(scene_depth.index, rotation=15)
    axes[1, 1].set_ylabel('Mean Depth (m)', fontsize=11)
    axes[1, 1].set_title('Average Depth by Scene Type', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 바 위에 평균값 표시
    for bar, mean_val in zip(bars, scene_depth['mean']):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean_val:.1f}m',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_depth_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ 깊이 분석: calibration_depth_analysis.png")
    plt.close()

def create_image_samples_grid(calib_meta, output_dir, n_samples=12):
    """샘플 이미지 그리드 표시 (랜덤하게 선택)"""
    import random
    
    # 랜덤하게 n_samples개 선택
    if len(calib_meta) > n_samples:
        samples = random.sample(calib_meta, n_samples)
    else:
        samples = calib_meta
        n_samples = len(samples)
    
    # 그리드 크기 결정
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for i, sample in enumerate(samples):
        try:
            img_path = sample['copied_path']
            img = Image.open(img_path)
            
            # 이미지 표시
            axes[i].imshow(img)
            axes[i].axis('off')
            
            # 파일명 표시
            filename = sample['filename']
            # 파일명이 너무 길면 줄임
            if len(filename) > 30:
                filename = filename[:27] + '...'
            axes[i].set_title(filename, fontsize=8)
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error loading\n{sample["filename"]}',
                        ha='center', va='center', fontsize=8)
            axes[i].axis('off')
    
    # 남은 빈 서브플롯 숨기기
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Calibration Dataset Sample Images (Random {n_samples})', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_sample_images.png', dpi=120, bbox_inches='tight')
    print(f"  ✅ 샘플 이미지: calibration_sample_images.png")
    plt.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze and visualize calibration dataset'
    )
    parser.add_argument('--calibration_meta', type=str,
                        default='calibration_images/calibration_metadata.json',
                        help='Calibration 메타데이터 JSON 파일')
    parser.add_argument('--ncdb_meta', type=str,
                        default='ncdb_train_metadata.csv',
                        help='NCDB 전체 메타데이터 CSV 파일')
    parser.add_argument('--output_dir', type=str,
                        default='calibration_analysis',
                        help='분석 결과 출력 디렉토리')
    
    args = parser.parse_args()
    
    # 분석 실행
    analyze_calibration_dataset(
        calibration_metadata_json=args.calibration_meta,
        ncdb_metadata_csv=args.ncdb_meta,
        output_dir=args.output_dir
    )
```

### 3.5.2. 실행

```bash
# Calibration 데이터셋 분석 및 시각화
python analyze_calibration_dataset.py \
    --calibration_meta calibration_images/calibration_metadata.json \
    --ncdb_meta ncdb_train_metadata.csv \
    --output_dir calibration_analysis
```

### 3.5.3. 생성되는 시각화 결과

실행 후 `calibration_analysis/` 디렉토리에 다음 파일들이 생성됩니다:

**1. `calibration_comprehensive_analysis.png`** (종합 분석)
   - 깊이 분포 히스토그램 (전체 vs calibration)
   - 깊이 박스플롯
   - Scene Type 분포 (파이 차트)
   - Depth Variant 분포 (바 차트)
   - Depth Range 분포 (4개 구간)
   - 누적 분포 함수 (CDF)

**2. `calibration_depth_analysis.png`** (깊이 상세 분석)
   - Mean vs Median Depth (산점도)
   - Depth 표준편차 분포
   - Valid Pixels 비율 분포
   - Scene Type별 평균 깊이 (에러바 포함)

**3. `calibration_sample_images.png`** (샘플 이미지 그리드)
   - 랜덤하게 선택된 12개 이미지 표시
   - 각 이미지의 파일명 표시

**4. `calibration_statistics.json`** (통계 리포트)
   - JSON 형식의 상세 통계 데이터

### 3.5.4. 결과 확인

```bash
# 생성된 파일 확인
ls -lh calibration_analysis/

# 통계 리포트 확인
cat calibration_analysis/calibration_statistics.json | python -m json.tool

# 이미지 뷰어로 시각화 결과 확인
# Linux
xdg-open calibration_analysis/calibration_comprehensive_analysis.png

# macOS
open calibration_analysis/calibration_comprehensive_analysis.png

# 또는 VSCode에서 직접 열기
code calibration_analysis/
```

**예상 출력**:
```
======================================================================
Calibration Dataset 분석
======================================================================

✅ 로드 완료:
  - 전체 NCDB 샘플: 4856개
  - Calibration 샘플: 300개
  - 복사된 이미지: 300개

======================================================================
깊이 분포 통계
======================================================================
평균 깊이: 8.45m (± 4.23m)
중앙값: 7.82m
범위: [0.50m, 98.50m]
백분위: p25=5.12m, p50=7.82m, p75=11.34m, p90=14.56m, p95=18.23m

Scene 타입 분포:
  outdoor_near   : 135개 ( 45.0%)
  indoor         :  98개 ( 32.7%)
  outdoor_far    :  67개 ( 22.3%)

Depth Variant 분포:
  newest_depth_maps              : 245개 ( 81.7%)
  newest_synthetic_depth_maps    :  42개 ( 14.0%)
  new_depth_maps                 :  13개 (  4.3%)

======================================================================
시각화 생성 중...
======================================================================
  ✅ 종합 분석: calibration_comprehensive_analysis.png
  ✅ 깊이 분석: calibration_depth_analysis.png
  ✅ 샘플 이미지: calibration_sample_images.png

✅ 분석 완료! 결과가 'calibration_analysis' 디렉토리에 저장되었습니다.
======================================================================
```

---

## Step 4: NPU PTQ 실행

### 4.1. 방법 1: 이미지 디렉토리 직접 사용

```bash
# NPU 툴체인이 디렉토리를 직접 읽을 수 있는 경우
npu_quantize \
    --model /path/to/resnetsan.onnx \
    --output resnetsan_int8.bin \
    --calibration_dir calibration_images/ \
    --num_samples 300
```

### 4.2. 방법 2: 이미지 리스트 파일 사용

```bash
# NPU 툴체인이 텍스트 파일로 경로 리스트를 받는 경우
npu_quantize \
    --model /path/to/resnetsan.onnx \
    --output resnetsan_int8.bin \
    --calibration_list calibration_images/image_list.txt \
    --num_samples 300
```

### 4.3. 방법 3: Python API 사용 (NPU 툴체인에 따라 다름)

```python
# 예시: NPU Python API를 사용하는 경우
import npu_toolkit  # 가상의 NPU 툴킷

calibration_images = []
with open('calibration_images/image_list.txt', 'r') as f:
    calibration_images = [line.strip() for line in f]

quantizer = npu_toolkit.PTQQuantizer(
    model_path='resnetsan.onnx',
    calibration_images=calibration_images,
    output_path='resnetsan_int8.bin'
)

quantizer.run()
```

---

## Step 5: 성능 평가

### 5.1. INT8 모델로 추론 실행

```bash
# NCDB validation set으로 평가
python scripts/infer.py \
    --checkpoint resnetsan_int8.bin \
    --input /data/ncdb/val_split.json \
    --output results_int8_calibrated/
```

### 5.2. Metric 계산

```bash
# Depth estimation metric 계산
python scripts/eval_depth.py \
    --pred_dir results_int8_calibrated/ \
    --gt_split /data/ncdb/val_split.json \
    --output metrics_int8_calibrated.json
```

### 5.3. 결과 비교

```bash
# FP32 vs INT8 (기존) vs INT8 (calibrated) 비교
python -c "
import json

# FP32 결과
fp32_metrics = {'abs_rel': 0.0304}

# INT8 (100 samples)
int8_old = {'abs_rel': 0.1133}

# INT8 (300 samples, calibrated)
with open('metrics_int8_calibrated.json', 'r') as f:
    int8_new = json.load(f)

print('='*60)
print('성능 비교')
print('='*60)
print(f'FP32 (baseline)        : abs_rel = {fp32_metrics[\"abs_rel\"]:.4f}')
print(f'INT8 (100 samples)     : abs_rel = {int8_old[\"abs_rel\"]:.4f}')
print(f'INT8 (300 samples, new): abs_rel = {int8_new[\"abs_rel\"]:.4f}')
print(f'')
print(f'개선율: {(int8_old[\"abs_rel\"] - int8_new[\"abs_rel\"]) / int8_old[\"abs_rel\"] * 100:.1f}%')
print('='*60)
"
```

---

## 📊 체크리스트

### 준비 단계
- [ ] NCDB 데이터셋 경로 확인 (`/data/ncdb`)
- [ ] `train_split.json` 파일 존재 확인
- [ ] Python 환경 확인 (pandas, numpy, PIL, tqdm, matplotlib)

### Step 1: 메타데이터 생성
- [ ] `create_ncdb_metadata.py` 작성
- [ ] 스크립트 실행 완료
- [ ] `ncdb_train_metadata.csv` 생성 확인
- [ ] CSV 파일 내용 검증 (깊이 통계, Scene 타입 등)

### Step 2: Calibration Split 생성
- [ ] `create_calibration_split.py` 작성
- [ ] 스크립트 실행 완료
- [ ] `calibration_split.json` 생성 확인 (300개 샘플)
- [ ] `calibration_split_distribution.png` 확인 (분포가 적절한지)

### Step 3: 이미지 복사 ⭐
- [ ] `copy_calibration_images.py` 작성
- [ ] 스크립트 실행 완료
- [ ] `calibration_images/` 디렉토리 생성 확인
- [ ] 이미지 300개 복사 완료 확인
- [ ] `image_list.txt` 생성 확인
- [ ] (선택) `analyze_calibration_dataset.py` 실행하여 시각화 생성
- [ ] (선택) `calibration_analysis/` 결과 확인

### Step 4: NPU PTQ
- [ ] ONNX 모델 준비 (`resnetsan.onnx`)
- [ ] NPU 툴체인 설치 및 설정 확인
- [ ] Calibration 이미지로 PTQ 실행
- [ ] INT8 모델 생성 확인 (`resnetsan_int8.bin`)

### Step 5: 평가
- [ ] Validation set 추론 실행
- [ ] Metric 계산 완료
- [ ] 성능 개선 확인 (`abs_rel < 0.090` 목표)

---

## 🎯 예상 결과

| 단계 | Calibration 설정 | abs_rel | 개선율 |
|------|-----------------|---------|--------|
| Baseline | 100 samples (random) | 0.1133 | - |
| **현재 목표** | **300 samples (stratified)** | **~0.085** | **~25%** |
| 확장 (선택) | 500 samples (stratified) | ~0.075 | ~34% |

**성공 기준**:
- ✅ **abs_rel < 0.090**: Phase 1 성공! → Phase 2 (Dual-Head) 진행 가능
- ⚠️ **abs_rel 0.090~0.100**: 데이터셋 500개로 확대 또는 Weight Normalization 시도
- ❌ **abs_rel > 0.100**: NPU 스펙 재확인 필요, 즉시 Phase 2로 진행 고려

---

## 🔧 트러블슈팅

### 문제 1: "No module named 'pandas'"

```bash
pip install pandas numpy pillow tqdm matplotlib
```

### 문제 2: "이미지 파일을 찾을 수 없습니다"

```bash
# NCDB 데이터셋 구조 확인
ls -la /data/ncdb/synced_data/scene_001/image_a6/

# Split 파일 경로 확인
cat /data/ncdb/train_split.json | head -n 5
```

### 문제 3: "메모리 부족"

메타데이터 생성 시 메모리 부족이 발생하면, 배치 처리를 추가합니다:

```python
# create_ncdb_metadata.py에서 청크 처리 추가
CHUNK_SIZE = 1000
for i in range(0, len(split_data), CHUNK_SIZE):
    chunk = split_data[i:i+CHUNK_SIZE]
    # 청크별 처리
```

### 문제 4: "복사 속도가 느림"

하드 링크를 사용하여 복사 대신 링크 생성:

```python
# copy_calibration_images.py에서 shutil.copy2 대신
os.link(src_path, dst_path)  # 하드 링크 (같은 파일시스템일 때만)
# 또는
os.symlink(src_path, dst_path)  # 심볼릭 링크
```

### 문제 5: "시각화 생성 실패"

이미지를 열 수 없는 경우:

```bash
# PIL 재설치
pip install --upgrade pillow

# 또는 손상된 이미지 파일 확인
python -c "
from PIL import Image
from pathlib import Path
import json

with open('calibration_images/calibration_metadata.json', 'r') as f:
    meta = json.load(f)

for item in meta:
    try:
        Image.open(item['copied_path']).verify()
    except Exception as e:
        print(f'손상된 이미지: {item[\"copied_path\"]} - {e}')
"
```

### 문제 6: "matplotlib에서 한글 깨짐"

한글 폰트 설정:

```python
# analyze_calibration_dataset.py 상단에 추가
import matplotlib.pyplot as plt
import platform

# 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:  # Linux
    plt.rc('font', family='NanumGothic')

plt.rc('axes', unicode_minus=False)  # 마이너스 기호 깨짐 방지
```

---

## 📝 다음 단계

이 Action Plan 완료 후:

1. **성능이 목표 도달 시** (`abs_rel < 0.090`):
   - ✅ Phase 1 완료!
   - 📖 `ST2_dual_head_architecture.md` 참조하여 Phase 2 진행

2. **성능이 목표 미달 시** (`abs_rel > 0.090`):
   - 🔄 Calibration 데이터셋 500개로 확대
   - 🔄 Weight Normalization 적용 (ST1_advanced_PTQ_Calibration.md 참조)
   - 🔄 NPU 스펙 재확인 (Asymmetric Quantization 지원 여부)

3. **문서 정리**:
   - 실제 실행 결과를 이 문서에 기록
   - 발생한 문제와 해결 방법 추가
