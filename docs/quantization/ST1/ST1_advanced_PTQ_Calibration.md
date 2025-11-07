# 전략 1: Advanced PTQ Calibration 상세 가이드

**문서 ID**: ST1_Advanced_PTQ_Calibration  
**버전**: 1.0  
**작성자**: GitHub Copilot (NPU & AI Expert)

---

## 1. 개요

**목표**: NPU의 자동화된 PTQ(Post-Training Quantization) 기능이 최상의 결과를 내도록, **모델과 데이터를 최적의 상태로 준비**하는 것.

**핵심 원리**:
NPU 툴체인은 `Activation Clipping`, `Bias Correction`, `Cross-Layer Equalization (CLE)`과 같은 고급 PTQ 기법들을 **자동으로 수행**합니다. 우리는 이 로직을 제어할 수 없습니다. 따라서 우리의 역할은 이 자동화된 기능들이 잘못된 결정을 내리지 않도록, **양질의 '재료'(Representative Dataset)를 제공**하고, **최적의 '요리법'(Weight Distribution)을 가진 모델**을 전달하는 것입니다.

이 전략은 **재학습 없이 시도할 수 있는 가장 빠르고 효과적인 방법**이며, 모든 후속 전략의 성공을 위한 기반이 됩니다.

**예상 성능 개선**: `abs_rel` 0.1133 → **0.085** (약 25% 개선)

---

## 2. Representative Calibration Dataset 확대 및 다양성 확보

NPU의 자동 Clipping 기능은 제공된 Calibration Dataset을 기반으로 전체 Activation의 통계적 분포를 추정합니다. 만약 데이터셋이 편향되거나 크기가 작으면, NPU는 잘못된 통계치를 학습하여 Outlier를 잘못 판단하고 최적의 Clipping 임계값을 찾는 데 실패합니다.

### 2.1. 왜 데이터셋 크기가 중요한가? (100개 → 200~500개)

- **통계적 안정성**: 데이터 수가 많을수록 통계적 추정(평균, 분산, 백분위 등)이 더 안정화됩니다. 특히 Per-channel Quantization이 없는 경우, Per-tensor 통계치의 정확성이 매우 중요합니다.
- **엣지 케이스(Edge Case) 커버**: 적은 데이터셋은 일반적인 케이스만 포함할 가능성이 높습니다. 데이터셋을 늘리면 어두운 환경, 특정 물체가 가까이 있는 경우 등 다양한 엣지 케이스를 포함시켜 NPU가 더 강건한(robust) 양자화 파라미터를 학습하게 할 수 있습니다.
- **자동화 로직의 신뢰성 확보**: 제어할 수 없는 자동화 로직에 대한 유일한 영향력은 '입력 데이터'입니다. 입력 데이터의 신뢰성을 높이는 것이 전체 결과의 신뢰성을 높이는 길입니다.

### 2.2. 다양성 확보 방안

단순히 데이터 개수만 늘리는 것은 의미가 없습니다. 데이터셋이 전체 데이터의 분포를 잘 '대표'해야 합니다.

- **Depth 분포 기반 샘플링**: 깊이 값은 양자화 정밀도에 가장 큰 영향을 미칩니다. 근거리, 중거리, 원거리 데이터를 골고루 포함해야 합니다.
- **Scene 및 조명 조건**: 실내, 야외, 낮, 밤 등 다양한 Scene과 조명 조건을 가진 데이터를 포함하여 Activation이 가질 수 있는 다양한 분포를 NPU에 알려줘야 합니다.

### 2.3. NCDB 데이터셋 구조 이해

NCDB 데이터셋은 다음과 같은 구조를 가지고 있습니다:

```
/data/ncdb/
├── train_split.json          # JSON 형식의 split 파일
├── val_split.json
├── synced_data/              # 실제 데이터가 저장된 디렉토리
│   ├── scene_001/
│   │   ├── image_a6/         # RGB 이미지
│   │   │   ├── frame_0001.png
│   │   │   └── frame_0002.png
│   │   ├── newest_depth_maps/     # 깊이 맵 (우선순위 1)
│   │   │   ├── frame_0001.png
│   │   │   └── frame_0002.png
│   │   ├── newest_synthetic_depth_maps/  # (우선순위 2)
│   │   ├── new_depth_maps/              # (우선순위 3)
│   │   └── depth_maps/                  # (우선순위 4)
│   └── scene_002/
│       └── ...
```

**Split 파일 형식** (`train_split.json`):
```json
[
  {
    "dataset_root": "synced_data/scene_001",
    "new_filename": "frame_0001"
  },
  {
    "dataset_root": "synced_data/scene_002",
    "new_filename": "frame_0005"
  },
  ...
]
```

**Depth 우선순위**:
- `ncdb_dataset.py`는 여러 depth variant를 우선순위 순서로 탐색합니다.
- 기본 순서: `newest_depth_maps` → `newest_synthetic_depth_maps` → `new_depth_maps` → `depth_maps`
- 환경 변수 `NCDB_DEPTH_VARIANT`로 변경 가능

### 2.4. 메타데이터 생성 스크립트

먼저 NCDB 데이터셋의 모든 이미지와 깊이 정보를 분석하여 메타데이터 CSV 파일을 생성해야 합니다.

#### 2.4.1. 메타데이터 생성 스크립트

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
        """
        variant 우선순위에 따라 존재하는 depth 경로 반환.
        ncdb_dataset.py의 _resolve_depth_path와 동일한 로직
        """
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
                # Depth가 없는 샘플은 스킵 (calibration에는 depth 필요)
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
        print("\n=== NCDB 데이터셋 깊이 분포 통계 ===")
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
        
        return df

# 실행 예시
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
    print(f"\n메타데이터가 '{args.output}'에 저장되었습니다.")
```

#### 2.4.2. 실행 방법

```bash
# NCDB 데이터셋 경로를 실제 경로로 변경
python create_ncdb_metadata.py \
    --dataset_root /data/ncdb \
    --split_file train_split.json \
    --output ncdb_train_metadata.csv
```

**출력 예시**:
```
총 5000개의 샘플 분석 중...
100%|████████████████████| 5000/5000 [02:30<00:00, 33.21it/s]

=== NCDB 데이터셋 깊이 분포 통계 ===
총 샘플 수: 4856
스킵된 샘플: 144

평균 깊이 통계:
  Mean: 8.45m (std: 4.23m)
  Median: 7.82m
  Range: [0.50m, 98.50m]

Scene 타입 분포:
outdoor_near    2456
indoor          1823
outdoor_far      577

Depth Variant 사용 분포:
newest_depth_maps              3821
newest_synthetic_depth_maps     892
new_depth_maps                  143
```

### 2.5. Representative Calibration Dataset 생성

이제 생성된 메타데이터를 사용하여 Representative Calibration Dataset을 만듭니다.

#### 2.5.1. Stratified Sampling 스크립트

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
        """
        Args:
            metadata_path: 앞서 생성한 메타데이터 CSV 경로
        """
        self.df = pd.read_csv(metadata_path)
        print(f"총 {len(self.df)}개의 샘플이 메타데이터에 있습니다.")
    
    def create_stratified_split(self, target_size=300, output_file='calibration_split.json',
                                depth_bins=None, sampling_ratios=None):
        """
        Depth 분포에 기반하여 계층화된 샘플링을 수행합니다.
        
        Args:
            target_size: 목표 샘플 개수 (300~500 권장)
            output_file: 출력 JSON 파일 경로
            depth_bins: 깊이 구간 (기본: [0, 3, 8, 15, 100])
            sampling_ratios: 각 구간별 샘플링 비율 (기본: [0.25, 0.40, 0.25, 0.10])
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
        print("\n=== Depth Range 분포 ===")
        range_counts = self.df['depth_range'].value_counts(sort=False)
        print(range_counts)
        print("\n비율:")
        print(self.df['depth_range'].value_counts(normalize=True, sort=False))
        
        # 각 구간별 샘플링 크기 결정
        sampled_dfs = []
        total_sampled = 0
        
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
            # 중거리에서 추가 샘플링 시도
            mid_available = range_counts.get('mid', 0) - int(target_size * sampling_ratios[1])
            if mid_available > 0:
                additional = min(shortage, mid_available)
                # 이미 샘플링된 것 제외
                already_sampled = sampled_dfs[1] if len(sampled_dfs) > 1 else pd.DataFrame()
                mid_pool = self.df[self.df['depth_range'] == 'mid']
                mid_pool = mid_pool[~mid_pool.index.isin(already_sampled.index)]
                
                if len(mid_pool) >= additional:
                    extra_samples = mid_pool.sample(n=additional, replace=False, random_state=42)
                    sampled_dfs.append(extra_samples)
                    total_sampled += additional
                    print(f"중거리에서 {additional}개 추가 샘플링")
        
        print(f"\n총 샘플링: {total_sampled}개")
        
        # 최종 데이터셋 병합
        representative_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # JSON 형식으로 변환 (ncdb_dataset.py 형식에 맞춤)
        calibration_data = []
        for _, row in representative_df.iterrows():
            calibration_data.append({
                'dataset_root': row['dataset_root'],
                'new_filename': row['filename']
            })
        
        # JSON 저장
        with open(output_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"'{output_file}' 생성 완료 ({len(calibration_data)}개 샘플)")
        
        # 시각화
        self.visualize_distribution(self.df, representative_df, output_file)
        
        return calibration_data
    
    def visualize_distribution(self, original_df, sampled_df, output_file):
        """
        원본 데이터셋과 샘플링된 데이터셋의 깊이 분포를 비교합니다.
        """
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
        
        # 3. Depth Range별 비교 (막대 그래프)
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
        print(f"분포 비교 그래프가 '{plot_file}'에 저장되었습니다.")
        plt.close()

# 실행 예시
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create calibration split from metadata')
    parser.add_argument('--metadata', type=str, required=True,
                        help='메타데이터 CSV 파일 경로')
    parser.add_argument('--target_size', type=int, default=300,
                        help='목표 샘플 개수 (기본: 300)')
    parser.add_argument('--output', type=str, default='calibration_split.json',
                        help='출력 JSON 파일명 (기본: calibration_split.json)')
    
    args = parser.parse_args()
    
    # Calibration Dataset 생성기 초기화
    creator = CalibrationDatasetCreator(args.metadata)
    
    # Stratified Sampling 수행
    creator.create_stratified_split(
        target_size=args.target_size,
        output_file=args.output
    )
```

#### 2.5.2. 실행 방법

```bash
# 앞서 생성한 메타데이터를 사용하여 Calibration Split 생성
python create_calibration_split.py \
    --metadata ncdb_train_metadata.csv \
    --target_size 300 \
    --output calibration_split.json
```

**출력 예시**:
```
총 4856개의 샘플이 메타데이터에 있습니다.

=== Depth Range 분포 ===
near         1823
mid          2456
far           492
very_far       85

비율:
near        0.375
mid         0.506
far         0.101
very_far    0.018

near       (  0.0-  3.0m): 목표  75, 실제  75 (가용 1823)
mid        (  3.0-  8.0m): 목표 120, 실제 120 (가용 2456)
far        (  8.0- 15.0m): 목표  75, 실제  75 (가용  492)
very_far   ( 15.0-100.0m): 목표  30, 실제  30 (가용   85)

총 샘플링: 300개
'calibration_split.json' 생성 완료 (300개 샘플)
분포 비교 그래프가 'calibration_split_distribution.png'에 저장되었습니다.
```

**생성된 `calibration_split.json` 형식**:
```json
[
  {
    "dataset_root": "synced_data/scene_001",
    "new_filename": "frame_0001"
  },
  {
    "dataset_root": "synced_data/scene_003",
    "new_filename": "frame_0125"
  },
  ...
]
```

---

## 3. Weight Normalization

**목적**: NPU의 자동화된 **Cross-Layer Equalization (CLE) 기능의 효과를 극대화**하는 것.

Per-channel Quantization이 미지원되는 상황에서, CLE는 채널 간 가중치 분포의 균형을 맞춰 Per-tensor 양자화의 손실을 줄이는 핵심 기능입니다. 하지만 특정 채널의 가중치 값이 다른 채널에 비해 너무 크거나 작으면 CLE가 제대로 동작하기 어렵습니다.

Weight Normalization은 학습 과정에서 레이어 내 채널 간 가중치 분산이 너무 커지지 않도록 규제(Regularization)를 가하여, CLE가 더 쉽게 동작할 수 있는 모델을 만들어 줍니다.

### 3.1. 구현 예시: Regularization Loss 추가

기존 학습 코드의 Loss 함수에 Weight Normalization을 위한 Regularization 항을 추가합니다.

```python
# packnet_sfm/losses.py 또는 학습 스크립트의 train_step 함수에 추가

def train_step(batch, model, optimizer):
    # ... 기존 학습 과정 ...
    outputs = model(batch['rgb'])
    photometric_loss, ssim_loss, smoothness_loss = calculate_base_losses(outputs, batch)
    
    # --- Weight Normalization Loss 추가 ---
    weight_norm_loss = 0.0
    # 하이퍼파라미터. 1e-7 ~ 1e-9 사이에서 튜닝
    wn_lambda = 1e-8 
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1: # Conv 레이어 등의 가중치에만 적용
            # 채널 간 분산 계산 (가중치를 [out_channels, -1]로 reshape)
            weights = param.view(param.shape[0], -1)
            # 각 채널별 가중치의 평균값
            channel_means = weights.mean(dim=1, keepdim=True)
            # 채널 간 평균값의 분산
            inter_channel_variance = torch.var(channel_means)
            
            weight_norm_loss += inter_channel_variance

    total_loss = photometric_loss + ssim_loss + smoothness_loss + (wn_lambda * weight_norm_loss)
    
    # ... 후속 학습 과정 ...
    total_loss.backward()
    optimizer.step()
    
    return total_loss, ...
```

### 3.2. 적용 시점

- **Full-training**: 학습 초기부터 Weight Normalization Loss를 추가하여 가중치 분포가 안정적으로 학습되도록 유도하는 것이 가장 이상적입니다.
- **Fine-tuning (QAF)**: 이미 학습된 모델에 대해 QAF를 수행할 때, 위 Loss를 추가하여 짧은 epoch 동안 가중치 분포를 미세 조정할 수 있습니다.

---

## 4. Action Plan

### Step 1: 데이터셋 메타데이터 생성 (1-2시간)

1.  **NCDB 데이터셋 구조 확인**:
    ```bash
    # NCDB 데이터셋의 실제 구조를 확인
    ls -la /data/ncdb/
    ls -la /data/ncdb/synced_data/scene_001/
    
    # Split 파일 확인
    cat /data/ncdb/train_split.json | head -n 20
    ```

2.  **메타데이터 생성 스크립트 실행**:
    ```bash
    # create_ncdb_metadata.py를 실행하여 ncdb_train_metadata.csv 생성
    python create_ncdb_metadata.py \
        --dataset_root /data/ncdb \
        --split_file train_split.json \
        --output ncdb_train_metadata.csv
    ```

3.  **메타데이터 검증**:
    ```bash
    # CSV 파일 확인
    head -n 10 ncdb_train_metadata.csv
    wc -l ncdb_train_metadata.csv
    
    # 통계 확인 (Python)
    python -c "
    import pandas as pd
    df = pd.read_csv('ncdb_train_metadata.csv')
    print(df.describe())
    print('\nDepth variant 분포:')
    print(df['depth_variant'].value_counts())
    "
    ```

### Step 2: Representative Calibration Dataset 생성 (30분)

1.  **Calibration Split 생성**:
    ```bash
    # create_calibration_split.py를 실행하여 calibration_split.json 생성
    python create_calibration_split.py \
        --metadata ncdb_train_metadata.csv \
        --target_size 300 \
        --output calibration_split.json
    ```

2.  **생성된 파일 확인**:
    ```bash
    # 샘플 개수 확인
    python -c "import json; data=json.load(open('calibration_split.json')); print(f'총 {len(data)}개 샘플')"
    
    # 처음 10개 샘플 확인
    cat calibration_split.json | head -n 30
    ```

3.  **분포 시각화 확인**:
    - `calibration_split_distribution.png` 파일을 열어서 원본 데이터셋과 샘플링된 데이터셋의 깊이 분포가 유사한지 확인합니다.
    - 만약 분포가 너무 다르다면, `create_calibration_split.py`의 `sampling_ratios`를 조정합니다:
      ```python
      # 예: 근거리를 더 늘리고 싶다면
      creator.create_stratified_split(
          target_size=300,
          sampling_ratios=[0.35, 0.35, 0.20, 0.10]  # 근거리 35%, 중거리 35%
      )
      ```

### Step 3: NPU PTQ에 Calibration Dataset 적용 (2-3시간)

1.  **NCDB Dataset으로 Calibration 수행**:
    ```bash
    # ONNX 모델을 NPU INT8로 변환할 때 calibration_split.json 사용
    # 실제 NPU 툴체인 명령어는 NPU 벤더에 따라 다름 (예시)
    
    # 방법 1: NPU 툴체인이 JSON split을 직접 지원하는 경우
    npu_quantize \
        --model resnetsan.onnx \
        --output resnetsan_int8.bin \
        --calibration_split calibration_split.json \
        --dataset_root /data/ncdb
    
    # 방법 2: 별도 calibration dataset 추출이 필요한 경우
    # (다음 섹션 참조)
    ```

2.  **Calibration Dataset 이미지 추출 (필요 시)**:
    
    일부 NPU 툴체인은 이미지 파일 경로 리스트만 받을 수 있습니다. 이 경우 다음 스크립트를 사용합니다:
    
    ```python
    # extract_calibration_images.py
    import json
    import shutil
    from pathlib import Path
    from tqdm import tqdm
    
    def extract_calibration_images(
        dataset_root,
        calibration_split_json,
        output_dir='calibration_images'
    ):
        """
        calibration_split.json에서 이미지만 별도 디렉토리로 복사
        """
        dataset_root = Path(dataset_root)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        with open(calibration_split_json, 'r') as f:
            split_data = json.load(f)
        
        print(f"총 {len(split_data)}개 이미지 복사 중...")
        
        image_list = []
        for entry in tqdm(split_data):
            base_dir = dataset_root / entry['dataset_root']
            stem = entry['new_filename']
            
            src_path = base_dir / 'image_a6' / f"{stem}.png"
            dst_path = output_dir / f"{stem}.png"
            
            if src_path.exists():
                shutil.copy(src_path, dst_path)
                image_list.append(str(dst_path.absolute()))
        
        # 이미지 경로 리스트 저장
        list_file = output_dir / 'image_list.txt'
        with open(list_file, 'w') as f:
            for img_path in image_list:
                f.write(f"{img_path}\n")
        
        print(f"이미지 복사 완료: {len(image_list)}개")
        print(f"이미지 리스트: {list_file}")
        
        return image_list
    
    if __name__ == '__main__':
        extract_calibration_images(
            dataset_root='/data/ncdb',
            calibration_split_json='calibration_split.json',
            output_dir='calibration_images'
        )
    ```
    
    실행:
    ```bash
    python extract_calibration_images.py
    
    # NPU 툴체인에 이미지 리스트 전달
    npu_quantize \
        --model resnetsan.onnx \
        --output resnetsan_int8.bin \
        --calibration_images calibration_images/image_list.txt
    ```

3.  **성능 평가**:
    ```bash
    # INT8 모델로 NCDB validation set 평가
    python scripts/infer.py \
        --checkpoint resnetsan_int8.bin \
        --input /data/ncdb/val_split.json \
        --output results_int8/
    
    # Metric 계산
    python scripts/eval_depth.py \
        --pred_dir results_int8/ \
        --gt_dir /data/ncdb/synced_data/
    ```

4.  **결과 비교**:
    - **Before** (100 samples): `abs_rel = 0.1133`
    - **Target** (300 samples): `abs_rel ≈ 0.085`
    - 만약 목표에 도달하지 못했다면:
      - Calibration 데이터셋 크기를 400~500개로 늘려보기
      - Depth variant 우선순위 확인 (`newest_depth_maps` 사용 중인지)
      - 데이터 다양성 재검토 (특정 범위가 부족한지 확인)

### Step 4: Weight Normalization 적용 (선택적, 1-2일)

만약 Step 3의 결과가 기대에 미치지 못한다면, Weight Normalization을 적용합니다.

1.  **학습 코드 수정**:
    ```python
    # packnet_sfm/trainers/horovod_trainer.py 또는 해당 학습 스크립트
    # _train_epoch 또는 train_step 함수에 Weight Normalization Loss 추가
    
    # 예시: _train_step 메서드 수정
    def _train_step(self, batch, epoch):
        # ... 기존 학습 과정 ...
        
        # Weight Normalization Loss 추가
        weight_norm_loss = 0.0
        wn_lambda = self.config.get('weight_norm_lambda', 1e-8)
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                weights = param.view(param.shape[0], -1)
                channel_means = weights.mean(dim=1, keepdim=True)
                inter_channel_variance = torch.var(channel_means)
                weight_norm_loss += inter_channel_variance
        
        # Total loss에 추가
        total_loss = loss + wn_lambda * weight_norm_loss
        
        return total_loss
    ```

2.  **Config 파일 수정**:
    ```yaml
    # configs/train_resnet_san_ncdb_weight_norm.yaml
    
    # 기존 설정 상속
    defaults:
      - train_resnet_san_ncdb_640x384
    
    # Weight Normalization 추가
    weight_norm_lambda: 1e-8  # 1e-7 ~ 1e-9 사이에서 튜닝
    
    # Fine-tuning 설정
    epochs: 5
    learning_rate: 1e-6
    ```

3.  **Fine-tuning 실행**:
    ```bash
    # 기존 체크포인트에서 시작하여 짧게 Fine-tuning
    python scripts/train.py \
        --config configs/train_resnet_san_ncdb_weight_norm.yaml \
        --checkpoint checkpoints/resnetsan01_640x384.ckpt
    ```

4.  **PTQ 재수행**:
    ```bash
    # Fine-tuned 모델을 ONNX로 export
    python scripts/export_onnx.py \
        --checkpoint checkpoints/resnetsan01_weight_norm_finetuned.ckpt \
        --output resnetsan_weight_norm.onnx
    
    # NPU PTQ 재수행 (동일한 calibration_split.json 사용)
    npu_quantize \
        --model resnetsan_weight_norm.onnx \
        --output resnetsan_weight_norm_int8.bin \
        --calibration_split calibration_split.json
    ```

5.  **성능 재평가**:
    - Weight Normalization이 적용된 모델의 INT8 성능을 측정합니다.
    - `wn_lambda` 값을 조정하며 최적값을 찾습니다 (1e-7, 1e-8, 1e-9 등).

---

## 5. 트러블슈팅

### 5.1. "특정 Depth Range에 샘플이 부족합니다" 오류

**원인**: 데이터셋에 해당 범위의 샘플이 충분하지 않음.

**해결책**:
```python
# create_calibration_split.py에서 샘플링 비율 조정
creator.create_stratified_split(
    target_size=300,
    sampling_ratios=[0.30, 0.50, 0.15, 0.05]  # 부족한 범위 비율 낮춤
)
```

### 5.2. "Depth 파일을 찾을 수 없습니다" 오류

**원인**: 
- Depth variant가 실제 데이터셋 구조와 맞지 않음
- JSON split 파일의 경로가 잘못됨

**해결책**:
```bash
# 1. 실제 depth 디렉토리 확인
ls /data/ncdb/synced_data/scene_001/

# 2. 환경 변수로 depth variant 지정
export NCDB_DEPTH_VARIANT=newest_depth_maps

# 3. 또는 메타데이터 생성 시 명시
python create_ncdb_metadata.py \
    --dataset_root /data/ncdb \
    --split_file train_split.json \
    --depth_variants newest_depth_maps,new_depth_maps
```

### 5.3. PTQ 성능이 여전히 낮음 (abs_rel > 0.09)

**가능한 원인**:
1. Calibration 데이터셋의 다양성이 부족
2. NPU가 Asymmetric Quantization을 지원하지 않음
3. 모델 자체가 양자화에 취약한 구조

**해결책**:

**A. 데이터 다양성 재검토**:
```python
# Scene 타입도 고려한 샘플링
# create_calibration_split.py에 추가
def create_balanced_split(self, target_size=300):
    # Depth range와 Scene type 모두 고려
    stratified = []
    for depth_range in ['near', 'mid', 'far']:
        for scene_type in ['indoor', 'outdoor_near', 'outdoor_far']:
            subset = self.df[
                (self.df['depth_range'] == depth_range) & 
                (self.df['scene_type'] == scene_type)
            ]
            n_samples = int(target_size * weight_matrix[depth_range][scene_type])
            if len(subset) >= n_samples:
                stratified.append(subset.sample(n=n_samples))
```

**B. NPU 스펙 재확인**:
```bash
# NPU 벤더에 문의
# - Asymmetric Quantization 지원 여부
# - Per-channel Quantization 지원 여부 (재확인)
# - FP16 Mixed Precision 지원 여부
```

**C. 다음 전략으로 진행**:
- Weight Normalization (현재 섹션)
- 전략 2: Dual-Head 모델 (INT8_OPTIMIZATION_STRATEGY.md 참조)

---

## 6. 예상 결과

| 단계 | Calibration 설정 | abs_rel (예상) | 개선율 |
|------|-----------------|---------------|--------|
| Baseline | 100 samples (random) | 0.1133 | - |
| **Step 2-3** | **300 samples (stratified)** | **~0.085** | **25%** |
| Step 2-3 (확장) | 500 samples (stratified) | ~0.075 | 34% |
| Step 4 추가 | 300 samples + Weight Norm | ~0.070 | 38% |

**성공 기준**:
- ✅ `abs_rel < 0.090`: Phase 1 성공 → Phase 2 (Dual-Head) 진행
- ⚠️ `abs_rel 0.090~0.100`: 데이터셋 크기를 500개로 확대 또는 Weight Normalization 적용
- ❌ `abs_rel > 0.100`: NPU 스펙 재확인 필요, 또는 즉시 Phase 2로 진행

**이 전략을 통해 우리는 NPU라는 '블랙박스'에 최상의 재료를 제공하여, 최종적으로 양자화 성능을 극대화할 수 있습니다.**
