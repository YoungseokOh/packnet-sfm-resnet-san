import os
import json
from pathlib import Path
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from packnet_sfm.datasets.transforms import get_transforms
from packnet_sfm.utils.image import load_image
from packnet_sfm.geometry.camera import FisheyeCamera  # ✅ 추가

# 캘리브레이션 데이터 하드코딩 (camera_lidar_projector.py에서 가져옴)
DEFAULT_CALIB_A6 = {
    "model": "vadas",
    "intrinsic": [-0.0004, 1.0136, -0.0623, 0.2852, -0.332, 0.1896, -0.0391,
                  1.0447, 0.0021, 44.9516, 2.48822, 0, 0.9965, -0.0067,
                  -0.0956, 0.1006, -0.054, 0.0106],
    "extrinsic": [0.0900425, -0.00450864, -0.356367, 0.00100918, -0.236104, -0.0219886],
    "image_size": None
}

DEFAULT_LIDAR_TO_WORLD = np.array([
    [-0.998752, -0.00237052, -0.0498847,  0.0375091],
    [ 0.00167658, -0.999901,   0.0139481,  0.0349093],
    [-0.0499128,  0.0138471,   0.998658,   0.771878],
    [ 0.,         0.,          0.,         1.       ]
])

class NcdbDataset(Dataset):
    """
    NCDB Dataset for Semi-supervised Learning with FisheyeCamera support.
    """
    # 🔔 해상도 감지 로그: 프로세스당 한 번만 출력
    _RESOLUTION_LOG_SHOWN = False

    # ✅ depth_type → 폴더명 매핑
    DEPTH_TYPE_MAPPING = {
        'distance': 'newest_distance_maps',           # 카메라 중심에서 3D 유클리드 거리
        'depth': 'newest_original_depth_maps',        # Z-axis depth (synthetic 없음, 원본)
        'depth_synthetic': 'newest_depth_maps',       # Z-axis depth (synthetic 포함)
    }
    
    # 기본값: distance (3D 유클리드 거리)
    DEFAULT_DEPTH_TYPE = 'distance'

    # ✅ 공통 PNG 로더: 항상 (미터*256) 규약이면 /256 복원 (default)
    def _load_depth_png(self, depth_path: Path):
        try:
            depth_png = Image.open(depth_path)
            arr16 = np.asarray(depth_png, dtype=np.uint16)
            depth = arr16.astype(np.float32)
            
            # 1. KITTI 스타일로 256으로 나누기
            if depth.max() > 255:
                depth /= 256.0
            
            # 2. 유효하지 않은 픽셀(값이 0이었던 부분)을 -1로 마스킹
            depth[arr16 == 0] = 0
            
            return depth
        except (FileNotFoundError, OSError) as e:
            print(f"[NcdbDataset] Depth load failed: {depth_path} ({e})")
            return None

    def __init__(self, dataset_root, split_file, transform=None, mask_file=None,
                 back_context=0, forward_context=0, strides=(1,), 
                 with_context=False, with_depth=True,
                 depth_type=None,        # ✅ 'distance', 'depth', or 'depth_synthetic' (config에서 설정)
                 depth_variants=None,    # str | list | None (하위호환용, depth_type 우선)
                 strict_depth=True,      # ⚠️ 기본 True: depth 파일 없으면 예외 발생
                 use_mask: bool = False, # ← 추가: 마스크 사용 여부 (기본 미사용)
                 min_depth: float = None,  # ← GT depth 필터링 최소값
                 max_depth: float = None,  # ← GT depth 필터링 최대값
                 dry_run: bool = False,  # ✅ dry-run 모드: 폴더 존재 여부 검증만 수행
                 **kwargs):
        super().__init__()
        
        # Dataset paths
        self.dataset_root = Path(dataset_root)
        self.use_mask = bool(use_mask)
        
        # Depth range for GT filtering
        self.min_depth = min_depth
        self.max_depth = max_depth
        if self.min_depth is not None or self.max_depth is not None:
            print(f"[NcdbDataset] GT depth filtering enabled: "
                  f"min_depth={self.min_depth}, max_depth={self.max_depth}")

        # Context parameters (KITTI style)
        self.backward_context = back_context
        self.forward_context = forward_context
        self.strides = strides
        self.with_context = with_context or (back_context > 0 or forward_context > 0)
        self.with_depth = with_depth
        
        # ✅ depth_type 처리 (config에서 설정 가능)
        # 우선순위: depth_type > 환경변수 > depth_variants > 기본값
        env_depth_type = os.getenv('NCDB_DEPTH_TYPE', '').strip().lower()
        env_variant = os.getenv('NCDB_DEPTH_VARIANT', '').strip()
        
        if depth_type is not None:
            # config에서 depth_type 지정
            depth_type = depth_type.lower()
            if depth_type not in self.DEPTH_TYPE_MAPPING:
                raise ValueError(f"[NcdbDataset] Invalid depth_type: '{depth_type}'. "
                               f"Valid options: {list(self.DEPTH_TYPE_MAPPING.keys())}")
            self.depth_type = depth_type
            self.depth_variants = [self.DEPTH_TYPE_MAPPING[depth_type]]
        elif env_depth_type:
            # 환경변수에서 depth_type 지정
            if env_depth_type not in self.DEPTH_TYPE_MAPPING:
                raise ValueError(f"[NcdbDataset] Invalid NCDB_DEPTH_TYPE: '{env_depth_type}'. "
                               f"Valid options: {list(self.DEPTH_TYPE_MAPPING.keys())}")
            self.depth_type = env_depth_type
            self.depth_variants = [self.DEPTH_TYPE_MAPPING[env_depth_type]]
        elif env_variant:
            # 환경변수에서 직접 폴더명 지정 (하위호환)
            self.depth_type = None
            self.depth_variants = [v.strip() for v in env_variant.split(',') if v.strip()]
        elif depth_variants is not None:
            # depth_variants 직접 지정 (하위호환)
            self.depth_type = None
            if isinstance(depth_variants, str):
                self.depth_variants = [depth_variants]
            else:
                self.depth_variants = list(depth_variants)
        else:
            # 기본값 사용
            self.depth_type = self.DEFAULT_DEPTH_TYPE
            self.depth_variants = [self.DEPTH_TYPE_MAPPING[self.DEFAULT_DEPTH_TYPE]]
        
        self.strict_depth = strict_depth
        
        # 중복 제거 (앞쪽 우선순위 유지)
        seen = set()
        ordered = []
        for v in self.depth_variants:
            if v not in seen:
                ordered.append(v)
                seen.add(v)
        self.depth_variants = ordered
        
        # ✅ 명확한 로그 출력
        if self.depth_type:
            print(f"[NcdbDataset] 📊 depth_type='{self.depth_type}' → folder: {self.depth_variants[0]}")
        else:
            print(f"[NcdbDataset] Using depth variants (priority order): {self.depth_variants}")
        
        # ✅ dry_run 모드 저장
        self.dry_run = dry_run
        
        # Context path storage
        self.backward_context_paths = []
        self.forward_context_paths = []
        
        # Cache for file validation
        self._file_cache = {}
        self._folder_cache = {}
        
        # Load split file
        self._load_split_file(split_file)
        
        # ✅ dry-run 모드: 폴더 및 파일 존재 여부 검증
        if dry_run:
            self._validate_depth_folders()
            return  # dry-run 모드에서는 여기서 종료
        
        # Load mask if provided (0/1 binary), but apply only when use_mask=True
        self.mask = None
        if mask_file:
            absolute_mask_path = self.dataset_root / mask_file
            if absolute_mask_path.exists():
                self.mask = (np.array(Image.open(absolute_mask_path).convert('L')) > 0).astype(np.uint8)
                print(f"[NcdbDataset] Loaded mask (0/1) shape={self.mask.shape} | use_mask={self.use_mask}")
        
        # Transform
        self.transform = transform
        
        # Filter paths with context if needed
        if self.with_context:
            self._filter_paths_with_context()

        # 🔔 메인 프로세스에서 한 번만 입력 해상도 로그
        self._log_input_resolution_once()
    
    def _load_split_file(self, split_file):
        """Load and validate split file"""
        absolute_split_path = self.dataset_root / split_file
        if Path(split_file).is_absolute():
            absolute_split_path = Path(split_file)

        if not absolute_split_path.exists():
            raise FileNotFoundError(f"Split file not found: {absolute_split_path}")
        
        with open(absolute_split_path, 'r') as f:
            mapping_data = json.load(f)
        
        if not isinstance(mapping_data, list):
            raise ValueError("Split file must contain a list of entries")
        
        # 표준화: image_path만 있는 항목을 (dataset_root, new_filename)로 변환
        normalized = []
        converted = 0
        for item in mapping_data:
            if 'dataset_root' in item and 'new_filename' in item:
                normalized.append({'dataset_root': item['dataset_root'],
                                   'new_filename': item['new_filename']})
                continue
            if 'image_path' in item:
                p = Path(item['image_path'])
                stem = p.stem  # 파일 스템
                base_dir = p.parent
                if base_dir.name == 'image_a6':
                    base_dir = base_dir.parent  # .../synced_data

                # self.dataset_root 기준 상대경로로 만들기 시도
                try:
                    rel_base = str(base_dir.relative_to(self.dataset_root))
                except Exception:
                    # 루트 밖 절대 경로면 그대로 저장(절대 경로 우선)
                    rel_base = str(base_dir)
                normalized.append({'dataset_root': rel_base, 'new_filename': stem})
                converted += 1
                continue
            raise ValueError(f"Split entry missing required keys: {list(item.keys())}")
        
        self.data_entries = normalized
        print(f"Loaded {len(self.data_entries)} entries from {absolute_split_path} (converted {converted} from image_path)")
    
    def _validate_depth_folders(self):
        """
        ✅ Dry-run validation: 폴더 존재 여부 및 파일 매칭 검증
        """
        print(f"\n{'='*60}")
        print(f"🔍 DRY-RUN VALIDATION")
        print(f"{'='*60}")
        print(f"Dataset root: {self.dataset_root}")
        print(f"Depth type: {self.depth_type}")
        print(f"Target folder: {self.depth_variants[0]}")
        print(f"Total entries: {len(self.data_entries)}")
        print(f"{'='*60}\n")
        
        # 1. 고유한 dataset_root 경로들 수집
        unique_roots = set(entry['dataset_root'] for entry in self.data_entries)
        print(f"📂 Unique dataset roots: {len(unique_roots)}")
        
        # 2. 각 root에서 depth 폴더 존재 여부 확인
        folder_status = {}
        for root in sorted(unique_roots):
            base_path = self.dataset_root / root
            depth_folder = base_path / self.depth_variants[0]
            
            if depth_folder.exists():
                # 폴더 내 파일 수 확인
                png_files = list(depth_folder.glob('*.png'))
                folder_status[root] = {
                    'exists': True,
                    'file_count': len(png_files),
                    'path': str(depth_folder)
                }
            else:
                folder_status[root] = {
                    'exists': False,
                    'file_count': 0,
                    'path': str(depth_folder)
                }
        
        # 3. 결과 출력
        exists_count = sum(1 for s in folder_status.values() if s['exists'])
        missing_count = len(folder_status) - exists_count
        
        print(f"\n📊 Folder Status:")
        print(f"   ✅ Found: {exists_count}/{len(folder_status)}")
        print(f"   ❌ Missing: {missing_count}/{len(folder_status)}")
        
        if missing_count > 0:
            print(f"\n⚠️  Missing folders:")
            for root, status in folder_status.items():
                if not status['exists']:
                    print(f"   ❌ {status['path']}")
        
        # 4. 샘플 단위 파일 매칭 검증 (처음 10개 + 랜덤 10개)
        print(f"\n📁 Sample file validation (first 10 entries):")
        matched = 0
        missing_files = []
        
        check_entries = self.data_entries[:min(10, len(self.data_entries))]
        for entry in check_entries:
            stem = entry['new_filename']
            depth_path = self.dataset_root / entry['dataset_root'] / self.depth_variants[0] / f"{stem}.png"
            rgb_path = self.dataset_root / entry['dataset_root'] / 'image_a6' / f"{stem}.png"
            if not rgb_path.exists():
                rgb_path = self.dataset_root / entry['dataset_root'] / 'image_a6' / f"{stem}.jpg"
            
            rgb_ok = rgb_path.exists()
            depth_ok = depth_path.exists()
            
            if rgb_ok and depth_ok:
                matched += 1
                status = "✅"
            else:
                status = "❌"
                missing_files.append({
                    'stem': stem,
                    'rgb': str(rgb_path) if not rgb_ok else None,
                    'depth': str(depth_path) if not depth_ok else None
                })
            
            print(f"   {status} {stem}: RGB={'✓' if rgb_ok else '✗'}, Depth={'✓' if depth_ok else '✗'}")
        
        # 5. 전체 파일 매칭률 계산 (샘플링)
        import random
        sample_size = min(100, len(self.data_entries))
        sample_entries = random.sample(self.data_entries, sample_size)
        
        total_matched = 0
        for entry in sample_entries:
            stem = entry['new_filename']
            depth_path = self.dataset_root / entry['dataset_root'] / self.depth_variants[0] / f"{stem}.png"
            if depth_path.exists():
                total_matched += 1
        
        match_rate = (total_matched / sample_size) * 100
        
        print(f"\n📈 Estimated match rate (sampled {sample_size} entries):")
        print(f"   Depth files found: {total_matched}/{sample_size} ({match_rate:.1f}%)")
        
        # 6. 최종 요약
        print(f"\n{'='*60}")
        if missing_count == 0 and match_rate >= 95:
            print(f"✅ VALIDATION PASSED")
            print(f"   depth_type='{self.depth_type}' → '{self.depth_variants[0]}'")
        else:
            print(f"⚠️  VALIDATION WARNING")
            if missing_count > 0:
                print(f"   - {missing_count} depth folders missing")
            if match_rate < 95:
                print(f"   - Low match rate: {match_rate:.1f}%")
        print(f"{'='*60}\n")
    
    def _filter_paths_with_context(self):
        """Filter paths that have valid context frames (KITTI style)"""
        print(f"Filtering for context (backward={self.backward_context}, forward={self.forward_context})")
        
        valid_entries = []
        valid_backward_contexts = []
        valid_forward_contexts = []
        
        for stride in self.strides:
            for idx, entry in enumerate(self.data_entries):
                backward_context, forward_context = self._get_sample_context(
                    idx, self.backward_context, self.forward_context, stride)
                
                if backward_context is not None:
                    valid_entries.append(entry)
                    valid_backward_contexts.append(backward_context)
                    valid_forward_contexts.append(forward_context)
        
        self.data_entries = valid_entries
        self.backward_context_paths = valid_backward_contexts
        self.forward_context_paths = valid_forward_contexts
        
        print(f"After context filtering: {len(self.data_entries)} valid samples")
    
    def _get_sample_context(self, idx, backward_context, forward_context, stride=1):
        """Get context frames for a sample (KITTI style)"""
        max_idx = len(self.data_entries) - 1
        
        # Check bounds
        if idx - backward_context * stride < 0 or idx + forward_context * stride > max_idx:
            return None, None
        
        # Validate backward context
        backward_context_files = []
        for offset in range(-backward_context, 0):
            context_idx = idx + offset * stride
            if not self._check_sample_exists(context_idx):
                return None, None
            backward_context_files.append(context_idx)
        
        # Validate forward context
        forward_context_files = []
        for offset in range(1, forward_context + 1):
            context_idx = idx + offset * stride
            if not self._check_sample_exists(context_idx):
                return None, None
            forward_context_files.append(context_idx)
        
        return backward_context_files, forward_context_files
    
    def _resolve_depth_path(self, entry, stem):
        """
        variant 우선순위에 따라 존재하는 depth 경로 반환.
        Returns (path, variant_name) or (None, None)
        """
        base = self.dataset_root / entry['dataset_root']
        for variant in self.depth_variants:
            p = base / variant / f"{stem}.png"
            if p.exists():
                return p, variant
        return None, None
    
    def _check_sample_exists(self, idx):
         """Check if sample files exist"""
         if idx in self._file_cache:
             return self._file_cache[idx]
         entry = self.data_entries[idx]
         stem = entry['new_filename']
         image_path = self.dataset_root / entry['dataset_root'] / 'image_a6' / f"{stem}.png"
         if not image_path.exists():
             image_path = self.dataset_root / entry['dataset_root'] / 'image_a6' / f"{stem}.jpg"
         if not image_path.exists():
             self._file_cache[idx] = False
             return False
         if self.with_depth:
            depth_path, variant = self._resolve_depth_path(entry, stem)
            if depth_path is None:
                if self.strict_depth:
                    self._file_cache[idx] = False
                    return False
                # strict가 아니면 depth 없이도 샘플 사용
         self._file_cache[idx] = True
         return True
    
    def __len__(self):
        return len(self.data_entries)
    
    def __getitem__(self, idx):
        entry = self.data_entries[idx]
        stem = entry['new_filename']
        
        # Construct paths (try .png first, fallback to .jpg)
        image_path = self.dataset_root / entry['dataset_root'] / 'image_a6' / f"{stem}.png"
        if not image_path.exists():
            image_path = self.dataset_root / entry['dataset_root'] / 'image_a6' / f"{stem}.jpg"
        
        depth_path = None
        depth_variant = None
        if self.with_depth:
            depth_path, depth_variant = self._resolve_depth_path(entry, stem)
            if depth_path is None and self.strict_depth:
                base_path = self.dataset_root / entry['dataset_root']
                raise FileNotFoundError(
                    f"❌ [STRICT] Depth file not found!\n"
                    f"   Sample: {stem}\n"
                    f"   Expected: {base_path / self.depth_variants[0] / f'{stem}.png'}\n"
                    f"   Variants tried: {self.depth_variants}")
        
        # Load image
        image = load_image(str(image_path))
        W, H = image.size
        depth_gt = None
        
        if self.with_depth and depth_path is not None:
            depth_gt = self._load_depth_png(depth_path)
            
            # ✅ GT depth 필터링: min_depth 미만과 max_depth 초과 값을 0으로 설정
            if depth_gt is not None and (self.min_depth is not None or self.max_depth is not None):
                original_valid = (depth_gt > 0).sum()
                
                if self.min_depth is not None:
                    depth_gt[depth_gt < self.min_depth] = 0
                
                if self.max_depth is not None:
                    depth_gt[depth_gt > self.max_depth] = 0
                
        # (추가) Depth 통계 계산
        depth_stats = None
        if depth_gt is not None:
            pos = depth_gt[depth_gt > 0]
            if pos.size > 0:
                depth_stats = {
                    'min': float(depth_gt.min()),
                    'max': float(depth_gt.max()),
                    'p50': float(np.percentile(pos, 50)),
                    'p90': float(np.percentile(pos, 90)),
                    'p95': float(np.percentile(pos, 95)),
                    'count_pos': int(pos.size),
                    'count_all': int(depth_gt.size),
                }
                # ✅ Debug output removed - was causing spam during training
            else:
                depth_stats = {
                    'min': float(depth_gt.min()),
                    'max': float(depth_gt.max()),
                    'p50': 0.0, 'p90': 0.0, 'p95': 0.0,
                    'count_pos': 0,
                    'count_all': int(depth_gt.size)
                }
                # ✅ Debug output removed

        # (선택) Binary mask per-sample 생성 (RGB에는 적용하지 않음)
        mask01 = None
        if self.use_mask and (self.mask is not None):
            if self.mask.shape[:2] != (H, W):
                mask_img = Image.fromarray((self.mask * 255).astype(np.uint8), mode='L')
                mask_img = mask_img.resize((W, H), Image.NEAREST)
                mask01 = (np.array(mask_img) > 0).astype(np.uint8)
            else:
                mask01 = self.mask

            # GT에만 마스크 적용 (0/1 곱)
            if depth_gt is not None:
                depth_gt = depth_gt * mask01

        # Calibration data
        calib_data = DEFAULT_CALIB_A6.copy()
        calib_data['image_size'] = (H, W)
        
        # Process intrinsics for FisheyeCamera
        intrinsics_list = torch.tensor(calib_data['intrinsic'], dtype=torch.float32)
        distortion_coeffs = {
            'k': torch.tensor(calib_data['intrinsic'][0:7], dtype=torch.float32),
            's': torch.tensor(calib_data['intrinsic'][7], dtype=torch.float32),
            'div': torch.tensor(calib_data['intrinsic'][8], dtype=torch.float32),
            'ux': torch.tensor(calib_data['intrinsic'][9], dtype=torch.float32),
            'uy': torch.tensor(calib_data['intrinsic'][10], dtype=torch.float32),
            'image_size': (H, W)
        }
        
        # ✅ FisheyeCamera 객체 생성
        camera = FisheyeCamera(
            intrinsics=distortion_coeffs,
            image_size=(H, W)
        )
        
        # Extrinsic and LiDAR data
        extrinsic_matrix = torch.tensor(calib_data['extrinsic'], dtype=torch.float32)
        lidar_to_world_matrix = torch.tensor(DEFAULT_LIDAR_TO_WORLD, dtype=torch.float32)
        
        # Apply mask if available
        if self.mask is not None:
            if self.mask.shape[:2] != (H, W):
                mask_img = Image.fromarray((self.mask * 255).astype(np.uint8), mode='L')
                mask_img = mask_img.resize((W, H), Image.NEAREST)
                self.mask = np.array(mask_img)
            
            image_np = np.array(image)
            image_np = (image_np * self.mask[:, :, np.newaxis]).astype(image_np.dtype)
            image = Image.fromarray(image_np)
            
            if depth_gt is not None:
                depth_gt = depth_gt * self.mask  # ← 여기가 문제일 가능성
        
        # Build sample
        sample = {
            'rgb': image,
            'idx': idx,
            'camera': camera,
            'intrinsics': intrinsics_list,
            'distortion_coeffs': distortion_coeffs,
            'extrinsic': extrinsic_matrix,
            'lidar_to_world': lidar_to_world_matrix,
            'filename': stem,
            'meta': {
                'image_path': str(image_path),
                'depth_path': str(depth_path) if (self.with_depth and depth_path is not None) else None,
                'depth_variant': depth_variant,
                'calibration_source': 'hardcoded_default',
                'depth_stats': depth_stats,
            }
        }
        
        # Add depth
        if depth_gt is not None:
            sample['depth'] = depth_gt

        # 필요 시 downstream에서 사용할 수 있도록 mask도 전달 (옵션)
        if mask01 is not None:
            sample['mask'] = mask01

        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
            
            # # ★ 디버깅: Transform 후 depth 확인
            # if 'depth' in sample and not hasattr(self, '_transform_depth_logged'):
            #     self._transform_depth_logged = True
            #     d = sample['depth']
            #     print(f"\n[NcdbDataset.__getitem__] After transform:")
            #     if isinstance(d, torch.Tensor):
            #         print(f"  Type: torch.Tensor")
            #         print(f"  Shape: {d.shape}")
            #         print(f"  Max: {d.max().item():.2f}")
            #         print(f"  Min: {d.min().item():.2f}")
            #         valid = d > 0
            #         if valid.any():
            #             print(f"  Valid range: [{d[valid].min().item():.2f}, {d[valid].max().item():.2f}]")
            #     elif isinstance(d, np.ndarray):
            #         print(f"  Type: numpy.ndarray")
            #         print(f"  Shape: {d.shape}")
            #         print(f"  Max: {np.max(d):.2f}")
            #         print(f"  Min: {np.min(d):.2f}")
            #         valid = d > 0
            #         if valid.any():
            #             print(f"  Valid range: [{d[valid].min().item():.2f}, {d[valid].max().item():.2f}]")
        
        return sample

    # ✅ 커스텀 collate_fn 추가: FisheyeCamera를 처리
    @staticmethod
    def custom_collate_fn(batch):
        """
        커스텀 collate 함수: FisheyeCamera 객체를 처리.
        - 배치는 딕셔너리 리스트 형태 (예: [{'rgb': ..., 'camera': FisheyeCamera, ...}]).
        - 'camera'와 'camera_context'는 리스트로 묶음.
        - 다른 키는 default_collate 사용.
        """
        if not batch:
            return {}
        
        # 첫 번째 아이템의 키를 기준으로 처리
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            values = [item[key] for item in batch if key in item]
            
            if key in ['camera', 'camera_context']:
                # FisheyeCamera 객체는 리스트로 묶음
                collated[key] = values
            else:
                # 다른 데이터는 default_collate 사용
                try:
                    collated[key] = default_collate(values)
                except TypeError:
                    # 실패 시 리스트로 유지
                    collated[key] = values
        
        return collated
    
    def _log_input_resolution_once(self):
        """첫 유효 샘플의 W×H를 한 번만 로그로 출력"""
        if NcdbDataset._RESOLUTION_LOG_SHOWN:
            return
        # 첫 유효 엔트리 탐색
        for entry in self.data_entries:
            stem = entry.get('new_filename')
            if not stem:
                continue
            img_path = self.dataset_root / entry['dataset_root'] / 'image_a6' / f"{stem}.png"
            if not img_path.exists():
                img_path = self.dataset_root / entry['dataset_root'] / 'image_a6' / f"{stem}.jpg"
            if not img_path.exists():
                continue
            try:
                with Image.open(img_path) as im:
                    W, H = im.size
            except Exception:
                continue
            msg = f"[NcdbDataset] Detected input resolution {W}x{H} (W×H)"
            if (W, H) == (640, 384):
                msg += " — OK (expected 640x384)."
            print(msg + " [once]")
            NcdbDataset._RESOLUTION_LOG_SHOWN = True
            break