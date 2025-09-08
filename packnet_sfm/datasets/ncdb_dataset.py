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
    
    Uses FisheyeCamera for accurate fisheye lens distortion modeling.
    """
    
    def __init__(self, dataset_root, split_file, transform=None, mask_file=None,
                 back_context=0, forward_context=0, strides=(1,), 
                 with_context=False, with_depth=True, **kwargs):
        super().__init__()
        
        # Dataset paths
        self.dataset_root = Path(dataset_root)
        
        # Context parameters (KITTI style)
        self.backward_context = back_context
        self.forward_context = forward_context
        self.strides = strides
        self.with_context = with_context or (back_context > 0 or forward_context > 0)
        self.with_depth = with_depth
        
        # Context path storage
        self.backward_context_paths = []
        self.forward_context_paths = []
        
        # Cache for file validation
        self._file_cache = {}
        self._folder_cache = {}
        
        # Load split file
        self._load_split_file(split_file)
        
        # Load mask if provided
        self.mask = None
        if mask_file:
            absolute_mask_path = self.dataset_root / mask_file
            if absolute_mask_path.exists():
                self.mask = (np.array(Image.open(absolute_mask_path).convert('L')) > 0).astype(np.uint8)
        
        # Transform
        self.transform = transform
        
        # Filter paths with context if needed
        if self.with_context:
            self._filter_paths_with_context()
    
    def _load_split_file(self, split_file):
        """Load and validate split file"""
        absolute_split_path = self.dataset_root / split_file
        if not absolute_split_path.exists():
            raise FileNotFoundError(f"Split file not found: {absolute_split_path}")
        
        with open(absolute_split_path, 'r') as f:
            mapping_data = json.load(f)
        
        if not isinstance(mapping_data, list):
            raise ValueError("Split file must contain a list of entries")
        
        self.data_entries = mapping_data
        print(f"Loaded {len(self.data_entries)} entries from {split_file}")
    
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
    
    def _check_sample_exists(self, idx):
        """Check if sample files exist"""
        if idx in self._file_cache:
            return self._file_cache[idx]
        
        entry = self.data_entries[idx]
        stem = entry['new_filename']
        
        # Check image file
        image_path = self.dataset_root / entry['dataset_root'] / 'image_a6' / f"{stem}.png"
        if not image_path.exists():
            self._file_cache[idx] = False
            return False
        
        # Check depth file if needed
        if self.with_depth:
            depth_path = self.dataset_root / entry['dataset_root'] / 'newest_depth_maps' / f"{stem}.png"
            if not depth_path.exists():
                self._file_cache[idx] = False
                return False
        
        self._file_cache[idx] = True
        return True
    
    def __len__(self):
        return len(self.data_entries)
    
    def __getitem__(self, idx):
        entry = self.data_entries[idx]
        stem = entry['new_filename']
        
        # Construct paths
        image_path = self.dataset_root / entry['dataset_root'] / 'image_a6' / f"{stem}.png"
        depth_path = self.dataset_root / entry['dataset_root'] / 'newest_depth_maps' / f"{stem}.png"
        
        # Load image
        image = load_image(str(image_path))
        W, H = image.size
        
        # Load depth if needed (안전하게 체크)
        depth_gt = None
        if hasattr(self, 'with_depth') and self.with_depth:
            try:
                depth_png = Image.open(depth_path)
                depth_gt_png = np.asarray(depth_png, dtype=np.uint16)
                depth_gt = depth_gt_png.astype(np.float32)
            except (FileNotFoundError, OSError) as e:
                print(f"Warning: Could not load depth file {depth_path}: {e}")
                depth_gt = None
        
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
                depth_gt = depth_gt * self.mask
        
        # Build sample
        sample = {
            'rgb': image,
            'idx': idx,
            'camera': camera,  # ✅ FisheyeCamera 객체 추가
            'intrinsics': intrinsics_list,
            'distortion_coeffs': distortion_coeffs,
            'extrinsic': extrinsic_matrix,
            'lidar_to_world': lidar_to_world_matrix,
            'filename': stem,
            'meta': {
                'image_path': str(image_path),
                'depth_path': str(depth_path) if hasattr(self, 'with_depth') and self.with_depth else None,
                'calibration_source': 'hardcoded_default',
            }
        }
        
        # Add depth (안전하게 체크)
        if depth_gt is not None:
            sample['depth'] = depth_gt
        
        # Add mask
        if self.mask is not None:
            sample['mask'] = self.mask
        
        # Add context frames with FisheyeCamera
        if hasattr(self, 'with_context') and self.with_context and idx < len(self.backward_context_paths):
            context_images = []
            context_cameras = []  # ✅ Context 카메라들도 FisheyeCamera
            
            # Load backward context
            for context_idx in self.backward_context_paths[idx]:
                context_entry = self.data_entries[context_idx]
                context_stem = context_entry['new_filename']
                context_image_path = self.dataset_root / context_entry['dataset_root'] / 'image_a6' / f"{context_stem}.png"
                
                if context_image_path.exists():
                    context_image = load_image(str(context_image_path))
                    context_images.append(context_image)
                    
                    # ✅ 각 context 프레임도 FisheyeCamera 생성
                    context_camera = FisheyeCamera(
                        intrinsics=distortion_coeffs,  # 동일한 캘리브레이션 사용
                        image_size=(H, W)
                    )
                    context_cameras.append(context_camera)
            
            # Load forward context
            for context_idx in self.forward_context_paths[idx]:
                context_entry = self.data_entries[context_idx]
                context_stem = context_entry['new_filename']
                context_image_path = self.dataset_root / context_entry['dataset_root'] / 'image_a6' / f"{context_stem}.png"
                
                if context_image_path.exists():
                    context_image = load_image(str(context_image_path))
                    context_images.append(context_image)
                    
                    # ✅ 각 context 프레임도 FisheyeCamera 생성
                    context_camera = FisheyeCamera(
                        intrinsics=distortion_coeffs,  # 동일한 캘리브레이션 사용
                        image_size=(H, W)
                    )
                    context_cameras.append(context_camera)
            
            if context_images:
                sample['rgb_context'] = context_images
                sample['camera_context'] = context_cameras  # ✅ Context 카메라들 추가
        
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
        
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