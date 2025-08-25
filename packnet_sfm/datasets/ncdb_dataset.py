import os
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from packnet_sfm.datasets.transforms import get_transforms

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
    Ncdb-Cls-Sample Dataset.

    Parameters
    ----------
    split_file : str
        Path to the split file (e.g., '/workspace/packnet-sfm/splits/combined_train.json').
    transform : callable, optional
        A function/transform that takes in a sample and returns a transformed version.
    mask_file : str, optional
        Path to the binary mask file (e.g., '/workspace/packnet-sfm/ncdb-cls/synced_data/binary_mask.png').
        This should be an absolute path if used.
    """
    def __init__(self, dataset_root, split_file, transform=None, mask_file=None, **kwargs):
        super().__init__()
        self.data_entries = []
        self.dataset_root = Path(dataset_root)

        # Initialize transforms based on kwargs (from config)
        self.transform = transform

        # kwargs에서 transform을 제거하여 중복 전달 방지
        kwargs.pop('transform', None)
        
        # split_file을 dataset_root에 대한 상대 경로로 처리
        absolute_split_path = self.dataset_root / split_file
        if not absolute_split_path.exists():
            raise FileNotFoundError(f"Split file not found at {absolute_split_path}")

        with open(absolute_split_path, 'r') as f:
            mapping_data = json.load(f)
            if isinstance(mapping_data, list):
                self.data_entries = mapping_data
            else:
                raise ValueError(f"Unsupported split file format: {split_file}. Expected a list of dictionaries.")

        self.mask = None
        if mask_file:
            # mask_file을 dataset_root에 대한 상대 경로로 처리
            absolute_mask_path = self.dataset_root / mask_file
            if not absolute_mask_path.exists():
                raise FileNotFoundError(f"Mask file not found at {absolute_mask_path}")
            self.mask = (np.array(Image.open(absolute_mask_path).convert('L')) > 0).astype(np.uint8)

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        entry = self.data_entries[idx]
        stem = entry['new_filename']
        # dataset_root = Path(entry['dataset_root']) # 이 줄은 더 이상 필요 없음
        
        # Construct paths using self.dataset_root
        image_path = self.dataset_root / entry['dataset_root'] / 'image_a6' / f"{stem}.jpg"
        depth_path = self.dataset_root / entry['dataset_root'] / 'depth_maps' / f"{stem}.png"

        # 캘리브레이션 데이터는 하드코딩된 DEFAULT_CALIB_A6 사용
        calib_data = DEFAULT_CALIB_A6

        # Load data
        try:
            image = Image.open(image_path).convert('RGB')
            depth_png = Image.open(depth_path)
            
            # 🆕 이미지 크기 추출
            W, H = image.size # PIL Image.size returns (width, height)
            
            # VADAS 모델의 intrinsic 리스트를 그대로 저장
            intrinsics_list = torch.tensor(calib_data['intrinsic'], dtype=torch.float32)
            
            # 왜곡 계수 (k, s, div, ux, uy)를 별도로 저장
            # VADAS intrinsic: [k0..k6, s, div, ux, uy, ...]
            distortion_coeffs = {
                'k': torch.tensor(calib_data['intrinsic'][0:7], dtype=torch.float32),
                's': torch.tensor(calib_data['intrinsic'][7], dtype=torch.float32),
                'div': torch.tensor(calib_data['intrinsic'][8], dtype=torch.float32),
                'ux': torch.tensor(calib_data['intrinsic'][9], dtype=torch.float32),
                'uy': torch.tensor(calib_data['intrinsic'][10], dtype=torch.float32),
                'image_size': (H, W) # 🆕 이미지 크기 추가
            }
            
            # extrinsic (camera to world) 정보 추가
            extrinsic_matrix = torch.tensor(calib_data['extrinsic'], dtype=torch.float32)
            
            # LiDAR to World 변환 정보 추가
            lidar_to_world_matrix = torch.tensor(DEFAULT_LIDAR_TO_WORLD, dtype=torch.float32)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find file for stem {stem}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading data for stem {stem}: {e}")

        # Convert depth map and apply scale factor
        depth_gt_png = np.asarray(depth_png, dtype=np.uint16)
        depth_gt = depth_gt_png.astype(np.float32)
        image_np = np.array(image)

        # Apply mask if available
        if self.mask is not None:
            # Ensure mask matches image/depth dimensions
            if self.mask.shape[:2] != image_np.shape[:2]:
                # Resize mask if dimensions do not match
                mask_img = Image.fromarray((self.mask * 255).astype(np.uint8), mode='L')
                mask_img = mask_img.resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST)
                self.mask = np.array(mask_img)

            # Apply mask to image (element-wise multiplication)
            image_np = (image_np * self.mask[:, :, np.newaxis]).astype(image_np.dtype)
            image = Image.fromarray(image_np)
            # Apply mask to depth (set masked-out areas to 0)
            depth_gt = depth_gt * self.mask

        sample = {
            'rgb': image,
            'depth': depth_gt,
            'idx': idx,
            'intrinsics': intrinsics_list, # VADAS intrinsic 리스트 전체
            'distortion_coeffs': distortion_coeffs, # 왜곡 계수 딕셔너리 (image_size 포함)
            'extrinsic': extrinsic_matrix, # 카메라 외부 파라미터
            'lidar_to_world': lidar_to_world_matrix, # LiDAR to World 변환
            'filename': stem,
            'meta': {
                'image_path': str(image_path),
                'depth_path': str(depth_path),
                'calibration_source': 'hardcoded_default',
            }
        }
        # 🆕 Add mask to sample if available
        if self.mask is not None:
            sample['mask'] = self.mask

        if self.transform:
            sample = self.transform(sample)

        return sample