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
    root_dir : str
        Path to the dataset root directory (e.g., 'ncdb-cls-sample/synced_data').
    split_file : str
        Path to the split file, which is 'mapping_data.json'.
    transform : callable, optional
        A function/transform that takes in a sample and returns a transformed version.
    mask_file : str, optional
        Path to the binary mask file (e.g., 'binary_mask.png').
    """
    def __init__(self, root_dir, split_file, transform=None, mask_file=None, **kwargs):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.file_stems = []

        # Initialize transforms based on kwargs (from config)
        self.transform = transform # Assuming 'train' mode for now

        split_path = self.root_dir / split_file
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found at {split_path}")

        with open(split_path, 'r') as f:
            # Handle both .txt and .json split files
            if split_path.suffix == '.txt':
                self.file_stems = [line.strip() for line in f if line.strip()]
            elif split_path.suffix == '.json':
                mapping_data = json.load(f)
                if isinstance(mapping_data, list):
                    # Handle list of dictionaries
                    self.file_stems = [item['new_filename'] for item in mapping_data if 'new_filename' in item]
                elif isinstance(mapping_data, dict):
                    # Handle dictionary of lists (original assumption)
                    self.file_stems = [Path(p).stem for p in mapping_data.get("image_a6", [])]

        # Load mask file if provided
        self.mask = None
        if mask_file:
            mask_path = self.root_dir / mask_file
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask file not found at {mask_path}")
            self.mask = np.array(Image.open(mask_path).convert('L')) # Convert to grayscale (assuming 0 or 1 values)

    def __len__(self):
        return len(self.file_stems)

    def __getitem__(self, idx):
        stem = self.file_stems[idx]
        
        # Construct paths (changed .png to .jpg for image_path)
        image_path = self.root_dir / 'image_a6' / f"{stem}.jpg"
        depth_path = self.root_dir / 'depth_maps' / f"{stem}.png" # Depth maps are typically PNG

        # 캘리브레이션 데이터는 하드코딩된 DEFAULT_CALIB_A6 사용
        calib_data = DEFAULT_CALIB_A6

        # Load data
        try:
            image = Image.open(image_path).convert('RGB')
            depth_png = Image.open(depth_path)
            
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
        depth_gt = np.asarray(depth_png, dtype=np.uint16)
        depth_gt = depth_gt.astype(np.float32) / 256.0

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
            'distortion_coeffs': distortion_coeffs, # 왜곡 계수 딕셔너리
            'extrinsic': extrinsic_matrix, # 카메라 외부 파라미터
            'lidar_to_world': lidar_to_world_matrix, # LiDAR to World 변환
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