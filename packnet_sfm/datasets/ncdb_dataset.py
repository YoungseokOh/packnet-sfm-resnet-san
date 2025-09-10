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
from packnet_sfm.datasets.augmentations import to_tensor # ✅ 추가
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
    Simplified NCDB Dataset for fisheye self-supervised learning
    """
    
    def __init__(self, dataset_root, split_file, transform=None, 
                 back_context=0, forward_context=0, strides=(1,), 
                 with_context=False, **kwargs):
        super().__init__()
        
        # Dataset paths
        self.dataset_root = Path(dataset_root)
        
        # Context parameters
        self.backward_context = back_context
        self.forward_context = forward_context
        self.strides = strides
        self.with_context = with_context or (back_context > 0 or forward_context > 0)
        
        # Transform
        self.transform = transform
        
        # Load split file
        self._load_split_file(split_file)
        
        # 🔧 KITTI 방식: Context가 필요한 경우 사전 필터링
        if self.with_context:
            self._filter_valid_samples_with_context()
        
    def _load_split_file(self, split_file):
        """Load and validate split file"""
        absolute_split_path = self.dataset_root / split_file
        if not absolute_split_path.exists():
            raise FileNotFoundError(f"Split file not found: {absolute_split_path}")
        
        with open(absolute_split_path, 'r') as f:
            mapping_data = json.load(f)
        
        if not isinstance(mapping_data, list):
            raise ValueError("Split file must contain a list of entries")
        
        # Process entries
        processed = []
        for e in mapping_data:
            if isinstance(e, dict):
                if ('new_filename' not in e or not e.get('new_filename')) and e.get('image_path'):
                    try:
                        e['new_filename'] = Path(e['image_path']).stem
                    except Exception:
                        pass
            processed.append(e)
        
        self.data_entries = processed
        print(f"Loaded {len(self.data_entries)} entries from {split_file}")

    def _filter_valid_samples_with_context(self):
        """KITTI 방식: Context가 유효한 샘플만 필터링"""
        print(f"🔄 Filtering samples with valid context (backward={self.backward_context}, forward={self.forward_context})")
        
        valid_samples = []
        skipped_count = 0
        
        for idx, entry in enumerate(self.data_entries):
            # Context 유효성 검사
            if self._validate_context_availability(entry):
                valid_samples.append(entry)
            else:
                skipped_count += 1
        
        print(f"📊 Context filtering results:")
        print(f"   Original samples: {len(self.data_entries)}")
        print(f"   Valid samples: {len(valid_samples)}")
        print(f"   Skipped samples: {skipped_count}")
        
        # 유효한 샘플만 유지
        self.data_entries = valid_samples
    
    def _validate_context_availability(self, entry):
        """특정 entry에 대해 context 가용성 검증"""
        stem = entry.get('new_filename')
        if not stem and entry.get('image_path'):
            stem = Path(entry['image_path']).stem
        
        image_path = self._resolve_image_path(entry, stem)
        dirp = image_path.parent
        
        try:
            base_num = int(image_path.stem)
        except ValueError:
            return False
        
        # 필요한 context 개수 확인
        required_backward = 0
        required_forward = 0
        
        # Backward context 확인
        for i in range(1, self.backward_context + 1):
            prev_num = base_num - i
            if prev_num >= 0:
                prev_path = dirp / f"{prev_num:010d}.png"
                if prev_path.exists():
                    required_backward += 1
        
        # Forward context 확인
        for i in range(1, self.forward_context + 1):
            next_num = base_num + i
            next_path = dirp / f"{next_num:010d}.png"
            if next_path.exists():
                required_forward += 1
        
        # 필요한 개수만큼 있는지 확인
        has_sufficient_context = (
            required_backward >= self.backward_context and
            required_forward >= self.forward_context
        )
        
        return has_sufficient_context

    def _resolve_image_path(self, entry, stem=None):
        """Resolve image path from entry"""
        if 'image_path' in entry and entry['image_path']:
            return Path(entry['image_path'])
        if stem is None:
            stem = entry.get('new_filename')
        root = Path(entry.get('dataset_root', ''))
        if not root.is_absolute():
            root = self.dataset_root / root
        return root / 'image_a6' / f"{stem}.png"

    def _resolve_depth_path(self, entry, stem=None):
        """Resolve depth path from entry (PNG depth map)"""
        # Prefer explicit path if provided
        if 'depth_path' in entry and entry['depth_path']:
            return Path(entry['depth_path'])
        # Infer from dataset_root and stem
        if stem is None:
            if entry.get('new_filename'):
                stem = entry['new_filename']
            elif entry.get('image_path'):
                try:
                    stem = Path(entry['image_path']).stem
                except Exception:
                    stem = None
        root = Path(entry.get('dataset_root', ''))
        if not root.is_absolute():
            root = self.dataset_root / root
        if stem is None:
            return None
        return root / 'newest_depth_maps' / f"{stem}.png"

    def _get_context_frames(self, idx):
        """Enhanced context frame discovery with proper validation (KITTI-style)"""
        entry = self.data_entries[idx]
        stem = entry.get('new_filename')
        if not stem and entry.get('image_path'):
            stem = Path(entry['image_path']).stem
        
        image_path = self._resolve_image_path(entry, stem)
        dirp = image_path.parent
        
        try:
            base_num = int(image_path.stem)
        except ValueError:
            # 🔧 실패 시 빈 리스트 대신 None 반환하여 명확한 처리
            return None
        
        context_images = []
        
        # 🔧 KITTI 방식: 사전에 존재하는 파일들만 확인
        available_frames = []
        
        # Find backward context
        for i in range(1, self.backward_context + 1):
            prev_num = base_num - i
            if prev_num >= 0:
                prev_path = dirp / f"{prev_num:010d}.png"
                if prev_path.exists():
                    available_frames.append(('backward', i, prev_path))
        
        # Find forward context  
        for i in range(1, self.forward_context + 1):
            next_num = base_num + i
            next_path = dirp / f"{next_num:010d}.png"
            if next_path.exists():
                available_frames.append(('forward', i, next_path))
        
        # 🔧 필수 context가 부족한 경우 None 반환
        required_contexts = self.backward_context + self.forward_context
        if len(available_frames) < required_contexts:
            return None
        
        # 🔧 정확한 순서로 context 이미지 로드 (backward -> forward)
        backward_images = []
        forward_images = []
        
        for direction, offset, path in available_frames:
            try:
                img = load_image(str(path))
                if direction == 'backward':
                    backward_images.append((offset, img))
                else:
                    forward_images.append((offset, img))
            except Exception as e:
                # 🔧 로딩 실패 시 None 반환
                print(f"⚠️ Failed to load context image {path}: {e}")
                return None
        
        # 🔧 정렬하여 올바른 순서 보장
        backward_images.sort(key=lambda x: -x[0])  # 역순 정렬 (가까운 것부터)
        forward_images.sort(key=lambda x: x[0])    # 정순 정렬
        
        # 최종 context list 구성
        for _, img in backward_images:
            context_images.append(img)
        for _, img in forward_images:
            context_images.append(img)
        
        return context_images

    def __len__(self):
        return len(self.data_entries)
    
    def __getitem__(self, idx):
        entry = self.data_entries[idx]
        stem = entry.get('new_filename')
        if not stem and entry.get('image_path'):
            stem = Path(entry['image_path']).stem
        
        # Load image
        image_path = self._resolve_image_path(entry, stem)
        image = load_image(str(image_path))
        W, H = image.size
        
        # Load depth (16-bit PNG -> float32). If missing, fallback to zeros to avoid KeyError
        depth_gt = None
        depth_path = self._resolve_depth_path(entry, stem)
        if depth_path and depth_path.exists():
            try:
                # Load 16-bit PNG depth map
                depth_img = Image.open(str(depth_path))
                depth_array = np.array(depth_img, dtype=np.float32)
                # Convert to meters (assuming depth is in mm)
                depth_array = depth_array / 1000.0
                depth_gt = depth_array
            except Exception as e:
                print(f"⚠️ Failed to load depth {depth_path}: {e}")
                depth_gt = None
        
        # 🔧 Ensure depth is never None for tensor conversion
        if depth_gt is None:
            # Create dummy depth map with zeros
            depth_gt = np.zeros((H, W), dtype=np.float32)
        
        # 🔧 Camera and distortion coefficients 설정 (모든 필수 키 포함)
        calib = DEFAULT_CALIB_A6.copy()
        calib['image_size'] = (H, W)
        
        # Extract all required parameters for FisheyeCamera
        intrinsic_vals = calib['intrinsic']
        k_coeffs = torch.tensor(intrinsic_vals[:7], dtype=torch.float32)  # k0~k6
        
        # VADAS 모델에서 추가 파라미터들 (intrinsic[7:] 사용)
        s_val = torch.tensor(intrinsic_vals[7], dtype=torch.float32)      # scale factor
        div_val = torch.tensor(intrinsic_vals[8], dtype=torch.float32)    # division factor
        ux_val = torch.tensor(W / 2.0, dtype=torch.float32)              # principal point x
        uy_val = torch.tensor(H / 2.0, dtype=torch.float32)              # principal point y
        
        # FisheyeCamera가 기대하는 완전한 형식
        fisheye_intrinsics = {
            'k': k_coeffs,
            's': s_val,
            'div': div_val,
            'ux': ux_val,
            'uy': uy_val
        }
        
        # Create FisheyeCamera object with complete intrinsics
        camera = FisheyeCamera(intrinsics=fisheye_intrinsics, image_size=(H, W))
        
        # Distortion coefficients for loss function (simplified)
        distortion_coeffs = {'k': k_coeffs}
        
        # Create intrinsics matrix for compatibility
        fx = fy = 1.0  # Normalized for fisheye
        cx, cy = W / 2.0, H / 2.0
        intrinsics_list = np.array([
            [fx, 0., cx],
            [0., fy, cy],
            [0., 0., 1.]
        ], dtype=np.float32)
        
        # Build sample
        sample = {
            'rgb': image,
            'idx': idx,
            'camera': camera,
            'intrinsics': intrinsics_list,
            'distortion_coeffs': {
                'k': k_coeffs,
                's': s_val,
                'div': div_val,
                'ux': ux_val,
                'uy': uy_val,
            },
            'filename': stem,
            'depth': depth_gt,  # ✅ ensure depth is always present
        }
        
        # Add context frames if needed
        if self.with_context:
            # 🔧 사전 필터링으로 인해 모든 샘플이 유효한 context를 가짐이 보장됨
            context_images = self._get_context_frames(idx)
            # 하지만 방어적 코딩으로 여전히 체크
            if context_images is not None and len(context_images) > 0:
                sample['rgb_context'] = context_images
                # Create context cameras with same intrinsics
                sample['camera_context'] = [
                    FisheyeCamera(intrinsics=fisheye_intrinsics, image_size=(H, W))
                    for _ in context_images
                ]
                # Generate dummy poses for context frames
                sample['pose_context'] = self._generate_dummy_poses(len(context_images))
            else:
                # 🚨 이 경우는 사전 필터링 후에는 발생하면 안됨
                print(f"⚠️ WARNING: Sample {idx} passed filtering but has no valid context!")
                sample['rgb_context'] = []
                sample['camera_context'] = []
                sample['pose_context'] = []
        else:
            sample['rgb_context'] = []
            sample['camera_context'] = []
            sample['pose_context'] = []
        
        sample['has_context'] = len(sample['rgb_context']) > 0
        
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)

        return sample

    def _generate_dummy_poses(self, num_contexts):
        """Generate dummy pose context for self-supervised learning
        
        Parameters
        ----------
        num_contexts : int
            Number of context frames
            
        Returns
        -------
        poses : list
            List of 4x4 transformation matrices (dummy poses)
        """
        poses = []
        for i in range(num_contexts):
            # Create identity transformation matrix as dummy pose
            # In real scenarios, this would be actual camera poses between frames
            pose_matrix = np.eye(4, dtype=np.float32)
            
            # Add small random perturbation to make it non-identity
            # This gives PoseNet something to learn from
            angle = np.random.uniform(-0.1, 0.1)  # Small rotation
            translation = np.random.uniform(-0.01, 0.01, 3)  # Small translation
            
            # Simple rotation around Y axis (typical for vehicle motion)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            pose_matrix[0, 0] = cos_a
            pose_matrix[0, 2] = sin_a
            pose_matrix[2, 0] = -sin_a
            pose_matrix[2, 2] = cos_a
            
            # Add translation
            pose_matrix[:3, 3] = translation
            
            poses.append(pose_matrix)
        
        return poses

    # Custom collate function (simplified)
    @staticmethod
    def custom_collate_fn(batch):
        """Simplified collate function for FisheyeCamera objects"""
        if not batch:
            return {}
        
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            values = [item[key] for item in batch if key in item]
            
            if key in ['camera', 'camera_context']:
                # Keep camera objects as lists
                collated[key] = values
            elif key == 'distortion_coeffs':
                # Stack distortion coefficients
                stacked = {}
                sample_keys = list(values[0].keys())
                for k in sample_keys:
                    vals = [d[k] for d in values]
                    if isinstance(vals[0], torch.Tensor):
                        try:
                            # Ensure scalars are unsqueezed
                            proc = []
                            for v in vals:
                                if v.dim() == 0:
                                    v = v.unsqueeze(0)
                                proc.append(v)
                            stacked[k] = torch.stack(proc, dim=0)
                        except Exception:
                            stacked[k] = vals
                    else:
                        stacked[k] = vals
                collated[key] = stacked
            elif key == 'depth':
                # Collate depth and add channel dimension
                collated[key] = default_collate(values).unsqueeze(1)
            elif key == 'rgb':
                # Stack Tensors
                collated[key] = torch.stack(values)
            elif key == 'rgb_context':
                if values and isinstance(values[0], list):
                    transposed_context_frames = zip(*values)
                    collated[key] = [torch.stack(ctx_frames) for ctx_frames in transposed_context_frames]
                else:
                    collated[key] = torch.stack(values)
            else:
                try:
                    collated[key] = default_collate(values)
                except Exception:
                    collated[key] = values
        
        return collated