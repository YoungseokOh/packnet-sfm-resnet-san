# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import sys
import time
import pickle
import hashlib
import numpy as np
from glob import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
from functools import partial
from torch.utils.data import Dataset
from tqdm import tqdm

from packnet_sfm.datasets.kitti_dataset_utils import \
    pose_from_oxts_packet, read_calib_file, transform_from_rot_trans
from packnet_sfm.utils.image import load_image
from packnet_sfm.geometry.pose_utils import invert_pose_numpy

########################################################################################################################

# Cameras from the stero pair (left is the origin)
IMAGE_FOLDER = {
    'left': 'image_02',
    'right': 'image_03',
}
# Name of different calibration files
CALIB_FILE = {
    'cam2cam': 'calib_cam_to_cam.txt',
    'velo2cam': 'calib_velo_to_cam.txt',
    'imu2velo': 'calib_imu_to_velo.txt',
}
PNG_DEPTH_DATASETS = ['groundtruth']
OXTS_POSE_DATA = 'oxts'

########################################################################################################################
#### FUNCTIONS
########################################################################################################################

def read_npz_depth(file, depth_type):
    """Reads a .npz depth map given a certain depth_type."""
    depth = np.load(file)[depth_type + '_depth'].astype(np.float32)
    return np.expand_dims(depth, axis=2)

def read_png_depth(file):
    """Reads a .png depth map."""
    depth_png = np.array(load_image(file), dtype=int)
    assert (np.max(depth_png) > 255), 'Wrong .png depth file'
    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return np.expand_dims(depth, axis=2)

########################################################################################################################
#### CACHE SYSTEM
########################################################################################################################

class FileCache:
    """파일 기반 캐싱 시스템"""
    
    def __init__(self, cache_dir="/tmp/packnet_kitti_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_age = 24 * 3600  # 24시간
    
    def get_cache_key(self, config_dict):
        """설정 기반 캐시 키 생성"""
        config_str = (f"{config_dict.get('root_dir', '')}_"
                     f"{config_dict.get('file_list', '')}_"
                     f"{config_dict.get('depth_type', '')}_"
                     f"{config_dict.get('input_depth_type', '')}_"
                     f"{config_dict.get('back_context', 0)}_"
                     f"{config_dict.get('forward_context', 0)}")
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def is_cache_valid(self, cache_file):
        """캐시 파일이 유효한지 확인"""
        if not cache_file.exists():
            return False
        
        cache_age = time.time() - cache_file.stat().st_mtime
        return cache_age < self.max_cache_age
    
    def load(self, cache_key):
        """캐시에서 로드"""
        cache_file = self.cache_dir / f"kitti_{cache_key}.pkl"
        
        if self.is_cache_valid(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    print(f"🚀 Loaded from cache: {len(data['paths'])} files")
                    return data
            except Exception as e:
                print(f"⚠️ Cache load failed: {e}")
                try:
                    cache_file.unlink()
                except:
                    pass
        
        return None
    
    def save(self, cache_key, data):
        """캐시에 저장"""
        cache_file = self.cache_dir / f"kitti_{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Saved to cache: {len(data['paths'])} files")
        except Exception as e:
            print(f"⚠️ Cache save failed: {e}")

########################################################################################################################
#### PARALLEL VALIDATION
########################################################################################################################

def validate_file_chunk(args):
    """파일 청크 검증 (멀티프로세싱용)"""
    lines_chunk, root_dir, depth_type, input_depth_type, with_depth, with_input_depth = args
    
    valid_paths = []
    
    for line in lines_chunk:
        try:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if len(parts) < 1:
                continue
            
            # RGB 파일 경로
            path = os.path.join(root_dir, parts[0])
            
            # RGB 파일 존재 확인
            if not os.path.exists(path):
                continue
            
            # Input depth 확인 (필요한 경우)
            if with_input_depth:
                input_depth_path = get_depth_file_static(path, input_depth_type)
                if not os.path.exists(input_depth_path):
                    continue
            
            # GT depth 확인 (필요한 경우)
            if with_depth:
                gt_depth_path = get_depth_file_static(path, depth_type)
                if not os.path.exists(gt_depth_path):
                    continue
            
            valid_paths.append(path)
        
        except Exception:
            continue
    
    return valid_paths

def get_depth_file_static(image_file, depth_type):
    """Static version of _get_depth_file for multiprocessing"""
    for cam in ['left', 'right']:
        if IMAGE_FOLDER[cam] in image_file:
            if depth_type in PNG_DEPTH_DATASETS:
                return image_file.replace(f'{IMAGE_FOLDER[cam]}/data', 
                                        f'proj_depth/groundtruth/{IMAGE_FOLDER[cam]}')
            else:
                return image_file.replace(f'{IMAGE_FOLDER[cam]}/data', 
                                        f'proj_depth/{depth_type}/{IMAGE_FOLDER[cam]}').replace('.png', '.npz')
    return image_file

def validate_context_chunk(args):
    """컨텍스트 검증 (멀티프로세싱용)"""
    paths_chunk, backward_context, forward_context, strides = args
    
    valid_paths = []
    
    for stride in strides:
        for path in paths_chunk:
            try:
                context_result = get_sample_context_static(
                    path, backward_context, forward_context, stride)
                
                if context_result is not None:
                    valid_paths.append(path)
            except Exception:
                continue
    
    return valid_paths

def get_sample_context_static(sample_name, backward_context, forward_context, stride=1):
    """Static version of _get_sample_context for multiprocessing"""
    try:
        base, ext = os.path.splitext(os.path.basename(sample_name))
        parent_folder = os.path.dirname(sample_name)
        f_idx = int(base)

        # Check number of files in folder
        max_num_files = len(glob(os.path.join(parent_folder, '*' + ext)))

        # Check bounds
        if (f_idx - backward_context * stride) < 0 or (
                f_idx + forward_context * stride) >= max_num_files:
            return None

        # Backward context
        c_idx = f_idx
        backward_context_idxs = []
        while len(backward_context_idxs) < backward_context and c_idx > 0:
            c_idx -= stride
            filename = get_next_file_static(c_idx, sample_name)
            if os.path.exists(filename):
                backward_context_idxs.append(c_idx)
        if c_idx < 0:
            return None

        # Forward context
        c_idx = f_idx
        forward_context_idxs = []
        while len(forward_context_idxs) < forward_context and c_idx < max_num_files:
            c_idx += stride
            filename = get_next_file_static(c_idx, sample_name)
            if os.path.exists(filename):
                forward_context_idxs.append(c_idx)
        if c_idx >= max_num_files:
            return None

        return backward_context_idxs, forward_context_idxs
    
    except Exception:
        return None

def get_next_file_static(idx, file):
    """Static version of _get_next_file for multiprocessing"""
    base, ext = os.path.splitext(os.path.basename(file))
    return os.path.join(os.path.dirname(file), str(idx).zfill(len(base)) + ext)

########################################################################################################################
#### OPTIMIZED DATASET
########################################################################################################################

class OptimizedKITTIDataset(Dataset):
    """
    KITTI dataset class with parallel processing and caching.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    file_list : str
        Split file, with paths to the images to be used
    train : bool
        True if the dataset will be used for training
    data_transform : Function
        Transformations applied to the sample
    depth_type : str
        Which depth type to load
    input_depth_type : str
        Which input depth type to load
    with_pose : bool
        True if returning ground-truth pose
    back_context : int
        Number of backward frames to consider as context
    forward_context : int
        Number of forward frames to consider as context
    strides : tuple
        List of context strides
    use_cache : bool
        Whether to use file caching
    max_workers : int
        Maximum number of worker processes
    """
    def __init__(self, root_dir, file_list, train=True,
                 data_transform=None, depth_type=None, input_depth_type=None,
                 with_pose=False, back_context=0, forward_context=0, strides=(1,),
                 use_cache=True, max_workers=None):
        
        # 🆕 file_list를 먼저 저장 (AttributeError 방지)
        self.file_list = file_list
        
        # Assertions
        backward_context = back_context
        assert backward_context >= 0 and forward_context >= 0, 'Invalid contexts'

        self.backward_context = backward_context
        self.backward_context_paths = []
        self.forward_context = forward_context
        self.forward_context_paths = []

        self.with_context = (backward_context != 0 or forward_context != 0)
        self.split = file_list.split('/')[-1].split('.')[0]

        self.train = train
        self.root_dir = root_dir
        self.data_transform = data_transform

        self.depth_type = depth_type
        self.with_depth = depth_type is not '' and depth_type is not None
        self.with_pose = with_pose

        self.input_depth_type = input_depth_type
        self.with_input_depth = input_depth_type is not '' and input_depth_type is not None

        # 🆕 병렬화 및 캐싱 설정
        self.use_cache = use_cache
        self.max_workers = max_workers or min(32, cpu_count())
        
        # 🆕 캐시 시스템 초기화
        if use_cache:
            self.cache = FileCache()
        
        # 원본과 동일한 캐시들
        self._cache = {}
        self.pose_cache = {}
        self.oxts_cache = {}
        self.calibration_cache = {}
        self.imu2velo_calib_cache = {}
        self.sequence_origin_cache = {}

        # 🆕 최적화된 파일 로딩
        self._load_paths_optimized()

    def _load_paths_optimized(self):
        """최적화된 파일 경로 로딩"""
        start_time = time.time()
        
        # 캐시 키 생성
        cache_config = {
            'root_dir': self.root_dir,
            'file_list': self.file_list,
            'depth_type': self.depth_type,
            'input_depth_type': self.input_depth_type,
            'back_context': self.backward_context,
            'forward_context': self.forward_context,
        }
        
        cache_key = None
        cached_data = None
        
        if self.use_cache:
            cache_key = self.cache.get_cache_key(cache_config)
            cached_data = self.cache.load(cache_key)
        
        if cached_data:
            # 캐시에서 로드 성공
            self.paths = cached_data['paths']
            load_time = time.time() - start_time
            print(f"⚡ Loaded from cache in {load_time:.2f}s")
            return
        
        # 캐시 미스 - 새로 스캔
        print(f"🔍 Scanning files with {self.max_workers} workers...")
        
        # 파일 읽기
        with open(self.file_list, "r") as f:
            data = f.readlines()
        
        print(f"📊 Processing {len(data)} lines from {self.file_list}")
        
        # 🆕 병렬 파일 검증
        valid_paths = self._validate_files_parallel(data)
        
        # 🆕 컨텍스트 필터링 (필요한 경우)
        if self.with_context:
            valid_paths = self._filter_paths_with_context_parallel(valid_paths)
        
        self.paths = valid_paths
        
        # 캐시에 저장
        if self.use_cache and cache_key:
            cache_data = {
                'paths': self.paths,
                'scan_time': time.time() - start_time,
            }
            self.cache.save(cache_key, cache_data)
        
        load_time = time.time() - start_time
        print(f"⏱️ File scan completed in {load_time:.2f}s")
        print(f"📈 Valid files: {len(self.paths)}/{len(data)} ({len(self.paths)/len(data)*100:.1f}%)")

    def _validate_files_parallel(self, lines):
        """병렬로 파일 검증"""
        chunk_size = max(100, len(lines) // self.max_workers)
        chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
        
        chunk_args = [
            (chunk, self.root_dir, self.depth_type, self.input_depth_type, 
             self.with_depth, self.with_input_depth)
            for chunk in chunks
        ]
        
        valid_paths = []
        
        if len(chunks) > 1:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(validate_file_chunk, args) for args in chunk_args]
                
                with tqdm(total=len(futures), desc="Validating files") as pbar:
                    for future in futures:
                        try:
                            chunk_results = future.result()
                            valid_paths.extend(chunk_results)
                            pbar.update(1)
                        except Exception as e:
                            print(f"⚠️ Chunk processing error: {e}")
                            pbar.update(1)
        else:
            chunk_results = validate_file_chunk(chunk_args[0])
            valid_paths.extend(chunk_results)
        
        return valid_paths

    def _filter_paths_with_context_parallel(self, valid_paths):
        """병렬로 컨텍스트 필터링"""
        print(f"🔄 Filtering for context (backward={self.backward_context}, forward={self.forward_context})")
        
        chunk_size = max(50, len(valid_paths) // self.max_workers)
        chunks = [valid_paths[i:i+chunk_size] for i in range(0, len(valid_paths), chunk_size)]
        
        chunk_args = [
            (chunk, self.backward_context, self.forward_context, (1,))  # strides는 (1,)로 고정
            for chunk in chunks
        ]
        
        paths_with_context = []
        
        if len(chunks) > 1:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(validate_context_chunk, args) for args in chunk_args]
                
                with tqdm(total=len(futures), desc="Checking context") as pbar:
                    for future in futures:
                        try:
                            chunk_results = future.result()
                            paths_with_context.extend(chunk_results)
                            pbar.update(1)
                        except Exception as e:
                            print(f"⚠️ Context check error: {e}")
                            pbar.update(1)
        else:
            chunk_results = validate_context_chunk(chunk_args[0])
            paths_with_context.extend(chunk_results)
        
        # 중복 제거
        paths_with_context = list(set(paths_with_context))
        print(f"📉 After context filtering: {len(paths_with_context)} files (was {len(valid_paths)})")
        
        return paths_with_context

########################################################################################################################
#### 원본과 동일한 메서드들
########################################################################################################################

    @staticmethod
    def _get_next_file(idx, file):
        """Get next file given next idx and current file."""
        base, ext = os.path.splitext(os.path.basename(file))
        return os.path.join(os.path.dirname(file), str(idx).zfill(len(base)) + ext)

    @staticmethod
    def _get_parent_folder(image_file):
        """Get the parent folder from image_file."""
        return os.path.abspath(os.path.join(image_file, "../../../.."))

    @staticmethod
    def _get_intrinsics(image_file, calib_data):
        """Get intrinsics from the calib_data dictionary."""
        for cam in ['left', 'right']:
            if IMAGE_FOLDER[cam] in image_file:
                return calib_data[f'P_rect_{IMAGE_FOLDER[cam][-2:]}'].reshape(3, 4)[:3, :3].astype(np.float32)

    @staticmethod
    def _read_raw_calib_file(folder):
        """Read raw calibration files from folder."""
        return read_calib_file(os.path.join(folder, CALIB_FILE['cam2cam']))

########################################################################################################################
#### DEPTH
########################################################################################################################

    def _read_depth(self, depth_file):
        """Get the depth map from a file."""
        if depth_file.endswith('.npz'):
            return read_npz_depth(depth_file, 'velodyne')
        elif depth_file.endswith('.png'):
            return read_png_depth(depth_file)
        else:
            raise NotImplementedError(f'Depth type {self.depth_type} not implemented')

    @staticmethod
    def _get_depth_file(image_file, depth_type):
        """Get the corresponding depth file from an image file."""
        for cam in ['left', 'right']:
            if IMAGE_FOLDER[cam] in image_file:
                if depth_type in PNG_DEPTH_DATASETS:
                    return image_file.replace(f'{IMAGE_FOLDER[cam]}/data', 
                                            f'proj_depth/groundtruth/{IMAGE_FOLDER[cam]}')
                else:
                    return image_file.replace(f'{IMAGE_FOLDER[cam]}/data', 
                                            f'proj_depth/{depth_type}/{IMAGE_FOLDER[cam]}').replace('.png', '.npz')

    def _get_sample_context(self, sample_name,
                            backward_context, forward_context, stride=1):
        """
        Get a sample context

        Parameters
        ----------
        sample_name : str
            Path + Name of the sample
        backward_context : int
            Size of backward context
        forward_context : int
            Size of forward context
        stride : int
            Stride value to consider when building the context

        Returns
        -------
        backward_context : list of int
            List containing the indexes for the backward context
        forward_context : list of int
            List containing the indexes for the forward context
        """
        base, ext = os.path.splitext(os.path.basename(sample_name))
        parent_folder = os.path.dirname(sample_name)
        f_idx = int(base)

        # Check number of files in folder
        if parent_folder in self._cache:
            max_num_files = self._cache[parent_folder]
        else:
            max_num_files = len(glob(os.path.join(parent_folder, '*' + ext)))
            self._cache[parent_folder] = max_num_files

        # Check bounds
        if (f_idx - backward_context * stride) < 0 or (
                f_idx + forward_context * stride) >= max_num_files:
            return None

        # Backward context
        c_idx = f_idx
        backward_context_idxs = []
        while len(backward_context_idxs) < backward_context and c_idx > 0:
            c_idx -= stride
            filename = self._get_next_file(c_idx, sample_name)
            if os.path.exists(filename):
                backward_context_idxs.append(c_idx)
        if c_idx < 0:
            return None

        # Forward context
        c_idx = f_idx
        forward_context_idxs = []
        while len(forward_context_idxs) < forward_context and c_idx < max_num_files:
            c_idx += stride
            filename = self._get_next_file(c_idx, sample_name)
            if os.path.exists(filename):
                forward_context_idxs.append(c_idx)
        if c_idx >= max_num_files:
            return None

        return backward_context_idxs, forward_context_idxs

    def _get_context_files(self, sample_name, idxs):
        """
        Returns image and depth context files

        Parameters
        ----------
        sample_name : str
            Name of current sample
        idxs : list of idxs
            Context indexes

        Returns
        -------
        image_context_paths : list of str
            List of image names for the context
        depth_context_paths : list of str
            List of depth names for the context
        """
        image_context_paths = [self._get_next_file(i, sample_name) for i in idxs]
        return image_context_paths, None

########################################################################################################################
#### POSE
########################################################################################################################

    def _get_imu2cam_transform(self, image_file):
        """Gets the transformation between IMU an camera from an image file"""
        parent_folder = self._get_parent_folder(image_file)
        if image_file in self.imu2velo_calib_cache:
            return self.imu2velo_calib_cache[image_file]

        cam2cam = read_calib_file(os.path.join(parent_folder, CALIB_FILE['cam2cam']))
        imu2velo = read_calib_file(os.path.join(parent_folder, CALIB_FILE['imu2velo']))
        velo2cam = read_calib_file(os.path.join(parent_folder, CALIB_FILE['velo2cam']))

        velo2cam_mat = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])
        imu2velo_mat = transform_from_rot_trans(imu2velo['R'], imu2velo['T'])
        cam_2rect_mat = transform_from_rot_trans(cam2cam['R_rect_00'], np.zeros(3))

        imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat
        self.imu2velo_calib_cache[image_file] = imu2cam
        return imu2cam

    @staticmethod
    def _get_oxts_file(image_file):
        """Gets the oxts file from an image file."""
        for cam in ['left', 'right']:
            if IMAGE_FOLDER[cam] in image_file:
                # 🆕 oxts/data 경로 수정
                return image_file.replace(f'{IMAGE_FOLDER[cam]}/data', f'{OXTS_POSE_DATA}/data').replace('.png', '.txt')
        raise ValueError('Invalid KITTI path for pose supervision.')

    def _get_oxts_data(self, image_file):
        """Gets the oxts data from an image file."""
        oxts_file = self._get_oxts_file(image_file)
        if oxts_file in self.oxts_cache:
            oxts_data = self.oxts_cache[oxts_file]
        else:
            oxts_data = np.loadtxt(oxts_file, delimiter=' ', skiprows=0)
            self.oxts_cache[oxts_file] = oxts_data
        return oxts_data

    def _get_pose(self, image_file):
        """Gets the pose information from an image file."""
        if image_file in self.pose_cache:
            return self.pose_cache[image_file]
        
        # Find origin frame in this sequence to determine scale & origin translation
        base, ext = os.path.splitext(os.path.basename(image_file))
        origin_frame = os.path.join(os.path.dirname(image_file), str(0).zfill(len(base)) + ext)
        # Get origin data
        origin_oxts_data = self._get_oxts_data(origin_frame)
        lat = origin_oxts_data[0]
        scale = np.cos(lat * np.pi / 180.)
        # Get origin pose
        origin_R, origin_t = pose_from_oxts_packet(origin_oxts_data, scale)
        origin_pose = transform_from_rot_trans(origin_R, origin_t)
        # Compute current pose
        oxts_data = self._get_oxts_data(image_file)
        R, t = pose_from_oxts_packet(oxts_data, scale)
        pose = transform_from_rot_trans(R, t)
        # Compute odometry pose
        imu2cam = self._get_imu2cam_transform(image_file)
        odo_pose = (imu2cam @ np.linalg.inv(origin_pose) @
                    pose @ np.linalg.inv(imu2cam)).astype(np.float32)
        # Cache and return pose
        self.pose_cache[image_file] = odo_pose
        return odo_pose

########################################################################################################################

    def __len__(self):
        """Dataset length."""
        return len(self.paths)

    def __getitem__(self, idx):
        """Get dataset sample given an index."""
        # Add image information
        sample = {
            'idx': idx,
            'filename': '%s_%010d' % (self.split, idx),
            'rgb': load_image(self.paths[idx]),
        }

        # Add intrinsics
        parent_folder = self._get_parent_folder(self.paths[idx])
        if parent_folder in self.calibration_cache:
            c_data = self.calibration_cache[parent_folder]
        else:
            c_data = self._read_raw_calib_file(parent_folder)
            self.calibration_cache[parent_folder] = c_data
        
        sample.update({
            'intrinsics': self._get_intrinsics(self.paths[idx], c_data),
        })

        # Add pose information if requested
        if self.with_pose:
            sample['pose'] = self._get_pose(self.paths[idx])

        # Add depth information if requested
        if self.with_depth:
            depth_path = self._get_depth_file(self.paths[idx], self.depth_type)
            sample['depth'] = self._read_depth(depth_path)

        # Add input depth information if requested
        if self.with_input_depth:
            input_depth_path = self._get_depth_file(self.paths[idx], self.input_depth_type)
            sample['input_depth'] = self._read_depth(input_depth_path)

        # Add context information if requested
        if self.with_context:
            sample_name = self.paths[idx]
            backward_context, forward_context = [], []
            
            for stride in (1,):  # strides is hardcoded to (1,) for simplicity
                context = self._get_sample_context(sample_name, self.backward_context, self.forward_context, stride)
                if context:
                    back_idxs, forward_idxs = context
                    back_imgs, _ = self._get_context_files(sample_name, back_idxs)
                    forward_imgs, _ = self._get_context_files(sample_name, forward_idxs)
                    backward_context.extend([load_image(f) for f in back_imgs])
                    forward_context.extend([load_image(f) for f in forward_imgs])
            
            sample['rgb_context'] = backward_context + forward_context

        # Apply transformations
        if self.data_transform:
            sample = self.data_transform(sample)

        # Return sample
        return sample

########################################################################################################################

# 원본과의 호환성을 위한 alias
class KITTIDataset(OptimizedKITTIDataset):
    """
    기존 KITTIDataset의 backward compatibility를 위한 래퍼
    """
    def __init__(self, *args, **kwargs):
        # 기본적으로 최적화 활성화
        if 'use_cache' not in kwargs:
            kwargs['use_cache'] = True
        if 'max_workers' not in kwargs:
            kwargs['max_workers'] = min(32, cpu_count())
        
        super().__init__(*args, **kwargs)

########################################################################################################################