#!/usr/bin/env python3
# filepath: /workspace/packnet-sfm/scripts/eval_onnx.py
# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import numpy as np
import os
import torch
import cv2
from PIL import Image
from glob import glob
from tqdm import tqdm
import onnxruntime as ort

from packnet_sfm.utils.depth import load_depth, compute_depth_metrics
from packnet_sfm.utils.config import parse_test_file
import torchvision.transforms as transforms


def parse_args():
    """Parse arguments for ONNX evaluation script"""
    parser = argparse.ArgumentParser(description='PackNet-SfM ONNX evaluation script')
    parser.add_argument('--onnx_model', type=str, required=True,
                        help='Path to ONNX model (.onnx)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Original checkpoint (.ckpt) for configuration')
    parser.add_argument('--config', type=str, default=None,
                        help='Configuration (.yaml)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to KITTI dataset')
    parser.add_argument('--split_file', type=str, 
                        default='data_splits/eigen_test_files.txt',
                        help='Split file for evaluation')
    parser.add_argument('--output_folder', type=str, default='outputs/onnx_eval',
                        help='Output folder for depth maps')
    parser.add_argument('--min_depth', type=float, default=0.0,
                        help='Minimum depth for evaluation')
    parser.add_argument('--max_depth', type=float, default=80.0,
                        help='Maximum depth for evaluation')
    parser.add_argument('--use_gt_scale', action='store_true',
                        help='Use ground-truth median scaling')
    parser.add_argument('--crop', type=str, default='garg', choices=['', 'garg'],
                        help='Crop type for evaluation')
    parser.add_argument('--save_depth_maps', action='store_true',
                        help='Save predicted depth maps')
    # 🆕 PackNet-SfM 호환성을 위한 추가 속성들
    parser.add_argument('--scale_output', type=str, default='top-center',
                        help='Scale output for depth evaluation')
    
    args = parser.parse_args()
    
    # Validate files
    assert os.path.exists(args.onnx_model), f'ONNX model not found: {args.onnx_model}'
    assert os.path.exists(args.checkpoint), f'Checkpoint not found: {args.checkpoint}'
    assert os.path.exists(args.data_path), f'Data path not found: {args.data_path}'
    
    return args


class ONNXDepthEstimator:
    """ONNX 기반 깊이 추정기"""
    
    def __init__(self, onnx_model_path):
        """ONNX 모델 초기화"""
        print(f"🔧 Loading ONNX model: {onnx_model_path}")
        
        # ONNX Runtime 세션 생성
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_model_path)
        
        # 입력/출력 정보 가져오기
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 🆕 모델의 실제 입력 크기 확인
        input_shape = self.session.get_inputs()[0].shape
        self.batch_size = input_shape[0] if input_shape[0] != -1 else 1
        self.channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        
        print(f"📐 ONNX Model Input Shape: {input_shape}")
        print(f"   Expected input size: {self.input_width}x{self.input_height}")
        print(f"   Input name: {self.input_name}")
        print(f"   Output name: {self.output_name}")
    
    def predict(self, image_path):
        """이미지에서 깊이 맵 예측"""
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # 🆕 모델의 실제 입력 크기로 리사이즈
        image_resized = image.resize((self.input_width, self.input_height), Image.LANCZOS)
        
        # 텐서로 변환 및 정규화
        image_array = np.array(image_resized, dtype=np.float32) / 255.0
        image_array = image_array.transpose(2, 0, 1)  # HWC -> CHW
        image_array = image_array[np.newaxis, :]  # 배치 차원 추가
        
        # ONNX 추론
        outputs = self.session.run([self.output_name], {self.input_name: image_array})
        inv_depth = outputs[0]  # [1, 1, H, W]
        
        # Inverse depth를 depth로 변환
        depth = 1.0 / np.clip(inv_depth, a_min=1e-6, a_max=None)
        depth = np.clip(depth, a_min=0.1, a_max=100.0)
        
        # 원본 크기로 리사이즈
        depth_resized = self.resize_depth(depth[0, 0], original_size)
        
        return depth_resized
    
    def resize_depth(self, depth, target_size):
        """Resize depth map to target size"""
        # depth: (H, W), target_size: (width, height)
        target_width, target_height = target_size
        
        # Use OpenCV for resizing
        depth_resized = cv2.resize(depth, (target_width, target_height), 
                                 interpolation=cv2.INTER_LINEAR)
        
        return depth_resized


def load_test_files(data_path, split_file):
    """Load test file list from split file"""
    # split_file이 절대 경로인지 확인
    if os.path.isabs(split_file):
        split_path = split_file
    else:
        # 상대 경로인 경우 workspace 기준으로 처리
        split_path = os.path.join('/workspace/packnet-sfm', split_file)
    
    if not os.path.exists(split_path):
        print(f"❌ Split file not found: {split_path}")
        return []
    
    print(f"📁 Loading split file: {split_path}")
    
    test_files = []
    with open(split_path, 'r') as f:
        lines = f.readlines()
        print(f"📊 Total lines in split file: {len(lines)}")
        
        for line_num, line in enumerate(lines):
            # KITTI split file format: "left_image_path right_image_path"
            parts = line.strip().split()
            if len(parts) >= 2:
                left_image_path, right_image_path = parts[0], parts[1]
                
                # Debug: print first few entries
                if line_num < 3:
                    print(f"  Line {line_num}: {left_image_path} | {right_image_path}")
                
                # 우리는 left camera만 사용 (image_02)
                image_rel_path = left_image_path
                
                # Construct full image path
                image_path = os.path.join(data_path, image_rel_path)
                
                # Construct groundtruth depth path
                # 예: 2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000069.png
                # -> 2011_09_26/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000069.png
                
                # Split the path parts
                path_parts = image_rel_path.split('/')
                if len(path_parts) >= 4:  # [date, drive, image_folder, data, filename]
                    date_folder = path_parts[0]
                    drive_folder = path_parts[1] 
                    image_folder = path_parts[2]  # image_02 or image_03
                    filename = path_parts[4]      # 0000000069.png
                    
                    # Construct GT depth path
                    gt_rel_path = f"{date_folder}/{drive_folder}/proj_depth/groundtruth/{image_folder}/{filename}"
                    gt_path = os.path.join(data_path, gt_rel_path)
                    
                    # Check if both files exist
                    if os.path.exists(image_path) and os.path.exists(gt_path):
                        test_files.append((image_path, gt_path))
                        if line_num < 3:  # Debug: show first few found files
                            print(f"  ✅ Found: {os.path.basename(image_path)} -> {os.path.basename(gt_path)}")
                    else:
                        if line_num < 10:  # Debug: show first few missing files
                            print(f"  ❌ Missing files for line {line_num}:")
                            print(f"    Image: {image_path} (exists: {os.path.exists(image_path)})")
                            print(f"    GT: {gt_path} (exists: {os.path.exists(gt_path)})")
                else:
                    if line_num < 5:  # Debug: show malformed paths
                        print(f"  ⚠️  Invalid path format {line_num}: {image_rel_path}")
            else:
                if line_num < 5:  # Debug: show malformed lines
                    print(f"  ⚠️  Malformed line {line_num}: {line.strip()}")
    
    print(f"✅ Loaded {len(test_files)} test files")
    
    # 추가 디버깅: 실제 데이터 구조 확인
    if len(test_files) == 0:
        print("\n🔍 Debugging: Checking actual data structure...")
        
        # 첫 번째 split 라인으로 실제 구조 확인
        with open(split_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                parts = first_line.split()
                if len(parts) >= 2:
                    left_image_path = parts[0]
                    
                    print(f"  First split entry: {left_image_path}")
                    
                    # 실제 존재하는 디렉토리 확인
                    full_image_path = os.path.join(data_path, left_image_path)
                    print(f"  Full image path: {full_image_path}")
                    print(f"  Image exists: {os.path.exists(full_image_path)}")
                    
                    # 상위 디렉토리들 확인
                    path_parts = left_image_path.split('/')
                    current_path = data_path
                    for i, part in enumerate(path_parts):
                        current_path = os.path.join(current_path, part)
                        exists = os.path.exists(current_path)
                        print(f"    Level {i}: {current_path} (exists: {exists})")
                        if not exists:
                            # 현재 레벨에서 사용 가능한 디렉토리/파일 보기
                            parent_dir = os.path.dirname(current_path)
                            if os.path.exists(parent_dir):
                                available = os.listdir(parent_dir)[:10]  # 처음 10개만
                                print(f"      Available in {parent_dir}: {available}")
                            break
    
    return test_files


def create_packnet_compatible_args(args, config):
    """PackNet-SfM compute_depth_metrics 함수와 호환되는 args 객체 생성"""
    from argparse import Namespace
    
    # 원래 config에서 필요한 속성들 가져오기
    model_params = config.model.params if hasattr(config.model, 'params') else {}
    
    compatible_args = Namespace()
    
    # 기본 평가 파라미터들
    compatible_args.min_depth = args.min_depth
    compatible_args.max_depth = args.max_depth
    compatible_args.crop = args.crop
    compatible_args.use_gt_scale = args.use_gt_scale
    
    # PackNet-SfM 특정 파라미터들 (config에서 가져오거나 기본값 사용)
    compatible_args.scale_output = model_params.get('scale_output', 'top-center')
    
    # 추가로 필요할 수 있는 속성들
    if hasattr(config.datasets, 'augmentation'):
        augmentation = config.datasets.augmentation
        if hasattr(augmentation, 'crop_eval_borders'):
            compatible_args.crop_eval_borders = augmentation.crop_eval_borders
    
    return compatible_args


def evaluate_onnx_model(args):
    """Main evaluation function"""
    
    # Load configuration from checkpoint
    config, _ = parse_test_file(args.checkpoint, args.config)
    
    # PackNet-SfM 호환 args 생성
    packnet_args = create_packnet_compatible_args(args, config)
    
    # Initialize ONNX model
    onnx_estimator = ONNXDepthEstimator(args.onnx_model)
    
    # Load test files
    test_files = load_test_files(args.data_path, args.split_file)
    
    if len(test_files) == 0:
        print("❌ No test files found!")
        return
    
    # Create output directory
    if args.save_depth_maps:
        os.makedirs(args.output_folder, exist_ok=True)
    
    # Evaluation metrics
    metrics_list = []
    
    print(f"🚀 Starting ONNX evaluation on {len(test_files)} images...")
    print(f"📋 Evaluation settings:")
    print(f"   Min depth: {packnet_args.min_depth}m")
    print(f"   Max depth: {packnet_args.max_depth}m")
    print(f"   Crop: {packnet_args.crop}")
    print(f"   Scale output: {packnet_args.scale_output}")
    print(f"   GT scaling: {packnet_args.use_gt_scale}")
    
    # Process each test file
    progress_bar = tqdm(test_files, desc="Evaluating")
    
    for i, (image_path, gt_path) in enumerate(progress_bar):
        try:
            # ONNX inference
            pred_depth = onnx_estimator.predict(image_path)
            
            # Load ground truth
            gt_depth = load_depth(gt_path)
            
            # Convert to tensors
            pred_tensor = torch.from_numpy(pred_depth).unsqueeze(0).unsqueeze(0).float()
            gt_tensor = torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(0).float()
            
            # Compute metrics using PackNet-SfM's function with compatible args
            metrics = compute_depth_metrics(
                packnet_args, gt_tensor, pred_tensor, use_gt_scale=args.use_gt_scale
            )
            
            metrics_list.append(metrics.detach().cpu().numpy())
            
            # Save depth map if requested
            if args.save_depth_maps:
                output_path = os.path.join(args.output_folder, f"depth_{i:06d}.npz")
                np.savez_compressed(output_path, depth=pred_depth)
            
            # Update progress
            if len(metrics_list) % 50 == 0:
                current_metrics = np.mean(metrics_list, axis=0)
                progress_bar.set_postfix({
                    'abs_rel': f'{current_metrics[0]:.4f}',
                    'rmse': f'{current_metrics[2]:.4f}'
                })
        
        except Exception as e:
            print(f"❌ Error processing {image_path}: {str(e)}")
            # 첫 번째 에러에서 상세 디버깅 정보 출력
            if len(metrics_list) == 0:
                print(f"🔍 Debug info for first error:")
                print(f"   packnet_args attributes: {dir(packnet_args)}")
                print(f"   pred_tensor shape: {pred_tensor.shape if 'pred_tensor' in locals() else 'Not created'}")
                print(f"   gt_tensor shape: {gt_tensor.shape if 'gt_tensor' in locals() else 'Not created'}")
                import traceback
                traceback.print_exc()
            continue
    
    # Calculate final metrics
    if len(metrics_list) == 0:
        print("❌ No successful evaluations!")
        return
    
    final_metrics = np.mean(metrics_list, axis=0)
    metric_names = ['abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
    
    print("\n" + "="*60)
    print("📊 ONNX MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {args.onnx_model}")
    print(f"Dataset: KITTI ({len(metrics_list)} images)")
    print(f"Input size: {onnx_estimator.input_width}x{onnx_estimator.input_height}")
    print(f"Depth range: {args.min_depth} - {args.max_depth}m")
    print(f"Crop: {args.crop}")
    print(f"GT scaling: {args.use_gt_scale}")
    print("-"*60)
    
    for name, metric in zip(metric_names, final_metrics):
        print(f"{name:>10s}: {metric:.4f}")
    
    print("="*60)
    
    # Save results
    results_file = os.path.join(args.output_folder, 'evaluation_results.txt')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("ONNX Model Evaluation Results\n")
        f.write("="*40 + "\n")
        f.write(f"Model: {args.onnx_model}\n")
        f.write(f"Dataset: KITTI ({len(metrics_list)} images)\n")
        f.write(f"Input size: {onnx_estimator.input_width}x{onnx_estimator.input_height}\n")
        f.write(f"Depth range: {args.min_depth} - {args.max_depth}m\n")
        f.write(f"Crop: {args.crop}\n")
        f.write(f"GT scaling: {args.use_gt_scale}\n")
        f.write("-"*40 + "\n")
        
        for name, metric in zip(metric_names, final_metrics):
            f.write(f"{name:>10s}: {metric:.4f}\n")
    
    print(f"📁 Results saved to: {results_file}")
    
    if args.save_depth_maps:
        print(f"📁 Depth maps saved to: {args.output_folder}")


def main():
    """Main function"""
    args = parse_args()
    
    # Check if onnxruntime is available
    try:
        import onnxruntime as ort
        print(f"✅ ONNXRuntime version: {ort.__version__}")
    except ImportError:
        print("❌ ONNXRuntime not found. Install with: pip install onnxruntime")
        return
    
    # Run evaluation
    evaluate_onnx_model(args)


if __name__ == '__main__':
    main()