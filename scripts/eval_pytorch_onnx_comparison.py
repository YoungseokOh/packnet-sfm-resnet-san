#!/usr/bin/env python3
# filepath: /workspace/packnet-sfm/scripts/eval_pytorch_onnx_comparison.py

import argparse
import torch
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import os
import onnxruntime as ort
import traceback

from packnet_sfm.utils.depth import load_depth, compute_depth_metrics
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.misc import parse_crop_borders
from scripts.convert_to_onnx import load_model_simple, SimpleDepthNet


def parse_args():
    """Parse arguments for evaluation script"""
    parser = argparse.ArgumentParser(description='PackNet-SfM PyTorch vs ONNX evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Original checkpoint (.ckpt)')
    parser.add_argument('--onnx_model', type=str, required=True,
                        help='ONNX model (.onnx)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to KITTI dataset')
    parser.add_argument('--split_file', type=str, 
                        default='data_splits/eigen_test_files.txt',
                        help='Split file for evaluation')
    parser.add_argument('--min_depth', type=float, default=0.0,
                        help='Minimum depth for evaluation')
    parser.add_argument('--max_depth', type=float, default=80.0,
                        help='Maximum depth for evaluation')
    parser.add_argument('--crop', type=str, default='garg', choices=['', 'garg'],
                        help='Crop type for evaluation')
    parser.add_argument('--use_gt_scale', action='store_true', default=True,
                        help='Use ground-truth median scaling')
    parser.add_argument('--num_images', type=int, default=None,
                        help='Number of images to evaluate (None for all)')
    
    return parser.parse_args()


def load_test_files(data_path, split_file):
    """Load test file list from split file"""
    if os.path.isabs(split_file):
        split_path = split_file
    else:
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
            parts = line.strip().split()
            if len(parts) >= 2:
                left_image_path = parts[0]
                
                # Construct full image path
                image_path = os.path.join(data_path, left_image_path)
                
                # Construct groundtruth depth path
                path_parts = left_image_path.split('/')
                if len(path_parts) >= 4:
                    date_folder = path_parts[0]
                    drive_folder = path_parts[1] 
                    image_folder = path_parts[2]
                    filename = path_parts[4]
                    
                    gt_rel_path = f"{date_folder}/{drive_folder}/proj_depth/groundtruth/{image_folder}/{filename}"
                    gt_path = os.path.join(data_path, gt_rel_path)
                    
                    if os.path.exists(image_path) and os.path.exists(gt_path):
                        test_files.append((image_path, gt_path))
    
    print(f"✅ Loaded {len(test_files)} test files")
    return test_files


def create_packnet_compatible_args(args, config):
    """PackNet-SfM compute_depth_metrics 함수와 호환되는 args 객체 생성"""
    from argparse import Namespace
    
    model_params = config.model.params if hasattr(config.model, 'params') else {}
    
    compatible_args = Namespace()
    compatible_args.min_depth = args.min_depth
    compatible_args.max_depth = args.max_depth
    compatible_args.crop = args.crop
    compatible_args.use_gt_scale = args.use_gt_scale
    compatible_args.scale_output = model_params.get('scale_output', 'top-center')
    
    if hasattr(config.datasets, 'augmentation'):
        augmentation = config.datasets.augmentation
        if hasattr(augmentation, 'crop_eval_borders'):
            compatible_args.crop_eval_borders = augmentation.crop_eval_borders
    
    return compatible_args


class PyTorchEvaluator:
    """PyTorch 모델 평가기"""
    
    def __init__(self, checkpoint_path):
        print("🔧 Loading PyTorch model...")
        self.depth_net, self.config = load_model_simple(checkpoint_path)
        self.model = SimpleDepthNet(self.depth_net)
        self.model.eval()
    
    def predict(self, rgb_image):
        """RGB 이미지에서 깊이 예측"""
        # PackNet-SfM과 동일한 전처리
        from torchvision import transforms
        to_tensor = transforms.ToTensor()
        
        # 1216x352로 리사이즈 (원본 PackNet-SfM 크기)
        rgb_resized = rgb_image.resize((1216, 352), Image.LANCZOS)
        rgb_tensor = to_tensor(rgb_resized).unsqueeze(0)
        
        with torch.no_grad():
            inv_depth = self.model(rgb_tensor)
            depth_pred = 1.0 / torch.clamp(inv_depth, min=1e-6)
            depth_pred = torch.clamp(depth_pred, min=0.1, max=100.0)
            depth_pred_np = depth_pred[0, 0].cpu().numpy()
        
        return depth_pred_np


class ONNXEvaluator:
    """ONNX 모델 평가기"""
    
    def __init__(self, onnx_model_path):
        print("🔧 Loading ONNX model...")
        self.session = ort.InferenceSession(onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        
        print(f"📐 ONNX Model Input Shape: {input_shape}")
    
    def predict(self, rgb_image):
        """RGB 이미지에서 깊이 예측"""
        # ONNX 모델 입력 크기로 리사이즈
        rgb_resized = rgb_image.resize((self.input_width, self.input_height), Image.LANCZOS)
        
        # 텐서 변환 및 정규화
        rgb_array = np.array(rgb_resized, dtype=np.float32) / 255.0
        rgb_array = rgb_array.transpose(2, 0, 1)[np.newaxis, :]
        
        # ONNX 추론
        outputs = self.session.run([self.output_name], {self.input_name: rgb_array})
        inv_depth = outputs[0]
        
        # Depth 변환
        depth = 1.0 / np.clip(inv_depth, a_min=1e-6, a_max=None)
        depth = np.clip(depth, a_min=0.1, a_max=100.0)
        depth_pred_np = depth[0, 0]
        
        return depth_pred_np


def evaluate_models(args):
    """메인 평가 함수"""
    
    # 설정 로드
    config, _ = parse_test_file(args.checkpoint, None)
    packnet_args = create_packnet_compatible_args(args, config)
    
    # 모델 로드
    pytorch_evaluator = PyTorchEvaluator(args.checkpoint)
    onnx_evaluator = ONNXEvaluator(args.onnx_model)
    
    # 테스트 파일 로드
    test_files = load_test_files(args.data_path, args.split_file)
    
    if len(test_files) == 0:
        print("❌ No test files found!")
        return
    
    # 평가할 이미지 수 제한
    if args.num_images is not None:
        test_files = test_files[:args.num_images]
        print(f"🔢 Evaluating on {len(test_files)} images")
    
    # 평가 메트릭
    pytorch_metrics_list = []
    onnx_metrics_list = []
    
    print(f"\n🚀 Starting evaluation on {len(test_files)} images...")
    print(f"📋 Settings:")
    print(f"   Min depth: {args.min_depth}m")
    print(f"   Max depth: {args.max_depth}m")
    print(f"   Crop: {args.crop}")
    print(f"   GT scaling: {args.use_gt_scale}")
    print(f"   Crop eval borders: {getattr(packnet_args, 'crop_eval_borders', 'Not set')}")
    
    # 각 테스트 파일 처리
    progress_bar = tqdm(test_files, desc="Evaluating PyTorch vs ONNX")
    
    for i, (image_path, gt_path) in enumerate(progress_bar):
        try:
            # 이미지와 GT 로드
            rgb_image = Image.open(image_path).convert('RGB')
            gt_depth = load_depth(gt_path)  # 원본 크기 유지!
            
            if i == 0:
                print(f"\n🔍 First image debug:")
                print(f"   Original RGB: {rgb_image.size}")
                print(f"   Original GT depth: {gt_depth.shape}")
            
            # 🆕 올바른 crop 처리 (원본 PackNet-SfM 방식)
            rgb_cropped = rgb_image
            if hasattr(packnet_args, 'crop_eval_borders'):
                crop_eval_borders = packnet_args.crop_eval_borders
                
                # parse_crop_borders 함수 사용 (원본 방식)
                image_hw = rgb_image.size[::-1]  # (height, width) for parse_crop_borders
                borders = parse_crop_borders(crop_eval_borders, image_hw)
                
                # PIL crop format: (left, top, right, bottom)
                crop_box = tuple(borders)
                rgb_cropped = rgb_image.crop(crop_box)
                
                if i == 0:
                    print(f"   Crop eval borders: {crop_eval_borders}")
                    print(f"   Image HW for parsing: {image_hw}")
                    print(f"   Parsed borders: {borders}")
                    print(f"   RGB after crop: {rgb_cropped.size}")
            
            # PyTorch 예측
            pytorch_depth = pytorch_evaluator.predict(rgb_cropped)
            
            # ONNX 예측
            onnx_depth = onnx_evaluator.predict(rgb_cropped)
            
            if i == 0:
                print(f"   PyTorch pred: {pytorch_depth.shape}")
                print(f"   ONNX pred: {onnx_depth.shape}")
                print(f"   GT depth (original): {gt_depth.shape}")
                print("   (compute_depth_metrics will handle size mismatch)")
                print("   (Detailed logging disabled for remaining images)")
            
            # GT 텐서 생성
            gt_tensor = torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(0).float()
            
            # PyTorch 메트릭 계산
            pytorch_pred_tensor = torch.from_numpy(pytorch_depth).unsqueeze(0).unsqueeze(0).float()
            pytorch_metrics = compute_depth_metrics(
                packnet_args, gt_tensor, pytorch_pred_tensor, use_gt_scale=args.use_gt_scale
            )
            pytorch_metrics_list.append(pytorch_metrics.detach().cpu().numpy())
            
            # ONNX 메트릭 계산
            onnx_pred_tensor = torch.from_numpy(onnx_depth).unsqueeze(0).unsqueeze(0).float()
            onnx_metrics = compute_depth_metrics(
                packnet_args, gt_tensor, onnx_pred_tensor, use_gt_scale=args.use_gt_scale
            )
            onnx_metrics_list.append(onnx_metrics.detach().cpu().numpy())
            
            # 프로그레스 업데이트
            if len(pytorch_metrics_list) % 50 == 0:
                pytorch_current = np.mean(pytorch_metrics_list, axis=0)
                onnx_current = np.mean(onnx_metrics_list, axis=0)
                progress_bar.set_postfix({
                    'PT_abs_rel': f'{pytorch_current[0]:.4f}',
                    'ONNX_abs_rel': f'{onnx_current[0]:.4f}',
                    'diff': f'{onnx_current[0] - pytorch_current[0]:+.4f}'
                })
        
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(image_path)}: {str(e)}")
            if i < 5:
                traceback.print_exc()
            continue
    
    # 결과 출력
    if len(pytorch_metrics_list) == 0 or len(onnx_metrics_list) == 0:
        print("❌ No successful evaluations!")
        return
    
    pytorch_final = np.mean(pytorch_metrics_list, axis=0)
    onnx_final = np.mean(onnx_metrics_list, axis=0)
    
    metric_names = ['abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
    
    print("\n" + "="*80)
    print("📊 PYTORCH vs ONNX MODEL COMPARISON")
    print("="*80)
    print(f"Dataset: KITTI ({len(pytorch_metrics_list)} images)")
    print(f"PyTorch checkpoint: {args.checkpoint}")
    print(f"ONNX model: {args.onnx_model}")
    print("-"*80)
    print(f"{'Metric':<12} | {'PyTorch':<10} | {'ONNX':<10} | {'Difference':<12}")
    print("-"*80)
    
    for i, name in enumerate(metric_names):
        pytorch_val = pytorch_final[i]
        onnx_val = onnx_final[i]
        diff = onnx_val - pytorch_val
        print(f"{name:<12} | {pytorch_val:<10.4f} | {onnx_val:<10.4f} | {diff:<+12.4f}")
    
    print("="*80)
    
    # 전체적인 성능 차이 요약
    abs_rel_diff = onnx_final[0] - pytorch_final[0]
    rmse_diff = onnx_final[2] - pytorch_final[2]
    
    print(f"\n📈 Performance Summary:")
    print(f"   abs_rel difference: {abs_rel_diff:+.4f} ({'better' if abs_rel_diff < 0 else 'worse'} for ONNX)")
    print(f"   rmse difference: {rmse_diff:+.4f} ({'better' if rmse_diff < 0 else 'worse'} for ONNX)")
    
    if abs(abs_rel_diff) < 0.001:
        print("   ✅ Models are very similar in performance!")
    elif abs_rel_diff < 0:
        print("   🎯 ONNX model performs better!")
    else:
        print("   ⚠️  PyTorch model performs better.")


def main():
    """메인 함수"""
    args = parse_args()
    
    # 파일 존재 확인
    assert os.path.exists(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    assert os.path.exists(args.onnx_model), f"ONNX model not found: {args.onnx_model}"
    assert os.path.exists(args.data_path), f"Data path not found: {args.data_path}"
    
    # ONNXRuntime 확인
    try:
        print(f"✅ ONNXRuntime version: {ort.__version__}")
    except ImportError:
        print("❌ ONNXRuntime not found. Install with: pip install onnxruntime")
        return
    
    # 평가 실행
    evaluate_models(args)


if __name__ == '__main__':
    main()