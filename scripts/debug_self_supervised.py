#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('/workspace/packnet-sfm')

from packnet_sfm.models.SemiSupCompletionModel import SemiSupCompletionModel
from packnet_sfm.datasets.augmentations import resize_image, resize_depth_preserve, to_tensor
from packnet_sfm.datasets.transforms import get_transforms
from packnet_sfm.utils.config import parse_train_config
from packnet_sfm.utils.load import set_debug
from packnet_sfm.datasets.ncdb_dataset import NcdbDataset
from torch.utils.data import DataLoader

def validate_self_supervised_basics(model, batch, save_dir="./debug_outputs"):
    """Self-supervised 기본 동작 검증"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print("=== Self-Supervised Debug Report ===")
    
    # 1. 기본 배치 정보 확인
    print(f"Batch keys: {list(batch.keys())}")
    print(f"RGB shape: {batch['rgb'].shape}")
    if 'rgb_context' in batch:
        print(f"Context frames: {len(batch['rgb_context'])}")
        for i, ctx in enumerate(batch['rgb_context']):
            print(f"  Context {i}: {ctx.shape}")
    
    # 2. Depth 예측 확인
    with torch.no_grad():
        model.eval()
        depth_output = model.depth_net(batch['rgb'])
        
        if isinstance(depth_output['inv_depths'], list):
            print(f"Depth scales: {len(depth_output['inv_depths'])}")
            for i, inv_depth in enumerate(depth_output['inv_depths']):
                depth = 1.0 / (inv_depth + 1e-7)
                print(f"  Scale {i}: {inv_depth.shape}, depth range: {depth.min():.3f} - {depth.max():.3f}m")
                
                # Save first scale depth visualization
                if i == 0:
                    depth_vis = depth[0, 0].cpu().numpy()
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(batch['rgb'][0].permute(1, 2, 0).cpu().numpy())
                    plt.title('Input RGB')
                    plt.axis('off')
                    plt.subplot(1, 2, 2)
                    plt.imshow(depth_vis, cmap='jet', vmin=0, vmax=50)
                    plt.colorbar()
                    plt.title('Predicted Depth (m)')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(save_dir / 'depth_prediction.png', dpi=150)
                    plt.close()
        else:
            depth = 1.0 / (depth_output['inv_depths'] + 1e-7)
            print(f"Single depth: {depth_output['inv_depths'].shape}, range: {depth.min():.3f} - {depth.max():.3f}m")
    
    # 3. Pose 예측 확인 (context frames 있을 때)
    if 'rgb_context' in batch and len(batch['rgb_context']) > 0 and hasattr(model, 'pose_net') and model.pose_net is not None:
        with torch.no_grad():
            pose_output = model.pose_net(batch['rgb'], batch['rgb_context'])
            print(f"Pose predictions: {pose_output.shape}")
            
            # Pose 값 범위 확인
            translation = pose_output[:, :, :3]  # [B, num_frames, 3]
            rotation = pose_output[:, :, 3:]     # [B, num_frames, 3] (axis-angle)
            print(f"Translation range: {translation.abs().mean():.6f} ± {translation.std():.6f}")
            print(f"Rotation range: {rotation.abs().mean():.6f} ± {rotation.std():.6f}")
    
    # 4. Distortion coeffs 확인
    if 'distortion_coeffs' in batch:
        print("Distortion coefficients found:")
        for key, value in batch['distortion_coeffs'].items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}, range: {value.min():.6f} - {value.max():.6f}")
            else:
                print(f"  {key}: {type(value)}")
    else:
        print("No distortion coefficients in batch")
    
    # 5. Self-supervised loss 시도
    if 'rgb_context' in batch and len(batch['rgb_context']) > 0:
        try:
            model.train()  # Training mode for loss calculation
            forward_output = model.forward(batch, return_logs=True)
            print(f"Forward pass successful!")
            print(f"Total loss: {forward_output['loss'].item():.6f}")
            
            if 'metrics' in forward_output:
                print("Loss components:")
                for key, value in forward_output['metrics'].items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        print(f"  {key}: {value.item():.6f}")
                        
        except Exception as e:
            print(f"Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No context frames - skipping self-supervised loss test")
    
    print("=== Debug Complete ===")

def main():
    # Config 로드 (수정됨)
    cfg_default = "configs/default_config"
    cfg_file = "/workspace/packnet-sfm/configs/train_resnet_san_ncdb_self_sup+supervised.yaml"
    config = parse_train_config(cfg_default, cfg_file)
    
    # 강제로 context frames 설정
    config.datasets.train.back_context = 1
    config.datasets.train.forward_context = 1
    
    print("📋 Configuration Summary:")
    print(f"   - Model: {config.model.name}")
    print(f"   - Supervised weight: {config.model.loss.supervised_loss_weight}")
    print(f"   - Back context: {config.datasets.train.back_context}")
    print(f"   - Forward context: {config.datasets.train.forward_context}")
    print(f"   - Batch size: {config.datasets.train.batch_size}")
    
    # Model 초기화 (SemiSupCompletionModel with supervised_loss_weight=0.1 for testing)
    config.model.loss.supervised_loss_weight = 0.1  # Light supervision for testing
    model = SemiSupCompletionModel(**config.model)
    model.cuda()
    
    print(f"✅ Model initialized: {type(model).__name__}")
    print(f"   - Has depth_net: {hasattr(model, 'depth_net')}")
    print(f"   - Has pose_net: {hasattr(model, 'pose_net')}")
    print(f"   - Has self_supervised_loss: {hasattr(model, 'self_supervised_loss')}")
    
    # Dataset 초기화 - 올바른 파라미터 이름 사용
    dataset_args = {
        'dataset_root': config.datasets.train.path[0],
        'split_file': config.datasets.train.split[0],
        'back_context': config.datasets.train.back_context,
        'forward_context': config.datasets.train.forward_context,
    }
    # 🔧 Ensure tensors via standard train transforms (no advanced aug)
    transform = get_transforms(
        mode='train',
        image_shape=config.datasets.augmentation.image_shape,
        jittering=config.datasets.augmentation.jittering,
        crop_train_borders=config.datasets.augmentation.crop_train_borders,
    )
    dataset = NcdbDataset(**dataset_args, transform=transform)
    
    print(f"✅ Dataset initialized: {len(dataset)} samples")
    print(f"   - Back context: {dataset.backward_context}")
    print(f"   - Forward context: {dataset.forward_context}")
    print(f"   - With context: {dataset.with_context}")
    
    # DataLoader 생성
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=dataset.custom_collate_fn)
    
    # 첫 번째 배치로 테스트
    print("\n🧪 Testing with first batch...")
    try:
        batch = next(iter(dataloader))
        print("✅ Batch loaded successfully")
        print(f"   - Batch keys: {batch.keys()}")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"   - {key}: {value.shape} ({value.dtype})")
            elif isinstance(value, list):
                print(f"   - {key}: list of {len(value)} items")
                if len(value) > 0 and isinstance(value[0], torch.Tensor):
                    for i, item in enumerate(value):
                        print(f"     [{i}]: {item.shape} ({item.dtype})")
            else:
                print(f"   - {key}: {type(value)}")
        
        # GPU로 이동
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.cuda()
            elif isinstance(value, list):
                batch[key] = [v.cuda() if isinstance(v, torch.Tensor) else v for v in value]
            elif isinstance(value, dict):
                batch[key] = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                             for k, v in value.items()}
        
        # 검증 실행
        validate_self_supervised_basics(model, batch)
        
    except Exception as e:
        print(f"❌ Failed to load or process batch: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()