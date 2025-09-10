import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Pillow 라이브러리 로드 시도 및 미설치 시 안내
try:
    from PIL import Image
except ImportError:
    print("경고: Pillow 라이브러리를 찾을 수 없습니다. 이미지 관련 시각화를 수행할 수 없습니다.", file=sys.stderr)
    print("라이브러리를 설치하려면 다음 명령을 실행하세요: pip install Pillow", file=sys.stderr)
    Image = None

def load_lut_data(lut_path: Path) -> dict:
    """지정된 경로에서 LUT pickle 파일을 로드합니다."""
    if not lut_path.exists():
        raise FileNotFoundError(f"LUT file not found at: {lut_path}")
    
    print(f"📂 Loading LUT file from: {lut_path}")
    with open(lut_path, 'rb') as f:
        lut_data = pickle.load(f)
    
    # 메타데이터와 LUT 데이터 유효성 검사
    if 'metadata' not in lut_data or 'angle_lut' not in lut_data or 'theta_lut' not in lut_data:
        raise ValueError("Invalid LUT file format. Required keys: 'metadata', 'angle_lut', 'theta_lut'")
        
    print("✅ LUT data loaded successfully.")
    return lut_data

def visualize_lut(lut_data: dict, output_dir: Path):
    """LUT 데이터를 순수 데이터 형태로 시각화하고 이미지 파일로 저장합니다."""
    
    meta = lut_data['metadata']
    H, W = meta['image_size']
    
    angle_lut_2d = lut_data['angle_lut'].reshape(H, W)
    theta_lut_2d = lut_data['theta_lut'].reshape(H, W)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"VADAS Lookup Table Visualization ({W}x{H})", fontsize=16)
    
    im1 = axes[0].imshow(np.degrees(angle_lut_2d), cmap='viridis')
    axes[0].set_title('Angle LUT (degrees)')
    axes[0].set_xlabel('Image Width')
    axes[0].set_ylabel('Image Height')
    fig.colorbar(im1, ax=axes[0], label='Angle (°)')
    
    im2 = axes[1].imshow(np.degrees(theta_lut_2d), cmap='hsv')
    axes[1].set_title('Theta LUT (degrees)')
    axes[1].set_xlabel('Image Width')
    axes[1].set_ylabel('Image Height')
    fig.colorbar(im2, ax=axes[1], label='Theta (°)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"lut_visualization_{W}x{H}.png"
    
    plt.savefig(output_path)
    print(f"💾 LUT-only visualization saved to: {output_path}")
    plt.close(fig)

def visualize_lut_on_image(lut_data: dict, image_path: Path, output_dir: Path):
    """LUT 데이터를 실제 어안 이미지 위에 오버레이 및 등고선 형태로 시각화합니다."""
    if Image is None:
        print("Skipping image visualization because Pillow library is not installed.")
        return

    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"❌ Error: Image file not found at {image_path}")
        return
    except Exception as e:
        print(f"❌ Error opening image file: {e}")
        return

    meta = lut_data['metadata']
    H, W = meta['image_size']
    
    if image.size != (W, H):
        print(f"⚠️ Warning: Image size {image.size} does not match LUT size {(W, H)}. Resizing image to fit.")
        image = image.resize((W, H))

    angle_lut_2d = lut_data['angle_lut'].reshape(H, W)
    theta_lut_2d = lut_data['theta_lut'].reshape(H, W)

    # 1. 오버레이 시각화
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
    fig.suptitle(f"LUT Overlay on '{image_path.name}'", fontsize=16)
    
    axes[0].imshow(image)
    im1 = axes[0].imshow(np.degrees(angle_lut_2d), cmap='viridis', alpha=0.5)
    axes[0].set_title('Angle LUT Overlay')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], label='Angle (°)')

    axes[1].imshow(image)
    im2 = axes[1].imshow(np.degrees(theta_lut_2d), cmap='hsv', alpha=0.5)
    axes[1].set_title('Theta LUT Overlay')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], label='Theta (°)')
    
    output_path = output_dir / f"lut_overlay_{W}x{H}.png"
    plt.savefig(output_path)
    print(f"💾 Overlay visualization saved to: {output_path}")
    plt.close(fig)

    # 2. 등고선 시각화
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
    fig.suptitle(f"LUT Contours on '{image_path.name}'", fontsize=16)

    axes[0].imshow(image)
    contour1 = axes[0].contour(np.degrees(angle_lut_2d), levels=15, cmap='viridis', linewidths=1.5)
    axes[0].clabel(contour1, inline=True, fontsize=9, fmt='%1.1f°')
    axes[0].set_title('Angle LUT Contours (degrees)')
    axes[0].axis('off')

    axes[1].imshow(image)
    # Theta 값의 주기적 특성을 고려하여 등고선 레벨 설정 (0-360도)
    levels = np.arange(0, 361, 30)
    contour2 = axes[1].contour(np.degrees(theta_lut_2d), levels=levels, cmap='hsv', linewidths=1.5)
    axes[1].clabel(contour2, inline=True, fontsize=9, fmt='%1.0f°')
    axes[1].set_title('Theta LUT Contours (degrees)')
    axes[1].axis('off')

    output_path = output_dir / f"lut_contours_{W}x{H}.png"
    plt.savefig(output_path)
    print(f"💾 Contour visualization saved to: {output_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="VADAS Lookup Table(LuT) 시각화 및 검증 스크립트.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--lut_path", type=str, required=True,
                        help="시각화할 LuT pickle 파일 경로.")
    parser.add_argument("--output_dir", type=str, default="debug_outputs/lut_verification",
                        help="시각화 결과 이미지를 저장할 디렉토리.")
    parser.add_argument("--image_path", type=str, default=None,
                        help="(선택 사항) LuT를 오버레이할 어안 이미지 경로.\n" 
                             "이 옵션을 사용하면 오버레이 및 등고선 시각화를 추가로 수행합니다.")
    
    args = parser.parse_args()
    
    lut_path = Path(args.lut_path)
    output_dir = Path(args.output_dir)
    
    try:
        lut_data = load_lut_data(lut_path)
        
        # 1. 기본 LuT 데이터 시각화 (항상 수행)
        visualize_lut(lut_data, output_dir)
        
        # 2. 이미지가 제공된 경우, 추가 시각화 수행
        if args.image_path:
            image_path = Path(args.image_path)
            visualize_lut_on_image(lut_data, image_path, output_dir)

        print("\n✅ Verification script finished successfully.")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()