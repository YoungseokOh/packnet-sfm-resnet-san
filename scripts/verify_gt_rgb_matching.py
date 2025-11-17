#!/usr/bin/env python3
"""
Verify GT-RGB matching by visualizing:
1. Original RGB image
2. GT depth as colored points (larger size for visibility)
3. GT depth projected on RGB (overlay)
"""

import numpy as np
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_gt_depth(gt_path):
    """Load GT depth from 16-bit PNG (value/256 = meters)"""
    gt_img = Image.open(gt_path)
    gt_depth = np.array(gt_img, dtype=np.float32) / 256.0
    return gt_depth


def visualize_gt_rgb_matching(rgb_path, gt_path, output_path, point_size=20):
    """
    Create 3-panel visualization:
    1. RGB image
    2. GT depth as colored scatter points
    3. GT depth projected on RGB
    """
    # Load images
    rgb_img = Image.open(rgb_path).convert('RGB')
    rgb_array = np.array(rgb_img)
    gt_depth = load_gt_depth(gt_path)
    
    height, width = gt_depth.shape
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    # Panel 1: Original RGB
    axes[0].imshow(rgb_array)
    axes[0].set_title('RGB Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: GT Depth as colored points
    # Create coordinate grids
    valid_mask = gt_depth > 0
    y_coords, x_coords = np.where(valid_mask)
    depth_values = gt_depth[valid_mask]
    
    # Normalize depth for colormap
    depth_min, depth_max = 0.5, 15.0
    depth_normalized = np.clip((depth_values - depth_min) / (depth_max - depth_min), 0, 1)
    
    # Use jet colormap for better visibility
    cmap = plt.get_cmap('jet')
    colors = cmap(depth_normalized)
    
    axes[1].set_facecolor('black')  # Black background for better contrast
    scatter = axes[1].scatter(x_coords, y_coords, c=depth_values, 
                              s=point_size, cmap='jet', 
                              vmin=depth_min, vmax=depth_max,
                              marker='s', alpha=0.8)  # Square markers for better coverage
    axes[1].set_xlim(0, width)
    axes[1].set_ylim(height, 0)  # Invert y-axis to match image coordinates
    axes[1].set_aspect('equal')
    axes[1].set_title(f'GT Depth Points (size={point_size})', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Depth (m)', fontsize=12)
    
    # Panel 3: GT projected on RGB (overlay)
    axes[2].imshow(rgb_array)
    
    # Create colored overlay
    overlay_scatter = axes[2].scatter(x_coords, y_coords, c=depth_values,
                                     s=point_size, cmap='jet',
                                     vmin=depth_min, vmax=depth_max,
                                     marker='s', alpha=0.6)  # Semi-transparent
    axes[2].set_xlim(0, width)
    axes[2].set_ylim(height, 0)
    axes[2].set_title('GT Depth Projected on RGB', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Add colorbar
    cbar2 = plt.colorbar(overlay_scatter, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label('Depth (m)', fontsize=12)
    
    # Add metadata
    image_name = Path(rgb_path).stem
    gt_name = Path(gt_path).stem
    info_text = f"RGB: {image_name} | GT: {gt_name} | Valid GT pixels: {len(depth_values):,}"
    fig.suptitle(info_text, fontsize=12, y=0.02)
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    print(f"   RGB: {image_name}")
    print(f"   GT:  {gt_name}")
    print(f"   Valid GT pixels: {len(depth_values):,}/{height*width} ({len(depth_values)/(height*width)*100:.1f}%)")
    print(f"   Depth range: {depth_values.min():.2f}m - {depth_values.max():.2f}m")


def main():
    # Base paths
    base_dir = Path('Fin_Test_Set_ncdb')
    rgb_dir = base_dir / 'RGB'
    gt_dir = base_dir / 'GT'
    output_dir = base_dir / 'gt_rgb_verification'
    output_dir.mkdir(exist_ok=True)
    
    # Get ALL RGB files from the directory
    rgb_files = sorted(rgb_dir.glob('*.png'))
    
    print("=" * 80)
    print("GT-RGB MATCHING VERIFICATION (ALL SAMPLES)")
    print("=" * 80)
    print(f"\nVisualizing {len(rgb_files)} samples with large point size for clarity\n")
    
    point_size = 15  # Larger points for better visibility
    
    for idx, rgb_path in enumerate(rgb_files, 1):
        # Get corresponding GT file (same filename)
        rgb_filename = rgb_path.name
        gt_path = gt_dir / rgb_filename
        
        # Check if files exist
        if not rgb_path.exists():
            print(f"❌ RGB not found: {rgb_path}")
            continue
        if not gt_path.exists():
            print(f"❌ GT not found: {gt_path}")
            continue
        
        # Create visualization
        output_path = output_dir / f"{Path(rgb_filename).stem}_verification.png"
        
        print(f"\n[{idx}/{len(rgb_files)}] Processing: {Path(rgb_filename).stem}")
        print("-" * 80)
        
        visualize_gt_rgb_matching(rgb_path, gt_path, output_path, point_size=point_size)
    
    print("\n" + "=" * 80)
    print(f"✅ Verification complete! {len(rgb_files)} images saved to: {output_dir}/")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Open the verification images to check if GT points align with RGB content")
    print("2. Look for obvious mismatches (e.g., GT shows objects not in RGB)")
    print("3. If filenames match but content doesn't align, there's a data issue")
    print(f"\n📊 Summary:")
    print(f"   Total samples: {len(rgb_files)}")
    print(f"   Output directory: {output_dir}/")
    print(f"   Browse: {output_dir}/index.html (if generated)")


if __name__ == '__main__':
    main()
