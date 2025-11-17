#!/usr/bin/env python3
"""
NCDB Video Projection Visualization
RGB에 GT depth와 NPU depth를 오버레이하여 비교 시각화
Proper 3D point cloud projection using VADAS fisheye camera model
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import cv2
from pathlib import Path
import argparse


# ========== VADAS Fisheye Camera Model (from ref code) ==========

class VADASFisheyeCameraModel:
    """VADAS Polynomial Fisheye Camera Model"""
    def __init__(self, intrinsic, image_size=(640, 384), original_size=None):
        self.original_intrinsic = intrinsic.copy()
        self.image_size = image_size
        
        # NO SCALING - assume intrinsic is already for target size
        self.k = intrinsic[0:7]
        self.s = intrinsic[7]
        self.div = intrinsic[8]
        self.ux = intrinsic[9]
        self.uy = intrinsic[10]

    def _poly_eval(self, coeffs, x):
        res = 0.0
        for c in reversed(coeffs):
            res = res * x + c
        return res

    def project_point(self, Xc, Yc, Zc):
        """
        Project 3D camera coordinate to 2D image pixel
        Returns: (u, v, valid)
        """
        nx = -Yc
        ny = -Zc
        
        dist = math.hypot(nx, ny)
        
        if dist < sys.float_info.epsilon:
            dist = sys.float_info.epsilon
        
        cosPhi = nx / dist
        sinPhi = ny / dist
        
        theta = math.atan2(dist, Xc)

        if Xc <= 0:  # Point is behind the camera
            return 0, 0, False
        
        xd = theta * self.s

        if abs(self.div) < 1e-9:
            return 0, 0, False
        
        rd = self._poly_eval(self.k, xd) / self.div

        if math.isinf(rd) or math.isnan(rd):
            return 0, 0, False

        img_w_half = self.image_size[0] / 2
        img_h_half = self.image_size[1] / 2

        u = rd * cosPhi + self.ux + img_w_half
        v = rd * sinPhi + self.uy + img_h_half
        
        return int(round(u)), int(round(v)), True

    def unproject_pixel_to_camera_coords(self, u, v, Xc_depth):
        """
        Unproject 2D pixel + Xc (forward distance) to full 3D camera coordinate
        
        GT depth stores Xc (camera forward distance), not ray distance.
        Given pixel (u,v) and Xc, we need to recover Yc and Zc.
        
        In VADAS fisheye model:
        - theta = angle from X axis (forward direction)
        - phi = azimuth around X axis
        - From pixel we can compute (theta, phi)
        - Then: Yc/Xc = -tan(theta)*cos(phi), Zc/Xc = -tan(theta)*sin(phi)
        
        Returns: (Xc, Yc, Zc) in camera coordinates
        """
        img_w_half = self.image_size[0] / 2
        img_h_half = self.image_size[1] / 2
        
        # Get normalized pixel coordinates
        xd = u - self.ux - img_w_half
        yd = v - self.uy - img_h_half
        
        rd = math.sqrt(xd**2 + yd**2)
        
        if rd < 1e-9:
            # Center pixel, looking straight ahead (no lateral offset)
            return Xc_depth, 0.0, 0.0
        
        cosPhi = xd / rd
        sinPhi = yd / rd
        
        # Approximate theta from rd
        # Ideally should invert the polynomial, but we use linear approximation
        theta = rd / self.s  # Simplified: assuming linear relationship
        
        # In VADAS fisheye projection:
        # nx = -Yc, ny = -Zc
        # dist = sqrt(nx^2 + ny^2) = sqrt(Yc^2 + Zc^2)
        # theta = atan2(dist, Xc)
        # 
        # Therefore: tan(theta) = dist / Xc = sqrt(Yc^2 + Zc^2) / Xc
        # And: cosPhi = nx/dist = -Yc/dist, sinPhi = ny/dist = -Zc/dist
        # 
        # Solving: Yc = -dist * cosPhi, Zc = -dist * sinPhi
        # And: dist = Xc * tan(theta)
        
        tan_theta = math.tan(theta)
        dist = Xc_depth * tan_theta
        
        Yc = -dist * cosPhi
        Zc = -dist * sinPhi
        
        return Xc_depth, Yc, Zc
        
        return Xc, Yc, Zc


# Default calibration for NCDB dataset
DEFAULT_CALIB = {
    "intrinsic": [-0.0004, 1.0136, -0.0623, 0.2852, -0.332, 0.1896, -0.0391,
                  1.0447, 0.0021, 44.9516, 2.48822,
                  0, 0.9965, -0.0067, -0.0956, 0.1006, -0.054, 0.0106],
    "image_size": (640, 384)
}



def load_gt_depth(gt_path):
    """Load GT depth from 16-bit PNG (value/256 = meters)"""
    gt_img = Image.open(gt_path)
    gt_depth = np.array(gt_img, dtype=np.float32) / 256.0
    return gt_depth


def load_rgb(rgb_path):
    """Load RGB image"""
    rgb = Image.open(rgb_path)
    return np.array(rgb)


def load_npu_depth(int_path, frac_path, max_depth=15.0):
    """Load and compose NPU dual-head depth"""
    integer_sigmoid = np.load(int_path).squeeze()
    fractional_sigmoid = np.load(frac_path).squeeze()
    depth = integer_sigmoid * max_depth + fractional_sigmoid
    return depth


def create_custom_colormap(min_depth=0.5, max_depth=15.0):
    """
    Create custom progressive coarsening colormap
    근거리: 0.1m 스텝, 중거리: 0.2m 스텝, 원거리: 0.5-1m 스텝
    """
    depth_range = max_depth - min_depth  # 14.5m
    
    colors_positions = [
        # 0.5m ~ 0.75m: Pure Red
        ((0.5 - 0.5) / depth_range, (1.0, 0.0, 0.0)),
        ((0.75 - 0.5) / depth_range, (1.0, 0.0, 0.0)),
        
        # 0.75m ~ 3.0m: 0.1m steps (Red → Orange → Yellow)
        ((0.85 - 0.5) / depth_range, (1.0, 0.1, 0.0)),
        ((0.95 - 0.5) / depth_range, (1.0, 0.2, 0.0)),
        ((1.05 - 0.5) / depth_range, (1.0, 0.3, 0.0)),
        ((1.15 - 0.5) / depth_range, (1.0, 0.4, 0.0)),
        ((1.25 - 0.5) / depth_range, (1.0, 0.5, 0.0)),
        ((1.35 - 0.5) / depth_range, (1.0, 0.55, 0.0)),
        ((1.45 - 0.5) / depth_range, (1.0, 0.6, 0.0)),
        ((1.55 - 0.5) / depth_range, (1.0, 0.65, 0.0)),
        ((1.65 - 0.5) / depth_range, (1.0, 0.7, 0.0)),
        ((1.75 - 0.5) / depth_range, (1.0, 0.75, 0.0)),
        ((1.85 - 0.5) / depth_range, (1.0, 0.8, 0.0)),
        ((1.95 - 0.5) / depth_range, (1.0, 0.85, 0.0)),
        ((2.05 - 0.5) / depth_range, (1.0, 0.88, 0.0)),
        ((2.15 - 0.5) / depth_range, (1.0, 0.91, 0.0)),
        ((2.25 - 0.5) / depth_range, (1.0, 0.94, 0.0)),
        ((2.35 - 0.5) / depth_range, (1.0, 0.97, 0.0)),
        ((2.45 - 0.5) / depth_range, (1.0, 1.0, 0.0)),
        ((2.55 - 0.5) / depth_range, (0.97, 1.0, 0.03)),
        ((2.65 - 0.5) / depth_range, (0.94, 1.0, 0.06)),
        ((2.75 - 0.5) / depth_range, (0.91, 1.0, 0.09)),
        ((2.85 - 0.5) / depth_range, (0.88, 1.0, 0.12)),
        ((2.95 - 0.5) / depth_range, (0.85, 1.0, 0.15)),
        
        # 3.0m ~ 6.0m: 0.2m steps (Yellow → Green)
        ((3.2 - 0.5) / depth_range, (0.8, 1.0, 0.2)),
        ((3.4 - 0.5) / depth_range, (0.7, 1.0, 0.3)),
        ((3.6 - 0.5) / depth_range, (0.6, 1.0, 0.4)),
        ((3.8 - 0.5) / depth_range, (0.5, 1.0, 0.5)),
        ((4.0 - 0.5) / depth_range, (0.4, 1.0, 0.6)),
        ((4.2 - 0.5) / depth_range, (0.35, 1.0, 0.65)),
        ((4.4 - 0.5) / depth_range, (0.3, 1.0, 0.7)),
        ((4.6 - 0.5) / depth_range, (0.25, 1.0, 0.75)),
        ((4.8 - 0.5) / depth_range, (0.2, 1.0, 0.8)),
        ((5.0 - 0.5) / depth_range, (0.15, 1.0, 0.85)),
        ((5.2 - 0.5) / depth_range, (0.1, 1.0, 0.9)),
        ((5.4 - 0.5) / depth_range, (0.05, 1.0, 0.95)),
        ((5.6 - 0.5) / depth_range, (0.0, 1.0, 1.0)),
        
        # 6.0m ~ 10.0m: 0.5m steps (Cyan → Blue)
        ((6.0 - 0.5) / depth_range, (0.0, 0.95, 1.0)),
        ((6.5 - 0.5) / depth_range, (0.0, 0.9, 1.0)),
        ((7.0 - 0.5) / depth_range, (0.0, 0.85, 1.0)),
        ((7.5 - 0.5) / depth_range, (0.0, 0.8, 1.0)),
        ((8.0 - 0.5) / depth_range, (0.0, 0.7, 1.0)),
        ((8.5 - 0.5) / depth_range, (0.0, 0.6, 1.0)),
        ((9.0 - 0.5) / depth_range, (0.0, 0.5, 1.0)),
        ((9.5 - 0.5) / depth_range, (0.0, 0.4, 1.0)),
        ((10.0 - 0.5) / depth_range, (0.0, 0.3, 1.0)),
        
        # 10.0m ~ 15.0m: 1m steps (Blue)
        ((11.0 - 0.5) / depth_range, (0.0, 0.2, 1.0)),
        ((12.0 - 0.5) / depth_range, (0.0, 0.15, 1.0)),
        ((13.0 - 0.5) / depth_range, (0.0, 0.1, 1.0)),
        ((14.0 - 0.5) / depth_range, (0.0, 0.05, 1.0)),
        ((15.0 - 0.5) / depth_range, (0.0, 0.0, 1.0)),
    ]
    
    positions = [p[0] for p in colors_positions]
    colors = [p[1] for p in colors_positions]
    
    custom_cmap = LinearSegmentedColormap.from_list('depth_custom', 
                                                     list(zip(positions, colors)), 
                                                     N=512)
    return custom_cmap


def depth_to_colormap_rgb(depth, cmap, min_depth=0.5, max_depth=15.0):
    """
    Convert depth to RGB using colormap
    
    Args:
        depth: depth map (H, W)
        cmap: matplotlib colormap
        min_depth, max_depth: depth range for normalization
    
    Returns:
        rgb: (H, W, 3) array with values in [0, 1]
    """
    # Normalize depth to [0, 1]
    depth_normalized = np.clip((depth - min_depth) / (max_depth - min_depth), 0, 1)
    
    # Apply colormap
    depth_colored = cmap(depth_normalized)  # (H, W, 4) RGBA
    
    # Return RGB only (drop alpha channel)
    return depth_colored[:, :, :3].astype(np.float32)


def blend_depth_on_rgb(rgb, depth, cmap, valid_mask,
                       min_depth=0.5, max_depth=15.0, 
                       alpha=0.9):
    """
    Blend colormap depth onto RGB using alpha blending
    
    Args:
        rgb: (H, W, 3) RGB image with values in [0, 255]
        depth: (H, W) depth map in meters
        cmap: matplotlib colormap
        valid_mask: (H, W) boolean mask of valid pixels
        alpha: blending coefficient (0=RGB only, 1=depth only)
    
    Returns:
        blended: (H, W, 3) blended image with values in [0, 255]
    """
    # Normalize RGB to [0, 1]
    rgb_normalized = rgb.astype(np.float32) / 255.0
    
    # Get depth colormap - normalize and apply colormap
    depth_normalized = np.clip((depth - min_depth) / (max_depth - min_depth), 0, 1)
    depth_colored_rgba = cmap(depth_normalized)  # (H, W, 4) RGBA
    depth_colored = depth_colored_rgba[:, :, :3]  # Extract RGB only
    
    # Alpha blending: result = rgb * (1-alpha) + depth * alpha
    blended = rgb_normalized * (1 - alpha) + depth_colored * alpha
    
    # Apply valid mask (only blend valid pixels)
    result = rgb_normalized.copy()
    result[valid_mask] = blended[valid_mask]
    
    # Convert back to [0, 255]
    result = (result * 255).astype(np.uint8)
    
    return result


def draw_depth_points_with_projection(rgb, depth, cmap, valid_mask, marker_size, alpha,
                                      camera_model, min_depth=0.5, max_depth=15.0):
    """
    Draw depth points on RGB image using proper 3D projection
    
    Steps:
    1. Unproject depth map to 3D points (using camera model inverse)
    2. Project 3D points back to image (using camera model forward)
    3. Draw points with depth-based colors
    
    Args:
        rgb: (H, W, 3) numpy array
        depth: (H, W) depth map
        cmap: matplotlib colormap
        valid_mask: (H, W) boolean mask
        marker_size: size of each point
        alpha: transparency (0-1)
        camera_model: VADASFisheyeCameraModel instance
    
    Returns:
        result: (H, W, 3) numpy array with depth points drawn
    """
    result = rgb.copy()
    H, W = depth.shape
    
    # Get valid depth coordinates and values
    v_coords, u_coords = np.where(valid_mask)
    depth_values = depth[valid_mask]
    
    # Normalize depth to [0, 1] for colormap
    depth_normalized = np.clip((depth_values - min_depth) / (max_depth - min_depth), 0, 1)
    
    # Get colors from colormap
    colors_rgba = cmap(depth_normalized)
    colors_rgb = (colors_rgba[:, :3] * 255).astype(np.uint8)  # RGB format (0-255)
    
    # Draw each point
    radius = max(1, marker_size // 2)
    
    projected_count = 0
    unprojected_count = 0
    
    for i in range(len(u_coords)):
        u_orig = u_coords[i]
        v_orig = v_coords[i]
        d = depth_values[i]
        
        # Step 1: Unproject pixel + depth to 3D camera coordinate
        try:
            Xc, Yc, Zc = camera_model.unproject_pixel_to_camera_coords(u_orig, v_orig, d)
            unprojected_count += 1
        except:
            continue
        
        # Step 2: Project 3D point back to image using camera model
        u_proj, v_proj, valid = camera_model.project_point(Xc, Yc, Zc)
        
        if not valid:
            continue
        
        # Validate projection is within image bounds
        if 0 <= u_proj < W and 0 <= v_proj < H:
            projected_count += 1
            color = tuple(int(c) for c in colors_rgb[i])  # RGB order
            
            # Draw filled circle with alpha blending
            overlay = result.copy()
            cv2.circle(overlay, (u_proj, v_proj), radius, color, -1)
            cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    
    # Debug info
    print(f"  Unprojected: {unprojected_count}/{len(u_coords)}, Projected: {projected_count}/{unprojected_count}")
    
    return result


def draw_depth_points_opencv(rgb, depth, cmap, valid_mask, marker_size, alpha,
                             min_depth=0.5, max_depth=15.0):
    """
    Draw depth points on RGB image using OpenCV
    
    Args:
        rgb: (H, W, 3) numpy array
        depth: (H, W) depth map
        cmap: matplotlib colormap
        valid_mask: (H, W) boolean mask
        marker_size: size of each point
        alpha: transparency (0-1)
    
    Returns:
        result: (H, W, 3) numpy array with depth points drawn
    """
    result = rgb.copy()
    
    # Get valid depth coordinates and values
    y_coords, x_coords = np.where(valid_mask)
    depth_values = depth[valid_mask]
    
    # Normalize depth to [0, 1]
    depth_normalized = np.clip((depth_values - min_depth) / (max_depth - min_depth), 0, 1)
    
    # Get colors from colormap
    colors_rgba = cmap(depth_normalized)
    colors_rgb = (colors_rgba[:, :3] * 255).astype(np.uint8)  # RGB format (0-255)
    
    # Draw each point
    radius = max(1, marker_size // 2)
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        # Keep RGB order (not BGR) since we're using PIL to save
        color = tuple(int(c) for c in colors_rgb[i])  # RGB order
        
        # Draw filled circle with alpha blending
        overlay = result.copy()
        cv2.circle(overlay, (x, y), radius, color, -1)
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    
    return result


def visualize_projection_single(rgb_path, gt_path, npu_int_path, npu_frac_path,
                                output_path, marker_size=30, alpha=0.9,
                                min_depth=0.5, max_depth=15.0, use_projection=True):
    """
    Create side-by-side projection visualization using OpenCV
    
    [RGB+GT points] | [RGB+NPU points]
    Output: 1280 x 384 (640x384 + 640x384)
    
    Args:
        use_projection: If True, use proper 3D projection with camera model
    """
    # Load data
    rgb = load_rgb(rgb_path)
    gt_depth = load_gt_depth(gt_path)
    npu_depth = load_npu_depth(npu_int_path, npu_frac_path, max_depth)
    
    # Ensure RGB is exactly 384x640
    if rgb.shape[:2] != (384, 640):
        rgb = cv2.resize(rgb, (640, 384))
    
    # Get valid mask from GT
    valid_mask = gt_depth > 0
    
    # Create colormap
    cmap = create_custom_colormap(min_depth, max_depth)
    
    # Initialize camera model if using projection
    if use_projection:
        camera_model = VADASFisheyeCameraModel(
            DEFAULT_CALIB["intrinsic"],
            DEFAULT_CALIB["image_size"]
        )
        # Draw depth points with proper projection
        left_img = draw_depth_points_with_projection(rgb, gt_depth, cmap, valid_mask, 
                                                     marker_size, alpha, camera_model,
                                                     min_depth, max_depth)
        right_img = draw_depth_points_with_projection(rgb, npu_depth, cmap, valid_mask, 
                                                      marker_size, alpha, camera_model,
                                                      min_depth, max_depth)
    else:
        # Draw depth points without projection (direct pixel mapping)
        left_img = draw_depth_points_opencv(rgb, gt_depth, cmap, valid_mask, 
                                            marker_size, alpha, min_depth, max_depth)
        right_img = draw_depth_points_opencv(rgb, npu_depth, cmap, valid_mask, 
                                             marker_size, alpha, min_depth, max_depth)
    
    # Combine side-by-side (1280 x 384) - direct numpy concatenation
    result = np.concatenate([left_img, right_img], axis=1)
    
    # Convert to PIL for text rendering
    result_img = Image.fromarray(result)
    
    # Add text labels at center top of each panel
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(result_img)
    
    # Try to use DejaVuSansMono-Bold
    font = None
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, 26)
            break
        except:
            continue
    
    if font is None:
        font = ImageFont.load_default()
    
    # Calculate text positions (centered at top)
    left_text = "LiDAR"
    right_text = "Monocular Depth (NPU)"
    
    # Get text bounding boxes for centering
    left_bbox = draw.textbbox((0, 0), left_text, font=font)
    right_bbox = draw.textbbox((0, 0), right_text, font=font)
    
    left_text_width = left_bbox[2] - left_bbox[0]
    right_text_width = right_bbox[2] - right_bbox[0]
    
    # Calculate centered x positions
    left_x = (640 - left_text_width) // 2
    right_x = 640 + (640 - right_text_width) // 2
    y_pos = 5  # Very close to top
    
    # Draw text with shadow effect
    # Left label with shadow
    draw.text((left_x - 2, y_pos - 2), left_text, fill=(0, 0, 0), font=font)
    draw.text((left_x, y_pos), left_text, fill=(255, 255, 255), font=font)
    
    # Right label with shadow
    draw.text((right_x - 2, y_pos - 2), right_text, fill=(0, 0, 0), font=font)
    draw.text((right_x, y_pos), right_text, fill=(255, 255, 255), font=font)
    
    # Save as JPG
    result_img.save(output_path, 'JPEG', quality=95)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='NCDB Video Projection Visualization')
    parser.add_argument('--rgb_dir', type=str, default='ncdb_video/rgb',
                       help='Path to RGB images directory')
    parser.add_argument('--gt_dir', type=str, default='ncdb_video/gt',
                       help='Path to GT depth directory')
    parser.add_argument('--npu_dir', type=str, 
                       default='ncdb_video/npu/resnetsan_dual_head_seperate_static',
                       help='Path to NPU predictions directory')
    parser.add_argument('--output_dir', type=str, default='res',
                       help='Output directory for result images')
    parser.add_argument('--marker_size', type=int, default=30,
                       help='Marker size for scatter points')
    parser.add_argument('--alpha', type=float, default=0.9,
                       help='Alpha for scatter points (0-1, higher=more opaque)')
    parser.add_argument('--use_projection', action='store_true',
                       help='Use proper 3D projection with camera model (default: direct pixel mapping)')
    parser.add_argument('--min_depth', type=float, default=0.5,
                       help='Minimum depth (m)')
    parser.add_argument('--max_depth', type=float, default=15.0,
                       help='Maximum depth (m)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: process single image')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index for test mode')
    parser.add_argument('--batch', action='store_true',
                       help='Batch mode: process all images')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get RGB files
    rgb_dir = Path(args.rgb_dir)
    gt_dir = Path(args.gt_dir)
    npu_dir = Path(args.npu_dir)
    
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png'))])
    
    if not rgb_files:
        print(f"❌ No RGB files found in {rgb_dir}")
        return
    
    print(f"Found {len(rgb_files)} RGB images")
    
    # Test mode: single image
    if args.test:
        if args.sample_idx >= len(rgb_files):
            print(f"❌ Sample index {args.sample_idx} out of range (0-{len(rgb_files)-1})")
            return
        
        rgb_file = rgb_files[args.sample_idx]
        # Get file extension
        file_ext = rgb_file.split('.')[-1]
        base_name = rgb_file.replace(f'.{file_ext}', '')
        
        print(f"\n🔍 Test mode: Processing {base_name}...")
        
        rgb_path = rgb_dir / rgb_file
        gt_path = gt_dir / f'{base_name}.png'
        npu_int_path = npu_dir / 'integer_sigmoid' / f'{base_name}.npy'
        npu_frac_path = npu_dir / 'fractional_sigmoid' / f'{base_name}.npy'
        
        # Check files
        if not all([rgb_path.exists(), gt_path.exists(), 
                   npu_int_path.exists(), npu_frac_path.exists()]):
            print(f"❌ Missing files for {base_name}")
            return
        
        output_path = output_dir / f'{base_name}_res.jpg'
        
        try:
            visualize_projection_single(rgb_path, gt_path, npu_int_path, npu_frac_path,
                                       output_path, args.marker_size, args.alpha,
                                       args.min_depth, args.max_depth, args.use_projection)
            print(f"✅ Test result saved: {output_path}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Batch mode: all images
    elif args.batch:
        print(f"\n🎬 Batch mode: Processing {len(rgb_files)} images...")
        
        successful = 0
        failed = 0
        
        for idx, rgb_file in enumerate(rgb_files):
            # Get file extension
            file_ext = rgb_file.split('.')[-1]
            base_name = rgb_file.replace(f'.{file_ext}', '')
            
            rgb_path = rgb_dir / rgb_file
            gt_path = gt_dir / f'{base_name}.png'
            npu_int_path = npu_dir / 'integer_sigmoid' / f'{base_name}.npy'
            npu_frac_path = npu_dir / 'fractional_sigmoid' / f'{base_name}.npy'
            
            # Check files
            if not all([rgb_path.exists(), gt_path.exists(), 
                       npu_int_path.exists(), npu_frac_path.exists()]):
                print(f"⚠️  Skipping {base_name}: missing files")
                failed += 1
                continue
            
            output_path = output_dir / f'{base_name}_res.jpg'
            
            try:
                visualize_projection_single(rgb_path, gt_path, npu_int_path, npu_frac_path,
                                           output_path, args.marker_size, args.alpha,
                                           args.min_depth, args.max_depth, args.use_projection)
                successful += 1
                
                if (idx + 1) % 50 == 0 or (idx + 1) == len(rgb_files):
                    print(f"Processed {idx+1}/{len(rgb_files)} images...")
            
            except Exception as e:
                print(f"❌ Error processing {base_name}: {str(e)}")
                failed += 1
        
        print(f"\n✅ Batch processing complete!")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Output directory: {output_dir}")
    
    else:
        print("Please specify --test or --batch mode")
        parser.print_help()


if __name__ == '__main__':
    main()
