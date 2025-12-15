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
from typing import List, Optional, Tuple
from datetime import datetime


# ========== VADAS Fisheye Camera Model (from ref code) ==========

class VADASFisheyeCameraModel:
    """VADAS Polynomial Fisheye Camera Model, assuming +X is forward."""
    def __init__(self, intrinsic: List[float], image_size: Optional[Tuple[int, int]] = None):
        if len(intrinsic) < 11:
            raise ValueError("VADAS intrinsic must have at least 11 parameters.")
        self.k = intrinsic[0:7]
        self.s = intrinsic[7]
        self.div = intrinsic[8]
        self.ux = intrinsic[9]
        self.uy = intrinsic[10]
        self.image_size = image_size
        self.original_intrinsic = intrinsic.copy()  # [ADD] Store original intrinsic
        self.scale_x = 1.0  # [ADD] Aspect ratio scale factors
        self.scale_y = 1.0

    def _poly_eval(self, coeffs: List[float], x: float) -> float:
        res = 0.0
        for c in reversed(coeffs):
            res = res * x + c
        return res
    
    def scale_intrinsics(self, scale_x: float, scale_y: float) -> None:
        """[ADD] Scale intrinsic parameters for different image sizes
        
        Based on verified test_640x384_div_comparison.py with aspect ratio support:
        - ux, uy scale by multiplying with scale factors
        - div remains UNCHANGED (original value)
        - scale_x, scale_y are stored and applied in project_point()
        - k, s coefficients do NOT scale (normalized coordinates)
        """
        # Principal point offset scales with image size
        self.ux = self.original_intrinsic[9] * scale_x
        self.uy = self.original_intrinsic[10] * scale_y
        
        # [CRITICAL] div stays at original value!
        # Aspect ratio scaling is applied directly in project_point()
        self.div = self.original_intrinsic[8]
        
        # Store scale factors for use in project_point()
        self.scale_x = scale_x
        self.scale_y = scale_y

    def project_point(self, Xc: float, Yc: float, Zc: float) -> Tuple[int, int, bool]:
        """
        Project 3D camera coordinates to 2D image coordinates.
        
        Based on ref_camera_lidar_projector.py with aspect ratio scaling support.
        Aspect ratio is applied via self.scale_x and self.scale_y to the final coordinates.
        """
        nx = -Yc
        ny = -Zc
        dist = math.hypot(nx, ny)
        if dist < sys.float_info.epsilon:
            dist = sys.float_info.epsilon
        cosPhi = nx / dist
        sinPhi = ny / dist
        theta = math.atan2(dist, Xc)

        if Xc < 0:
            return 0, 0, False

        xd = theta * self.s
        if abs(self.div) < 1e-9:
            return 0, 0, False
        
        rd = self._poly_eval(self.k, xd) / self.div
        if math.isinf(rd) or math.isnan(rd):
            return 0, 0, False

        img_w_half = (self.image_size[0] / 2) if self.image_size else 0
        img_h_half = (self.image_size[1] / 2) if self.image_size else 0

        # [ADD] Apply aspect ratio scaling to rd components
        u = rd * cosPhi * self.scale_x + self.ux + img_w_half
        v = rd * sinPhi * self.scale_y + self.uy + img_h_half
        
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
        
        # Step 1: (u, v) → (rd_x, rd_y) → rd
        # In project_point: u = rd * cosPhi * scale_x + ux + img_w_half
        rd_x = (u - self.ux - img_w_half) / self.scale_x
        rd_y = (v - self.uy - img_h_half) / self.scale_y
        
        rd = math.sqrt(rd_x**2 + rd_y**2)
        
        if rd < 1e-9:
            # Center pixel, looking straight ahead (no lateral offset)
            return Xc_depth, 0.0, 0.0
        
        cosPhi = rd_x / rd
        sinPhi = rd_y / rd
        
        # Step 2: rd → xd (inverse of polynomial)
        # Forward: rd = polynomial(xd) / div
        # Inverse: xd = polynomial_inverse(rd * div)
        target = rd * self.div
        
        # Newton-Raphson iteration to solve polynomial(xd) = target
        xd = target  # Initial guess
        for _ in range(10):  # 10 iterations should be enough
            # Evaluate polynomial and its derivative
            poly_val = self._poly_eval(self.k, xd)
            
            # Derivative of polynomial
            poly_deriv = 0.0
            for i, c in enumerate(reversed(self.k[1:])):  # Skip constant term
                poly_deriv = poly_deriv * xd + c * (len(self.k) - i - 1)
            
            if abs(poly_deriv) < 1e-9:
                break
            
            # Newton-Raphson update
            xd = xd - (poly_val - target) / poly_deriv
        
        # Step 3: xd → theta
        theta = xd / self.s
        
        # Step 4: (theta, phi, Xc) → (Yc, Zc)
        # In VADAS fisheye projection:
        # nx = -Yc, ny = -Zc
        # dist = sqrt(nx^2 + ny^2) = sqrt(Yc^2 + Zc^2)
        # theta = atan2(dist, Xc)
        # 
        # Therefore: tan(theta) = dist / Xc
        # And: cosPhi = nx/dist = -Yc/dist, sinPhi = ny/dist = -Zc/dist
        # 
        # Solving: Yc = -dist * cosPhi, Zc = -dist * sinPhi
        # And: dist = Xc * tan(theta)
        
        tan_theta = math.tan(theta)
        dist = Xc_depth * tan_theta
        
        Yc = -dist * cosPhi
        Zc = -dist * sinPhi
        
        return Xc_depth, Yc, Zc


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


def create_custom_colormap(min_depth=0.5, max_depth=15.0, desaturate=False):
    """
    Create custom progressive coarsening colormap
    
    Args:
        min_depth: Minimum depth value (default 0.5m)
        max_depth: Maximum depth value (default 15.0m)
        desaturate: If True, make colors more subtle/pastel
    """
    depth_range = max_depth - min_depth
    
    if desaturate:
        # More subtle, desaturated colors (pastel-like) for 0.5-15m range
        depth_points = [
            (0.5,  (0.8, 0.3, 0.3)),    # Muted Red: very near
            (1.0,  (0.8, 0.3, 0.3)),    # Muted Red
            (1.5,  (0.8, 0.5, 0.3)),    # Muted Orange
            (2.0,  (0.8, 0.8, 0.4)),    # Muted Yellow
            (3.0,  (0.7, 0.8, 0.5)),    # Muted Yellow-Green
            (4.0,  (0.5, 0.8, 0.6)),    # Muted Green
            (5.5,  (0.4, 0.8, 0.7)),    # Muted Cyan-Green
            (7.0,  (0.4, 0.7, 0.8)),    # Muted Cyan
            (10.0, (0.4, 0.6, 0.8)),    # Muted Cyan-Blue
            (15.0, (0.3, 0.4, 0.7)),    # Muted Blue: far
        ]
    else:
        # Original vibrant colors for 0.5-15m range
        depth_points = [
            (0.5,  (1.0, 0.0, 0.0)),    # Red: very near
            (1.0,  (1.0, 0.0, 0.0)),    # Red
            (1.5,  (1.0, 0.5, 0.0)),    # Orange
            (2.0,  (1.0, 1.0, 0.0)),    # Yellow
            (3.0,  (0.8, 1.0, 0.2)),    # Yellow-Green
            (4.0,  (0.5, 1.0, 0.5)),    # Green
            (5.5,  (0.0, 1.0, 0.8)),    # Cyan-Green
            (7.0,  (0.0, 1.0, 1.0)),    # Cyan
            (10.0, (0.0, 0.5, 1.0)),    # Cyan-Blue
            (15.0, (0.0, 0.0, 1.0)),    # Blue: far
        ]
    
    positions = [(d - min_depth) / depth_range for d, c in depth_points]
    colors = [c for d, c in depth_points]
    
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
                       min_depth=0.1, max_depth=30.0, 
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
                                      camera_model, min_depth=0.1, max_depth=30.0):
    """
    Draw depth points on RGB image using proper 3D projection
    
    GT depth is from LiDAR projection - treat it like LiDAR data!
    Need to unproject to 3D, then reproject with camera model.
    
    Args:
        rgb: (H, W, 3) numpy array
        depth: (H, W) depth map (from LiDAR projection)
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
    
    # GT depth는 이미 올바른 픽셀 위치에 projection된 결과이므로
    # 픽셀 좌표 (u, v)를 직접 사용합니다 (unprojection 불필요!)
    
    # Fast drawing: create overlay once, draw all points, then blend
    overlay = result.copy()
    for i in range(len(u_coords)):
        u = u_coords[i]
        v = v_coords[i]
        color = tuple(int(c) for c in colors_rgb[i])  # RGB order
        cv2.circle(overlay, (u, v), radius, color, -1)
    
    # Single alpha blend at the end
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    
    # Debug info
    print(f"  Drew {len(u_coords):,} GT depth points directly at pixel locations")
    
    return result


def draw_depth_points_opencv(rgb, depth, cmap, valid_mask, marker_size, alpha,
                             min_depth=0.1, max_depth=30.0):
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
    result = rgb.copy().astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    # Get valid depth coordinates and values
    y_coords, x_coords = np.where(valid_mask)
    depth_values = depth[valid_mask]
    
    # Normalize depth to [0, 1]
    depth_normalized = np.clip((depth_values - min_depth) / (max_depth - min_depth), 0, 1)
    
    # Get colors from colormap
    colors_rgba = cmap(depth_normalized)
    colors_rgb = colors_rgba[:, :3]  # RGB format (0-1)
    
    # Fast pixel-level blending without circles (much faster)
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
            result[y, x] = result[y, x] * (1 - alpha) + colors_rgb[i] * alpha
    
    # Convert back to [0, 255]
    result = (result * 255).astype(np.uint8)
    
    return result


def visualize_projection_single(rgb_path, gt_path, npu_int_path, npu_frac_path,
                                output_path, marker_size=30, alpha=0.9,
                                min_depth=0.5, max_depth=15.0, use_projection=True, desaturate=False):
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
    
    # Create colormap (with desaturation option)
    cmap = create_custom_colormap(min_depth, max_depth, desaturate=desaturate)
    
    # Initialize camera model if using projection
    if use_projection:
        camera_model = VADASFisheyeCameraModel(
            DEFAULT_CALIB["intrinsic"],
            DEFAULT_CALIB["image_size"]
        )
        # Apply scaling (original intrinsic is for 1920x1536, target is 640x384)
        # scale_x = 640 / 1920 = 1/3, scale_y = 384 / 1536 = 1/4
        camera_model.scale_intrinsics(scale_x=640/1920, scale_y=384/1536)
        
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


def create_video_from_images(image_dir, output_video, fps=30):
    """
    Create MP4 video from JPG images
    
    Args:
        image_dir: Directory containing _res.jpg files
        output_video: Output MP4 file path
        fps: Frames per second (default: 30)
    
    Returns:
        True if successful, False otherwise
    """
    image_dir = Path(image_dir)
    
    # Get all _res.jpg files sorted by name
    image_files = sorted([f for f in image_dir.iterdir() if f.name.endswith('_res.jpg')])
    
    if not image_files:
        print(f"❌ No _res.jpg files found in {image_dir}")
        return False
    
    print(f"\n📹 Creating video from {len(image_files)} images...")
    print(f"   FPS: {fps}")
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"❌ Failed to read first image: {image_files[0]}")
        return False
    
    height, width = first_image.shape[:2]
    print(f"   Image size: {width}x{height}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ Failed to create video writer: {output_video}")
        return False
    
    try:
        # Write frames
        for idx, image_file in enumerate(image_files):
            frame = cv2.imread(str(image_file))
            
            if frame is None:
                print(f"⚠️  Failed to read image: {image_file.name}")
                continue
            
            out.write(frame)
            
            if (idx + 1) % 100 == 0 or (idx + 1) == len(image_files):
                print(f"   Written {idx+1}/{len(image_files)} frames...")
        
        out.release()
        
        # Verify output file
        if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
            file_size_mb = os.path.getsize(output_video) / (1024 * 1024)
            print(f"\n✅ Video created successfully!")
            print(f"   Output: {output_video}")
            print(f"   Size: {file_size_mb:.1f} MB")
            print(f"   Frames: {len(image_files)}")
            print(f"   Duration: {len(image_files)/fps:.1f} seconds")
            return True
        else:
            print(f"❌ Video file was not created or is empty")
            return False
    
    except Exception as e:
        print(f"❌ Error creating video: {str(e)}")
        return False


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
    parser.add_argument('--use_projection', action='store_true', default=True,
                       help='Use proper 3D projection with camera model (default: True)')
    parser.add_argument('--min_depth', type=float, default=0.5,
                       help='Minimum depth (m, default: 0.5)')
    parser.add_argument('--max_depth', type=float, default=15.0,
                       help='Maximum depth (m, default: 15.0)')
    parser.add_argument('--desaturate', action='store_true',
                       help='Use desaturated (pastel) colors for colormap')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: process single image')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index for test mode')
    parser.add_argument('--batch', action='store_true',
                       help='Batch mode: process all images')
    parser.add_argument('--make-video', action='store_true',
                       help='Create MP4 video from generated images (use with --batch)')
    parser.add_argument('--no-infer', action='store_true',
                       help='Skip inference, only create video from existing images (use with --make-video)')
    parser.add_argument('--fps', type=int, default=30,
                       help='FPS for video output (default: 30)')
    
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
                                       args.min_depth, args.max_depth, args.use_projection, args.desaturate)
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
                                           args.min_depth, args.max_depth, args.use_projection, args.desaturate)
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
        
        # Create video if requested
        if args.make_video:
            video_filename = f"{output_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            output_video = Path(args.output_dir).parent / video_filename
            create_video_from_images(output_dir, output_video, fps=args.fps)
    
    # Make video only mode: create video from existing images without inference
    elif args.make_video and args.no_infer:
        print(f"\n📹 Make-video mode (no inference): Creating video from images in {output_dir}...")
        
        # Count existing images
        existing_images = sorted([f for f in output_dir.glob('*_res.jpg')])
        if not existing_images:
            print(f"❌ No _res.jpg files found in {output_dir}")
            return
        
        print(f"   Found {len(existing_images)} images")
        
        video_filename = f"{output_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_video = output_dir.parent / video_filename
        create_video_from_images(output_dir, output_video, fps=args.fps)
    
    # Make video mode: create video from existing images
    elif args.make_video:
        print(f"\n📹 Make-video mode: Creating video from images in {output_dir}...")
        
        # Count existing images
        existing_images = sorted([f for f in output_dir.glob('*_res.jpg')])
        if not existing_images:
            print(f"❌ No _res.jpg files found in {output_dir}")
            return
        
        print(f"   Found {len(existing_images)} images")
        
        video_filename = f"{output_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_video = output_dir.parent / video_filename
        create_video_from_images(output_dir, output_video, fps=args.fps)
    
    else:
        print("Please specify --test, --batch, or --make-video mode")
        print("  Examples:")
        print("    --test --sample_idx 0              (test single image)")
        print("    --batch                            (process all images)")
        print("    --batch --make-video               (batch + create video)")
        print("    --make-video --no-infer            (video only, skip inference)")
        parser.print_help()


if __name__ == '__main__':
    main()
