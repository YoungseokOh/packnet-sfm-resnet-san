import argparse
import sys
from pathlib import Path
import random
import math

import cv2
import numpy as np
from tqdm import tqdm

# --- Core Projection Logic (Copied from create_depth_maps.py for data validation) ---
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

# Note: The following classes are simplified and included here for the sole purpose of
# validating the depth data integrity without external dependencies.
class VADASFisheyeCameraModel:
    def __init__(self, intrinsic: list, image_size: tuple = None):
        self.k, self.s, self.div, self.ux, self.uy = intrinsic[0:7], intrinsic[7], intrinsic[8], intrinsic[9], intrinsic[10]
        self.image_size = image_size
    def _poly_eval(self, coeffs: list, x: float) -> float:
        res = 0.0
        for c in reversed(coeffs): res = res * x + c
        return res
    def project_point(self, Xc: float, Yc: float, Zc: float) -> tuple:
        nx, ny = -Yc, -Zc
        dist = math.hypot(nx, ny)
        if dist < 1e-9: dist = 1e-9
        cosPhi, sinPhi = nx / dist, ny / dist
        theta = math.atan2(dist, Xc)
        if Xc < 0: return 0, 0, False
        xd = theta * self.s
        if abs(self.div) < 1e-9: return 0, 0, False
        rd = self._poly_eval(self.k, xd) / self.div
        if math.isinf(rd) or math.isnan(rd): return 0, 0, False
        img_w_half = self.image_size[0] / 2
        img_h_half = self.image_size[1] / 2
        u = rd * cosPhi + self.ux + img_w_half
        v = rd * sinPhi + self.uy + img_h_half
        return int(round(u)), int(round(v)), True

DEFAULT_CALIB = {
  "a6": { "model": "vadas", "intrinsic": [-0.0004, 1.0136, -0.0623, 0.2852, -0.332, 0.1896, -0.0391, 1.0447, 0.0021, 44.9516, 2.48822, 0, 0.9965, -0.0067, -0.0956, 0.1006, -0.054, 0.0106], "extrinsic": [0.0900425, -0.00450864, -0.356367, 0.00100918, -0.236104, -0.0219886] }
}
DEFAULT_LIDAR_TO_WORLD = np.array([ [-0.998752, -0.00237052, -0.0498847, 0.0375091], [0.00167658, -0.999901, 0.0139481, 0.0349093], [-0.0499128, 0.0138471, 0.998658, 0.771878], [0., 0., 0., 1.] ])
# --- End of Core Projection Logic ---

def load_pcd_xyz(path: Path) -> np.ndarray:
    if not OPEN3D_AVAILABLE: raise ImportError("Open3D is required for data validation.")
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points, dtype=np.float64) if pcd.has_points() else np.empty((0, 3))

def load_depth_map(path: Path) -> np.ndarray:
    depth_map_uint16 = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_map_uint16 is None: raise IOError(f"Could not read depth map at {path}")
    return depth_map_uint16.astype(np.float32) / 256.0

def validate_data_integrity(pcd_path: Path, depth_map: np.ndarray, image_size: tuple):
    """Projects a few points from PCD and compares with the loaded depth map."""
    print("\n--- Running Data Integrity Check ---")
    cloud_xyz = load_pcd_xyz(pcd_path)
    
    # Setup camera model for validation
    cam_data = DEFAULT_CALIB["a6"]
    model = VADASFisheyeCameraModel(cam_data["intrinsic"], image_size=image_size)
    rvec = np.array(cam_data["extrinsic"][3:6])
    tvec = np.array(cam_data["extrinsic"][0:3]).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    extrinsic = np.eye(4)
    extrinsic[0:3, 0:3] = R
    extrinsic[0:3, 3:4] = tvec
    
    lidar_to_camera_transform = extrinsic @ DEFAULT_LIDAR_TO_WORLD
    cloud_xyz_hom = np.hstack((cloud_xyz, np.ones((cloud_xyz.shape[0], 1))))
    points_cam_hom = (lidar_to_camera_transform @ cloud_xyz_hom.T).T
    points_cam = points_cam_hom[:, :3]

    validation_passed = True
    points_checked = 0
    for i in range(0, len(points_cam), 500): # Check every 500th point
        Xc, Yc, Zc = points_cam[i]
        if Xc <= 0: continue
        
        u, v, valid = model.project_point(Xc, Yc, Zc)
        if valid and 0 <= u < image_size[0] and 0 <= v < image_size[1]:
            points_checked += 1
            original_depth = Xc
            loaded_depth = depth_map[v, u]
            
            # Check if the loaded depth is the closest one at this pixel
            is_occluded = False
            if loaded_depth < original_depth - 0.01: # Allow for small precision diff
                is_occluded = True

            if not is_occluded and not np.isclose(original_depth, loaded_depth, atol=0.005):
                print(f"  [FAIL] Point {i}: Original depth {original_depth:.3f}m != Loaded depth {loaded_depth:.3f}m at ({u},{v})")
                validation_passed = False
            
            if points_checked >= 10: break # Check up to 10 valid points

    if validation_passed:
        print("  [PASS] Data integrity check passed. PNG values match original PCD values.")
    else:
        print("  [FAIL] Data integrity check failed. Discrepancies found between PCD and PNG.")
    print("--- End of Data Integrity Check ---\n")
    return validation_passed

def colorize_depth_log_scale(depth_map: np.ndarray) -> np.ndarray:
    """Visualizes depth using a log scale to enhance detail for closer points."""
    valid_mask = depth_map > 0
    if not np.any(valid_mask):
        return np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)

    # Apply log scale: log(1 + depth) to handle zero values gracefully
    log_depth = np.log1p(depth_map[valid_mask])
    min_log, max_log = log_depth.min(), log_depth.max()

    # Normalize the log-scaled depth to 0-255
    normalized_log_depth = np.zeros_like(depth_map, dtype=np.uint8)
    if max_log > min_log:
        normalized_log_depth[valid_mask] = ((np.log1p(depth_map[valid_mask]) - min_log) / (max_log - min_log) * 255).astype(np.uint8)
    
    # Apply a perceptually uniform and vibrant colormap
    color_map = cv2.applyColorMap(normalized_log_depth, cv2.COLORMAP_VIRIDIS)
    color_map[~valid_mask] = 0
    return color_map

def draw_points_on_image(image: np.ndarray, depth_map: np.ndarray, radius: int) -> np.ndarray:
    """Draws log-scale colorized depth points on the image."""
    colorized_depth = colorize_depth_log_scale(depth_map)
    output_image = image.copy()
    valid_pixels = np.argwhere(depth_map > 0)
    
    for y, x in valid_pixels:
        color = colorized_depth[y, x].tolist()
        cv2.circle(output_image, (x, y), radius=radius, color=color, thickness=-1)
        
    return output_image

def main():
    parser = argparse.ArgumentParser(description="Advanced verification of depth maps with data integrity check and log-scale visualization.")
    parser.add_argument("--synced-data-dir", type=str, required=True, help="Directory containing 'image_a6', 'pcd', and 'depth_maps'.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the verification images.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of random samples to verify. Verifies all if not specified.")
    parser.add_argument("--point-radius", type=int, default=2, help="Radius of the points drawn on the verification image.")
    args = parser.parse_args()

    synced_data_path = Path(args.synced_data_dir)
    image_dir = synced_data_path / "image_a6"
    pcd_dir = synced_data_path / "pcd"
    depth_dir = synced_data_path / "depth_maps"
    output_path = Path(args.output_dir)

    if not all([image_dir.exists(), pcd_dir.exists(), depth_dir.exists()]):
        print(f"Error: 'image_a6', 'pcd', or 'depth_maps' directory not found under {synced_data_path}", file=sys.stderr)
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)
    depth_files = sorted(list(depth_dir.glob("*.png")))
    if not depth_files:
        print(f"Error: No depth maps found in {depth_dir}", file=sys.stderr)
        sys.exit(1)

    sample_files = random.sample(depth_files, args.num_samples) if args.num_samples and args.num_samples < len(depth_files) else depth_files

    # --- Perform data integrity check on the first sample ---
    first_depth_file = sample_files[0]
    first_image_file = image_dir / first_depth_file.name
    first_pcd_file = pcd_dir / f"{first_depth_file.stem}.pcd"
    
    if not first_image_file.exists() or not first_pcd_file.exists():
        print("Error: Cannot find corresponding image/PCD for first sample to run integrity check.", file=sys.stderr)
        sys.exit(1)
        
    first_image = cv2.imread(str(first_image_file))
    first_depth_map = load_depth_map(first_depth_file)
    
    if not validate_data_integrity(first_pcd_file, first_depth_map, (first_image.shape[1], first_image.shape[0])):
        print("Aborting visualization due to data integrity concerns.", file=sys.stderr)
        sys.exit(1)

    print(f"Starting visualization for {len(sample_files)} samples. Results will be in {output_path}")
    for depth_file in tqdm(sample_files):
        try:
            image_file = image_dir / depth_file.name
            if not image_file.exists(): continue

            original_image = cv2.imread(str(image_file))
            depth_map = load_depth_map(depth_file)
            output_image = draw_points_on_image(original_image, depth_map, radius=args.point_radius)
            cv2.imwrite(str(output_path / f"verify_log_{depth_file.name}"), output_image)

        except Exception as e:
            print(f"Error processing {depth_file.name}: {e}", file=sys.stderr)

    print("Verification complete.")

if __name__ == "__main__":
    main()
