

import sys
import os
import json
import math
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

import numpy as np
from PIL import Image
import cv2

# Try importing open3d, provide a fallback if not available
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: open3d not found. Falling back to basic ASCII PCD parser.", file=sys.stderr)


# Re-use CameraModelBase, VADASFisheyeCameraModel, SensorInfo, CalibrationDB, load_pcd_xyz, load_image
# from create_depth_maps.py
# (Copy-pasting these for self-contained script, or import if possible, but for CLI, self-contained is better)

class CameraModelBase:
    """Base class for camera projection models."""
    def project_point(self, Xc: float, Yc: float, Zc: float) -> Tuple[int, int, bool]:
        raise NotImplementedError

class VADASFisheyeCameraModel(CameraModelBase):
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

    def _poly_eval(self, coeffs: List[float], x: float) -> float:
        res = 0.0
        for c in reversed(coeffs):
            res = res * x + c
        return res

    def project_point(self, Xc: float, Yc: float, Zc: float) -> Tuple[int, int, bool]:
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

        u = rd * cosPhi + self.ux + img_w_half
        v = rd * sinPhi + self.uy + img_h_half
        
        return int(round(u)), int(round(v)), True

class SensorInfo:
    """Holds camera sensor information."""
    def __init__(self, name: str, model: CameraModelBase, intrinsic: List[float], extrinsic: np.ndarray, image_size: Optional[Tuple[int, int]] = None):
        self.name = name
        self.model = model
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.image_size = image_size

class CalibrationDB:
    """Manages camera calibration data."""
    def __init__(self, calib_dict: Dict[str, Any], lidar_to_world: Optional[np.ndarray] = None):
        self.sensors: Dict[str, SensorInfo] = {}
        self.lidar_to_world = lidar_to_world if lidar_to_world is not None else np.eye(4)

        for cam_name, calib_data in calib_dict.items():
            model_type = calib_data["model"]
            intrinsic = calib_data["intrinsic"]
            extrinsic_raw = calib_data["extrinsic"]
            image_size = tuple(calib_data["image_size"]) if "image_size" in calib_data and calib_data["image_size"] else None

            extrinsic_matrix = self._rodrigues_to_matrix(extrinsic_raw) if len(extrinsic_raw) == 6 else np.array(extrinsic_raw).reshape(4, 4)

            if model_type == "vadas":
                camera_model = VADASFisheyeCameraModel(intrinsic, image_size=image_size)
            else:
                raise ValueError(f"Unsupported camera model: {model_type}. This script is configured for 'vadas' only.")
            
            self.sensors[cam_name] = SensorInfo(cam_name, camera_model, intrinsic, extrinsic_matrix, image_size)

    def _rodrigues_to_matrix(self, rvec_tvec: List[float]) -> np.ndarray:
        tvec = np.array(rvec_tvec[0:3]).reshape(3, 1)
        rvec = np.array(rvec_tvec[3:6])
        theta = np.linalg.norm(rvec)
        if theta < 1e-6:
            R = np.eye(3)
        else:
            r = rvec / theta
            K = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
        
        transform_matrix = np.eye(4)
        transform_matrix[0:3, 0:3] = R
        transform_matrix[0:3, 3:4] = tvec
        return transform_matrix

    def get(self, name: str) -> SensorInfo:
        if name not in self.sensors:
            raise ValueError(f"Sensor '{name}' not found in calibration database.")
        return self.sensors[name]

def load_pcd_xyz(path: Path) -> np.ndarray:
    if OPEN3D_AVAILABLE:
        try:
            pcd = o3d.io.read_point_cloud(str(path))
            return np.asarray(pcd.points, dtype=np.float64) if pcd.has_points() else np.empty((0, 3))
        except Exception as e:
            print(f"Warning: open3d failed to read {path}. Falling back. Error: {e}", file=sys.stderr)

    points = []
    with open(path, 'r', encoding='utf-8') as f:
        data_started = False
        for line in f:
            if data_started:
                try:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except (ValueError, IndexError):
                    continue
            elif line.startswith("DATA ascii"):
                data_started = True
    return np.array(points, dtype=np.float64)

def load_image(path: Path) -> Image.Image:
    return Image.open(path)

# --- New comparison logic ---
def compare_depth_map_with_pcd(
    projector_calib_db: CalibrationDB,
    sensor_name: str,
    original_pcd_path: Path,
    generated_depth_map_path: Path,
    image_size: Tuple[int, int]
) -> Dict[str, Any]:
    """
    Compares a generated depth map with its original LiDAR point cloud.
    Calculates statistics on the depth differences for projected points.
    """
    
    sensor_info = projector_calib_db.get(sensor_name)
    camera_model = sensor_info.model
    cam_extrinsic = sensor_info.extrinsic
    image_width, image_height = image_size

    if isinstance(camera_model, VADASFisheyeCameraModel) and camera_model.image_size is None:
        camera_model.image_size = (image_width, image_height)

    # Load generated depth map
    try:
        generated_depth_img = Image.open(generated_depth_map_path)
        # Ensure it's 16-bit and convert to numpy array
        generated_depth_array = np.array(generated_depth_img, dtype=np.float32) / 256.0 # Convert to meters
    except Exception as e:
        print(f"Error loading generated depth map {generated_depth_map_path}: {e}", file=sys.stderr)
        return {"error": f"Failed to load depth map: {e}"}

    # Load original PCD
    try:
        original_cloud_xyz = load_pcd_xyz(original_pcd_path)
    except Exception as e:
        print(f"Error loading original PCD {original_pcd_path}: {e}", file=sys.stderr)
        return {"error": f"Failed to load PCD: {e}"}

    if original_cloud_xyz.shape[0] == 0:
        return {"message": "Original PCD is empty, no comparison possible."}

    # Transform original PCD points to camera coordinates
    original_cloud_xyz_hom = np.hstack((original_cloud_xyz, np.ones((original_cloud_xyz.shape[0], 1))))
    lidar_to_camera_transform = cam_extrinsic @ projector_calib_db.lidar_to_world
    points_cam_hom = (lidar_to_camera_transform @ original_cloud_xyz_hom.T).T
    points_cam = points_cam_hom[:, :3]

    depth_differences = []
    num_projected_points = 0
    num_valid_comparisons = 0

    for i in range(points_cam.shape[0]):
        Xc, Yc, Zc = points_cam[i]
        
        if Xc <= 0: # Points behind the camera
            continue

        u, v, valid_projection = camera_model.project_point(Xc, Yc, Zc)

        if valid_projection and 0 <= u < image_width and 0 <= v < image_height:
            num_projected_points += 1
            D_map = generated_depth_array[v, u] # Depth from generated map

            if D_map > 0: # Only compare if the depth map has a valid (non-zero) depth at this pixel
                diff = abs(Xc - D_map)
                depth_differences.append(diff)
                num_valid_comparisons += 1
    
    results = {
        "total_original_pcd_points": original_cloud_xyz.shape[0],
        "points_projected_into_image": num_projected_points,
        "valid_depth_map_comparisons": num_valid_comparisons,
        "mean_absolute_error": 0.0,
        "rmse": 0.0,
        "max_absolute_error": 0.0,
        "min_absolute_error": 0.0,
        "percentage_compared": 0.0,
    }

    if num_valid_comparisons > 0:
        depth_differences_np = np.array(depth_differences)
        results["mean_absolute_error"] = np.mean(depth_differences_np)
        results["rmse"] = np.sqrt(np.mean(depth_differences_np**2))
        results["max_absolute_error"] = np.max(depth_differences_np)
        results["min_absolute_error"] = np.min(depth_differences_np)
        results["percentage_compared"] = (num_valid_comparisons / num_projected_points) * 100 if num_projected_points > 0 else 0.0

    return results

# Default calibration and lidar_to_world from create_depth_maps.py
DEFAULT_CALIB = {
  "a6": {
    "model": "vadas",
    "intrinsic": [-0.0004, 1.0136, -0.0623, 0.2852, -0.332, 0.1896, -0.0391,
                  1.0447, 0.0021, 44.9516, 2.48822, 0, 0.9965, -0.0067,
                  -0.0956, 0.1006, -0.054, 0.0106],
    "extrinsic": [ 0.293769, -0.0542026, -0.631615, -0.00394431, -0.33116, -0.00963617],
    "image_size": None
  }
}

DEFAULT_LIDAR_TO_WORLD = np.array([
    [-0.998752, -0.00237052, -0.0498847,  0.0375091],
    [ 0.00167658, -0.999901,   0.0139481,  0.0349093],
    [-0.0499128,  0.0138471,   0.998658,   0.771878],
    [ 0.,         0.,          0.,         1.       ]
])

def main():
    parser = argparse.ArgumentParser(description="Compare generated depth maps with original LiDAR point clouds.")
    parser.add_argument("--parent", type=str, required=True,
                        help="Parent folder containing the 'synced_data' directory (for original PCDs and images).")
    parser.add_argument("--generated-depth-dir", type=str, required=True,
                        help="Directory containing the generated depth maps (PNGs).")
    parser.add_argument("--cam", type=str, default="a6",
                        help="Camera name (must be 'a6' with this configuration).")
    parser.add_argument("--calib_json", type=str, default=None,
                        help="Path to a JSON file with calibration data. Uses default if not provided.")
    parser.add_argument("--lidar_to_world", type=str, default=None,
                        help="Path to a text file with a 4x4 LiDAR to World matrix. Uses default if not provided.")
    
    args = parser.parse_args()

    if args.cam != 'a6':
        print(f"Warning: This script is configured for '--cam a6' only. You provided '{args.cam}'.", file=sys.stderr)

    parent_folder = Path(args.parent)
    generated_depth_dir = Path(args.generated_depth_dir)
    
    calib_data = DEFAULT_CALIB
    if args.calib_json:
        with open(args.calib_json, 'r', encoding='utf-8') as f:
            calib_data = json.load(f)

    lidar_to_world_matrix = DEFAULT_LIDAR_TO_WORLD
    if args.lidar_to_world:
        lidar_to_world_matrix = np.loadtxt(args.lidar_to_world).reshape(4, 4)

    try:
        calib_db = CalibrationDB(calib_data, lidar_to_world=lidar_to_world_matrix)
        
        synced_data_dir = parent_folder / "synced_data"
        mapping_file = synced_data_dir / "mapping_data.json"
        
        if not mapping_file.exists():
            print(f"Error: mapping_data.json not found at {mapping_file}", file=sys.stderr)
            sys.exit(1)

        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)

        if isinstance(mapping_data, dict) and "image_a6" in mapping_data and "pcd" in mapping_data:
            image_rel_paths = mapping_data["image_a6"]
            pcd_rel_paths = mapping_data["pcd"]
            
            if len(image_rel_paths) != len(pcd_rel_paths):
                print(f"Warning: Mismatch in number of image and PCD entries in mapping_data.json. Using minimum count.", file=sys.stderr)
            
            num_samples = min(len(image_rel_paths), len(pcd_rel_paths))
            print(f"Comparing {num_samples} files.")

            all_comparison_results = []

            for i in tqdm(range(num_samples)):
                image_path = synced_data_dir / image_rel_paths[i]
                pcd_path = synced_data_dir / pcd_rel_paths[i]
                
                # Handle potential .jpg fallback for image
                if not image_path.exists():
                    image_path = image_path.with_suffix('.jpg')
                    if not image_path.exists():
                        print(f"Skipping {image_rel_paths[i]}: Image file not found.", file=sys.stderr)
                        continue
                
                # Handle potential .bin fallback for pcd
                if not pcd_path.exists():
                    pcd_path = pcd_path.with_suffix('.bin')
                    if not pcd_path.exists():
                        print(f"Skipping {pcd_rel_paths[i]}: PCD file not found.", file=sys.stderr)
                        continue

                generated_depth_map_path = generated_depth_dir / f"{image_path.stem}.png"
                if not generated_depth_map_path.exists():
                    print(f"Skipping {image_path.stem}: Generated depth map not found at {generated_depth_map_path}", file=sys.stderr)
                    continue

                try:
                    pil_image = load_image(image_path) # Used to get image_size
                    comparison_results = compare_depth_map_with_pcd(
                        projector_calib_db=calib_db,
                        sensor_name=args.cam,
                        original_pcd_path=pcd_path,
                        generated_depth_map_path=generated_depth_map_path,
                        image_size=pil_image.size
                    )
                    if "error" not in comparison_results:
                        print(f"--- Comparison for {image_path.stem}.png ---", file=sys.stderr)
                        for key, value in comparison_results.items():
                            print(f"  {key}: {value}", file=sys.stderr)
                        all_comparison_results.append(comparison_results)
                    else:
                        print(f"Error comparing {image_path.stem}.png: {comparison_results['error']}", file=sys.stderr)

                except Exception as e:
                    print(f"Error processing {image_path.name} for comparison: {e}", file=sys.stderr)
                    traceback.print_exc()
        else:
            print(f"Error: mapping_data.json is not in an expected format (dictionary with 'image_a6' and 'pcd' keys).", file=sys.stderr)
            sys.exit(1)

        if all_comparison_results:
            # Aggregate overall statistics
            total_original_pcd_points = sum(r["total_original_pcd_points"] for r in all_comparison_results)
            points_projected_into_image = sum(r["points_projected_into_image"] for r in all_comparison_results)
            valid_depth_map_comparisons = sum(r["valid_depth_map_comparisons"] for r in all_comparison_results)
            
            # For mean/rmse/max/min, we need to re-calculate from all individual differences,
            # but for simplicity here, we'll average the averages.
            # A more robust approach would be to collect all differences and then calculate.
            # For now, let's just average the means.
            
            mean_mae = np.mean([r["mean_absolute_error"] for r in all_comparison_results if r["valid_depth_map_comparisons"] > 0])
            mean_rmse = np.mean([r["rmse"] for r in all_comparison_results if r["valid_depth_map_comparisons"] > 0])
            max_overall_error = np.max([r["max_absolute_error"] for r in all_comparison_results if r["valid_depth_map_comparisons"] > 0])
            min_overall_error = np.min([r["min_absolute_error"] for r in all_comparison_results if r["valid_depth_map_comparisons"] > 0])

            print("\n--- Overall Comparison Statistics ---", file=sys.stderr)
            print(f"Total Original PCD Points: {total_original_pcd_points}", file=sys.stderr)
            print(f"Total Points Projected into Image: {points_projected_into_image}", file=sys.stderr)
            print(f"Total Valid Depth Map Comparisons: {valid_depth_map_comparisons}", file=sys.stderr)
            print(f"Average Mean Absolute Error (MAE): {mean_mae:.4f}", file=sys.stderr)
            print(f"Average Root Mean Squared Error (RMSE): {mean_rmse:.4f}", file=sys.stderr)
            print(f"Overall Max Absolute Error: {max_overall_error:.4f}", file=sys.stderr)
            print(f"Overall Min Absolute Error: {min_overall_error:.4f}", file=sys.stderr)
            print(f"Percentage of Projected Points Compared: {(valid_depth_map_comparisons / points_projected_into_image) * 100:.2f}%" if points_projected_into_image > 0 else "0.00%", file=sys.stderr)
        else:
            print("No valid comparisons were made.", file=sys.stderr)

    except Exception as e:
        print(f"An unexpected error occurred during comparison: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
