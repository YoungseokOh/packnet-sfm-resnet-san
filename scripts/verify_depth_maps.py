import argparse
import sys
from pathlib import Path
import random

import cv2
import numpy as np
from tqdm import tqdm

def load_depth_map(path: Path) -> np.ndarray:
    """Loads a 16-bit depth map and converts it to meters."""
    depth_map_uint16 = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_map_uint16 is None:
        raise IOError(f"Could not read depth map at {path}")
    return depth_map_uint16.astype(np.float32) / 256.0

def colorize_and_draw_points(image: np.ndarray, depth_map: np.ndarray, radius: int) -> np.ndarray:
    """
    Draws depth points on an image using a custom JET-like colormap,
    normalized to the specific depth range of the current image.
    """
    output_image = image.copy()
    
    valid_pixels = np.argwhere(depth_map > 0)
    if valid_pixels.shape[0] == 0:
        return output_image

    # Use the actual max distance in the current frame for normalization
    # max_dist = depth_map.max()
    max_dist = 10.0
    if max_dist <= 0: # Avoid division by zero if all depths are 0
        return output_image

    for y, x in valid_pixels:
        distance = depth_map[y, x]
        
        # --- Colormap logic from sync_tool.py ---
        normalized_dist = max(0.0, min(1.0, distance / max_dist))
        v = normalized_dist
        four_v = 4.0 * v
        r = min(four_v - 1.5, -four_v + 4.5)
        g = min(four_v - 0.5, -four_v + 3.5)
        b = min(four_v + 0.5, -four_v + 2.5)
        r_byte = int(max(0.0, min(1.0, r)) * 255)
        g_byte = int(max(0.0, min(1.0, g)) * 255)
        b_byte = int(max(0.0, min(1.0, b)) * 255)
        color = (b_byte, g_byte, r_byte) # BGR for OpenCV
        # --- End of colormap logic ---

        cv2.circle(output_image, (x, y), radius=radius, color=color, thickness=-1)
        
    return output_image

def main():
    """Main function to verify depth maps."""
    parser = argparse.ArgumentParser(description="Verify generated depth maps by overlaying them on original images.")
    parser.add_argument("--synced-data-dir", type=str, required=True,
                        help="Directory containing 'image_a6' and 'depth_maps'.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the verification images.")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of random samples to verify. Verifies all if not specified.")
    parser.add_argument("--point-radius", type=int, default=2,
                        help="Radius of the points drawn on the verification image.")

    args = parser.parse_args()

    synced_data_path = Path(args.synced_data_dir)
    image_dir = synced_data_path / "image_a6"
    depth_dir = synced_data_path / "depth_maps"
    output_path = Path(args.output_dir)

    if not image_dir.exists() or not depth_dir.exists():
        print(f"Error: 'image_a6' or 'depth_maps' directory not found under {synced_data_path}", file=sys.stderr)
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    depth_files = sorted(list(depth_dir.glob("*.png")))
    if not depth_files:
        print(f"Error: No depth maps found in {depth_dir}", file=sys.stderr)
        sys.exit(1)

    if args.num_samples and args.num_samples < len(depth_files):
        sample_files = random.sample(depth_files, args.num_samples)
    else:
        sample_files = depth_files

    print(f"Verifying {len(sample_files)} depth maps. Results will be in {output_path}")

    for depth_file in tqdm(sample_files):
        try:
            image_file = image_dir / depth_file.name
            if not image_file.exists():
                print(f"Warning: Corresponding image for {depth_file.name} not found. Skipping.", file=sys.stderr)
                continue

            original_image = cv2.imread(str(image_file))
            depth_map = load_depth_map(depth_file)

            # Use the new integrated function
            output_image = colorize_and_draw_points(original_image, depth_map, radius=args.point_radius)

            output_file = output_path / f"verify_{depth_file.name}"
            cv2.imwrite(str(output_file), output_image)

        except Exception as e:
            print(f"Error processing {depth_file.name}: {e}", file=sys.stderr)

    print("Verification complete.")


if __name__ == "__main__":
    main()

