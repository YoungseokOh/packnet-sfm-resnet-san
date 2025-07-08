import os
import glob
import cv2
import argparse

def create_sample(kitti_dir, width, height, output_path):
    # 예제에서는 image_02/data 폴더를 사용합니다.
    image_dir = os.path.join(kitti_dir, 'image_02', 'data')
    images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    sample_image_path = images[0]
    img = cv2.imread(sample_image_path, cv2.IMREAD_COLOR)
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_path, resized)
    print(f"Saved resized sample to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Create a single resized sample from KITTI images"
    )
    parser.add_argument(
        '--kitti_dir', required=True,
        help="Path to KITTI root directory (contains image_02/data)"
    )
    parser.add_argument('--width',  type=int, required=True, help="Target width")
    parser.add_argument('--height', type=int, required=True, help="Target height")
    parser.add_argument(
        '--output', default='sample_kitti.png',
        help="Output filename (default: sample_kitti.png)"
    )
    args = parser.parse_args()
    create_sample(args.kitti_dir, args.width, args.height, args.output)

if __name__ == "__main__":
    main()