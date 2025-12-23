import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def copy_depth_maps():
    # 설정
    json_path = Path("/workspace/data/ncdb-cls-indoor/splits_indoor_combined/combined_test.json")
    test_set_dir = Path("/workspace/data/ncdb-cls-indoor/test_set")
    target_depth_dir = test_set_dir / "newest_original_depth_maps"
    images_dir = test_set_dir / "images"
    
    depth_subdir_name = "newest_original_depth_maps"

    # 타겟 디렉토리 생성
    target_depth_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Target directory created: {target_depth_dir}")

    # JSON 로드
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"[INFO] Loaded {len(data)} entries from {json_path}")

    copied_count = 0
    missing_count = 0
    
    # 복사 진행
    for entry in tqdm(data, desc="Copying depth maps"):
        dataset_root = Path(entry['dataset_root'])
        stem = entry['new_filename']
        
        # 원본 경로 구성 (확장자는 png로 가정)
        src_path = dataset_root / depth_subdir_name / f"{stem}.png"
        dst_path = target_depth_dir / f"{stem}.png"
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            # 혹시 확장자가 다를 수 있으니 체크 (보통 depth map은 png)
            print(f"[WARNING] Source file not found: {src_path}")
            missing_count += 1

    print(f"\n[SUMMARY] Copied: {copied_count}, Missing: {missing_count}")

    # 검증: images 폴더의 파일들과 비교
    print("\n[VERIFICATION] Checking match with images in test_set/images...")
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    image_stems = {f.stem for f in image_files}
    
    depth_files = list(target_depth_dir.glob("*.png"))
    depth_stems = {f.stem for f in depth_files}
    
    common = image_stems.intersection(depth_stems)
    only_in_images = image_stems - depth_stems
    only_in_depth = depth_stems - image_stems
    
    print(f"Total Images: {len(image_stems)}")
    print(f"Total Depth Maps: {len(depth_stems)}")
    print(f"Matched Pairs: {len(common)}")
    
    if len(only_in_images) > 0:
        print(f"[WARNING] {len(only_in_images)} images have no corresponding depth map.")
        # print(f"Examples: {list(only_in_images)[:5]}")
    
    if len(only_in_depth) > 0:
        print(f"[WARNING] {len(only_in_depth)} depth maps have no corresponding image (unexpected).")

    if len(common) == len(image_stems) and len(common) > 0:
        print("\n[SUCCESS] All images have corresponding depth maps!")
    else:
        print("\n[FAIL] Some files are missing or mismatched.")

if __name__ == "__main__":
    copy_depth_maps()
