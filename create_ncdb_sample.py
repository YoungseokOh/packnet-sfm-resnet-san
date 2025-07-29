
import os
import shutil
import json
import random

SOURCE_DIR = "/workspace/packnet-sfm/ncdb-cls/synced_data"
TARGET_DIR = "/workspace/packnet-sfm/ncdb-cls-sample/synced_data"
NUM_SAMPLES = 100

def create_sample_dataset(num_samples):
    # Clear target directory (already done by previous step, but good for script re-run)
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR)

    # Create subdirectories in target
    os.makedirs(os.path.join(TARGET_DIR, "image_a6"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, "pcd"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, "projected_results"), exist_ok=True)

    # Get all sample IDs from image_a6
    image_files = [f for f in os.listdir(os.path.join(SOURCE_DIR, "image_a6")) if f.endswith(".png")]
    all_sample_ids = sorted([os.path.splitext(f)[0] for f in image_files])

    if len(all_sample_ids) < num_samples:
        print(f"Warning: Not enough samples in source directory. Found {len(all_sample_ids)}, requested {num_samples}.")
        selected_sample_ids = all_sample_ids
    else:
        selected_sample_ids = random.sample(all_sample_ids, num_samples)

    print(f"Selected {len(selected_sample_ids)} samples.")

    # Prepare mapping data for the new format
    new_mapping_data = {
        "image_a6": [],
        "pcd": [],
        "projected_results": [] # Add this if projected_results also needs to be mapped
    }

    # Copy files for selected samples and populate new_mapping_data
    for sample_id in selected_sample_ids:
        # Copy image_a6
        src_image_path = os.path.join(SOURCE_DIR, "image_a6", f"{sample_id}.png")
        tgt_image_path = os.path.join(TARGET_DIR, "image_a6", f"{sample_id}.png")
        if os.path.exists(src_image_path):
            shutil.copy2(src_image_path, tgt_image_path)
            new_mapping_data["image_a6"].append(os.path.join("image_a6", f"{sample_id}.png"))

        # Copy pcd (assuming .pcd files with same ID)
        src_pcd_path = os.path.join(SOURCE_DIR, "pcd", f"{sample_id}.pcd") # Changed from .bin to .pcd
        tgt_pcd_path = os.path.join(TARGET_DIR, "pcd", f"{sample_id}.pcd") # Changed from .bin to .pcd
        if os.path.exists(src_pcd_path):
            shutil.copy2(src_pcd_path, tgt_pcd_path)
            new_mapping_data["pcd"].append(os.path.join("pcd", f"{sample_id}.pcd")) # Changed from .bin to .pcd
        
        # Copy projected_results (assuming .json files with same ID)
        src_proj_path = os.path.join(SOURCE_DIR, "projected_results", f"{sample_id}.json")
        tgt_proj_path = os.path.join(TARGET_DIR, "projected_results", f"{sample_id}.json")
        if os.path.exists(src_proj_path):
            shutil.copy2(src_proj_path, tgt_proj_path)
            new_mapping_data["projected_results"].append(os.path.join("projected_results", f"{sample_id}.json"))

    # Save the new mapping_data.json
    tgt_mapping_path = os.path.join(TARGET_DIR, "mapping_data.json")
    with open(tgt_mapping_path, 'w') as f:
        json.dump(new_mapping_data, f, indent=4)
    print(f"Generated new mapping_data.json at {tgt_mapping_path}")

if __name__ == "__main__":
    create_sample_dataset(NUM_SAMPLES)
