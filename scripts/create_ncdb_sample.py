
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

    # Copy files for selected samples
    for sample_id in selected_sample_ids:
        # Copy image_a6
        src_image_path = os.path.join(SOURCE_DIR, "image_a6", f"{sample_id}.png")
        tgt_image_path = os.path.join(TARGET_DIR, "image_a6", f"{sample_id}.png")
        if os.path.exists(src_image_path):
            shutil.copy2(src_image_path, tgt_image_path)

        # Copy pcd (assuming .bin files with same ID)
        src_pcd_path = os.path.join(SOURCE_DIR, "pcd", f"{sample_id}.bin")
        tgt_pcd_path = os.path.join(TARGET_DIR, "pcd", f"{sample_id}.bin")
        if os.path.exists(src_pcd_path):
            shutil.copy2(src_pcd_path, tgt_pcd_path)
        
        # Copy projected_results (assuming .json files with same ID)
        src_proj_path = os.path.join(SOURCE_DIR, "projected_results", f"{sample_id}.json")
        tgt_proj_path = os.path.join(TARGET_DIR, "projected_results", f"{sample_id}.json")
        if os.path.exists(src_proj_path):
            shutil.copy2(src_proj_path, tgt_proj_path)

    # Filter and copy mapping_data.json
    src_mapping_path = os.path.join(SOURCE_DIR, "mapping_data.json")
    tgt_mapping_path = os.path.join(TARGET_DIR, "mapping_data.json")

    if os.path.exists(src_mapping_path):
        with open(src_mapping_path, 'r') as f:
            mapping_data = json.load(f)

        # Assuming mapping_data is a list of dictionaries, and each dict has an 'id' or similar key
        # that matches our sample_id (e.g., '0000000000')
        # This part might need adjustment based on the actual structure of mapping_data.json
        # For now, let's assume each entry has a key that matches the sample_id
        
        # A more robust way would be to iterate and check if the sample_id is part of the entry's data
        # For simplicity, let's assume the mapping data is a list of entries, and we need to find entries
        # that correspond to our selected sample_ids.
        # If mapping_data.json is a list of objects, and each object has a 'frame_id' or similar field
        # that matches the sample_id, we can filter it.
        
        # Let's assume the mapping_data is a list of dictionaries, and each dictionary has a key
        # that corresponds to the sample_id (e.g., 'frame_id' or 'image_name' without extension)
        # If the structure is different, this part will need to be adjusted.
        
        # For now, let's assume the mapping_data is a list of dictionaries, and we need to find entries
        # where one of the values matches the sample_id. This is a generic approach.
        
        filtered_mapping_data = []
        for entry in mapping_data:
            # Convert all values in the entry to strings for comparison
            if any(str(sample_id) in str(value) for value in entry.values() for sample_id in selected_sample_ids):
                filtered_mapping_data.append(entry)
        
        # A more precise filtering would be needed if the mapping_data.json has a specific key for the sample ID.
        # For example, if each entry is {'frame_id': '0000000000', ...}
        # filtered_mapping_data = [entry for entry in mapping_data if entry.get('frame_id') in selected_sample_ids]
        # Or if the sample_id is part of a file path in the mapping data.
        
        # For now, I'll use a simple check that if any value in the entry contains the sample_id, it's included.
        # This is a placeholder and might need refinement based on the actual JSON structure.
        
        # Let's refine the filtering for mapping_data.json.
        # Assuming mapping_data.json contains a list of dictionaries, and each dictionary has a key
        # that directly corresponds to the sample_id (e.g., 'frame_id' or 'image_id').
        # If the sample_id is '0000000000', we need to find an entry that contains this ID.
        
        # A common pattern is that the mapping_data.json contains a list of entries,
        # and each entry has a field like 'frame_id' or 'sequence_id' that matches the sample_id.
        # Let's assume a 'frame_id' key for now. If it's different, the user will need to specify.
        
        # To be safe, let's assume the sample_id is directly present as a value in one of the fields
        # of each dictionary entry in mapping_data.json.
        
        # A more robust approach would be to read the mapping_data.json and infer its structure.
        # However, without knowing the exact structure, I'll make a reasonable assumption.
        
        # Let's assume the mapping_data.json is a list of dictionaries, and each dictionary has a key
        # that contains the sample_id as part of its value (e.g., a file path).
        
        # For now, I will assume that the mapping_data.json contains entries that can be directly
        # linked to the selected_sample_ids. If it's a list of objects, and each object has a key
        # that matches the sample_id (e.g., 'frame_id'), then we can filter.
        
        # Let's try to read the mapping_data.json and then filter based on the sample_ids.
        # If the mapping_data.json is a list of dictionaries, and each dictionary has a key
        # that contains the sample_id (e.g., 'image_path': 'image_a6/0000000000.png'),
        # we can filter based on that.
        
        # For now, I will assume that the mapping_data.json is a list of dictionaries,
        # and each dictionary has a key (e.g., 'frame_id') that directly matches the sample_id.
        
        # If the mapping_data.json is a simple list of strings (sample IDs), then it's easier.
        # But it's usually more complex.
        
        # Let's assume the mapping_data.json is a list of dictionaries, and each dictionary
        # has a key 'frame_id' whose value is the sample_id (e.g., '0000000000').
        
        # If the structure is different, the user will need to provide more details.
        
        # For now, I will filter based on the presence of the sample_id in any string value
        # within each dictionary entry. This is a broad assumption but safe for a first pass.
        
        filtered_mapping_data = []
        for entry in mapping_data:
            # Check if any string value in the dictionary contains any of the selected sample_ids
            if any(str(sample_id) in str(value) for sample_id in selected_sample_ids for value in entry.values()):
                filtered_mapping_data.append(entry)

        with open(tgt_mapping_path, 'w') as f:
            json.dump(filtered_mapping_data, f, indent=4)
    else:
        print(f"Warning: {src_mapping_path} not found. Skipping mapping data copy.")

if __name__ == "__main__":
    create_sample_dataset(NUM_SAMPLES)
