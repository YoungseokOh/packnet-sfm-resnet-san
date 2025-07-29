import json
from pathlib import Path

def prepare_data_files(input_json_path, output_dir, samples=100):
    """
    Reads the list-based mapping_data.json, creates a dictionary-based version
    for create_depth_maps.py, and generates a split file for training.
    """
    input_path = Path(input_json_path)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON file not found at {input_path}")

    with open(input_path, 'r') as f:
        data_list = json.load(f)

    image_a6_paths = []
    pcd_paths = []
    split_filenames = []

    for item in data_list:
        # Create relative paths for the new dict-based json
        image_a6_paths.append(f"image_a6/{item['new_filename']}.png")
        pcd_paths.append(f"pcd/{item['new_filename']}.pcd")
        # Store filenames for the split file
        split_filenames.append(item['new_filename'])

    # Create the dictionary for create_depth_maps.py
    output_dict = {
        "image_a6": image_a6_paths,
        "pcd": pcd_paths
    }

    # Write the new JSON file
    dict_json_path = output_dir_path / "mapping_data_dict.json"
    with open(dict_json_path, 'w') as f:
        json.dump(output_dict, f, indent=4)
    print(f"Successfully created dictionary-based mapping file at: {dict_json_path}")

    # Create the split file
    split_file_path = output_dir_path / f"sample_{samples}.txt"
    with open(split_file_path, 'w') as f:
        for name in split_filenames[:samples]:
            f.write(name + '\n')
    print(f"Successfully created split file at: {split_file_path}")

if __name__ == "__main__":
    prepare_data_files(
        "ncdb-cls-sample/synced_data/mapping_data.json",
        "ncdb-cls-sample/synced_data",
        samples=100
    )
