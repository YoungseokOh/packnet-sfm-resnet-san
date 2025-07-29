import os
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from packnet_sfm.datasets.transforms import get_transforms

class NcdbDataset(Dataset):
    """
    Ncdb-Cls-Sample Dataset.

    Parameters
    ----------
    root_dir : str
        Path to the dataset root directory (e.g., 'ncdb-cls-sample/synced_data').
    split_file : str
        Path to the split file, which is 'mapping_data.json'.
    transform : callable, optional
        A function/transform that takes in a sample and returns a transformed version.
    """
    def __init__(self, root_dir, split_file, transform=None, **kwargs):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.file_stems = []

        # Initialize transforms based on kwargs (from config)
        self.transform = get_transforms('train', **kwargs) # Assuming 'train' mode for now

        split_path = self.root_dir / split_file
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found at {split_path}")

        with open(split_path, 'r') as f:
            # Handle both .txt and .json split files
            if split_path.suffix == '.txt':
                self.file_stems = [line.strip() for line in f if line.strip()]
            elif split_path.suffix == '.json':
                mapping_data = json.load(f)
                if isinstance(mapping_data, list):
                    # Handle list of dictionaries
                    self.file_stems = [item['new_filename'] for item in mapping_data if 'new_filename' in item]
                elif isinstance(mapping_data, dict):
                    # Handle dictionary of lists (original assumption)
                    self.file_stems = [Path(p).stem for p in mapping_data.get("image_a6", [])]


    def __len__(self):
        return len(self.file_stems)

    def __getitem__(self, idx):
        stem = self.file_stems[idx]
        
        # Construct paths
        image_path = self.root_dir / 'image_a6' / f"{stem}.png"
        depth_path = self.root_dir / 'depth_maps' / f"{stem}.png"

        # Load data
        try:
            image = Image.open(image_path).convert('RGB')
            depth_png = Image.open(depth_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find file for stem {stem}: {e}")

        # Convert depth map and apply scale factor
        depth_gt = np.asarray(depth_png, dtype=np.uint16)
        depth_gt = depth_gt.astype(np.float32) / 256.0

        image_np = np.array(image)

        sample = {
            'rgb': image_np,
            'depth': depth_gt,
            'idx': idx,
            'meta': {
                'image_path': str(image_path),
                'depth_path': str(depth_path),
            }
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
