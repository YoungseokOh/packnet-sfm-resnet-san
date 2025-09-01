# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
from tqdm import tqdm
from packnet_sfm.utils.logging import prepare_dataset_prefix
import numpy as np  # add if not present

def sample_to_cuda(data, dtype=None):
    """
    Recursively move tensors in a nested structure to CUDA.
    Leave non-tensors (e.g., Camera objects, strings) unchanged.
    """
    # Tensor -> GPU (with optional dtype cast for float tensors)
    if isinstance(data, torch.Tensor):
        if dtype is not None and torch.is_floating_point(data):
            data = data.to(dtype=dtype)
        return data.cuda(non_blocking=True)

    # NumPy -> Tensor -> GPU
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
        if dtype is not None and torch.is_floating_point(tensor):
            tensor = tensor.to(dtype=dtype)
        return tensor.cuda(non_blocking=True)

    # Mapping
    if isinstance(data, dict):
        return {k: sample_to_cuda(v, dtype) for k, v in data.items()}

    # Sequence
    if isinstance(data, (list, tuple)):
        return type(data)(sample_to_cuda(v, dtype) for v in data)

    # Primitives stay on CPU
    if isinstance(data, (int, float, bool)):
        return data

    # Anything else (e.g., FisheyeCamera) untouched
    return data


class BaseTrainer:
    def __init__(self, min_epochs=0, max_epochs=50,
                 validate_first=False, checkpoint=None, **kwargs):

        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.validate_first = validate_first

        self.checkpoint = checkpoint
        self.module = None

    @property
    def proc_rank(self):
        raise NotImplementedError('Not implemented for BaseTrainer')

    @property
    def world_size(self):
        raise NotImplementedError('Not implemented for BaseTrainer')

    @property
    def is_rank_0(self):
        return self.proc_rank == 0

    def check_and_save(self, module, output):
        if self.checkpoint:
            self.checkpoint.check_and_save(module, output)

    def train_progress_bar(self, dataloader, config):
        """Enhanced training progress bar with evaluation info"""
        if self.is_rank_0:
            # 🆕 enumerate를 포함한 tqdm 반환
            return tqdm(enumerate(dataloader),
                        total=len(dataloader),
                        desc=f'Training',
                        ncols=160,
                        leave=True,
                        dynamic_ncols=False)
        else:
            # 🆕 non-rank 0에서는 enumerate 반환
            return enumerate(dataloader)

    def val_progress_bar(self, dataloader, config, n=0, ncols=120):
        return tqdm(enumerate(dataloader, 0),
                    unit=' images', unit_scale=self.world_size * config.batch_size,
                    total=len(dataloader), smoothing=0,
                    disable=not self.is_rank_0, ncols=ncols,
                    desc=prepare_dataset_prefix(config, n)
                    )

    def test_progress_bar(self, dataloader, config, n=0, ncols=120):
        return tqdm(enumerate(dataloader, 0),
                    unit=' images', unit_scale=self.world_size * config.batch_size,
                    total=len(dataloader), smoothing=0,
                    disable=not self.is_rank_0, ncols=ncols,
                    desc=prepare_dataset_prefix(config, n)
                    )
