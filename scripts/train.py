# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import argparse
import os

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.models.model_checkpoint import ModelCheckpoint
from packnet_sfm.trainers.horovod_trainer import HorovodTrainer
from packnet_sfm.trainers.base_trainer import BaseTrainer
from packnet_sfm.utils.config import parse_train_file
from packnet_sfm.utils.load import set_debug, filter_args_create
from packnet_sfm.utils.horovod import hvd_init, rank
from packnet_sfm.loggers import WandbLogger, TensorboardLogger


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='PackNet-SfM training script')
    parser.add_argument('file', type=str, help='Input file (.ckpt or .yaml)')
    args = parser.parse_args()
    assert args.file.endswith(('.ckpt', '.yaml')), \
        'You need to provide a .ckpt of .yaml file'
    return args


def train(file):
    """
    Monocular depth estimation training script.

    Parameters
    ----------
    file : str
        Filepath, can be either a
        **.yaml** for a yacs configuration file or a
        **.ckpt** for a pre-trained checkpoint file.
    """
    # Initialize horovod (mock for single GPU)
    hvd_init()

    # Produce configuration and checkpoint from filename
    config, ckpt = parse_train_file(file)

    # Set debug if requested
    set_debug(config.debug)

    # Loggers list
    loggers = []

    # Wandb Logger
    if not config.wandb.dry_run and rank() == 0:
        loggers.append(filter_args_create(WandbLogger, config.wandb))

    # Tensorboard Logger
    if not config.tensorboard.dry_run and rank() == 0:
        # Ensure log_dir is set for Tensorboard
        if config.tensorboard.log_dir == '':
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            config.tensorboard.log_dir = os.path.join(config.save.folder, f'tensorboard_logs_{timestamp}')
        loggers.append(TensorboardLogger(log_dir=config.tensorboard.log_dir, config=config.tensorboard))

    # model checkpoint
    checkpoint = None if config.checkpoint.filepath == '' or rank() > 0 else \
        filter_args_create(ModelCheckpoint, config.checkpoint)

    # Initialize model wrapper
    model_wrapper = ModelWrapper(config, resume=ckpt, loggers=loggers)

    # Create trainer with arch parameters
    trainer = HorovodTrainer(**config.arch, checkpoint=checkpoint)
    
    # Train model
    trainer.fit(model_wrapper)


if __name__ == '__main__':
    args = parse_args()
    train(args.file)
