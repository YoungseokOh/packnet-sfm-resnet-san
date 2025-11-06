#!/usr/bin/env python3
"""
Unified evaluation script using official eval.py pipeline
Evaluates PyTorch FP32, ONNX FP32, ONNX INT8, and NPU INT8 models
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.trainers.horovod_trainer import HorovodTrainer
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.horovod import hvd_init


def evaluate_pytorch(checkpoint_path, config_path, split='val'):
    """
    Evaluate PyTorch model using official pipeline
    
    Args:
        checkpoint_path: Path to .ckpt file
        config_path: Path to config .yaml file
        split: 'val' or 'test'
    """
    print("\n" + "="*80)
    print("🔥 PyTorch FP32 Evaluation (Official Pipeline)")
    print("="*80)
    
    # Initialize horovod
    hvd_init()
    
    # Parse configuration
    config, state_dict = parse_test_file(checkpoint_path, config_path)
    
    # Override test split if evaluating on validation set
    if split == 'val':
        print(f"📝 Using validation split for evaluation")
        # Replace test dataset config with validation config
        if not hasattr(config.datasets, 'test') or config.datasets.test is None:
            config.datasets.test = config.datasets.validation
        else:
            config.datasets.test.dataset = config.datasets.validation.dataset
            config.datasets.test.path = config.datasets.validation.path
            config.datasets.test.split = config.datasets.validation.split
            config.datasets.test.mask_file = config.datasets.validation.mask_file
            config.datasets.test.use_mask = config.datasets.validation.use_mask
    
    # Set debug if requested
    set_debug(config.debug)
    
    # Initialize model
    model_wrapper = ModelWrapper(config)
    model_wrapper.load_state_dict(state_dict, strict=False)
    model_wrapper = model_wrapper.to('cuda')
    
    # Use FP32
    config.arch["dtype"] = None
    
    # Create trainer
    trainer = HorovodTrainer(**config.arch)
    
    # Get validation dataloaders
    if split == 'val':
        print(f"📊 Using validation_dataloader()")
        dataloaders = model_wrapper.val_dataloader()
        # Use validate() instead of test() for validation set
        result = trainer.validate(dataloaders, model_wrapper)
    else:
        print(f"📊 Using test_dataloader()")
        dataloaders = model_wrapper.test_dataloader()
        # Use evaluate() for test set
        result = trainer.evaluate(dataloaders, model_wrapper)
    
    print("✅ PyTorch FP32 evaluation complete!\n")


def evaluate_onnx_fp32(onnx_model_path, checkpoint_path, config_path, split='val'):
    """
    Evaluate ONNX FP32 model by wrapping it in PyTorch inference
    
    This requires creating a custom ModelWrapper that uses ONNX inference
    """
    print("\n" + "="*80)
    print("🔷 ONNX FP32 Evaluation (Official Pipeline)")
    print("="*80)
    print("⚠️  ONNX evaluation requires custom wrapper - not implemented yet")
    print("    Recommendation: Use separate ONNX evaluation script")
    print()


def evaluate_npu_int8(npu_results_dir, checkpoint_path, config_path, split='val'):
    """
    Evaluate NPU INT8 results by loading pre-computed depth maps
    
    This requires creating a custom ModelWrapper that loads .npy files
    """
    print("\n" + "="*80)
    print("📱 NPU INT8 Evaluation (Official Pipeline)")
    print("="*80)
    print("⚠️  NPU evaluation requires custom wrapper - not implemented yet")
    print("    Recommendation: Use separate NPU evaluation script")
    print()


def main():
    parser = argparse.ArgumentParser(description='Unified model evaluation using official pipeline')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='PyTorch checkpoint (.ckpt)')
    parser.add_argument('--config', type=str, required=True,
                       help='Configuration file (.yaml)')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['pytorch'],
                       choices=['pytorch', 'onnx_fp32', 'onnx_int8', 'npu_int8'],
                       help='Models to evaluate')
    parser.add_argument('--onnx_model', type=str,
                       help='ONNX model path (for ONNX evaluation)')
    parser.add_argument('--npu_results', type=str,
                       help='NPU results directory (for NPU evaluation)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎯 UNIFIED MODEL EVALUATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config:     {args.config}")
    print(f"Split:      {args.split}")
    print(f"Models:     {', '.join(args.models)}")
    print("="*80)
    
    # Evaluate each requested model
    for model_type in args.models:
        if model_type == 'pytorch':
            evaluate_pytorch(args.checkpoint, args.config, args.split)
        
        elif model_type == 'onnx_fp32':
            if not args.onnx_model:
                print("❌ --onnx_model required for ONNX evaluation")
                continue
            evaluate_onnx_fp32(args.onnx_model, args.checkpoint, args.config, args.split)
        
        elif model_type == 'npu_int8':
            if not args.npu_results:
                print("❌ --npu_results required for NPU evaluation")
                continue
            evaluate_npu_int8(args.npu_results, args.checkpoint, args.config, args.split)
        
        else:
            print(f"⚠️  {model_type} evaluation not implemented yet")
    
    print("\n" + "="*80)
    print("✅ ALL EVALUATIONS COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
