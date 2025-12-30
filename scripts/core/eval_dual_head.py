#!/usr/bin/env python3
"""
Dual-Head Depth Estimation Evaluation Script

This script is based on eval.py but specifically designed for Dual-Head models.
It uses the SAME evaluation logic as training validation (model_wrapper.evaluate_depth).

Key differences from evaluate_dual_head.py:
- Uses the exact same code path as training validation
- Loads model and runs inference, not pre-computed NPU results
- Ensures consistency between training metrics and evaluation metrics

Usage:
    python scripts/core/eval_dual_head.py --checkpoint <path_to_ckpt>
    
    # With custom depth_type:
    python scripts/core/eval_dual_head.py --checkpoint <path_to_ckpt> --depth_type depth_synthetic
    
    # With custom split:
    python scripts/core/eval_dual_head.py --checkpoint <path_to_ckpt> --split /path/to/split.json
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.trainers.horovod_trainer import HorovodTrainer
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.horovod import hvd_init


def parse_args():
    """Parse arguments for evaluation script"""
    parser = argparse.ArgumentParser(description='Dual-Head Depth Estimation Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint (.ckpt)')
    parser.add_argument('--config', type=str, default=None, help='Configuration (.yaml) to override')
    parser.add_argument('--half', action="store_true", help='Use half precision (fp16)')
    
    # Dataset overrides
    parser.add_argument('--split', type=str, default=None, 
                        help='Override test split file (JSON)')
    parser.add_argument('--depth_type', type=str, default=None,
                        choices=['depth', 'depth_synthetic', 'distance', 'distance_original'],
                        help='Override depth type for GT loading')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Override dataset path')
    
    # Output
    parser.add_argument('--output_json', type=str, default=None,
                        help='Save results to JSON file')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed per-sample metrics')
    
    args = parser.parse_args()
    
    assert args.checkpoint.endswith('.ckpt'), \
        'You need to provide a .ckpt file as checkpoint'
    
    return args


def test(args):
    """
    Dual-Head depth estimation evaluation.
    Uses the same code path as training validation for consistency.
    """
    # Initialize horovod
    hvd_init()

    # Parse checkpoint
    config, state_dict = parse_test_file(args.checkpoint, args.config)

    # Apply overrides if provided
    if args.split is not None:
        print(f"📝 Overriding split: {args.split}")
        config.datasets.test.split = [args.split]
    
    if args.depth_type is not None:
        print(f"📝 Overriding depth_type: {args.depth_type}")
        config.datasets.test.depth_type = [args.depth_type]
    
    if args.dataset_path is not None:
        print(f"📝 Overriding dataset path: {args.dataset_path}")
        config.datasets.test.path = [args.dataset_path]
    
    # Set debug if requested
    set_debug(config.debug)

    # Print configuration
    print("\n" + "=" * 80)
    print("🔍 DUAL-HEAD EVALUATION CONFIGURATION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {config.datasets.test.split}")
    print(f"Depth type: {getattr(config.datasets.test, 'depth_type', 'default')}")
    print(f"Dataset path: {config.datasets.test.path}")
    print(f"Depth range: [{config.model.params.min_depth}, {config.model.params.max_depth}]m")
    print("=" * 80 + "\n")

    # Initialize model wrapper
    model_wrapper = ModelWrapper(config)
    
    # Restore model state
    model_wrapper.load_state_dict(state_dict)
    
    # Verify it's a dual-head model
    depth_net = model_wrapper.model.depth_net
    use_dual_head = getattr(depth_net, 'use_dual_head', False)
    if not use_dual_head:
        print("⚠️  WARNING: This model is NOT a dual-head model!")
        print("   Proceeding with standard evaluation...")
    else:
        print("✅ Confirmed: Dual-Head model")

    # Change to half precision for evaluation if requested
    config.arch["dtype"] = torch.float16 if args.half else None

    # Create trainer with args.arch parameters
    trainer = HorovodTrainer(**config.arch)

    # Test model - this uses model_wrapper.evaluate_depth() internally
    # which is the SAME function used during training validation
    trainer.test(model_wrapper)
    
    # Save results if requested
    if args.output_json:
        # Collect results from the last evaluation
        # The trainer.test() already prints results, but we can also save them
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Results would be saved to: {output_path}")
        print("   (Note: Detailed JSON saving requires extending HorovodTrainer)")


def main():
    """Main entry point."""
    args = parse_args()
    test(args)


if __name__ == '__main__':
    main()
