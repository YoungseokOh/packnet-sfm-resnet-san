# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import numpy as np
import os
import torch
from PIL import Image  # for mask loading

from glob import glob
from cv2 import imwrite
from pathlib import Path
import json
import yaml
from typing import Optional
from argparse import Namespace
from tqdm import tqdm

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth, load_depth, compute_depth_metrics, post_process_inv_depth
from packnet_sfm.utils.logging import pcolor


def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def _pil_interp(name: str):
    name = (name or '').lower()
    if name in ('nearest', 'nn'):
        return Image.NEAREST
    if name in ('bilinear', 'linear'):
        return Image.BILINEAR
    if name in ('bicubic', 'cubic'):
        return Image.BICUBIC
    # default: high-quality
    return Image.LANCZOS


def resize_to(image: Image.Image, height: int, width: int, interp: str = 'lanczos') -> Image.Image:
    """
    Resize PIL image to (width, height) using the given interpolation.
    height, width are ints. Interp in {nearest, bilinear, bicubic, lanczos}.
    """
    resample = _pil_interp(interp)
    return image.resize((int(width), int(height)), resample=resample)


def parse_args():
    parser = argparse.ArgumentParser(description='PackNet-SfM inference of depth maps from images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint (.ckpt)')
    parser.add_argument('--config', type=str, required=True, help='YAML config (fallback for image_shape, mask, etc.)')
    parser.add_argument('--split_json', type=str, default=None,
                        help='Split JSON with entries containing "image_path" (or dataset_root/new_filename)')
    parser.add_argument('--split_name', type=str, default='test',
                        choices=['test', 'validation', 'val'],
                        help="Which YAML split to use when --split_json is not provided")
    parser.add_argument('--output', type=str, required=True, help='Output folder')
    parser.add_argument('--image_shape', type=int, nargs='+', default=None,
                        help='Input and output image shape ' 
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')
    # 🔧 직접 (W,H)로 지정하고 싶을 때 사용 (우선순위: resize_* > image_shape > YAML/ckpt)
    parser.add_argument('--resize_w', type=int, default=None, help='Resize width (e.g., 640)')
    parser.add_argument('--resize_h', type=int, default=None, help='Resize height (e.g., 384)')
    parser.add_argument('--interp', type=str, default='lanczos',
                        choices=['nearest', 'bilinear', 'bicubic', 'lanczos'],
                        help='Interpolation for resizing (default: lanczos)')
    parser.add_argument('--half', action="store_true", help='Use half precision (fp16)')
    parser.add_argument('--save', type=str, choices=['npz', 'png'], default=None,
                        help='Save format (npz or png). Default is None (no depth map is saved).')
    parser.add_argument('--mask_file', type=str, default=None,
                        help='Path to the binary mask file (e.g., /workspace/packnet-sfm/ncdb-cls/synced_data/binary_mask.png).')
    parser.add_argument('--apply_mask_to_input', action='store_true',
                        help='If set, multiply the input RGB by the ROI mask. Default: off (parity with eval.py).')
    parser.add_argument('--flip_tta', action='store_true',
                        help='If set, run horizontal flip TTA and fuse predictions (mean).')
    # Evaluation options
    parser.add_argument('--eval', action='store_true', help='Compute metrics during inference')
    parser.add_argument('--depth_variants', type=str,
                        default='newest_depth_maps,newest_synthetic_depth_maps,new_depth_maps,depth_maps',
                        help='Comma-separated priority of depth variants to search for GT (for --eval)')
    parser.add_argument('--use_gt_scale', action='store_true',
                        help='Use ground-truth median scaling on predicted depth maps (for --eval)')
    parser.add_argument('--min_depth', type=float, default=0.3,
                        help='Minimum distance to consider during evaluation (for --eval)')
    parser.add_argument('--max_depth', type=float, default=100.0,
                        help='Maximum distance to consider during evaluation (for --eval)')
    parser.add_argument('--crop', type=str, default='', choices=['', 'garg'],
                        help='Which crop to use during evaluation (for --eval)')
    parser.add_argument('--scale_output', type=str, default='top-center',
                        help='How to scale output to GT resolution (for --eval)')
    parser.add_argument('--eval_output_file', type=str, default=None,
                        help='Where to save eval metrics (default: <output>/metrics.txt)')
    args = parser.parse_args()
    assert args.checkpoint.endswith('.ckpt'), \
        'You need to provide a .ckpt file as checkpoint'
    if args.image_shape is not None:
        assert len(args.image_shape) == 2, 'image_shape must be 2 ints (H W)'
    if (args.resize_w is None) ^ (args.resize_h is None):
        raise AssertionError('Both --resize_w and --resize_h must be provided together')
    # output은 폴더만 허용
    if os.path.splitext(args.output)[1]:
        raise AssertionError('Output must be a folder when using a YAML split')
    # normalize split alias
    if args.split_name == 'val':
        args.split_name = 'validation'
    return args

 
def _rebase_under_dataset_root(p: Path, dataset_root: Path) -> Path:
    """'synced_data'를 앵커로 dataset_root 아래로 강제 귀속"""
    s = str(p)
    if 'synced_data' in s:
        suffix = s.split('synced_data', 1)[1].lstrip('/\\')
        return dataset_root / 'synced_data' / suffix
    return p


def collect_files_from_yaml(yaml_path: str, split_name: str):
    """YAML의 split(test/validation)에서 이미지 파일 목록, image_shape, mask_file 추출"""
    with open(yaml_path, 'r') as f:
        y = yaml.safe_load(f)
    ds = y.get('datasets', {})
    aug = ds.get('augmentation', {})
    subset = ds.get(split_name, {})
    paths = subset.get('path', []) or []
    splits = subset.get('split', []) or []
    mask_files = subset.get('mask_file', []) or []
    files = []
    for i, (root, split) in enumerate(zip(paths, splits)):
        dataset_root = Path(root)
        split_path = Path(split)
        if not split_path.is_absolute():
            split_path = dataset_root / split_path
        if not split_path.exists():
            raise FileNotFoundError(f"Split not found: {split_path}")
        with open(split_path, 'r') as sf:
            entries = json.load(sf)
        for e in entries:
            if 'image_path' in e:
                # image_path가 있으면 그대로 사용
                files.append(str(e['image_path']))
            elif 'dataset_root' in e and 'new_filename' in e:
                base = Path(e['dataset_root'])
                if base.is_absolute():
                    base = _rebase_under_dataset_root(base, dataset_root)
                else:
                    base = dataset_root / base
                files.append(str(base / 'image_a6' / f"{e['new_filename']}.png"))
            else:
                raise ValueError(f"Unsupported split entry: keys={list(e.keys())}")
    # image_shape(H,W)
    ishape = aug.get('image_shape', None)
    if ishape is not None:
        ishape = (int(ishape[0]), int(ishape[1]))
    # mask_file: 첫 항목만 사용(여러 테스트셋인 경우 필요시 확장)
    mfile = mask_files[0] if isinstance(mask_files, list) and mask_files else None
    if mfile and not os.path.isabs(mfile) and paths:
        mfile = str(Path(paths[0]) / mfile)
    return files, ishape, mfile


def collect_files_from_split_json(split_json: str):
    """Split JSON에서 이미지 파일 목록 수집: image_path 우선 사용"""
    p = Path(split_json)
    if not p.exists():
        raise FileNotFoundError(f"Split not found: {p}")
    with open(p, 'r') as sf:
        entries = json.load(sf)
    files = []
    for e in entries:
        if 'image_path' in e:
            files.append(str(e['image_path']))
        elif 'dataset_root' in e and 'new_filename' in e:
            base = Path(e['dataset_root'])
            files.append(str(base / 'image_a6' / f"{e['new_filename']}.png"))
        else:
            raise ValueError(f"Unsupported split entry: keys={list(e.keys())}")
    return files


def _find_synced_base(image_path: Path) -> Path:
    """Return the base '<...>/synced_data' directory for a given image path."""
    parts = image_path.parts
    if 'synced_data' in parts:
        idx = parts.index('synced_data')
        return Path(*parts[:idx + 1])
    # Fallback: if parent is image_a6, use its parent
    return image_path.parent.parent if image_path.parent.name == 'image_a6' else image_path.parent


def _resolve_gt_from_image(image_path: str, variants):
    """Resolve GT PNG path for an image by checking variant folders under synced_data."""
    p = Path(image_path)
    stem = p.stem
    base_synced = _find_synced_base(p)
    for v in variants:
        cand = base_synced / v / f'{stem}.png'
        if cand.exists():
            return str(cand)
    return None


def _load_mask(mask_file: str, h: int, w: int):
    """Binary mask resized to (h,w), like evaluation scripts."""
    m = (np.array(Image.open(mask_file).convert('L')) > 0).astype(np.uint8)
    if m.shape[0] != h or m.shape[1] != w:
        m_img = Image.fromarray((m * 255).astype(np.uint8), mode='L').resize((w, h), Image.NEAREST)
        m = (np.array(m_img) > 0).astype(np.uint8)
    return m


@torch.no_grad()
def infer_and_save_depth(input_file: str,
                         output_file: str,
                         model_wrapper,
                         image_shape,  # (H, W)
                         half: bool,
                         save: Optional[str],
                         interp: str = 'lanczos',
                         do_eval: bool = False,
                         eval_args: Optional[Namespace] = None,
                         eval_mask_file: Optional[str] = None,
                         eval_variants: Optional[list] = None,
                         metrics_accumulator: Optional[list] = None,
                         missing_gt_counter: Optional[list] = None,
                         apply_mask_to_input: bool = False,
                         flip_tta: bool = False):
    """
    Process a single input file to produce and save visualization

    Parameters
    ----------
    input_file : str
        Image file
    output_file : str
        Output file, or folder where the output will be saved
    model_wrapper : nn.Module
        Model wrapper used for inference
    image_shape : Image shape
        Input image shape
    half: bool
        use half precision (fp16)
    save: str
        Save format (npz or png)
    """
    if not is_image(output_file):
        # If not an image, assume it's a folder and append the input name
        os.makedirs(output_file, exist_ok=True)
        output_file = os.path.join(output_file, os.path.basename(input_file))

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    # Load image
    image = load_image(input_file)
    # Resize and to tensor (our own resizer, explicit (H,W))
    image = resize_to(image, height=image_shape[0], width=image_shape[1], interp=interp)
    image_tensor = to_tensor(image).unsqueeze(0) # Renamed to avoid conflict with numpy image

    # Load mask if mask_file is provided in config
    mask = None
    if hasattr(model_wrapper.config.datasets, 'mask_file') and model_wrapper.config.datasets.mask_file:
        mask_path = model_wrapper.config.datasets.mask_file
        if os.path.exists(mask_path):
            mask = (np.array(Image.open(mask_path).convert('L')) > 0).astype(np.uint8)
            # Resize mask to image_shape if dimensions do not match
            if mask.shape[:2] != image_shape:
                mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                mask_img = mask_img.resize((image_shape[1], image_shape[0]), Image.NEAREST)  # (W,H)
                mask = np.array(mask_img)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() # Add channel and batch dim
            # Optionally apply mask to input image (default off for parity with eval.py)
            if apply_mask_to_input:
                image_tensor = image_tensor * mask_tensor

    # Send image to GPU if available
    if torch.cuda.is_available():
        image_tensor = image_tensor.to('cuda:{}'.format(rank()), dtype=dtype)
        model_wrapper = model_wrapper.to('cuda:{}'.format(rank()), dtype=dtype)
        if mask is not None:
            mask_tensor = mask_tensor.to('cuda:{}'.format(rank()), dtype=dtype)

    # Depth inference (returns predicted inverse depth)
    pred_inv_depth = model_wrapper.depth(image_tensor)['inv_depths'][0]

    # Flip TTA if requested: run on flipped image and fuse
    if flip_tta:
        flipped = torch.flip(image_tensor, dims=[3])
        pred_inv_depth_flipped = model_wrapper.depth(flipped)['inv_depths'][0]
        pred_inv_depth = post_process_inv_depth(pred_inv_depth, pred_inv_depth_flipped, method='mean')

    # 🔍 DEBUG: Store the depth for comparison
    pred_depth_for_save = inv2depth(pred_inv_depth)
    
    if save == 'npz' or save == 'png':
        # Get depth from predicted depth map and save to different formats
        filename = '{}.{}'.format(os.path.splitext(output_file)[0], save)
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(filename, 'magenta', attrs=['bold'])))
        write_depth(filename, depth=pred_depth_for_save)
        
        # 🔍 DEBUG: Verify saved file immediately
        if save == 'npz':
            loaded_back = load_depth(filename)
            if torch.is_tensor(pred_depth_for_save):
                pred_np = pred_depth_for_save.detach().cpu().numpy().squeeze()
            else:
                pred_np = pred_depth_for_save.squeeze()
            diff = np.abs(loaded_back - pred_np)
            if diff.max() > 1e-5:
                print(f"⚠️ NPZ save/load mismatch! Max diff: {diff.max():.6f}")
                print(f"   Original: min={pred_np.min():.3f}, max={pred_np.max():.3f}, mean={pred_np.mean():.3f}")
                print(f"   Loaded:   min={loaded_back.min():.3f}, max={loaded_back.max():.3f}, mean={loaded_back.mean():.3f}")
    else:
        # Prepare RGB image (original image, not masked for visualization)
        rgb = (image_tensor[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        # Prepare inverse depth
        viz_pred_inv_depth = (viz_inv_depth(pred_inv_depth[0]) * 255).astype(np.uint8)
        # Concatenate both vertically
        image_concat = np.concatenate([rgb, viz_pred_inv_depth], 0)
        # Save visualization
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(output_file, 'magenta', attrs=['bold'])))
        imwrite(output_file, image_concat[:, :, ::-1])

    # On-the-fly evaluation (compute metrics using in-memory prediction vs GT)
    if do_eval:
        try:
            gt_path = _resolve_gt_from_image(input_file, eval_variants or [])
            if gt_path is None or not os.path.exists(gt_path):
                if missing_gt_counter is not None:
                    missing_gt_counter[0] += 1
                return
            gt_np = load_depth(gt_path)
            # Apply ROI mask on GT at its resolution
            if eval_mask_file and os.path.exists(eval_mask_file):
                mh, mw = gt_np.shape[:2]
                m = _load_mask(eval_mask_file, mh, mw).astype(np.float32)
                gt_np = gt_np * m

            # Prepare tensors
            gt_t = torch.tensor(gt_np).unsqueeze(0).unsqueeze(0)
            # Ensure GT is on same device/dtype as prediction
            gt_t = gt_t.to(device=pred_inv_depth.device, dtype=pred_inv_depth.dtype)
            # 🔍 Use the same depth we saved
            pred_depth = pred_depth_for_save
            
            # 🔍 DEBUG: Compare on-the-fly vs saved for first sample
            if save == 'npz' and len(metrics_accumulator) == 0:
                # Compute metrics for saved version
                saved_filename = '{}.{}'.format(os.path.splitext(output_file)[0], save)
                saved_depth = load_depth(saved_filename)
                saved_t = torch.tensor(saved_depth).unsqueeze(0).unsqueeze(0).to(device=gt_t.device, dtype=gt_t.dtype)
                
                metrics_live = compute_depth_metrics(eval_args, gt_t, pred_depth, use_gt_scale=eval_args.use_gt_scale)
                metrics_saved = compute_depth_metrics(eval_args, gt_t, saved_t, use_gt_scale=eval_args.use_gt_scale)
                
                print(f"\n🔍 DEBUG: First sample comparison")
                print(f"   Live abs_rel:  {metrics_live[0].item():.6f}")
                print(f"   Saved abs_rel: {metrics_saved[0].item():.6f}")
                print(f"   Depth stats - Live:  min={pred_depth.min().item():.3f}, max={pred_depth.max().item():.3f}")
                if torch.is_tensor(saved_depth):
                    print(f"   Depth stats - Saved: min={saved_depth.min().item():.3f}, max={saved_depth.max().item():.3f}")
                else:
                    print(f"   Depth stats - Saved: min={saved_depth.min():.3f}, max={saved_depth.max().item():.3f}")
            
            metrics_accumulator.append(compute_depth_metrics(eval_args, gt_t, pred_depth, use_gt_scale=eval_args.use_gt_scale))
        except Exception as e:
            print(f"Evaluation error for {input_file}: {e}")


def main(args):
     # Initialize horovod
     hvd_init()
     # Parse arguments
     config, state_dict = parse_test_file(args.checkpoint)
 
     # Collect test files (split JSON 우선, 없으면 YAML test 세트)
     if args.split_json:
         test_files = collect_files_from_split_json(args.split_json)
         ishape_yaml, mask_yaml = None, None
     else:
         test_files, ishape_yaml, mask_yaml = collect_files_from_yaml(args.config, args.split_name)
 
    # If no image shape is provided, prefer explicit (--resize_*) > --image_shape > YAML > checkpoint
     if args.resize_w is not None and args.resize_h is not None:
        image_shape = (int(args.resize_h), int(args.resize_w))  # (H,W)
     else:
        image_shape = args.image_shape or ishape_yaml or getattr(config.datasets.augmentation, 'image_shape', None)
     assert image_shape is not None, "image_shape must be provided (arg | YAML | checkpoint)"
     image_shape = (int(image_shape[0]), int(image_shape[1]))  # normalize (H,W)
     print0(f'Using resize (W x H) = {image_shape[1]} x {image_shape[0]} | interp={args.interp}')

    # Add mask_file to config if provided
     mask_to_use = args.mask_file or mask_yaml
     if mask_to_use:
        config.datasets.update({'mask_file': mask_to_use})
 
     # Set debug if requested
     set_debug(config.debug)
 
     # Initialize model wrapper from checkpoint arguments
     model_wrapper = ModelWrapper(config, load_datasets=False)
     model_wrapper.load_state_dict(state_dict)
 
     # change to half precision for evaluation if requested
     dtype = torch.float16 if args.half else None
 
     if torch.cuda.is_available():
         model_wrapper = model_wrapper.to('cuda:{}'.format(rank()), dtype=dtype)
 
     model_wrapper.eval()
 
     os.makedirs(args.output, exist_ok=True)
     files = sorted(test_files)
     src_txt = 'split JSON' if args.split_json else f'YAML:{args.split_name}'
     print0('Found {} images from {}'.format(len(files), src_txt))
 
     # Prepare eval config if requested
     eval_variants = [v.strip() for v in (args.depth_variants or '').split(',') if v.strip()]
     metrics_list = []
     missing_gt = [0]
     eval_ns = Namespace(
         min_depth=args.min_depth,
         max_depth=args.max_depth,
         crop=args.crop,
         scale_output=args.scale_output,
         use_gt_scale=args.use_gt_scale,
         flip_tta=args.flip_tta,
     ) if args.eval else None

     iter_files = files[rank()::world_size()]
     iterator = tqdm(iter_files, total=len(iter_files)) if args.eval else iter_files
     for fn in iterator:
         output_path = os.path.join(args.output, os.path.basename(fn))
         infer_and_save_depth(
             fn, output_path, model_wrapper, image_shape, args.half, args.save, args.interp,
             do_eval=args.eval, eval_args=eval_ns, eval_mask_file=(args.mask_file or None),
             eval_variants=eval_variants, metrics_accumulator=metrics_list, missing_gt_counter=missing_gt,
             apply_mask_to_input=args.apply_mask_to_input, flip_tta=args.flip_tta)

     # Summarize evaluation
     if args.eval:
         if len(metrics_list) == 0:
             print0('No evaluated pairs (check GT availability/variants).')
         else:
             metrics = (sum(metrics_list) / len(metrics_list)).detach().cpu().numpy()
             names = ['abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
             out_file = args.eval_output_file or os.path.join(args.output, 'metrics.txt')
             os.makedirs(os.path.dirname(out_file), exist_ok=True)
             print0(f"Evaluated pairs: {len(metrics_list)} | missing_gt: {missing_gt[0]}")
             with open(out_file, 'w') as f:
                 for name, metric in zip(names, metrics):
                     f.write(f'{name} = {metric}\n')
                     print0(f'{name} = {metric}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
