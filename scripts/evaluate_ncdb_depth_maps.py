#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from argparse import Namespace
from glob import glob
from pathlib import Path
from tqdm import tqdm

from packnet_sfm.utils.depth import load_depth, compute_depth_metrics, inv2depth, post_process_inv_depth
from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.image import load_image
from packnet_sfm.datasets.augmentations import to_tensor


def _ensure_parent_dir(path: Optional[str]):
    if not path:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def parse_args():
    """Parse arguments for NCDB evaluation"""
    parser = argparse.ArgumentParser(description='PackNet-SfM NCDB evaluation')
    parser.add_argument('--pred_folder', type=str, required=True,
                        help='Folder containing predicted depth maps (.npz or .png)')
    parser.add_argument('--pred_ext', type=str, default='png', choices=['png', 'npz'],
                        help='Prediction file extension to use (default: png).')
    # 기존 경로 직접 지정(레거시 모드)
    parser.add_argument('--gt_folder', type=str, default=None,
                        help='[Legacy] Folder containing GT depth maps (.npz or .png)')
    # 새 모드: split로 GT를 variants에서 자동 탐색
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='Dataset root for NCDB (e.g., /workspace/data/ncdb-cls-640x384)')
    parser.add_argument('--split', type=str, default=None,
                        help='Split JSON (relative to dataset_root or absolute)')
    parser.add_argument('--depth_variants', type=str,
                        default='newest_depth_maps,newest_synthetic_depth_maps,new_depth_maps,depth_maps',
                        help='Comma-separated priority of depth variants to search for GT')
    parser.add_argument('--mask_file', type=str, default=None,
                        help='Binary ROI mask image (png). Applied at GT resolution.')
    parser.add_argument('--use_gt_scale', action='store_true',
                        help='Use ground-truth median scaling on predicted depth maps')
    parser.add_argument('--min_depth', type=float, default=0.3,
                        help='Minimum distance to consider during evaluation')
    parser.add_argument('--max_depth', type=float, default=100.0,
                        help='Maximum distance to consider during evaluation')
    parser.add_argument('--crop', type=str, default='', choices=['', 'garg'],
                        help='Which crop to use during evaluation')
    parser.add_argument('--scale_output', type=str, default='top-center',
                        help='How to scale output to GT resolution')
    parser.add_argument('--output_file', type=str, default='metrics.txt',
                        help='File to save the metrics to')
    # Optional: compare against on-the-fly model evaluation (parity with eval.py)
    parser.add_argument('--compare_with_model', action='store_true',
                        help='Also run model inference on split images and report metrics side-by-side to analyze differences')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint (.ckpt) for compare_with_model mode')
    parser.add_argument('--image_shape', type=int, nargs='+', default=None,
                        help='Explicit image shape (H W) for compare_with_model; otherwise tries from checkpoint config')
    parser.add_argument('--resize_w', type=int, default=None, help='Resize width for compare_with_model')
    parser.add_argument('--resize_h', type=int, default=None, help='Resize height for compare_with_model')
    parser.add_argument('--interp', type=str, default='lanczos', choices=['nearest', 'bilinear', 'bicubic', 'lanczos'],
                        help='Interpolation for resizing (compare_with_model)')
    parser.add_argument('--flip_tta', action='store_true',
                        help='Use horizontal flip TTA when comparing with model (like eval.py)')
    parser.add_argument('--per_sample_report', type=str, default=None,
                        help='Optional JSON file to dump per-sample metrics for deeper analysis')
    parser.add_argument('--model_name', type=str, default=None,
                        help='표시에 사용할 모델 이름 (미지정 시 pred_folder 이름 사용)')
    
    # 디버깅 옵션 추가
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with verbose output')
    parser.add_argument('--debug_samples', type=int, default=3,
                        help='Number of samples to debug in detail (0 for all)')
    parser.add_argument('--debug_output', type=str, default=None,
                        help='Directory to save debug outputs (tensors, etc.)')
    
    args = parser.parse_args()

    # 유효성: gt_folder 없으면 dataset_root+split 필수
    if args.gt_folder is None:
        assert args.dataset_root and args.split, \
            'When --gt_folder is not provided, you must pass --dataset_root and --split.'
    if args.model_name is None:
        default_name = Path(args.pred_folder).name
        args.model_name = default_name[:-6] if default_name.endswith('_debug') else default_name
    return args


def _normalize_split_entries(dataset_root: Path, split_path: Path):
    """NcdbDataset와 동일한 규칙으로 split 엔트리를 표준화"""
    if not split_path.is_absolute():
        split_path = dataset_root / split_path
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('Split must be a list of entries')

    entries = []
    converted = 0
    for item in data:
        if 'new_filename' not in item:
            raise ValueError(f"Missing 'new_filename' in split entry: {item}")

        stem = item['new_filename']
        img_path = None
        if item.get('image_path'):
            img_path = Path(item['image_path'])
        else:
            base = Path(item.get('dataset_root', ''))
            if not base.is_absolute():
                base = dataset_root / base
            img_path = base / 'image_a6' / f'{stem}.png'
            converted += 1

        if not img_path.is_absolute():
            img_path = dataset_root / img_path

        if img_path.parent.name == 'image_a6':
            base_dir = img_path.parent.parent
        else:
            base_dir = img_path.parent

        entries.append({
            'stem': stem,
            'image_path': img_path,
            'base_dir': base_dir,
            'dataset_root': item.get('dataset_root'),
            'original_entry': item,
        })

    print(f"Loaded {len(entries)} entries from {split_path} (converted {converted} entries)")
    return entries


def _resolve_gt_path(dataset_root: Path, entry: dict, variants: List[str]) -> Optional[Path]:
    """variant 우선순위에 따라 GT 경로 탐색"""
    base = entry.get('base_dir')
    if base is None:
        base = Path(entry.get('dataset_root', ''))
        if not base.is_absolute():
            base = dataset_root / base
    if 'synced_data' not in str(base):
        base = base / 'synced_data'

    stem = entry['stem']
    for v in variants:
        p = base / v / f'{stem}.png'
        if p.exists():
            return p
    return None


def _image_path_from_entry(dataset_root: Path, entry: dict) -> Path:
    """Reconstruct the RGB image path (image_a6/<stem>.png) from a split entry."""
    return entry['image_path']


def _collect_pred_files_by_stem(pred_folder: str, ext_preference: str = 'png'):
    """예측 파일 수집, key=stem. 지정된 확장자만 사용(png 또는 npz)"""
    files = glob(os.path.join(pred_folder, f'*.{ext_preference}'))
    files.sort()
    return {Path(f).stem: f for f in files}


def _collect_gt_files_by_stem(gt_folder: str):
    """GT 파일(.npz/.png) 수집, key=stem"""
    files = []
    for ext in ('npz', 'png'):
        files.extend(glob(os.path.join(gt_folder, f'*.{ext}')))
    files.sort()
    return {Path(f).stem: f for f in files}


def _load_mask(mask_file: str, h: int, w: int):
    """GT 해상도(h,w)에 맞춘 이진 마스크 로드"""
    m = (np.array(Image.open(mask_file).convert('L')) > 0).astype(np.uint8)
    if m.shape[0] != h or m.shape[1] != w:
        m_img = Image.fromarray((m * 255).astype(np.uint8), mode='L').resize((w, h), Image.NEAREST)
        m = (np.array(m_img) > 0).astype(np.uint8)
    return m


def _make_eval_args(user_args):
    """compute_depth_metrics에 넘길 Namespace 구성"""
    return Namespace(
        min_depth=user_args.min_depth,
        max_depth=user_args.max_depth,
        crop=user_args.crop,
        scale_output=user_args.scale_output
    )


def _debug_tensors(pred_t, gt_t, stem, debug_output, idx=0):
    """디버깅용: 텐서 정보 출력 및 저장"""
    print(f"\n{'='*60}")
    print(f"DEBUG SAMPLE {idx}: {stem}")
    print(f"{'='*60}")
    
    # Shape 정보
    print(f"Shapes:")
    print(f"  Pred tensor: {pred_t.shape}")
    print(f"  GT tensor:   {gt_t.shape}")
    
    # 값 범위
    pred_np = pred_t.cpu().numpy().squeeze()
    gt_np = gt_t.cpu().numpy().squeeze()
    
    print(f"Value ranges:")
    print(f"  Pred: min={pred_np.min():.4f}, max={pred_np.max():.4f}, mean={pred_np.mean():.4f}")
    print(f"  GT:   min={gt_np.min():.4f}, max={gt_np.max():.4f}, mean={gt_np.mean():.4f}")
    
    # 유효 픽셀 수
    valid_gt = gt_np > 0
    print(f"Valid pixels:")
    print(f"  GT > 0: {valid_gt.sum()} / {valid_gt.size} ({100*valid_gt.mean():.2f}%)")
    
    if valid_gt.any():
        print(f"  Pred at valid GT: min={pred_np[valid_gt].min():.4f}, max={pred_np[valid_gt].max():.4f}")
    
    # 디버그 출력 저장
    if debug_output:
        os.makedirs(debug_output, exist_ok=True)
        debug_file = os.path.join(debug_output, f'debug_{stem}.npz')
        np.savez(debug_file, 
                 pred=pred_np, 
                 gt=gt_np,
                 pred_shape=pred_t.shape,
                 gt_shape=gt_t.shape)
        print(f"  Saved debug tensors to: {debug_file}")


def eval_with_gt_folder(args):
    """기존 방식: pred_folder와 gt_folder를 stem으로 매칭"""
    pred_by_stem = _collect_pred_files_by_stem(args.pred_folder, args.pred_ext)
    gt_by_stem = _collect_gt_files_by_stem(args.gt_folder)

    common_stems = sorted(set(pred_by_stem.keys()) & set(gt_by_stem.keys()))
    missing_pred = len(set(gt_by_stem.keys()) - set(pred_by_stem.keys()))
    missing_gt = len(set(pred_by_stem.keys()) - set(gt_by_stem.keys()))
    print(f'Matched stems: {len(common_stems)} | missing_pred: {missing_pred} | missing_gt: {missing_gt}')

    if args.debug:
        print(f"\n{'='*60}")
        print("DEBUG MODE ENABLED")
        print(f"{'='*60}")
        print(f"Arguments:")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        print(f"{'='*60}")

    metrics = []
    progress_bar = tqdm(common_stems, total=len(common_stems))
    eval_args = _make_eval_args(args)
    
    debug_count = 0
    max_debug = args.debug_samples if args.debug_samples > 0 else len(common_stems)
    
    for i, stem in enumerate(progress_bar):
        gt_np = load_depth(gt_by_stem[stem])     # GT 해상도
        pred_np = load_depth(pred_by_stem[stem]) # pred 해상도(다를 수 있음)

        # ROI 마스크: GT 해상도로 적용
        if args.mask_file and os.path.exists(args.mask_file):
            mh, mw = gt_np.shape[:2]
            m = _load_mask(args.mask_file, mh, mw).astype(np.float32)
            gt_np = gt_np * m

        gt_t = torch.tensor(gt_np).unsqueeze(0).unsqueeze(0)
        pred_t = torch.tensor(pred_np).unsqueeze(0).unsqueeze(0)
        
        # 디버깅
        if args.debug and debug_count < max_debug:
            _debug_tensors(pred_t, gt_t, stem, args.debug_output, debug_count)
            debug_count += 1
        
        sample_metrics = compute_depth_metrics(eval_args, gt_t, pred_t, use_gt_scale=args.use_gt_scale)
        metrics.append(sample_metrics)
        
        # 첫 몇 샘플 메트릭 출력
        if args.debug and i < max_debug:
            m_np = sample_metrics.cpu().numpy()
            names = ['abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
            print(f"Metrics for {stem}:")
            for name, val in zip(names, m_np):
                print(f"  {name}: {val:.6f}")

    if len(metrics) == 0:
        raise RuntimeError('No matched (GT, PRED) pairs to evaluate.')

    mean_metrics = (sum(metrics) / len(metrics)).detach().cpu().numpy()
    names = ['abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
    mean_list = mean_metrics.tolist()
    
    if args.debug:
        print(f"\n{'='*60}")
        print("FINAL AVERAGED METRICS")
        print(f"{'='*60}")
    
    _print_metrics_table("Evaluation (Saved predictions)", names, [(args.model_name, mean_list)])
    _ensure_parent_dir(args.output_file)
    with open(args.output_file, 'w') as f:
        for name, metric in zip(names, mean_list):
            f.write(f'{name} = {metric:.4f}\n')


def eval_with_split_and_variants(args):
    """새 방식: split에서 stem을 읽고 variants에서 GT를 찾음"""
    dataset_root = Path(args.dataset_root)
    split_path = Path(args.split)
    variants = [v.strip() for v in args.depth_variants.split(',') if v.strip()]
    if not variants:
        variants = ['newest_depth_maps', 'newest_synthetic_depth_maps', 'new_depth_maps', 'depth_maps']
    names = ['abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']  # 메트릭 이름 고정

    if args.debug:
        print(f"\n{'='*60}")
        print("DEBUG MODE ENABLED")
        print(f"{'='*60}")
        print(f"Arguments:")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        print(f"Dataset root: {dataset_root}")
        print(f"Split path: {split_path}")
        print(f"Depth variants: {variants}")
        print(f"{'='*60}")

    entries = _normalize_split_entries(dataset_root, split_path)
    pred_by_stem = _collect_pred_files_by_stem(args.pred_folder, args.pred_ext)
    
    if args.debug:
        print(f"\nFound {len(pred_by_stem)} prediction files")
        print(f"First 5 prediction stems: {list(pred_by_stem.keys())[:5]}")
    
    # list of (gt_path, pred_path, stem, image_path)
    pairs = [] 
    missing_pred = 0
    missing_gt = 0

    for e in entries:
        stem = e['stem']
        pred_path = pred_by_stem.get(stem, None)
        if pred_path is None:
            missing_pred += 1
            if args.debug and missing_pred <= 3:
                print(f"  Missing pred for: {stem}")
            continue
        gt_path = _resolve_gt_path(dataset_root, e, variants)
        if gt_path is None:
            missing_gt += 1
            if args.debug and missing_gt <= 3:
                print(f"  Missing GT for: {stem}")
            continue
        img_path = _image_path_from_entry(dataset_root, e)
        pairs.append((gt_path, pred_path, stem, img_path))

        if args.debug and len(pairs) <= 3:
            print(f"  Matched pair {len(pairs)}:")
            print(f"    Stem: {stem}")
            print(f"    Pred: {pred_path}")
            print(f"    GT:   {gt_path}")
            print(f"    Image:{img_path}")

    print(f'Matched pairs: {len(pairs)} | missing_pred: {missing_pred} | missing_gt: {missing_gt}')

    metrics = []
    progress_bar = tqdm(pairs, total=len(pairs))
    eval_args = _make_eval_args(args)
    per_sample = []
    
    if args.debug:
        print(f"\nEval args passed to compute_depth_metrics:")
        print(f"  min_depth: {eval_args.min_depth}")
        print(f"  max_depth: {eval_args.max_depth}")
        print(f"  crop: '{eval_args.crop}'")
        print(f"  scale_output: '{eval_args.scale_output}'")
        print(f"  use_gt_scale: {args.use_gt_scale}")
    
    debug_count = 0
    max_debug = args.debug_samples if args.debug_samples > 0 else len(pairs)
    
    # 1) Evaluate SAVED predictions
    for i, (gt, pred, stem, img_path) in enumerate(progress_bar):
        gt_np = load_depth(str(gt))     # GT 해상도
        pred_np = load_depth(str(pred)) # pred 해상도(다를 수 있음)

        # ROI 마스크는 GT 해상도로만 적용(유효 픽셀 지정용)
        if args.mask_file and os.path.exists(args.mask_file):
            mh, mw = gt_np.shape[:2]
            m = _load_mask(args.mask_file, mh, mw).astype(np.float32)
            if args.debug and debug_count < max_debug:
                print(f"  Applied mask: original shape {Image.open(args.mask_file).size} -> GT shape ({mw}, {mh})")
            gt_np = gt_np * m

        gt_t = torch.tensor(gt_np).unsqueeze(0).unsqueeze(0)
        pred_t = torch.tensor(pred_np).unsqueeze(0).unsqueeze(0)
        
        # 디버깅
        if args.debug and debug_count < max_debug:
            _debug_tensors(pred_t, gt_t, stem, args.debug_output, debug_count)
            debug_count += 1
        
        m_saved = compute_depth_metrics(eval_args, gt_t, pred_t, use_gt_scale=args.use_gt_scale)
        metrics.append(m_saved)
        
        # 첫 몇 샘플 메트릭 출력
        if args.debug and i < max_debug:
            m_np = m_saved.cpu().numpy()
            print(f"Metrics for {stem}:")
            for name, val in zip(names, m_np):
                print(f"  {name}: {val:.6f}")
        
        if args.per_sample_report:
            per_sample.append({'stem': stem, 'saved': [x.item() for x in m_saved.detach().cpu()]})

    if len(metrics) == 0:
        raise RuntimeError('No matched (GT, PRED) pairs to evaluate.')

    saved_avg = (sum(metrics) / len(metrics)).detach().cpu().numpy()
    saved_avg_list = saved_avg.tolist()
    
    if args.debug:
        print(f"\n{'='*60}")
        print("FINAL AVERAGED METRICS (Saved predictions)")
        print(f"{'='*60}")
    
    _print_metrics_table("Evaluation (Saved predictions)", names, [(args.model_name, saved_avg_list)])
    _ensure_parent_dir(args.output_file)
    with open(args.output_file, 'w') as f:
        f.write('[Saved predictions]\n')
        for name, metric in zip(names, saved_avg_list):
            f.write(f'{name} = {metric:.4f}\n')

    # 2) Optional: Evaluate MODEL predictions on-the-fly for comparison
    if args.compare_with_model:
        assert args.checkpoint and args.checkpoint.endswith('.ckpt'), 'Provide --checkpoint (.ckpt) for compare_with_model'
        
        if args.debug:
            print(f"\n{'='*60}")
            print("COMPARE WITH MODEL (on-the-fly inference)")
            print(f"{'='*60}")
        
        # Load model
        config, state_dict = parse_test_file(args.checkpoint)
        model_wrapper = ModelWrapper(config, load_datasets=False)
        model_wrapper.load_state_dict(state_dict)
        dtype = torch.float16 if getattr(config.arch, 'dtype', None) == torch.float16 else None
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model_wrapper = model_wrapper.to(device, dtype=dtype)
        model_wrapper.eval()

        # Decide image shape
        if args.resize_w is not None and args.resize_h is not None:
            image_shape = (int(args.resize_h), int(args.resize_w))
        elif args.image_shape is not None and len(args.image_shape) == 2:
            image_shape = (int(args.image_shape[0]), int(args.image_shape[1]))
        else:
            image_shape = getattr(config.datasets.augmentation, 'image_shape', None)
            assert image_shape is not None, 'image_shape unknown; pass --resize_w/--resize_h or --image_shape'
            image_shape = (int(image_shape[0]), int(image_shape[1]))
        
        if args.debug:
            print(f"Model image shape: {image_shape}")
            print(f"Using device: {device}, dtype: {dtype}")

        # Helper: resize PIL
        def _pil_interp(name: str):
            from PIL import Image as _Image
            n = (name or '').lower()
            return {'nearest': _Image.NEAREST, 'nn': _Image.NEAREST,
                    'bilinear': _Image.BILINEAR, 'linear': _Image.BILINEAR,
                    'bicubic': _Image.BICUBIC, 'cubic': _Image.BICUBIC}.get(n, _Image.LANCZOS)

        def _resize_to(img, h, w, interp: str):
            resample = _pil_interp(interp)
            return img.resize((int(w), int(h)), resample=resample)

        # Evaluate
        model_metrics = []
        debug_count = 0
        
        for i, (gt, pred, stem, img_path) in enumerate(tqdm(pairs, total=len(pairs))):
            # load image and resize
            img = load_image(str(img_path))
            img = _resize_to(img, image_shape[0], image_shape[1], args.interp)
            img_t = to_tensor(img).unsqueeze(0).to(device=device, dtype=dtype)
            
            with torch.no_grad():
                inv = model_wrapper.depth(img_t)['inv_depths'][0]
                if args.flip_tta:
                    flipped = torch.flip(img_t, dims=[3])
                    inv_f = model_wrapper.depth(flipped)['inv_depths'][0]
                    inv = post_process_inv_depth(inv, inv_f, method='mean')
                pred_depth = inv2depth(inv)
            
            # load GT and mask
            gt_np = load_depth(str(gt))
            if args.mask_file and os.path.exists(args.mask_file):
                mh, mw = gt_np.shape[:2]
                m = _load_mask(args.mask_file, mh, mw).astype(np.float32)
                gt_np = gt_np * m
            
            gt_t = torch.tensor(gt_np).unsqueeze(0).unsqueeze(0).to(device=pred_depth.device, dtype=pred_depth.dtype)
            
            # 디버깅 (모델 예측)
            if args.debug and debug_count < max_debug:
                print(f"\n[Model prediction] {stem}")
                print(f"  Model pred shape: {pred_depth.shape}")
                print(f"  GT shape: {gt_t.shape}")
                pred_np_model = pred_depth.cpu().numpy().squeeze()
                print(f"  Model pred range: [{pred_np_model.min():.4f}, {pred_np_model.max():.4f}]")
                debug_count += 1
            
            m_model = compute_depth_metrics(eval_args, gt_t, pred_depth, use_gt_scale=args.use_gt_scale)
            model_metrics.append(m_model)
            
            # 첫 몇 샘플 메트릭 출력 및 비교
            if args.debug and i < max_debug:
                m_np = m_model.cpu().numpy()
                saved_np = metrics[i].cpu().numpy() if i < len(metrics) else None
                print(f"Model metrics for {stem}:")
                for j, name in enumerate(names):
                    model_val = m_np[j]
                    saved_val = saved_np[j] if saved_np is not None else 0
                    diff = model_val - saved_val
                    print(f"  {name}: model={model_val:.6f}, saved={saved_val:.6f}, diff={diff:.6f}")
            
            if args.per_sample_report:
                rec = per_sample[-1] if per_sample and per_sample[-1]['stem'] == stem else {'stem': stem}
                rec['model'] = [x.item() for x in m_model.detach().cpu()]
                if not (per_sample and per_sample[-1]['stem'] == stem):
                    per_sample.append(rec)

        model_avg = (sum(model_metrics) / len(model_metrics)).detach().cpu().numpy()
        model_avg_list = model_avg.tolist()
        
        if args.debug:
            print(f"\n{'='*60}")
            print("FINAL AVERAGED METRICS (Model on-the-fly)")
            print(f"{'='*60}")
        
        print('\n[Model (on-the-fly) predictions]')
        _print_metrics_table("Evaluation (Model predictions)", names,
                             [(f"{args.model_name} (model)", model_avg_list)])
        delta = (model_avg - saved_avg)
        delta_list = delta.tolist()
        _print_metrics_table("Evaluation Δ (Model - Saved)", names, [("Δ(Model-Saved)", delta_list)])
        with open(args.output_file, 'a') as f:
            f.write('\n[Model (on-the-fly) predictions]\n')
            for name, metric in zip(names, model_avg_list):
                f.write(f'{name} = {metric:.4f}\n')
            
            f.write('\n[Delta: model - saved]\n')
            for name, d in zip(names, delta_list):
                f.write(f'{name} = {d:.4f}\n')

        # Optional per-sample JSON report
        if args.per_sample_report:
            import json as _json
            _ensure_parent_dir(args.per_sample_report)
            with open(args.per_sample_report, 'w') as jf:
                _json.dump({'names': names, 'samples': per_sample}, jf, indent=2)
            print(f"\nPer-sample report saved to: {args.per_sample_report}")
    
    return pairs


def _print_metrics_table(title: str, metric_names: List[str], rows: List[tuple]):
    """모델 이름과 함께 메트릭을 가로 방향으로 출력"""
    print(f"\n{title}")
    header = ['Model'] + metric_names
    col_widths = []
    col_widths.append(max(len(header[0]), max(len(row[0]) for row in rows)))
    for idx, metric in enumerate(metric_names):
        value_width = max(len(metric), max(len(f"{row[1][idx]:.4f}") for row in rows))
        col_widths.append(value_width)
    total_width = sum(col_widths) + 2 * (len(col_widths) - 1)
    print('-' * total_width)
    header_line = header[0].ljust(col_widths[0]) + '  ' + '  '.join(
        metric.rjust(col_widths[i + 1]) for i, metric in enumerate(metric_names)
    )
    print(header_line)
    print('-' * total_width)
    for model_name, values in rows:
        row_line = model_name.ljust(col_widths[0]) + '  ' + '  '.join(
            f"{values[i]:.4f}".rjust(col_widths[i + 1]) for i in range(len(metric_names))
        )
        print(row_line)
    # End line
    print('-' * total_width)


def eval_and_print_by_distance(args, pairs: List[tuple]):
    """거리별로 정량 평가를 수행하고 결과를 표로 출력합니다."""
    print("\nEvaluation by Distance")
    
    eval_args = _make_eval_args(args)
    metric_names = ['abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
    
    # 거리 범위 정의
    dist_ranges = [
        ("D < 1m", 0.0, 1.0),
        ("1m < D < 2m", 1.0, 2.0),
        ("2m < D < 3m", 2.0, 3.0),
        ("D > 3m", 3.0, args.max_depth)
    ]
    
    # 결과를 저장할 리스트
    results_by_range = []

    for name, min_d, max_d in dist_ranges:
        metrics_for_range = []
        
        # 전체 데이터셋에 대해 반복
        for gt_path, pred_path, stem, img_path in tqdm(pairs, desc=name, leave=False):
            gt_np = load_depth(str(gt_path))
            pred_np = load_depth(str(pred_path))

            # 기본 ROI 마스크 적용
            if args.mask_file and os.path.exists(args.mask_file):
                mh, mw = gt_np.shape[:2]
                m = _load_mask(args.mask_file, mh, mw).astype(np.bool_)
                gt_np[~m] = 0

            # 거리별 마스크 생성 및 적용
            dist_mask = (gt_np >= min_d) & (gt_np < max_d)
            
            # 해당 거리에 유효한 GT가 없으면 건너뛰기
            if not np.any(dist_mask):
                continue

            # 임시 GT 생성 (거리 밖은 0으로)
            gt_temp = np.zeros_like(gt_np)
            gt_temp[dist_mask] = gt_np[dist_mask]

            gt_t = torch.from_numpy(gt_temp).unsqueeze(0).unsqueeze(0)
            pred_t = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0)

            # 메트릭 계산
            sample_metrics = compute_depth_metrics(eval_args, gt_t, pred_t, use_gt_scale=args.use_gt_scale)
            metrics_for_range.append(sample_metrics)

        if not metrics_for_range:
            # 해당 범위에 데이터가 없는 경우
            avg_metrics = [0.0] * len(metric_names)
        else:
            avg_metrics_tensor = sum(metrics_for_range) / len(metrics_for_range)
            avg_metrics = avg_metrics_tensor.cpu().numpy().tolist()
        
        # RMSE 단위를 m -> cm로 변경
        rmse_idx = metric_names.index('rmse')
        avg_metrics[rmse_idx] *= 100
        
        results_by_range.append((name, avg_metrics))

    # 결과 테이블 출력
    # RMSE(cm)으로 헤더 변경
    table_metric_names = metric_names[:]
    table_metric_names[table_metric_names.index('rmse')] = 'rmse(cm)'
    
    header = ['Range'] + table_metric_names
    col_widths = [max(len(row[0]) for row in results_by_range + [(header[0], [])])]
    for i, name in enumerate(table_metric_names):
        max_val_len = max((len(f"{row[1][i]:.4f}") for row in results_by_range), default=len(name))
        col_widths.append(max(len(name), max_val_len))

    total_width = sum(col_widths) + 2 * (len(col_widths) - 1)
    print('-' * total_width)
    header_line = header[0].ljust(col_widths[0]) + '  ' + '  '.join(
        h.rjust(col_widths[i + 1]) for i, h in enumerate(table_metric_names)
    )
    print(header_line)
    print('-' * total_width)

    for name, values in results_by_range:
        row_line = name.ljust(col_widths[0]) + '  ' + '  '.join(
            f"{v:.4f}".rjust(col_widths[i + 1]) for i, v in enumerate(values)
        )
        print(row_line)
    print('-' * total_width)


def main(args):
    if args.gt_folder:
        # 레거시 모드는 거리별 평가를 지원하지 않음
        eval_with_gt_folder(args)
    else:
        # eval_with_split_and_variants가 pair 리스트를 반환하도록 수정 필요
        pairs = eval_with_split_and_variants(args)
        if pairs:
            eval_and_print_by_distance(args, pairs)


if __name__ == '__main__':
    args = parse_args()
    main(args)

