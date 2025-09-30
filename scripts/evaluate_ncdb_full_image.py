#!/usr/bin/env python3
"""NCDB full image depth evaluation with visualization.

전체 이미지에 대한 depth 평가 및 시각화를 수행합니다.
Object mask 없이 전체 픽셀에 대해 GT, Pred, Error heatmap을 생성합니다.
"""

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from packnet_sfm.datasets.augmentations import to_tensor
from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.depth import compute_depth_metrics, inv2depth, load_depth, post_process_inv_depth
from packnet_sfm.utils.image import load_image


DEFAULT_ALL_SPLITS = ["combined_train.json", "combined_val.json", "combined_test.json"]
METRIC_NAMES = ["abs_rel", "sqr_rel", "rmse", "rmse_log", "a1", "a2", "a3"]


@dataclass
class SampleEntry:
    stem: str
    sequence_root: Path
    image_path: Path
    gt_path: Path
    prediction_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full image depth on NCDB")
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="데이터셋 루트 경로")
    parser.add_argument("--split-files", type=str, nargs="*",
                        help="평가에 사용할 split JSON")
    parser.add_argument("--use-all-splits", action="store_true",
                        help="combined_train/val/test 세 가지 split 을 모두 로드")
    parser.add_argument("--splits-dir", type=str, default="splits",
                        help="split-files 디렉토리")

    parser.add_argument("--pred-root", type=str, required=True,
                        help="예측 depth 저장 폴더")
    parser.add_argument("--gt-root", type=str, required=True,
                        help="GT depth 폴더명")
    parser.add_argument("--image-subdir", type=str, default="image_a6",
                        help="RGB 이미지 서브폴더")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="체크포인트 파일 경로")
    parser.add_argument("--image-shape", type=int, nargs=2, default=None,
                        help="추론 시 이미지 크기 (height width)")
    parser.add_argument("--flip-tta", action="store_true",
                        help="좌우 반전 TTA 사용 여부")

    parser.add_argument("--output-file", type=str, default=None,
                        help="결과 CSV 저장 경로")
    parser.add_argument("--per-sample-json", type=str, default=None,
                        help="샘플별 메트릭을 저장할 JSON 파일 경로")

    parser.add_argument("--min-depth", type=float, default=None,
                        help="평가에 사용할 최소 깊이 (None이면 모델에서 자동)")
    parser.add_argument("--max-depth", type=float, default=None,
                        help="평가에 사용할 최대 깊이 (None이면 모델에서 자동)")
    parser.add_argument("--crop", type=str, choices=["", "garg"], default="",
                        help="Eigen crop 적용 여부")
    parser.add_argument("--scale-output", type=str, default="top-center",
                        help="스케일 정렬 방식")
    parser.add_argument("--use-gt-scale", action="store_true",
                        help="GT의 중앙값으로 스케일 정렬")

    parser.add_argument("--device", type=str, default="cuda:0",
                        help="추론 디바이스")
    parser.add_argument("--dtype", type=str, choices=["fp16", "fp32"], default="fp32",
                        help="추론 시 데이터 타입")
    parser.add_argument("--debug", action="store_true",
                        help="디버그 출력 활성화")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="평가할 최대 샘플 수 (디버깅용)")
    parser.add_argument("--visualize-dir", type=str, default=None,
                        help="시각화 결과를 저장할 디렉토리")

    return parser.parse_args()


def discover_split_files(args: argparse.Namespace) -> List[Path]:
    """Split JSON 파일 경로 목록 반환."""
    dataset_root = Path(args.dataset_root)
    splits_dir = dataset_root / args.splits_dir

    if args.use_all_splits:
        return [splits_dir / s for s in DEFAULT_ALL_SPLITS]
    
    if not args.split_files:
        raise ValueError("--split-files 또는 --use-all-splits 중 하나는 필수입니다.")
    
    result = []
    for sf in args.split_files:
        p = Path(sf)
        if p.is_absolute():
            result.append(p)
        else:
            result.append(splits_dir / p)
    return result


def load_split_entries(args: argparse.Namespace, split_paths: List[Path]) -> List[dict]:
    """Split JSON들을 읽어 하나의 리스트로 병합."""
    merged = []
    for sp in split_paths:
        if not sp.exists():
            raise FileNotFoundError(f"Split 파일을 찾을 수 없습니다: {sp}")
        with open(sp, "r") as f:
            data = json.load(f)
            merged.extend(data)
    return merged


def get_checkpoint_id(checkpoint_path: str) -> str:
    """체크포인트 경로에서 ID 추출 (캐시 폴더명 생성용)."""
    p = Path(checkpoint_path)
    return p.stem


def normalize_entry(args: argparse.Namespace, dataset_root: Path, entry: dict, checkpoint_id: str) -> SampleEntry:
    """Split entry를 SampleEntry로 변환."""
    stem = entry["new_filename"]
    sequence_root_raw = entry.get("dataset_root", "")
    sequence_root = Path(sequence_root_raw) if sequence_root_raw else dataset_root

    # RGB 경로 - dataset_root 우선
    image_path = sequence_root / args.image_subdir / f"{stem}.png"
    if not image_path.exists():
        image_path_raw = entry.get("image_path")
        if image_path_raw:
            image_path = dataset_root / image_path_raw

    # GT depth 경로
    gt_path = sequence_root / args.gt_root / f"{stem}.png"

    # 예측 depth 저장 경로 (체크포인트별 폴더)
    pred_dir = sequence_root / args.pred_root / checkpoint_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = pred_dir / f"{stem}.npz"

    return SampleEntry(
        stem=stem,
        sequence_root=sequence_root,
        image_path=image_path,
        gt_path=gt_path,
        prediction_path=prediction_path,
    )


def prepare_model(args: argparse.Namespace):
    """모델 로딩 및 설정."""
    print("### Preparing Model")
    config, state_dict = parse_test_file(args.checkpoint)
    model_wrapper = ModelWrapper(config, load_datasets=False)
    model_wrapper.load_state_dict(state_dict)

    device = torch.device(args.device)
    model_wrapper.to(device)
    model_wrapper.eval()

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    if dtype == torch.float16:
        model_wrapper.half()

    # 모델에서 depth 범위 가져오기
    model_min_depth = getattr(model_wrapper.depth_net, 'min_depth', 0.1)
    model_max_depth = getattr(model_wrapper.depth_net, 'max_depth', 100.0)
    
    print(f"\n📊 모델에서 읽어온 depth 범위:")
    print(f"   min_depth: {model_min_depth}")
    print(f"   max_depth: {model_max_depth}")
    
    if args.min_depth is None or args.max_depth is None:
        args.min_depth = model_min_depth
        args.max_depth = model_max_depth
        print(f"   ⚠️  평가 설정을 모델 값으로 자동 조정")

    return {
        "wrapper": model_wrapper,
        "device": device,
        "dtype": dtype,
    }


def make_eval_namespace(args: argparse.Namespace):
    """compute_depth_metrics에 전달할 설정."""
    return argparse.Namespace(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        crop=args.crop,
        scale_output=args.scale_output,
        use_gt_scale=args.use_gt_scale,
    )


def run_inference(model_context: dict, image_path: Path, flip_tta: bool = False) -> np.ndarray:
    """이미지를 읽어 depth 추론 (학습 시 평가와 동일한 방식)."""
    wrapper = model_context["wrapper"]
    device = model_context["device"]
    dtype = model_context["dtype"]

    image = load_image(str(image_path))
    batch = {
        "rgb": to_tensor(image).unsqueeze(0).to(device, dtype=dtype),
    }

    with torch.no_grad():
        # 학습 시와 동일하게 wrapper.model() 직접 호출
        output = wrapper.model(batch)
    
    # 모델 출력에서 inv_depths 추출
    if 'inv_depths' in output:
        inv_depth = output['inv_depths'][0]  # 첫 번째 스케일
    else:
        raise KeyError(f"Cannot find inv_depths in output keys: {output.keys()}")
    
    if flip_tta:
        batch_flip = {"rgb": torch.flip(batch["rgb"], dims=[-1])}
        with torch.no_grad():
            output_flip = wrapper.model(batch_flip)
        
        inv_depth_flip = torch.flip(output_flip['inv_depths'][0], dims=[-1])
        inv_depth = post_process_inv_depth(inv_depth, inv_depth_flip, method='mean')
    
    depth = inv2depth(inv_depth)[0, 0].cpu().numpy()
    return depth


def load_prediction(pred_path: Path) -> Optional[np.ndarray]:
    """캐시된 예측 depth 로드."""
    if not pred_path.exists():
        return None
    data = np.load(str(pred_path))
    return data["depth"]


def save_prediction(pred_path: Path, depth: np.ndarray) -> None:
    """예측 depth를 NPZ로 저장."""
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(pred_path), depth=depth)


def ensure_dir(path: Path) -> None:
    """디렉토리 생성."""
    path.mkdir(parents=True, exist_ok=True)


def visualize_full_image(
    rgb_path: Path,
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    stem: str,
    save_path: Path,
    min_depth: float = 0.05,
    max_depth: float = 100.0,
) -> None:
    """전체 이미지에 대한 4-panel 시각화 (RGB, GT, Pred, Error heatmap)."""
    
    # RGB 로드
    rgb = np.array(Image.open(rgb_path))
    
    # Valid mask
    valid_mask = (gt_depth > min_depth) & (gt_depth < max_depth)
    
    # Gradient colormap with gray for invalid areas (객체 마스크 코드와 동일하게)
    from matplotlib.colors import ListedColormap
    colors_list = ['#cccccc']  # Invalid 영역용 회색
    gradient_colors = ['#00ff00', '#ffff00', '#ff8000', '#ff0000']  # green->yellow->orange->red
    n_gradient = 256
    gradient_cmap = LinearSegmentedColormap.from_list('gradient', gradient_colors, N=n_gradient)
    colors_list.extend([gradient_cmap(i) for i in range(n_gradient)])
    
    combined_cmap = ListedColormap(colors_list)
    
    # Error 계산 (valid 픽셀만)
    error_map = np.full_like(gt_depth, -1.0)  # -1로 초기화 (invalid)
    if valid_mask.any():
        gt_valid = gt_depth[valid_mask]
        pred_valid = pred_depth[valid_mask]
        abs_rel_valid = np.abs(gt_valid - pred_valid) / (gt_valid + 1e-7)
        error_map[valid_mask] = np.clip(abs_rel_valid, 0, 0.5)  # 0.5 이상은 0.5로 클리핑
    
    # Display map: -1 -> 0 (회색), 0~0.5 -> 1~256 (그라디언트)
    display_map = error_map.copy()
    display_map[error_map >= 0] = (error_map[error_map >= 0] / 0.5) * (n_gradient - 1) + 1
    display_map[error_map < 0] = 0
    
    # Error 분류 (현재 기준)
    error_bins = [
        ("[Excellent]", 0.00, 0.05),
        ("[Good]",      0.05, 0.10),
        ("[Fair]",      0.10, 0.20),
        ("[Poor]",      0.20, 0.30),
        ("[Bad]",       0.30, float('inf'))
    ]
    
    bin_counts = []
    valid_errors = error_map[error_map >= 0]  # -1 제외
    total_valid = len(valid_errors)
    
    for label, low, high in error_bins:
        count = np.sum((valid_errors >= low) & (valid_errors < high))
        bin_counts.append((label, count, count / total_valid * 100 if total_valid > 0 else 0))
    
    # 메트릭 계산
    if total_valid > 0:
        abs_rel_mean = valid_errors.mean()
        a1 = np.mean(np.maximum(gt_valid / pred_valid, pred_valid / gt_valid) < 1.25)
    else:
        abs_rel_mean = np.nan
        a1 = np.nan
    
    # 4-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f'RGB Image\n{stem}', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Panel 2: GT Depth (객체 마스크 방식과 동일하게)
    gt_display = gt_depth.copy()
    gt_display[~valid_mask] = np.nan
    # 99 percentile을 사용하여 outlier 제거한 범위 설정
    gt_valid_values = gt_depth[valid_mask]
    if len(gt_valid_values) > 0:
        gt_vmax = np.percentile(gt_valid_values, 99)
    else:
        gt_vmax = max_depth
    
    im1 = axes[0, 1].imshow(gt_display, cmap='viridis', vmin=0, vmax=gt_vmax)
    axes[0, 1].set_title(f'GT Depth (masked)\nRange: [0, {gt_vmax:.1f}]m (99%ile)', 
                         fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], label='Depth (m)', fraction=0.046, pad=0.04)
    
    # Panel 3: Pred Depth (GT와 동일한 범위 사용)
    pred_display = pred_depth.copy()
    pred_display[~valid_mask] = np.nan
    im2 = axes[1, 0].imshow(pred_display, cmap='viridis', vmin=0, vmax=gt_vmax)
    axes[1, 0].set_title(f'Predicted Depth\nRange: [0, {gt_vmax:.1f}]m', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], label='Depth (m)', fraction=0.046, pad=0.04)
    
    # Panel 4: Error Heatmap
    im3 = axes[1, 1].imshow(display_map, cmap=combined_cmap, vmin=0, vmax=n_gradient)
    axes[1, 1].set_title(f'Error Heatmap (abs_rel)\nMean: {abs_rel_mean:.4f}, a1: {a1:.4f}', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Colorbar는 0~0.5 범위만 표시
    cbar = plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar.set_label('Absolute Relative Error', rotation=270, labelpad=20)
    cbar.set_ticks([1, n_gradient//4, n_gradient//2, 3*n_gradient//4, n_gradient])
    cbar.set_ticklabels(['0.0', '0.125', '0.25', '0.375', '0.5+'])
    
    # Error distribution text box
    stats_text = f"Error Distribution ({total_valid:,} pixels):\n"
    stats_text += "-" * 40 + "\n"
    for label, count, pct in bin_counts:
        bar = "▓" * int(pct / 10) + "░" * (10 - int(pct / 10))
        stats_text += f"{label:12s}: {count:7,} ({pct:5.1f}%) {bar}\n"
    
    axes[1, 1].text(1.25, 0.5, stats_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    split_paths = discover_split_files(args)
    raw_entries = load_split_entries(args, split_paths)
    
    checkpoint_id = get_checkpoint_id(args.checkpoint)
    print(f"\n📁 체크포인트 ID: {checkpoint_id}")
    print(f"   예측 파일 저장 위치: {args.pred_root}/{checkpoint_id}/\n")
    
    samples = [normalize_entry(args, dataset_root, entry, checkpoint_id) for entry in raw_entries]

    if args.debug:
        print(f"총 {len(samples)}개 샘플 로드")

    model_context = prepare_model(args)
    eval_namespace = make_eval_namespace(args)

    # 결과 저장
    all_metrics = []
    sample_results = []
    
    visualize_dir = Path(args.visualize_dir) if args.visualize_dir else None
    if visualize_dir:
        ensure_dir(visualize_dir)
        print(f"\n시각화 결과를 저장할 디렉토리: {visualize_dir}")

    processed_samples = 0
    
    for sample_idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        # max_samples 체크
        if args.max_samples is not None and processed_samples >= args.max_samples:
            if args.debug:
                print(f"\n최대 샘플 수 도달: {args.max_samples}, 평가 종료")
            break
        
        gt_data = load_depth(str(sample.gt_path)) if sample.gt_path.exists() else None
        if gt_data is None:
            if args.debug:
                print(f"GT 누락으로 스킵: {sample.gt_path}")
            continue

        prediction = load_prediction(sample.prediction_path)
        if prediction is None:
            prediction = run_inference(model_context, sample.image_path, args.flip_tta)
            save_prediction(sample.prediction_path, prediction)
        elif args.debug:
            print(f"캐시된 예측 사용: {sample.prediction_path}")

        # 메트릭 계산
        gt_tensor = torch.from_numpy(gt_data).unsqueeze(0).unsqueeze(0)
        pred_tensor = torch.from_numpy(prediction).unsqueeze(0).unsqueeze(0)
        
        metrics = compute_depth_metrics(
            config=eval_namespace,
            gt=gt_tensor,
            pred=pred_tensor,
            use_gt_scale=eval_namespace.use_gt_scale,
        )
        
        all_metrics.append(metrics.cpu().numpy())
        
        # 샘플별 결과 저장
        sample_results.append({
            'stem': sample.stem,
            'metrics': metrics.cpu().numpy().tolist(),
        })
        
        # 시각화
        if visualize_dir:
            viz_path = visualize_dir / f"{sample_idx:04d}_{sample.stem}.png"
            visualize_full_image(
                rgb_path=sample.image_path,
                gt_depth=gt_data,
                pred_depth=prediction,
                stem=sample.stem,
                save_path=viz_path,
                min_depth=eval_namespace.min_depth,
                max_depth=eval_namespace.max_depth,
            )
            if args.debug:
                print(f"시각화 저장: {viz_path}")
        
        processed_samples += 1

    # 전체 평균 계산
    if all_metrics:
        mean_metrics = np.stack(all_metrics).mean(axis=0)
        
        print("\n" + "="*80)
        print("평가 요약 (전체 이미지 기준)")
        print("="*80)
        print(f"Samples: {len(all_metrics)}")
        print("-" * 80)
        for i, name in enumerate(METRIC_NAMES):
            print(f"{name:12s}: {mean_metrics[i]:.4f}")
        print("="*80)
        
        # JSON 저장
        if args.per_sample_json:
            output = {
                'metric_names': METRIC_NAMES,
                'samples': sample_results,
                'mean': mean_metrics.tolist(),
            }
            with open(args.per_sample_json, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\n✅ 샘플별 결과 저장: {args.per_sample_json}")
        
        # CSV 저장
        if args.output_file:
            import csv
            with open(args.output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['stem'] + METRIC_NAMES)
                for result in sample_results:
                    writer.writerow([result['stem']] + result['metrics'])
            print(f"✅ CSV 저장: {args.output_file}")
    else:
        print("\n⚠️  평가된 샘플이 없습니다.")


if __name__ == "__main__":
    main()
