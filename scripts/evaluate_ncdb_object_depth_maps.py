#!/usr/bin/env python3
"""NCDB object-masked depth evaluation.

segmentation_results/class_masks 에서 Mask2Former 인스턴스 마스크를 읽어 해당 객체
영역에 한정해 깊이 예측 품질을 정량 평가합니다. 예측 깊이는 지정한 체크포인트로
image_a6 이미지를 즉시 추론하거나, 기존 캐시를 재활용할 수 있습니다.
"""

import argparse
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
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
    segmentation_root: Path
    prediction_path: Path


@dataclass
class InstanceResult:
    stem: str
    class_name: str
    mask_path: Path
    valid_pixels: int
    metrics: List[float]
    gt_mean_depth: float = 0.0
    gt_median_depth: float = 0.0


@dataclass
class SampleData:
    """픽셀 레벨 거리별 평가를 위한 원본 데이터"""
    stem: str
    gt_depth: np.ndarray
    pred_depth: np.ndarray
    mask: np.ndarray
    class_name: str


class MetricsAccumulator:
    """단순 평균 메트릭 누산기."""

    def __init__(self, metric_names: Sequence[str]):
        self.metric_names = list(metric_names)
        self._rows: List[np.ndarray] = []

    def add(self, tensor: torch.Tensor) -> None:
        self._rows.append(tensor.detach().cpu().numpy())

    def count(self) -> int:
        return len(self._rows)

    def mean(self) -> List[float]:
        if not self._rows:
            return [math.nan for _ in self.metric_names]
        stacked = np.stack(self._rows, axis=0)
        return stacked.mean(axis=0).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate depth on NCDB object masks")
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="데이터셋 루트 경로 (splits 디렉토리를 포함하는 최상위 폴더)")
    parser.add_argument("--split-files", type=str, nargs="*",
                        help="평가에 사용할 split JSON (dataset-root 기준 상대경로 혹은 절대경로)")
    parser.add_argument("--use-all-splits", action="store_true",
                        help="combined_train/val/test 세 가지 split 을 모두 로드")
    parser.add_argument("--splits-dir", type=str, default="splits",
                        help="split-files 가 상대경로일 때 기준이 될 디렉토리")

    parser.add_argument("--segmentation-root", type=str, required=True,
                        help="segmentation 결과가 위치한 폴더명 또는 절대경로 (예: segmentation_results)")
    parser.add_argument("--class-mask-subdir", type=str, default="class_masks",
                        help="segmentation-root 하위에서 클래스별 마스크가 위치한 서브폴더명")
    parser.add_argument("--pred-root", type=str, required=True,
                        help="예측 깊이맵을 저장/불러올 폴더명 또는 절대경로")
    parser.add_argument("--gt-root", type=str, required=True,
                        help="GT 깊이맵이 위치한 폴더명 또는 절대경로 (예: newest_depth_maps)")
    parser.add_argument("--image-subdir", type=str, default="image_a6",
                        help="RGB 입력 이미지가 위치한 서브폴더명")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="PackNet-SfM 체크포인트(.ckpt). 지정 모델로 on-the-fly 추론")
    parser.add_argument("--image-shape", type=int, nargs=2, default=None,
                        help="모델 입력으로 사용할 (H W). 미지정 시 체크포인트 config")
    parser.add_argument("--flip-tta", action="store_true",
                        help="좌우 flip test-time augmentation 적용")

    parser.add_argument("--classes", type=str, nargs="*", default=None,
                        help="평가할 클래스 이름 목록. 미지정 시 첫 샘플에서 자동 추론")
    parser.add_argument("--output-file", type=str, default="metrics_object_masks.txt",
                        help="최종 요약 메트릭을 저장할 파일 경로")
    parser.add_argument("--per-instance-json", type=str, default=None,
                        help="각 인스턴스별 세부 메트릭을 JSON 으로 저장")

    parser.add_argument("--min-depth", type=float, default=0.3, help="평가 최소 깊이")
    parser.add_argument("--max-depth", type=float, default=100.0, help="평가 최대 깊이")
    parser.add_argument("--crop", type=str, default="", choices=["", "garg"], help="적용할 crop")
    parser.add_argument("--scale-output", type=str, default="top-center",
                        help="예측 깊이를 GT 해상도로 맞출 때 사용할 모드")
    parser.add_argument("--use-gt-scale", action="store_true",
                        help="GT median scaling 적용 여부")

    parser.add_argument("--device", type=str, default=None,
                        help="torch device (예: cuda:0). 생략 시 GPU 우선")
    parser.add_argument("--dtype", type=str, choices=["fp16", "fp32"], default=None,
                        help="모델 추론 dtype override")
    parser.add_argument("--debug", action="store_true", help="추가 로그 출력")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="디버깅용: 처리할 최대 샘플 수 제한")
    parser.add_argument("--visualize-dir", type=str, default=None,
                        help="시각화 결과를 저장할 디렉토리")
    parser.add_argument("--distance-metric", type=str, default="median", choices=["mean", "median"],
                        help="거리 범위 분류 시 사용할 GT 통계 (기본값: median)")
    
    # Output structure arguments
    parser.add_argument("--output-root", type=str, default="outputs",
                        help="모든 결과를 저장할 최상위 디렉토리")
    parser.add_argument("--save-rgb", action="store_true", default=False,
                        help="RGB 이미지 복사 여부")
    parser.add_argument("--save-gt", action="store_true", default=False,
                        help="GT depth 복사 여부")
    parser.add_argument("--save-pred", action="store_true", default=False,
                        help="예측 depth 복사 여부")

    args = parser.parse_args()

    if not args.split_files and not args.use_all_splits:
        parser.error("--split-files 또는 --use-all-splits 중 하나는 반드시 지정해야 합니다.")

    return args


def resolve_path(base: Path, maybe_path: str) -> Path:
    path = Path(maybe_path)
    return path if path.is_absolute() else base / path


def discover_split_files(args: argparse.Namespace) -> List[Path]:
    dataset_root = Path(args.dataset_root)
    split_paths: List[Path] = []

    if args.use_all_splits:
        splits_dir = resolve_path(dataset_root, args.splits_dir)
        for name in DEFAULT_ALL_SPLITS:
            candidate = splits_dir / name
            if not candidate.exists():
                raise FileNotFoundError(f"split 파일을 찾을 수 없습니다: {candidate}")
            split_paths.append(candidate)

    if args.split_files:
        for item in args.split_files:
            candidate = Path(item)
            if not candidate.is_absolute():
                candidate = resolve_path(dataset_root, args.splits_dir) / item
            if not candidate.exists():
                raise FileNotFoundError(f"split 파일을 찾을 수 없습니다: {candidate}")
            split_paths.append(candidate)

    unique_paths: List[Path] = []
    seen = set()
    for path in split_paths:
        if path in seen:
            continue
        unique_paths.append(path)
        seen.add(path)
    return unique_paths


def load_split_entries(args: argparse.Namespace, split_paths: Iterable[Path]) -> List[dict]:
    entries: List[dict] = []
    for split_path in split_paths:
        with open(split_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"split 형식이 리스트가 아닙니다: {split_path}")
        entries.extend(data)
    if not entries:
        raise RuntimeError("split 에 샘플이 없습니다.")
    return entries


def get_checkpoint_id(checkpoint_path: str) -> str:
    """체크포인트 경로에서 고유 ID 추출"""
    ckpt_path = Path(checkpoint_path)
    
    # 파일명에서 .ckpt 제거
    basename = ckpt_path.stem
    
    # 경로에 특정 패턴이 있으면 사용
    # 예: checkpoints/resnetsan01_640x384_newest_test_fixed_method_0.3_100_silog_1.0/... 
    #     -> resnetsan01_640x384_newest_test_fixed_method_0.3_100_silog_1.0
    if ckpt_path.parent.name and ckpt_path.parent.name.startswith('resnetsan'):
        return ckpt_path.parent.name
    
    # 단순 파일명 사용 (예: ResNet-SAN_0.5to100.ckpt -> ResNet-SAN_0.5to100)
    return basename


def normalize_entry(args: argparse.Namespace, dataset_root: Path, entry: dict, checkpoint_id: str) -> SampleEntry:
    if "new_filename" not in entry:
        raise ValueError(f"split 항목에 new_filename 이 없습니다: {entry}")

    stem = entry["new_filename"]

    sequence_root_raw = entry.get("dataset_root")
    if sequence_root_raw:
        sequence_root = Path(sequence_root_raw)
        if not sequence_root.is_absolute():
            sequence_root = dataset_root / sequence_root
    else:
        sequence_root = dataset_root

    if not sequence_root.exists():
        raise FileNotFoundError(f"sequence_root 를 찾을 수 없습니다: {sequence_root}")

    # ✅ RGB 이미지는 항상 dataset_root/sequence_root와 같은 위치에서 가져옴
    # (GT/Mask와 정렬을 위해 image_path는 무시)
    image_path = resolve_path(sequence_root, args.image_subdir) / f"{stem}.png"
    
    if not image_path.exists():
        # fallback: split에 명시된 image_path 사용
        image_path_raw = entry.get("image_path")
        if image_path_raw:
            fallback_path = Path(image_path_raw) if Path(image_path_raw).is_absolute() else sequence_root / image_path_raw
            if fallback_path.exists():
                print(f"⚠️  WARNING: RGB 이미지를 다른 시퀀스에서 가져옴: {fallback_path}")
                image_path = fallback_path
            else:
                raise FileNotFoundError(f"RGB 이미지를 찾을 수 없습니다: {image_path}")

    segmentation_root = resolve_path(sequence_root, args.segmentation_root)
    class_mask_root = segmentation_root / args.class_mask_subdir if args.class_mask_subdir else segmentation_root

    gt_candidates = [resolve_path(sequence_root, args.gt_root) / f"{stem}.png",
                     resolve_path(sequence_root, args.gt_root) / f"{stem}.npz"]
    gt_path = next((p for p in gt_candidates if p.exists()), gt_candidates[0])

    # ✅ 체크포인트별 폴더 생성: pred_root/checkpoint_id/
    pred_root_base = resolve_path(sequence_root, args.pred_root)
    pred_root = pred_root_base / checkpoint_id
    pred_path = pred_root / f"{stem}.npz"

    return SampleEntry(
        stem=stem,
        sequence_root=sequence_root,
        image_path=image_path,
        gt_path=gt_path,
        segmentation_root=class_mask_root,
        prediction_path=pred_path,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def create_output_structure(checkpoint_id: str, output_root: str) -> Dict[str, Path]:
    """체크포인트별 출력 폴더 구조 생성"""
    base = Path(output_root) / f"{checkpoint_id}_results"
    
    structure = {
        'rgb': base / 'rgb',
        'gt': base / 'gt',
        'pred': base / 'pred',
        'viz': base / 'viz',
        'metrics': base / 'metrics'
    }
    
    for path in structure.values():
        ensure_dir(path)
    
    return structure


def copy_file_to_output(src: Path, dst_dir: Path, new_name: Optional[str] = None) -> None:
    """파일을 출력 디렉토리로 복사"""
    if not src.exists():
        return
    
    dst_name = new_name if new_name else src.name
    dst_path = dst_dir / dst_name
    
    shutil.copy2(src, dst_path)


def save_depth_as_png(depth: np.ndarray, path: Path, scale: float = 256.0) -> None:
    """Depth를 16-bit PNG로 저장 (meter to mm 변환)"""
    # depth: (H, W) in meters
    depth_mm = (depth * scale).astype(np.uint16)
    Image.fromarray(depth_mm).save(path)


def load_mask(mask_path: Path, target_shape: Tuple[int, int]) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    if mask.size != (target_shape[1], target_shape[0]):
        mask = mask.resize((target_shape[1], target_shape[0]), Image.NEAREST)
    mask_arr = (np.array(mask) > 0).astype(np.float32)
    return mask_arr


def collect_masks_for_stem(segmentation_root: Path, class_names: Sequence[str], stem: str) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {}
    if not segmentation_root.exists():
        return result
    for class_name in class_names:
        class_dir = segmentation_root / class_name
        if not class_dir.exists():
            continue
        pattern = f"{stem}*.png"
        files = sorted(class_dir.glob(pattern))
        if files:
            result[class_name] = files
    return result


@dataclass
class ModelContext:
    wrapper: ModelWrapper
    device: torch.device
    dtype: torch.dtype
    image_shape: Tuple[int, int]


def prepare_model(args: argparse.Namespace) -> ModelContext:
    config, state_dict = parse_test_file(args.checkpoint)

    wrapper = ModelWrapper(config, load_datasets=False)
    wrapper.load_state_dict(state_dict, strict=False)

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "fp32":
        dtype = torch.float32
    else:
        dtype = torch.float16 if getattr(config.arch, "dtype", None) == torch.float16 else torch.float32

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wrapper = wrapper.to(device=device, dtype=dtype)
    wrapper.eval()

    if args.image_shape is not None:
        image_shape = (int(args.image_shape[0]), int(args.image_shape[1]))
    else:
        aug_shape = getattr(getattr(config, "datasets", None), "augmentation", None)
        if aug_shape is not None:
            image_shape = tuple(map(int, getattr(aug_shape, "image_shape")))
        else:
            raise RuntimeError("image_shape 를 config 또는 인자로부터 결정할 수 없습니다.")
    
    # 모델에서 min_depth, max_depth 읽어오기
    depth_net = wrapper.depth_net if hasattr(wrapper, 'depth_net') else None
    if depth_net is not None and hasattr(depth_net, 'min_depth') and hasattr(depth_net, 'max_depth'):
        model_min_depth = float(depth_net.min_depth)
        model_max_depth = float(depth_net.max_depth)
        print(f"\n📊 모델에서 읽어온 depth 범위:")
        print(f"   min_depth: {model_min_depth}")
        print(f"   max_depth: {model_max_depth}")
        
        # args에 설정되지 않은 경우 모델 값으로 오버라이드
        if args.min_depth == 0.3 and args.max_depth == 100.0:  # 기본값인 경우
            print(f"   ⚠️  평가 설정을 모델 값으로 자동 조정")
            args.min_depth = model_min_depth
            args.max_depth = model_max_depth
        elif abs(args.min_depth - model_min_depth) > 0.01 or abs(args.max_depth - model_max_depth) > 0.1:
            print(f"   ⚠️  WARNING: 평가 설정과 모델 학습 설정이 다릅니다!")
            print(f"   평가: min={args.min_depth}, max={args.max_depth}")
            print(f"   모델: min={model_min_depth}, max={model_max_depth}")

    return ModelContext(wrapper=wrapper, device=device, dtype=dtype, image_shape=image_shape)


def run_inference(context: ModelContext, image_path: Path, flip_tta: bool) -> np.ndarray:
    img = load_image(str(image_path)).convert("RGB")
    if img.size != (context.image_shape[1], context.image_shape[0]):
        img = img.resize((context.image_shape[1], context.image_shape[0]), Image.LANCZOS)
    img_tensor = to_tensor(img).unsqueeze(0).to(device=context.device, dtype=context.dtype)

    with torch.no_grad():
        inv_depth = context.wrapper.depth(img_tensor)["inv_depths"][0]
        if flip_tta:
            flipped = torch.flip(img_tensor, dims=[3])
            inv_depth_f = context.wrapper.depth(flipped)["inv_depths"][0]
            inv_depth = post_process_inv_depth(inv_depth, inv_depth_f, method="mean")
        depth = inv2depth(inv_depth).squeeze().detach().cpu().float().numpy()

    return depth


def load_prediction(prediction_path: Path) -> Optional[np.ndarray]:
    if not prediction_path.exists():
        return None
    try:
        data = np.load(prediction_path)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "depth" in data:
                return data["depth"]
            return data[list(data.files)[0]]
        return data
    except Exception:
        return None


def save_prediction(prediction_path: Path, depth: np.ndarray) -> None:
    ensure_dir(prediction_path.parent)
    np.savez_compressed(prediction_path, depth=depth.astype(np.float32))


def make_eval_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        crop=args.crop,
        scale_output=args.scale_output,
    )


def print_summary_table(class_metrics: Dict[str, Tuple[List[float], int]], 
                        overall: Tuple[List[float], int],
                        full_image_metrics: Optional[Tuple[List[float], int]] = None) -> None:
    header = ["Class", "Count"] + METRIC_NAMES
    rows = []
    for class_name, (metrics, count) in sorted(class_metrics.items()):
        rows.append([class_name, str(count)] + [f"{m:.4f}" if not math.isnan(m) else "nan" for m in metrics])
    rows.append(["car+road", str(overall[1])] + [f"{m:.4f}" if not math.isnan(m) else "nan" for m in overall[0]])
    
    # 전체 픽셀 메트릭 추가 (full image)
    if full_image_metrics is not None:
        rows.append(["ALL", str(full_image_metrics[1])] + [f"{m:.4f}" if not math.isnan(m) else "nan" for m in full_image_metrics[0]])

    col_widths = [max(len(row[i]) for row in rows + [header]) for i in range(len(header))]

    def print_row(row: Sequence[str]) -> None:
        print("  ".join(word.ljust(col_widths[i]) for i, word in enumerate(row)))

    print("\n평가 요약 (객체 마스크 및 전체 기준)")
    print_row(header)
    print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in rows:
        print_row(row)


def visualize_sample(
    image_path: Path,
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    mask: np.ndarray,
    save_path: Path,
    class_name: str,
    stem: str,
    metrics: Optional[List[float]] = None,
) -> None:
    """이미지, GT, Pred, 마스크를 Gradient 에러 히트맵으로 시각화 (4-panel)"""
    from matplotlib.colors import LinearSegmentedColormap
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # [1] RGB + 마스크 오버레이
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    
    # 마스크 영역을 반투명 초록색으로 오버레이
    overlay = img_array.copy()
    mask_bool = mask > 0
    overlay[mask_bool, 1] = np.clip(overlay[mask_bool, 1] + 100, 0, 255)  # Green channel boost
    
    # 알파 블렌딩 (70% 원본, 30% 오버레이)
    img_with_mask = (0.7 * img_array + 0.3 * overlay).astype(np.uint8)
    
    axes[0, 0].imshow(img_with_mask)
    axes[0, 0].set_title(f"RGB + Mask Overlay\n{stem}", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # [2] GT Depth (마스크 영역만)
    gt_masked = gt_depth.copy()
    gt_masked[mask == 0] = np.nan
    gt_max = np.nanpercentile(gt_masked, 99) if np.any(~np.isnan(gt_masked)) else 1.0
    im1 = axes[0, 1].imshow(gt_masked, cmap='viridis', vmin=0, vmax=gt_max)
    axes[0, 1].set_title(f"GT Depth (masked)\n{class_name}", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], label='Depth (m)', fraction=0.046, pad=0.04)
    
    # [3] Pred Depth (마스크 영역만)
    pred_masked = pred_depth.copy()
    pred_masked[mask == 0] = np.nan
    im2 = axes[1, 0].imshow(pred_masked, cmap='viridis', vmin=0, vmax=gt_max)
    axes[1, 0].set_title(f"Predicted Depth", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], label='Depth (m)', fraction=0.046, pad=0.04)
    
    # [4] Error Heatmap (Gradient Green->Yellow->Orange->Red)
    valid_mask = (mask > 0) & (gt_depth > 0)
    
    # 에러 맵 생성: 마스크 밖은 회색(NaN 대신 -1), 마스크 안은 abs_rel 값
    abs_rel_map = np.full_like(gt_depth, -1.0, dtype=np.float32)  # 기본값 -1 (회색으로 표시)
    
    if np.any(valid_mask):
        # 유효한 픽셀만 abs_rel 계산
        abs_rel_values = np.abs(pred_depth[valid_mask] - gt_depth[valid_mask]) / gt_depth[valid_mask]
        abs_rel_map[valid_mask] = np.clip(abs_rel_values, 0, 0.5)  # 0.5 이상은 0.5로 클리핑
    
    # Gradient colormap with gray for masked-out areas
    # -1: 회색(마스크 밖), 0: 초록(완벽), 0.5: 빨강(나쁨)
    from matplotlib.colors import ListedColormap
    colors_list = ['#cccccc']  # -1 값용 회색
    gradient_colors = ['#00ff00', '#ffff00', '#ff8000', '#ff0000']  # green->yellow->orange->red
    n_gradient = 256
    gradient_cmap = LinearSegmentedColormap.from_list('gradient', gradient_colors, N=n_gradient)
    colors_list.extend([gradient_cmap(i) for i in range(n_gradient)])
    
    # -1~0.5 범위를 0~257 인덱스로 매핑
    # -1 -> 0 (회색), 0~0.5 -> 1~256 (그라디언트)
    display_map = abs_rel_map.copy()
    display_map[abs_rel_map >= 0] = (abs_rel_map[abs_rel_map >= 0] / 0.5) * (n_gradient - 1) + 1
    display_map[abs_rel_map < 0] = 0
    
    combined_cmap = ListedColormap(colors_list)
    im3 = axes[1, 1].imshow(display_map, cmap=combined_cmap, vmin=0, vmax=n_gradient)
    axes[1, 1].set_title(f"Error Heatmap (abs_rel)\nGreen=Good, Red=Bad, Gray=No mask", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Colorbar는 0~0.5 범위만 표시
    cbar = plt.colorbar(im3, ax=axes[1, 1], label='Absolute Relative Error', fraction=0.046, pad=0.04, 
                        ticks=[1, n_gradient//4, n_gradient//2, 3*n_gradient//4, n_gradient])
    cbar.ax.set_yticklabels(['0.0', '0.125', '0.25', '0.375', '0.5+'])
    cbar.ax.set_ylabel('abs_rel (0.0=Perfect, 0.5+=Bad)', rotation=270, labelpad=20)
    
    # 통계 텍스트
    valid_errors = abs_rel_map[abs_rel_map >= 0]  # -1이 아닌 실제 에러 값만
    if len(valid_errors) > 0:
        stats_text = f"""Statistics (masked area):
  Mean abs_rel: {valid_errors.mean():.4f}  |  Median abs_rel: {np.median(valid_errors):.4f}  |  Total pixels: {len(valid_errors):,}

Pixel Distribution:
  [Excellent] (0.00-0.05): {(valid_errors < 0.05).sum():6,d} ({100*(valid_errors < 0.05).mean():5.1f}%)
  [Good]      (0.05-0.10): {((valid_errors >= 0.05) & (valid_errors < 0.10)).sum():6,d} ({100*((valid_errors >= 0.05) & (valid_errors < 0.10)).mean():5.1f}%)
  [Fair]      (0.10-0.20): {((valid_errors >= 0.10) & (valid_errors < 0.20)).sum():6,d} ({100*((valid_errors >= 0.10) & (valid_errors < 0.20)).mean():5.1f}%)
  [Poor]      (0.20-0.30): {((valid_errors >= 0.20) & (valid_errors < 0.30)).sum():6,d} ({100*((valid_errors >= 0.20) & (valid_errors < 0.30)).mean():5.1f}%)
  [Bad]       (0.30+):     {(valid_errors >= 0.30).sum():6,d} ({100*(valid_errors >= 0.30).mean():5.1f}%)"""
        
        if metrics:
            stats_text += f"\n\nDepth Metrics: abs_rel={metrics[0]:.4f}, rmse={metrics[2]:.4f}, a1={metrics[4]:.4f}"
        
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    ensure_dir(save_path.parent)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def analyze_by_distance_ranges(
    instance_results: List[InstanceResult],
    dist_ranges: List[Tuple[str, float, float]],
    use_median: bool = True
) -> None:
    """인스턴스를 거리 범위별로 분류하고 메트릭을 집계 (구 방식 - 참고용)"""
    
    # 거리 범위별 그룹화
    distance_groups: Dict[str, List[InstanceResult]] = {name: [] for name, _, _ in dist_ranges}
    
    for instance in instance_results:
        # "_ALL" 통합 마스크는 제외 (개별 인스턴스만 분석)
        if instance.class_name.endswith("_ALL"):
            continue
            
        representative_distance = instance.gt_median_depth if use_median else instance.gt_mean_depth
        
        # 거리 범위 결정
        for range_name, min_d, max_d in dist_ranges:
            if min_d <= representative_distance < max_d:
                distance_groups[range_name].append(instance)
                break
    
    # 결과 테이블 생성
    header = ["Range", "Count"] + METRIC_NAMES
    rows = []
    
    for range_name, min_d, max_d in dist_ranges:
        instances = distance_groups[range_name]
        count = len(instances)
        
        if count == 0:
            # 데이터가 없는 경우 NaN으로 표시
            rows.append([range_name, "0"] + ["nan"] * len(METRIC_NAMES))
        else:
            # 메트릭 평균 계산
            metrics_array = np.array([inst.metrics for inst in instances])
            avg_metrics = metrics_array.mean(axis=0)
            rows.append([range_name, str(count)] + [f"{m:.4f}" for m in avg_metrics])
    
    # 출력
    col_widths = [max(len(row[i]) for row in rows + [header]) for i in range(len(header))]
    
    def print_row(row: Sequence[str]) -> None:
        print("  ".join(word.ljust(col_widths[i]) for i, word in enumerate(row)))
    
    distance_type = "Median" if use_median else "Mean"
    print(f"\n거리별 평가 결과 (Instance {distance_type} 기준 - 구 방식)")
    print_row(header)
    print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in rows:
        print_row(row)


def analyze_by_distance_ranges_pixel_level(
    samples_data: List[SampleData],
    dist_ranges: List[Tuple[str, float, float]],
    eval_namespace: argparse.Namespace,
    class_filter: Optional[str] = None
) -> List[Tuple[str, int, List[float]]]:
    """픽셀 레벨로 거리 범위별 평가 (신규 정확한 방식)
    
    Args:
        samples_data: 샘플 데이터 리스트
        dist_ranges: 거리 범위 리스트
        eval_namespace: 평가 설정
        class_filter: 특정 클래스만 필터링 (None이면 전체)
        
    Returns:
        List of (range_name, pixel_count, metrics_list)
    """
    
    # 1. 거리별로 픽셀 수집
    range_pixels: Dict[str, Dict[str, List[float]]] = {}
    
    for sample in samples_data:
        # 클래스 필터 적용
        if class_filter is not None and sample.class_name != class_filter:
            continue
            
        valid_mask = (sample.mask > 0) & (sample.gt_depth > 0)
        gt_valid = sample.gt_depth[valid_mask]
        pred_valid = sample.pred_depth[valid_mask]
        
        for range_name, min_d, max_d in dist_ranges:
            # 이 범위에 속하는 픽셀 필터링
            in_range = (gt_valid >= min_d) & (gt_valid < max_d)
            
            if not np.any(in_range):
                continue
            
            if range_name not in range_pixels:
                range_pixels[range_name] = {'gt': [], 'pred': []}
            
            range_pixels[range_name]['gt'].extend(gt_valid[in_range].tolist())
            range_pixels[range_name]['pred'].extend(pred_valid[in_range].tolist())
    
    # 2. 각 범위별 메트릭 계산 (직접 계산 - min/max depth 필터링 없이)
    header = ["Range", "Pixels"] + METRIC_NAMES
    rows = []
    results = []  # (range_name, pixel_count, metrics_list) 저장
    
    for range_name, min_d, max_d in dist_ranges:
        if range_name not in range_pixels or len(range_pixels[range_name]['gt']) == 0:
            rows.append([range_name, "0"] + ["nan"] * len(METRIC_NAMES))
            results.append((range_name, 0, [float('nan')] * len(METRIC_NAMES)))
            continue
        
        # GT/Pred 배열 (이미 필터링된 데이터)
        gt_array = np.array(range_pixels[range_name]['gt'])
        pred_array = np.array(range_pixels[range_name]['pred'])
        
        # ✅ 메트릭 직접 계산 (compute_depth_metrics의 min/max 필터링 우회)
        gt_tensor = torch.from_numpy(gt_array).float()
        pred_tensor = torch.from_numpy(pred_array).float()
        
        # abs_rel, sqr_rel, rmse, rmse_log, a1, a2, a3
        thresh = torch.max((gt_tensor / pred_tensor), (pred_tensor / gt_tensor))
        a1 = (thresh < 1.25).float().mean()
        a2 = (thresh < 1.25 ** 2).float().mean()
        a3 = (thresh < 1.25 ** 3).float().mean()
        
        diff = gt_tensor - pred_tensor
        abs_rel = torch.mean(torch.abs(diff) / gt_tensor)
        sq_rel = torch.mean(diff ** 2 / gt_tensor)
        rmse = torch.sqrt(torch.mean(diff ** 2))
        rmse_log = torch.sqrt(torch.mean((torch.log(gt_tensor) - torch.log(pred_tensor)) ** 2))
        
        metrics_list = [abs_rel.item(), sq_rel.item(), rmse.item(), rmse_log.item(),
                       a1.item(), a2.item(), a3.item()]
        
        rows.append([range_name, str(len(gt_array))] + 
                   [f"{m:.4f}" for m in metrics_list])
        results.append((range_name, len(gt_array), metrics_list))
    
    # 테이블 출력
    col_widths = [max(len(row[i]) for row in rows + [header]) for i in range(len(header))]
    
    def print_row(row: Sequence[str]) -> None:
        print("  ".join(word.ljust(col_widths[i]) for i, word in enumerate(row)))
    
    # 클래스별 제목 출력
    if class_filter:
        print(f"\n거리별 평가 결과 [{class_filter.upper()}] (픽셀 레벨)")
    else:
        print(f"\n거리별 평가 결과 [ALL] (픽셀 레벨)")
    print_row(header)
    print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in rows:
        print_row(row)
    
    return results


def print_distance_error_distribution(
    samples_data: List[SampleData],
    dist_ranges: List[Tuple[str, float, float]]
) -> None:
    """거리별 에러 등급 분포 출력"""
    
    error_bins = [
        ("[Excellent]", 0.00, 0.05),
        ("[Good]",      0.05, 0.10),
        ("[Fair]",      0.10, 0.20),
        ("[Poor]",      0.20, 0.30),
        ("[Bad]",       0.30, float('inf'))
    ]
    
    print("\n거리별 에러 분포")
    print("=" * 70)
    
    for range_name, min_d, max_d in dist_ranges:
        # 픽셀 수집 및 abs_rel 계산
        all_abs_rel = []
        
        for sample in samples_data:
            valid_mask = (sample.mask > 0) & (sample.gt_depth > 0)
            gt_valid = sample.gt_depth[valid_mask]
            pred_valid = sample.pred_depth[valid_mask]
            
            # 이 범위에 속하는 픽셀 필터링
            in_range = (gt_valid >= min_d) & (gt_valid < max_d)
            
            if not np.any(in_range):
                continue
            
            # abs_rel 계산
            abs_rel = np.abs(pred_valid[in_range] - gt_valid[in_range]) / gt_valid[in_range]
            all_abs_rel.extend(abs_rel.tolist())
        
        if len(all_abs_rel) == 0:
            print(f"\n{range_name}: No pixels")
            continue
        
        abs_rel_array = np.array(all_abs_rel)
        total = len(abs_rel_array)
        
        print(f"\n{range_name} ({total:,} pixels)")
        print("-" * 70)
        
        # 에러 등급별 카운트
        for bin_name, bin_min, bin_max in error_bins:
            count = ((abs_rel_array >= bin_min) & (abs_rel_array < bin_max)).sum()
            pct = 100 * count / total
            bar = "▓" * int(pct / 10) + "░" * (10 - int(pct / 10))
            print(f"  {bin_name:15s}: {count:6,d} ({pct:5.1f}%)  {bar}")
    
    print("=" * 70)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    split_paths = discover_split_files(args)
    raw_entries = load_split_entries(args, split_paths)
    
    # ✅ 체크포인트 ID 추출
    checkpoint_id = get_checkpoint_id(args.checkpoint)
    print(f"\n📁 체크포인트 ID: {checkpoint_id}")
    print(f"   예측 파일 저장 위치: {args.pred_root}/{checkpoint_id}/\n")
    
    samples = [normalize_entry(args, dataset_root, entry, checkpoint_id) for entry in raw_entries]

    if args.debug:
        print(f"총 {len(samples)}개 샘플 로드")

    model_context = prepare_model(args)
    eval_namespace = make_eval_namespace(args)

    detected_classes: Optional[List[str]] = args.classes if args.classes else None
    instance_records: List[InstanceResult] = []
    all_samples_data: List[SampleData] = []  # ✅ 픽셀 레벨 분석용 데이터
    class_accumulators: Dict[str, MetricsAccumulator] = defaultdict(lambda: MetricsAccumulator(METRIC_NAMES))
    overall_accumulator = MetricsAccumulator(METRIC_NAMES)

    reference_shape: Optional[Tuple[int, int]] = None
    
    # 출력 디렉토리 구조 생성
    output_dirs = create_output_structure(checkpoint_id, args.output_root)
    print(f"\n📁 출력 디렉토리 생성: {Path(args.output_root) / f'{checkpoint_id}_results'}")
    
    # 디버깅용 카운터
    processed_samples = 0
    visualize_dir = output_dirs['viz']  # 항상 viz 디렉토리에 저장
    print(f"   시각화 저장 위치: {visualize_dir}")

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

        if reference_shape is None:
            reference_shape = gt_data.shape
            if args.debug:
                print(f"GT 기준 해상도: {reference_shape[::-1]} (w x h)")
        elif gt_data.shape != reference_shape:
            raise ValueError(f"GT 해상도가 일관되지 않습니다: {sample.gt_path} -> {gt_data.shape}")

        prediction = load_prediction(sample.prediction_path)
        if prediction is None:
            prediction = run_inference(model_context, sample.image_path, args.flip_tta)
            save_prediction(sample.prediction_path, prediction)
        elif args.debug:
            print(f"캐시된 예측 사용: {sample.prediction_path}")
        
        # 파일 복사 (선택적)
        if args.save_rgb:
            copy_file_to_output(sample.image_path, output_dirs['rgb'], f"{sample.stem}.png")
        
        if args.save_gt:
            copy_file_to_output(sample.gt_path, output_dirs['gt'], f"{sample.stem}.png")
        
        if args.save_pred:
            # 예측 depth를 16-bit PNG로 저장
            save_depth_as_png(prediction, output_dirs['pred'] / f"{sample.stem}.png")

        if prediction.shape != gt_data.shape and args.debug:
            print(f"예측/GT 해상도 불일치: pred {prediction.shape}, gt {gt_data.shape}")

        if detected_classes is None:
            class_root = sample.segmentation_root
            if not class_root.exists():
                raise FileNotFoundError(f"class_masks 디렉토리를 찾을 수 없습니다: {class_root}")
            detected_classes = sorted([d.name for d in class_root.iterdir() if d.is_dir()])
            if args.debug:
                print(f"자동 탐지된 클래스: {detected_classes}")

        mask_groups = collect_masks_for_stem(sample.segmentation_root, detected_classes or [], sample.stem)
        if not mask_groups:
            if args.debug:
                print(f"마스크 미존재로 샘플 스킵: {sample.stem}")
            continue

        pred_tensor = torch.tensor(prediction, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        sample_has_valid_mask = False
        
        # 이미지 전체 객체를 합친 마스크 (클래스별로 분리)
        combined_masks_by_class: Dict[str, np.ndarray] = {}

        for class_name, mask_paths in mask_groups.items():
            # 클래스별 통합 마스크 초기화
            if class_name not in combined_masks_by_class:
                combined_masks_by_class[class_name] = np.zeros_like(gt_data, dtype=np.float32)
            
            for mask_idx, mask_path in enumerate(mask_paths):
                mask = load_mask(mask_path, gt_data.shape)
                valid_mask = (mask > 0) & (gt_data > 0)
                if not np.any(valid_mask):
                    if args.debug:
                        print(f"유효 픽셀이 없어 스킵: {mask_path}")
                    continue

                sample_has_valid_mask = True
                
                # 클래스별 통합 마스크에 추가 (OR 연산)
                combined_masks_by_class[class_name] = np.maximum(combined_masks_by_class[class_name], mask)
                
                # 디버그 출력
                if args.debug:
                    print(f"\n{'='*80}")
                    print(f"[Sample {processed_samples}] {sample.stem} - {class_name} - mask #{mask_idx}")
                    print(f"{'='*80}")
                    print(f"Image path: {sample.image_path}")
                    print(f"GT path: {sample.gt_path}")
                    print(f"Mask path: {mask_path}")
                    print(f"\nGT depth (전체 이미지):")
                    gt_valid_all = gt_data > 0
                    if gt_valid_all.any():
                        print(f"  Range: [{gt_data[gt_valid_all].min():.4f}, {gt_data[gt_valid_all].max():.4f}]")
                        print(f"  Mean: {gt_data[gt_valid_all].mean():.4f}")
                        print(f"  Median: {np.median(gt_data[gt_valid_all]):.4f}")
                        print(f"  Valid pixels: {gt_valid_all.sum()} / {gt_data.size}")
                    
                    print(f"\nPred depth (전체 이미지):")
                    print(f"  Range: [{prediction.min():.4f}, {prediction.max():.4f}]")
                    print(f"  Mean: {prediction.mean():.4f}")
                    print(f"  Median: {np.median(prediction):.4f}")
                    
                    print(f"\nMask info:")
                    print(f"  Unique values: {np.unique(mask)}")
                    print(f"  Masked pixels: {(mask > 0).sum()}")
                    
                    print(f"\nGT depth (마스크 영역만):")
                    gt_in_mask = gt_data[valid_mask]
                    print(f"  Range: [{gt_in_mask.min():.4f}, {gt_in_mask.max():.4f}]")
                    print(f"  Mean: {gt_in_mask.mean():.4f}")
                    print(f"  Median: {np.median(gt_in_mask):.4f}")
                    print(f"  Valid pixels: {len(gt_in_mask)}")
                    
                    print(f"\nPred depth (마스크 영역만):")
                    pred_in_mask = prediction[valid_mask]
                    print(f"  Range: [{pred_in_mask.min():.4f}, {pred_in_mask.max():.4f}]")
                    print(f"  Mean: {pred_in_mask.mean():.4f}")
                    print(f"  Median: {np.median(pred_in_mask):.4f}")

                # ✅ GT와 Pred 모두 마스크 영역만 추출하여 비교
                # compute_depth_metrics는 4D tensor (B, C, H, W)를 기대하므로
                # 마스크된 영역을 원본 크기로 재구성 (마스크 밖은 0)
                gt_masked_full = gt_data * mask
                pred_masked_full = prediction * mask
                
                gt_masked_tensor = torch.tensor(gt_masked_full, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                pred_masked_tensor = torch.tensor(pred_masked_full, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                metrics = compute_depth_metrics(eval_namespace, gt_masked_tensor, pred_masked_tensor, use_gt_scale=args.use_gt_scale)

                # GT 거리 통계 계산
                gt_valid_values = gt_data[valid_mask]
                gt_mean = float(gt_valid_values.mean())
                gt_median = float(np.median(gt_valid_values))

                if args.debug:
                    print(f"\nGT distance statistics:")
                    print(f"  Mean: {gt_mean:.4f}m")
                    print(f"  Median: {gt_median:.4f}m")
                    print(f"\nComputed metrics (인스턴스별):")
                    for i, name in enumerate(METRIC_NAMES):
                        print(f"  {name}: {metrics[i].item():.6f}")

                class_accumulators[class_name].add(metrics)
                overall_accumulator.add(metrics)

                instance_records.append(InstanceResult(
                    stem=sample.stem,
                    class_name=class_name,
                    mask_path=mask_path,
                    valid_pixels=int(valid_mask.sum()),
                    metrics=metrics.detach().cpu().numpy().tolist(),
                    gt_mean_depth=gt_mean,
                    gt_median_depth=gt_median,
                ))
                
                # 시각화 (인스턴스별)
                if visualize_dir is not None and args.debug:  # 디버그 모드에서만 인스턴스별 저장
                    vis_filename = f"{sample_idx:04d}_{sample.stem}_{class_name}_inst{mask_idx}.png"
                    vis_path = visualize_dir / vis_filename
                    visualize_sample(
                        sample.image_path,
                        gt_data,
                        prediction,
                        mask,
                        vis_path,
                        f"{class_name} (inst {mask_idx})",
                        sample.stem,
                        metrics.detach().cpu().numpy().tolist(),
                    )
                    if args.debug:
                        print(f"\n시각화 저장 (인스턴스): {vis_path}")
        
        # 클래스별 통합 마스크로 평가 (이미지당 모든 객체)
        for class_name, combined_mask in combined_masks_by_class.items():
            if not np.any(combined_mask > 0):
                continue
            
            valid_combined = (combined_mask > 0) & (gt_data > 0)
            if not np.any(valid_combined):
                continue
            
            # ✅ GT와 Pred 모두 마스크 영역만 추출하여 비교
            # compute_depth_metrics는 4D tensor를 기대하므로 원본 크기 유지 (마스크 밖은 0)
            gt_combined_full = gt_data * combined_mask
            pred_combined_full = prediction * combined_mask
            
            gt_combined_tensor = torch.tensor(gt_combined_full, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            pred_combined_tensor = torch.tensor(pred_combined_full, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            combined_metrics = compute_depth_metrics(eval_namespace, gt_combined_tensor, pred_combined_tensor, use_gt_scale=args.use_gt_scale)
            
            if args.debug:
                print(f"\n{'='*80}")
                print(f"[Sample {processed_samples}] {sample.stem} - {class_name} - ALL INSTANCES COMBINED")
                print(f"{'='*80}")
                print(f"Combined mask pixels: {(combined_mask > 0).sum()}")
                print(f"Valid pixels: {valid_combined.sum()}")
                print(f"\nComputed metrics (통합):")
                for i, name in enumerate(METRIC_NAMES):
                    print(f"  {name}: {combined_metrics[i].item():.6f}")
            
            # 통합 마스크 GT 거리 통계
            gt_combined_valid = gt_data[valid_combined]
            gt_combined_mean = float(gt_combined_valid.mean())
            gt_combined_median = float(np.median(gt_combined_valid))
            
            # 통합 마스크 결과를 별도로 저장
            instance_records.append(InstanceResult(
                stem=sample.stem,
                class_name=f"{class_name}_ALL",  # 통합임을 표시
                mask_path=Path(f"combined_{class_name}"),
                valid_pixels=int(valid_combined.sum()),
                metrics=combined_metrics.detach().cpu().numpy().tolist(),
                gt_mean_depth=gt_combined_mean,
                gt_median_depth=gt_combined_median,
            ))
            
            # 시각화 (통합 마스크)
            if visualize_dir is not None:
                vis_filename = f"{sample_idx:04d}_{sample.stem}_{class_name}_ALL.png"
                vis_path = visualize_dir / vis_filename
                visualize_sample(
                    sample.image_path,
                    gt_data,
                    prediction,
                    combined_mask,
                    vis_path,
                    f"{class_name} (ALL)",
                    sample.stem,
                    combined_metrics.detach().cpu().numpy().tolist(),
                )
                if args.debug:
                    print(f"\n시각화 저장 (통합): {vis_path}")
            
            # ✅ 픽셀 레벨 분석을 위한 데이터 저장
            all_samples_data.append(SampleData(
                stem=sample.stem,
                gt_depth=gt_data.copy(),
                pred_depth=prediction.copy(),
                mask=combined_mask.copy(),
                class_name=class_name,
            ))
        
        # ✅ 전체 이미지 픽셀 수집 (마스크 없이 GT > 0인 모든 픽셀)
        if sample_has_valid_mask:
            # 전체 이미지를 "ALL" 클래스로 저장
            full_image_mask = (gt_data > 0).astype(np.float32)
            all_samples_data.append(SampleData(
                stem=sample.stem,
                gt_depth=gt_data.copy(),
                pred_depth=prediction.copy(),
                mask=full_image_mask,
                class_name="ALL",
            ))
        
        if sample_has_valid_mask:
            processed_samples += 1

    if not instance_records:
        raise RuntimeError("평가 가능한 객체 마스크가 없습니다.")

    class_metrics: Dict[str, Tuple[List[float], int]] = {}
    for class_name, acc in class_accumulators.items():
        class_metrics[class_name] = (acc.mean(), acc.count())

    overall_metrics = (overall_accumulator.mean(), overall_accumulator.count())
    
    # ✅ 전체 이미지 픽셀 메트릭 계산 (마스크 무시, GT > 0인 모든 픽셀)
    full_image_accumulator = MetricsAccumulator(METRIC_NAMES)
    for sample in samples:
        gt_data = load_depth(str(sample.gt_path)) if sample.gt_path.exists() else None
        if gt_data is None:
            continue
        
        prediction = load_prediction(sample.prediction_path)
        if prediction is None:
            continue
        
        # 전체 이미지에서 GT > 0인 모든 픽셀로 평가
        gt_tensor = torch.tensor(gt_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        pred_tensor = torch.tensor(prediction, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        full_metrics = compute_depth_metrics(eval_namespace, gt_tensor, pred_tensor, use_gt_scale=args.use_gt_scale)
        full_image_accumulator.add(full_metrics)
    
    full_image_metrics = (full_image_accumulator.mean(), full_image_accumulator.count()) if full_image_accumulator.count() > 0 else None

    print_summary_table(class_metrics, overall_metrics, full_image_metrics)
    
    # 거리 범위 정의
    dist_ranges = [
        ("D < 1m", 0.0, 1.0),
        ("1m < D < 2m", 1.0, 2.0),
        ("2m < D < 3m", 2.0, 3.0),
        ("D > 3m", 3.0, args.max_depth)
    ]
    
    # ✅ 클래스별 거리별 평가
    distance_results = {}  # {class_name: [(range_name, pixels, metrics), ...]}
    
    if all_samples_data:
        # 각 클래스별로 거리 평가 수행 (ALL 포함)
        detected_classes = sorted(set(sample.class_name for sample in all_samples_data))
        for class_name in detected_classes:
            results = analyze_by_distance_ranges_pixel_level(all_samples_data, dist_ranges, eval_namespace, class_filter=class_name)
            distance_results[class_name] = results
        
        # car+road 합친 것 (car와 road 샘플만 사용)
        car_road_samples = [s for s in all_samples_data if s.class_name in ['car', 'road']]
        if car_road_samples:
            car_road_results = analyze_by_distance_ranges_pixel_level(car_road_samples, dist_ranges, eval_namespace, class_filter=None)
            distance_results['car+road'] = car_road_results
        
        print_distance_error_distribution(all_samples_data, dist_ranges)

    # 메트릭 저장 - output_dirs['metrics']에 자동 저장
    summary_path = output_dirs['metrics'] / "summary.csv"
    with open(summary_path, "w") as f:
        f.write("Class,Count," + ",".join(METRIC_NAMES) + "\n")
        for class_name, (metrics, count) in sorted(class_metrics.items()):
            metric_str = ",".join(f"{m:.6f}" if not math.isnan(m) else "nan" for m in metrics)
            f.write(f"{class_name},{count},{metric_str}\n")
        metric_str = ",".join(f"{m:.6f}" if not math.isnan(m) else "nan" for m in overall_metrics[0])
        f.write(f"car+road,{overall_metrics[1]},{metric_str}\n")
        if full_image_metrics is not None:
            metric_str = ",".join(f"{m:.6f}" if not math.isnan(m) else "nan" for m in full_image_metrics[0])
            f.write(f"ALL,{full_image_metrics[1]},{metric_str}\n")
    print(f"\n✅ 요약 메트릭 저장: {summary_path}")
    
    # ✅ 거리별 메트릭 저장
    if distance_results:
        distance_path = output_dirs['metrics'] / "summary_by_distance.csv"
        with open(distance_path, "w") as f:
            f.write("Class,Range,Pixels," + ",".join(METRIC_NAMES) + "\n")
            for class_name in sorted(distance_results.keys()):
                for range_name, pixel_count, metrics in distance_results[class_name]:
                    metric_str = ",".join(f"{m:.6f}" if not math.isnan(m) else "nan" for m in metrics)
                    f.write(f"{class_name},{range_name},{pixel_count},{metric_str}\n")
        print(f"✅ 거리별 메트릭 저장: {distance_path}")

    # Per-instance JSON 저장
    json_path = output_dirs['metrics'] / "per_instance.json"
    with open(json_path, "w") as f:
        json.dump({
            "metric_names": METRIC_NAMES,
            "instances": [
                {
                    "stem": item.stem,
                    "class": item.class_name,
                    "mask_path": str(item.mask_path),
                    "valid_pixels": item.valid_pixels,
                    "gt_mean_depth": item.gt_mean_depth,
                    "gt_median_depth": item.gt_median_depth,
                    "metrics": item.metrics,
                }
                for item in instance_records
            ],
        }, f, indent=2)
    print(f"✅ 인스턴스별 메트릭 저장: {json_path}")
    
    # 실행 정보 README 생성
    readme_path = output_dirs['metrics'].parent / "README.txt"
    with open(readme_path, "w") as f:
        f.write(f"Evaluation Results for {checkpoint_id}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.dataset_root}\n")
        f.write(f"Min/Max Depth: {args.min_depth} / {args.max_depth}\n")
        f.write(f"Image Shape: {args.image_shape}\n")
        f.write(f"Flip TTA: {args.flip_tta}\n")
        f.write(f"GT Scale: {args.use_gt_scale}\n")
        f.write(f"Processed Samples: {processed_samples}\n")
        f.write(f"Total Instances: {len(instance_records)}\n")
        f.write(f"\nOutput Structure:\n")
        f.write(f"  - rgb/: RGB images\n")
        f.write(f"  - gt/: Ground truth depth maps\n")
        f.write(f"  - pred/: Predicted depth maps (16-bit PNG)\n")
        f.write(f"  - viz/: Visualization results (4-panel)\n")
        f.write(f"  - metrics/: Evaluation metrics (CSV, JSON)\n")
    print(f"✅ README 생성: {readme_path}")

    if args.output_file:
        out_path = Path(args.output_file)
        ensure_dir(out_path.parent if out_path.parent != Path("") else Path("."))
        with open(out_path, "w") as f:
            f.write("Class,Count," + ",".join(METRIC_NAMES) + "\n")
            for class_name, (metrics, count) in sorted(class_metrics.items()):
                metric_str = ",".join(f"{m:.6f}" if not math.isnan(m) else "nan" for m in metrics)
                f.write(f"{class_name},{count},{metric_str}\n")
            metric_str = ",".join(f"{m:.6f}" if not math.isnan(m) else "nan" for m in overall_metrics[0])
            f.write(f"ALL,{overall_metrics[1]},{metric_str}\n")

    if args.per_instance_json:
        json_path = Path(args.per_instance_json)
        ensure_dir(json_path.parent if json_path.parent != Path("") else Path("."))
        with open(json_path, "w") as f:
            json.dump({
                "metric_names": METRIC_NAMES,
                "instances": [
                    {
                        "stem": item.stem,
                        "class": item.class_name,
                        "mask_path": str(item.mask_path),
                        "valid_pixels": item.valid_pixels,
                        "gt_mean_depth": item.gt_mean_depth,
                        "gt_median_depth": item.gt_median_depth,
                        "metrics": item.metrics,
                    }
                    for item in instance_records
                ],
            }, f, indent=2)


if __name__ == "__main__":
    main()

