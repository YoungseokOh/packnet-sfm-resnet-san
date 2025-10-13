#!/usr/bin/env python3
"""NCDB object-masked depth evaluation.

segmentation_results/class_masks 에서 Mask2Former 인스턴스 마스크를 읽어 해당 객체
영역에 한정해 깊이 예측 품질을 정량 평가합니다. 예측 깊이는 지정한 체크포인트로
image_a6 이미지를 즉시 추론하거나, 기존 캐시를 재활용할 수 있습니다.
"""

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def normalize_entry(args: argparse.Namespace, dataset_root: Path, entry: dict) -> SampleEntry:
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

    image_path_raw = entry.get("image_path")
    if image_path_raw:
        image_path = Path(image_path_raw)
        if not image_path.is_absolute():
            image_path = sequence_root / image_path
    else:
        image_path = resolve_path(sequence_root, args.image_subdir) / f"{stem}.png"

    segmentation_root = resolve_path(sequence_root, args.segmentation_root)
    class_mask_root = segmentation_root / args.class_mask_subdir if args.class_mask_subdir else segmentation_root

    gt_candidates = [resolve_path(sequence_root, args.gt_root) / f"{stem}.png",
                     resolve_path(sequence_root, args.gt_root) / f"{stem}.npz"]
    gt_path = next((p for p in gt_candidates if p.exists()), gt_candidates[0])

    pred_root = resolve_path(sequence_root, args.pred_root)
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


def print_summary_table(class_metrics: Dict[str, Tuple[List[float], int]], overall: Tuple[List[float], int]) -> None:
    header = ["Class", "Count"] + METRIC_NAMES
    rows = []
    for class_name, (metrics, count) in sorted(class_metrics.items()):
        rows.append([class_name, str(count)] + [f"{m:.4f}" if not math.isnan(m) else "nan" for m in metrics])
    rows.append(["ALL", str(overall[1])] + [f"{m:.4f}" if not math.isnan(m) else "nan" for m in overall[0]])

    col_widths = [max(len(row[i]) for row in rows + [header]) for i in range(len(header))]

    def print_row(row: Sequence[str]) -> None:
        print("  ".join(word.ljust(col_widths[i]) for i, word in enumerate(row)))

    print("\n평가 요약 (객체 마스크 기준)")
    print_row(header)
    print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in rows:
        print_row(row)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    split_paths = discover_split_files(args)
    raw_entries = load_split_entries(args, split_paths)
    samples = [normalize_entry(args, dataset_root, entry) for entry in raw_entries]

    if args.debug:
        print(f"총 {len(samples)}개 샘플 로드")

    model_context = prepare_model(args)
    eval_namespace = make_eval_namespace(args)

    detected_classes: Optional[List[str]] = args.classes if args.classes else None
    instance_records: List[InstanceResult] = []
    class_accumulators: Dict[str, MetricsAccumulator] = defaultdict(lambda: MetricsAccumulator(METRIC_NAMES))
    overall_accumulator = MetricsAccumulator(METRIC_NAMES)

    reference_shape: Optional[Tuple[int, int]] = None

    for sample in tqdm(samples, desc="Evaluating"):
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

        for class_name, mask_paths in mask_groups.items():
            for mask_path in mask_paths:
                mask = load_mask(mask_path, gt_data.shape)
                valid_mask = (mask > 0) & (gt_data > 0)
                if not np.any(valid_mask):
                    if args.debug:
                        print(f"유효 픽셀이 없어 스킵: {mask_path}")
                    continue

                gt_masked = torch.tensor(gt_data * mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                metrics = compute_depth_metrics(eval_namespace, gt_masked, pred_tensor, use_gt_scale=args.use_gt_scale)

                class_accumulators[class_name].add(metrics)
                overall_accumulator.add(metrics)

                instance_records.append(InstanceResult(
                    stem=sample.stem,
                    class_name=class_name,
                    mask_path=mask_path,
                    valid_pixels=int(valid_mask.sum()),
                    metrics=metrics.detach().cpu().numpy().tolist(),
                ))

    if not instance_records:
        raise RuntimeError("평가 가능한 객체 마스크가 없습니다.")

    class_metrics: Dict[str, Tuple[List[float], int]] = {}
    for class_name, acc in class_accumulators.items():
        class_metrics[class_name] = (acc.mean(), acc.count())

    overall_metrics = (overall_accumulator.mean(), overall_accumulator.count())

    print_summary_table(class_metrics, overall_metrics)

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
                        "metrics": item.metrics,
                    }
                    for item in instance_records
                ],
            }, f, indent=2)


if __name__ == "__main__":
    main()

