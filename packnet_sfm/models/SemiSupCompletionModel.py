# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import torch
import torch.nn.functional as F  # ❗ 추가: F.interpolate를 사용하기 위해 임포트
from torchvision.utils import save_image
from packnet_sfm.models.SelfSupModel import SfmModel, SelfSupModel
from packnet_sfm.losses.supervised_loss import SupervisedLoss
from packnet_sfm.models.model_utils import merge_outputs
from packnet_sfm.utils.depth import depth2inv, inv2depth
# ❗ YOLOv8SAN01 모델을 임포트하여 타입 체크에 사용
from packnet_sfm.networks.depth.YOLOv8SAN01 import YOLOv8SAN01
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class SemiSupCompletionModel(SelfSupModel):
    """
    Semi-Supervised model for depth prediction and completion.

    Parameters
    ----------
    supervised_loss_weight : float
        Weight for the supervised loss
    kwargs : dict
        Extra parameters
    """
    def __init__(self, supervised_loss_weight=0.9, weight_rgbd=1.0,
                 consistency_loss_weight=0.0,
                 min_depth=0.5, max_depth=80.0,  # ← YAML에서 넘어오는 값 그대로 이름 유지
                 use_log_space=False,  # 🆕 Log space interpolation option
                 **kwargs):
        # Initializes SelfSupModel
        super().__init__(**kwargs)
        # If supervision weight is 0.0, use SelfSupModel directly
        assert 0. < supervised_loss_weight <= 1., "Model requires (0, 1] supervision"
        # Store weight and initializes supervised loss
        self.supervised_loss_weight = supervised_loss_weight
        # ✅ 먼저 YAML에서 전달된 min/max를 정규화/저장
        if max_depth <= 0: max_depth = 80.0
        if min_depth <= 0: min_depth = 0.5
        if max_depth <= min_depth: max_depth = min_depth + 1.0
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        # 🆕 Log space option
        self.use_log_space = use_log_space
        
        # 🆕 Print transformation mode
        transform_mode = "LOG SPACE" if use_log_space else "LINEAR SPACE"
        print(f"🔧 SemiSupCompletionModel: Using {transform_mode} interpolation")
        print(f"   Depth range: [{self.min_depth}, {self.max_depth}]m")

        # SupervisedLoss에 YAML min/max depth를 명시적으로 전달
        self._supervised_loss = SupervisedLoss(
            min_depth=self.min_depth, max_depth=self.max_depth, **kwargs)
        # ❗ 일관성 손실 가중치 저장
        self.consistency_loss_weight = consistency_loss_weight

        # Pose network is only required if there is self-supervision
        if self.supervised_loss_weight == 1:
            self._network_requirements.remove('pose_net')
        # GT depth is only required if there is supervision
        if self.supervised_loss_weight > 0:
            self._train_requirements.append('gt_depth')

        self._input_keys = ['rgb', 'input_depth', 'intrinsics']

        self.weight_rgbd = weight_rgbd

        self._one_step_viz_done = False  # ✅ 1회 저장 플래그

    @property
    def logs(self):
        """Return logs."""
        return {
            **super().logs,
            **self._supervised_loss.logs
        }

    def supervised_loss(self, inv_depths, gt_inv_depths,
                        return_logs=False, progress=0.0):
        """
        Calculates the supervised loss.
        
        🆕 Dual-Head 모델 자동 감지 및 처리

        Parameters
        ----------
        inv_depths : torch.Tensor [B,1,H,W] or dict
            Predicted inverse depth maps from the original image
            OR Dual-Head outputs dict with ("integer", 0) and ("fractional", 0)
        gt_inv_depths : torch.Tensor [B,1,H,W]
            Ground-truth inverse depth maps from the original image
        return_logs : bool
            True if logs are stored
        progress : float
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar and "metrics" dictionary
        """
        # ========================================
        # 🆕 Dual-Head 모델 자동 감지
        # ========================================
        if hasattr(self, 'depth_net') and hasattr(self.depth_net, 'is_dual_head') and self.depth_net.is_dual_head:
            # Dual-Head 모델인 경우
            # Dual-Head loss 초기화 (첫 호출 시)
            if not hasattr(self, '_dual_head_loss'):
                from packnet_sfm.losses.dual_head_depth_loss import DualHeadDepthLoss
                self._dual_head_loss = DualHeadDepthLoss(
                    max_depth=self.max_depth,
                    min_depth=self.min_depth,
                    integer_weight=1.0,
                    fractional_weight=10.0,
                    consistency_weight=0.5
                )
            
            # inv_depths는 이미 decoder의 dict 출력 (("integer", 0), ("fractional", 0))
            # gt_inv_depths는 inverse depth이므로 depth로 변환
            gt_depth = inv2depth(gt_inv_depths)
            
            # Dual-Head loss 계산
            return self._dual_head_loss(
                inv_depths,  # dict with ("integer", 0) and ("fractional", 0)
                gt_depth,
                return_logs=return_logs,
                progress=progress
            )
        else:
            # Standard Single-Head 모델
            return self._supervised_loss(
                inv_depths, gt_inv_depths,
                return_logs=return_logs, progress=progress)

    def _save_one_step_viz(self, batch, pred_inv_list, tag="rgb"):
        if self._one_step_viz_done:
            return
        save_dir = os.environ.get("ONE_STEP_VIZ_DIR", "one_step_viz")
        os.makedirs(save_dir, exist_ok=True)

        # 첫 배치 첫 샘플만
        rgb = batch['rgb'][0:1]  # (1,3,H,W)
        gt_depth = batch.get('depth', None)
        pred_inv = pred_inv_list[0][0:1]  # scale0 (1,1,H,W)
        pred_depth = inv2depth(pred_inv)

        # 정규화 함수
        def norm01(x, mask=None):
            x = x.clone()
            if mask is not None:
                valid = mask & torch.isfinite(x)
            else:
                valid = torch.isfinite(x)
            if valid.sum() < 10:
                return torch.zeros_like(x)
            v = x[valid]
            mn, mx = v.min(), v.max()
            if (mx - mn) < 1e-9:
                return torch.zeros_like(x)
            x[~valid] = mn
            x = (x - mn) / (mx - mn + 1e-12)
            return x.clamp(0,1)

        # RGB 저장
        save_image(rgb, f"{save_dir}/step0_rgb.png")

        # 예측 inv-depth 저장(그대로 0~1 가정, 혹시 범위 벗어나면 정규화)
        inv_viz = norm01(pred_inv)
        save_image(inv_viz, f"{save_dir}/step0_pred_inv.png")

        # 예측 depth 저장 (log 분포 대비 단순 min-max)
        depth_viz = norm01(pred_depth)
        save_image(depth_viz, f"{save_dir}/step0_pred_depth.png")

        if gt_depth is not None:
            gt_depth_ = gt_depth[0:1]
            gt_depth_viz = norm01(gt_depth_)
            save_image(gt_depth_viz, f"{save_dir}/step0_gt_depth.png")
            gt_inv = depth2inv(gt_depth_)
            gt_inv_viz = norm01(gt_inv)
            save_image(gt_inv_viz, f"{save_dir}/step0_gt_inv.png")

        print(f"[ONE_STEP_VIZ] Saved to {save_dir}")
        self._one_step_viz_done = True

    def _save_loss_inv_debug(self, pred_inv_list, gt_inv_full):
        """
        1회(또는 매 step) supervised loss 입력 분포 시각화 + 히스토그램 저장.
        ENV:
          LOSS_INV_VIZ_ONCE=1  -> 처음 1회만
          LOSS_INV_VIZ_EVERY=1 -> 모든 step (디버그용)
          LOSS_INV_VIZ_DIR=dir
        """
        every = os.environ.get("LOSS_INV_VIZ_EVERY","0") == "1"
        once  = os.environ.get("LOSS_INV_VIZ_ONCE","0") == "1"
        if not (every or once):
            return
        if once:
            os.environ["LOSS_INV_VIZ_ONCE"] = "0"

        save_dir = os.environ.get("LOSS_INV_VIZ_DIR", "loss_inv_viz")
        os.makedirs(save_dir, exist_ok=True)

        pred_inv = pred_inv_list[0].detach()          # scale 0
        gt_inv   = gt_inv_full.detach()

        # 해상도 맞추기
        if pred_inv.shape[-2:] != gt_inv.shape[-2:]:
            gt_inv = torch.nn.functional.interpolate(
                gt_inv, size=pred_inv.shape[-2:], mode='nearest'
            )

        p0 = pred_inv[0:1]   # (1,1,H,W)
        g0 = gt_inv[0:1]
        abs_diff = (p0 - g0).abs()

        def tensor_stats(x):
            v = x[x.isfinite()]
            if v.numel() == 0:
                return {}
            q = torch.quantile(v, torch.tensor([0.0,0.01,0.05,0.5,0.95,0.99,1.0], device=v.device))
            return {
                "min": float(q[0]), "p1": float(q[1]), "p5": float(q[2]),
                "median": float(q[3]), "p95": float(q[4]), "p99": float(q[5]),
                "max": float(q[6]),
                "mean": float(v.mean()), "std": float(v.std()),
                "numel": int(v.numel())
            }

        # 히스토그램 도우미
        def save_hist(x, name, bins=80):
            v = x[x.isfinite()].flatten().cpu()
            if v.numel() == 0:
                return
            vmin = float(v.min())
            vmax = float(v.max())
            if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax - vmin < 1e-12:
                return
            # torch.histc 사용 (낮은 버전 호환)
            counts = torch.histc(v, bins=bins, min=vmin, max=vmax)
            edges = torch.linspace(vmin, vmax, steps=bins + 1)
            # PNG
            plt.figure(figsize=(4,3), dpi=120)
            width = (edges[1]-edges[0]).item()
            plt.bar(edges[:-1].numpy(), counts.numpy(), width=width, align='edge')
            plt.title(name)
            plt.tight_layout()
            out_png = f"{save_dir}/step0_{name}_hist.png"
            plt.savefig(out_png)
            plt.close()
            # JSON
            hist_data = {
                "edges": edges.tolist(),
                "counts": counts.tolist(),
                "min": vmin,
                "max": vmax
            }
            with open(f"{save_dir}/step0_{name}_hist.json","w") as f:
                json.dump(hist_data, f)

        save_hist(p0, "pred_inv")
        save_hist(g0, "gt_inv")
        save_hist(abs_diff, "abs_diff")

        st_pred = tensor_stats(p0)
        st_gt   = tensor_stats(g0)
        st_abs  = tensor_stats(abs_diff)

        sat_hi = float((p0 > 0.99).float().mean())
        sat_lo = float((p0 < 0.01).float().mean())
        # 현재 구조 상 pred max=1.0 → overflow 후보는 gt_inv > 1.0
        overflow = float((g0 > 1.0).float().mean())

        summary = {
            "pred_inv": st_pred,
            "gt_inv": st_gt,
            "abs_diff": st_abs,
            "frac_pred>0.99": sat_hi,
            "frac_pred<0.01": sat_lo,
            "frac_gt_inv>1.0": overflow
        }
        with open(f"{save_dir}/step0_stats.json","w") as f:
            json.dump(summary, f, indent=2)

        print("[LOSS_INV_VIZ] pred_inv:", st_pred)
        print("[LOSS_INV_VIZ] gt_inv  :", st_gt)
        print("[LOSS_INV_VIZ] abs_diff:", st_abs)
        print(f"[LOSS_INV_VIZ] sat_hi={sat_hi:.3f} sat_lo={sat_lo:.3f} overflow(gt_inv>1)={overflow:.3f}")
        print(f"[LOSS_INV_VIZ] Saved hist & stats to {save_dir}")

    def _debug_gt_depth(self, depth_tensor: torch.Tensor):
        every = os.environ.get("GT_DEPTH_DEBUG_EVERY","0") == "1"
        once  = os.environ.get("GT_DEPTH_DEBUG_ONCE","0") == "1"
        if not (every or once):
            return
        if once:
            os.environ["GT_DEPTH_DEBUG_ONCE"] = "0"

        save_dir = os.environ.get("GT_DEPTH_DEBUG_DIR", "gt_depth_debug")
        os.makedirs(save_dir, exist_ok=True)

        d = depth_tensor.detach()
        # 기본 마스크(>0)
        valid = (d > 0) & torch.isfinite(d)
        v = d[valid]

        stats = {}
        if v.numel() > 0:
            # torch.quantile fallback 처리
            if hasattr(torch, "quantile"):
                qs = torch.tensor([0.0,0.01,0.05,0.5,0.95,0.99,1.0], device=v.device)
                qv = torch.quantile(v, qs)
                qmap = dict(zip(
                    ["min","p1","p5","median","p95","p99","max"],
                    [float(x) for x in qv]))
            else:
                # 간단 fallback (근사)
                sv, idx = torch.sort(v)
                def qpick(p):
                    k = min(int((sv.numel()-1)*p), sv.numel()-1)
                    return float(sv[k])
                qmap = {
                    "min": qpick(0.0),
                    "p1": qpick(0.01),
                    "p5": qpick(0.05),
                    "median": qpick(0.5),
                    "p95": qpick(0.95),
                    "p99": qpick(0.99),
                    "max": qpick(1.0)
                }
            stats.update(qmap)
            stats["mean"] = float(v.mean())
            stats["std"]  = float(v.std())
        else:
            stats.update(dict(min=None,p1=None,p5=None,median=None,p95=None,p99=None,max=None,mean=None,std=None))

        total = int(d.numel())
        stats["numel_total"] = total
        stats["numel_valid"] = int(valid.sum())
        stats["numel_zero_or_neg"] = int((d <= 0).sum())
        for th in [0.01,0.02,0.05,0.1]:
            stats[f"frac_depth<{th}"] = float(((d>0)&(d<th)).float().mean())

        # 가장 작은 depth 샘플
        smallest_samples = []
        if v.numel() > 0:
            k = min(10, v.numel())
            # topk on -v (가장 작은)
            flat = d.view(-1)
            # 유효 위치 인덱스
            flat_valid = valid.view(-1)
            valid_indices = torch.nonzero(flat_valid, as_tuple=False).view(-1)
            valid_vals = flat[valid_indices]
            if valid_vals.numel() > 0:
                vals, order = torch.sort(valid_vals)  # 오름차순
                sel_idx = valid_indices[order[:k]]
                H, W = d.shape[-2], d.shape[-1]
                for idx_int, val in zip(sel_idx.tolist(), vals[:k].tolist()):
                    y = idx_int // W
                    x = idx_int % W
                    smallest_samples.append({"y": int(y), "x": int(x), "depth": float(val)})
        stats["smallest_samples"] = smallest_samples

        # depth2inv 검증 (0 처리 방식 체킹)
        from packnet_sfm.utils.depth import depth2inv
        inv = depth2inv(d)
        inv_valid = inv[valid]
        stats["inv_numel_valid"] = int(inv_valid.numel())
        stats["inv_max"] = float(inv_valid.max()) if inv_valid.numel() else None
        stats["inv_min"] = float(inv_valid.min()) if inv_valid.numel() else None
        stats["inv_frac>2"] = float((inv_valid > 2.0).float().mean()) if inv_valid.numel() else 0.0
        stats["inv_frac>10"] = float((inv_valid > 10.0).float().mean()) if inv_valid.numel() else 0.0
        stats["inv_frac_inf_or_nan"] = float((~torch.isfinite(inv_valid)).float().mean()) if inv_valid.numel() else 0.0

        # 히스토그램 저장 (valid depth 대상)
        if v.numel() > 0:
            try:
                bins = int(os.environ.get("GT_DEPTH_HIST_BINS","80"))
                vcpu = v.cpu()
                vmin = float(vcpu.min())
                vmax = float(vcpu.max())
                if vmax - vmin > 1e-12:
                    counts = torch.histc(vcpu, bins=bins, min=vmin, max=vmax)
                    edges = torch.linspace(vmin, vmax, steps=bins+1)
                    # PNG
                    plt.figure(figsize=(4,3), dpi=120)
                    width = (edges[1]-edges[0]).item()
                    plt.bar(edges[:-1].numpy(), counts.numpy(), width=width, align='edge')
                    plt.title("gt_depth(m)")
                    plt.tight_layout()
                    plt.savefig(f"{save_dir}/step0_gt_depth_hist.png")
                    plt.close()
                    # JSON
                    hist = {
                        "edges": edges.tolist(),
                        "counts": counts.tolist(),
                        "min": vmin,
                        "max": vmax
                    }
                    with open(f"{save_dir}/step0_gt_depth_hist.json","w") as f:
                        json.dump(hist, f, indent=2)
            except Exception as e:
                print("[GT_DEPTH_DEBUG][HIST_ERROR]", e)

        # JSON 저장
        with open(f"{save_dir}/step0_gt_depth_stats.json","w") as f:
            json.dump(stats, f, indent=2)

        # 콘솔 요약
        print("[GT_DEPTH_DEBUG] depth stats:",
              " ".join([f"{k}={stats[k]:.4g}" for k in
                        ["min","p1","p5","median","p95","p99","max","mean","std"]
                        if stats[k] is not None]))
        print(f"[GT_DEPTH_DEBUG] small_depth_fracs:",
              f"<0.01={stats['frac_depth<0.01']:.4f}",
              f"<0.02={stats['frac_depth<0.02']:.4f}",
              f"<0.05={stats['frac_depth<0.05']:.4f}",
              f"<0.1={stats['frac_depth<0.1']:.4f}")
        print(f"[GT_DEPTH_DEBUG] zero_or_neg={stats['numel_zero_or_neg']} / {stats['numel_total']} "
              f"valid={stats['numel_valid']}")
        if smallest_samples:
            print("[GT_DEPTH_DEBUG] smallest_samples:",
                  ", ".join([f"(y={s['y']},x={s['x']},d={s['depth']:.4f})" for s in smallest_samples]))
        print(f"[GT_DEPTH_DEBUG] inv_max={stats['inv_max']} inv_frac>2={stats['inv_frac>2']:.4f} "
              f"inv_frac>10={stats['inv_frac>10']:.4f} inf_or_nan={stats['inv_frac_inf_or_nan']:.4f}")
        print(f"[GT_DEPTH_DEBUG] saved JSON to {save_dir}")

    def forward(self, batch, return_logs=False, progress=0.0, **kwargs):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar and different metrics and predictions
            for logging and downstream usage.
        """
        if not self.training:
            # If not training, no need for self-supervised loss
            return SfmModel.forward(self, batch, return_logs=return_logs, **kwargs)
        else:
            if self.supervised_loss_weight == 1.:
                # If no self-supervision, no need to calculate loss
                self_sup_output = SfmModel.forward(self, batch, return_logs=return_logs, **kwargs)
                loss = torch.tensor([0.]).type_as(batch['rgb'])
            else:
                # Otherwise, calculate and weight self-supervised loss
                self_sup_output = SelfSupModel.forward(
                    self, batch, return_logs=return_logs, progress=progress, **kwargs)
                loss = (1.0 - self.supervised_loss_weight) * self_sup_output['loss']

            # ✅ GT depth 통계/히스토그램 (A~D 검증) – supervised_loss 계산 직전에 호출
            try:
                if 'depth' in batch:
                    self._debug_gt_depth(batch['depth'])
            except Exception as e:
                print("[GT_DEPTH_DEBUG][ERROR]", e)

            # ================== 핵심 추가 부분 (호출식은 유지) ==================
            # depth2inv(batch['depth']) 호출 전에 batch['depth']를 범위 클램프
            if 'depth' in batch and batch['depth'] is not None:
                d = batch['depth']
                if d.dim() == 3:  # [B,H,W] -> [B,1,H,W]
                    d = d.unsqueeze(1)
                valid = (d > 0) & torch.isfinite(d)
                if valid.any():
                    d_clamped = d.clone()
                    # min_depth / max_depth 는 self.min_depth / self.max_depth (이미 __init__ 저장)
                    d_clamped[valid] = d_clamped[valid].clamp(self.min_depth, self.max_depth)
                    batch['depth'] = d_clamped
                else:
                    batch['depth'] = d  # 그대로 유지
            # ====================================================================

            # ✅ CRITICAL FIX: Convert sigmoid to bounded inverse depth
            # Model outputs sigmoid [0, 1], but we need to convert it to
            # inverse depth [1/max_depth, 1/min_depth] to match evaluation!
            # 🆕 Now supports both LINEAR and LOG space interpolation!
            from packnet_sfm.utils.post_process_depth import sigmoid_to_inv_depth
            
            # ========================================
            # Handle both Single-Head and Dual-Head outputs
            # ========================================
            if 'inv_depths' in self_sup_output:
                # Single-Head: outputs have 'inv_depths' key
                sigmoid_outputs = self_sup_output['inv_depths']
                bounded_inv_depths = [
                    sigmoid_to_inv_depth(sig, self.min_depth, self.max_depth, use_log_space=self.use_log_space)
                    for sig in sigmoid_outputs
                ]
                # Pass to supervised_loss (which handles Single-Head)
                sup_output = self.supervised_loss(
                    bounded_inv_depths, depth2inv(batch['depth']),
                    return_logs=return_logs, progress=progress)
            else:
                # Dual-Head: outputs have ('integer', i) and ('fractional', i) keys
                # Pass the original dict directly to supervised_loss 
                # (which will detect Dual-Head and use DualHeadDepthLoss)
                sup_output = self.supervised_loss(
                    self_sup_output,  # Pass original dict with tuple keys
                    depth2inv(batch['depth']),
                    return_logs=return_logs, progress=progress)

            # Debug visualization (only for Single-Head, Dual-Head doesn't need this)
            if 'inv_depths' in self_sup_output:
                try:
                    self._save_loss_inv_debug(bounded_inv_depths, depth2inv(batch['depth']))
                except Exception as e:
                    print("[LOSS_INV_VIZ][ERROR]", e)

            loss += self.supervised_loss_weight * sup_output['loss']

            if 'inv_depths_rgbd' in self_sup_output:
                # Also convert RGBD outputs
                sigmoid_rgbd = self_sup_output['inv_depths_rgbd']
                bounded_inv_rgbd = [
                    sigmoid_to_inv_depth(sig, self.min_depth, self.max_depth, use_log_space=self.use_log_space)
                    for sig in sigmoid_rgbd
                ]
                sup_output2 = self.supervised_loss(
                    bounded_inv_rgbd, depth2inv(batch['depth']),
                    return_logs=return_logs, progress=progress)
                loss += self.weight_rgbd * self.supervised_loss_weight * sup_output2['loss']
                if 'depth_loss' in self_sup_output:
                    loss += self_sup_output['depth_loss']

                # YOLOv8 consistency (원래 코드 유지)
                if self.training and isinstance(self.depth_net, YOLOv8SAN01) and self.consistency_loss_weight > 0:
                    pred_rgb = self_sup_output['inv_depths']
                    pred_rgbd = self_sup_output['inv_depths_rgbd']
                    consistency_loss = 0.0
                    num_scales = min(len(pred_rgb), len(pred_rgbd))
                    if num_scales > 0:
                        for i in range(num_scales):
                            pr = pred_rgb[i]
                            prd = pred_rgbd[i]
                            if pr.shape[-2:] != prd.shape[-2:]:
                                pr = F.interpolate(pr, size=prd.shape[-2:], mode='bilinear', align_corners=False)
                            consistency_loss += torch.abs(pr - prd.detach()).mean()
                        consistency_loss /= num_scales
                        loss += self.consistency_loss_weight * consistency_loss
                        if return_logs:
                            self_sup_output['metrics']['consistency_loss'] = consistency_loss

            # try:
            #     self._save_one_step_viz(batch, self_sup_output['inv_depths'])
            # except Exception as e:
            #     print("[ONE_STEP_VIZ][ERROR]", e)

            # Merge and return outputs
            return {
                'loss': loss,
                **merge_outputs(self_sup_output, sup_output),
            }
