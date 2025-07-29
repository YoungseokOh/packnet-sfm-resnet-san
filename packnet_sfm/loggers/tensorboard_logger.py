import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import numpy as np

class TensorboardLogger:
    def __init__(self, log_dir, **kwargs):
        self.log_directory = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.config = kwargs.get('config', {}) # Pass config as a kwarg if needed

    def log_metrics(self, metrics, step):
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, dict):
                self.writer.add_scalars(metric_name, metric_value, global_step=step)
            else:
                self.writer.add_scalar(metric_name, metric_value, global_step=step)

    def log_depth(self, mode, batch, output, args, dataset, world_size, config, step, batch_idx=0):
        # Log depth maps and images sparingly to avoid excessive log file size
        # Select a random index from the batch
        batch_size = batch['rgb'].shape[0] if 'rgb' in batch else 1
        idx_to_log = np.random.randint(0, batch_size)

        # Input image
        if 'rgb' in batch:
            input_image = batch['rgb'][idx_to_log].cpu()
            self.writer.add_image(f'{mode}/Input Image', vutils.make_grid(input_image, normalize=True), global_step=step)

        # Predicted Depth
        if 'viz_pred_inv_depth' in output:
            # Assuming viz_pred_inv_depth is already a numpy array of shape (H, W, C)
            # If it's a batch, we need to select the correct one.
            # For now, assuming it's already processed for a single image or needs batching.
            # If output['viz_pred_inv_depth'] is a batch, it needs to be indexed.
            # Let's assume it's already a single image for simplicity, or needs to be handled if it's a batch.
            if isinstance(output['viz_pred_inv_depth'], np.ndarray) and output['viz_pred_inv_depth'].ndim == 4: # (B, H, W, C)
                predicted_depth_viz = torch.from_numpy(output['viz_pred_inv_depth'][idx_to_log]).permute(2, 0, 1)
            else: # (H, W, C) or other single image format
                predicted_depth_viz = torch.from_numpy(output['viz_pred_inv_depth']).permute(2, 0, 1)
            self.writer.add_image(f'{mode}/Predicted Depth', vutils.make_grid(predicted_depth_viz, normalize=False), global_step=step)

        # Ground Truth Depth (if available)
        if 'depth' in batch and batch['depth'] is not None:
            gt_depth = batch['depth'][idx_to_log].cpu()
            # Normalize for visualization
            gt_depth_norm = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-8)
            self.writer.add_image(f'{mode}/Ground Truth Depth', vutils.make_grid(gt_depth_norm, normalize=False), global_step=step)

    def finish(self):
        self.writer.close()
