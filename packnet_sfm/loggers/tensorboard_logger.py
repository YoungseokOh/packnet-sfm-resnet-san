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

    def log_depth(self, mode, batch, output, args, dataset, world_size, config, step, batch_idx=0, image_idx=None):
        # Log depth maps and images sparingly to avoid excessive log file size
        # Select a random index from the batch, or use the provided image_idx
        batch_size = batch['rgb'].shape[0] if 'rgb' in batch else 1
        idx_to_log = image_idx if image_idx is not None else np.random.randint(0, batch_size)

        # Input image (already masked by NcdbDataset)
        if 'rgb' in batch:
            input_image = batch['rgb'][idx_to_log].cpu()
            # Ensure image is float and in [0,1] or [0,255] range for proper normalization
            if input_image.dtype == torch.uint8:
                input_image = input_image.float() / 255.0
            elif input_image.max() > 1.0: # Assuming float images are typically 0-255 if max > 1
                input_image = input_image / 255.0
            self.writer.add_image(f'{mode}/Input Image (Masked)', vutils.make_grid(input_image, normalize=True), global_step=step)

        # Predicted Depth
        if 'viz_pred_inv_depth' in output:
            if isinstance(output['viz_pred_inv_depth'], np.ndarray) and output['viz_pred_inv_depth'].ndim == 4: # (B, H, W, C)
                predicted_depth_viz = torch.from_numpy(output['viz_pred_inv_depth'][idx_to_log]).permute(2, 0, 1)
            else: # (H, W, C) or other single image format
                predicted_depth_viz = torch.from_numpy(output['viz_pred_inv_depth']).permute(2, 0, 1)
            self.writer.add_image(f'{mode}/Predicted Depth', vutils.make_grid(predicted_depth_viz, normalize=False), global_step=step)

        # Ground Truth Depth (if available and already masked by NcdbDataset)
        if 'depth' in batch and batch['depth'] is not None:
            gt_depth = batch['depth'][idx_to_log].cpu()
            # Normalize for visualization, but handle zeros from mask
            # Option 1: Simple normalization (zeros will be min value)
            gt_depth_norm = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-8)
            self.writer.add_image(f'{mode}/Ground Truth Depth (Masked)', vutils.make_grid(gt_depth_norm, normalize=False), global_step=step)

        # Log the raw mask if it's available in the batch
        if 'mask' in batch and batch['mask'] is not None:
            mask_viz = batch['mask'][idx_to_log].cpu()
            # Ensure mask is 1 channel and float for visualization
            if mask_viz.ndim == 2: # (H, W)
                mask_viz = mask_viz.unsqueeze(0) # (1, H, W)
            self.writer.add_image(f'{mode}/Binary Mask (Raw)', vutils.make_grid(mask_viz, normalize=False), global_step=step)

    def finish(self):
        self.writer.close()