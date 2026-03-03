from __future__ import annotations

import torch
import torch.nn as nn


class TransitionLoss(nn.Module):
    """Combined loss for transition curve prediction.

    Components:
    - MAE: primary reconstruction loss
    - Monotonicity penalty: fader_prev should decrease, fader_next should increase
    - Smoothness: penalizes jittery curves (L1 on 2nd derivative)
    """

    def __init__(self, mono_weight=0.1, smooth_weight=0.05):
        super().__init__()
        self.mono_weight = mono_weight
        self.smooth_weight = smooth_weight

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 4, 8, T) predicted curves
            target: (B, 4, 8, T) ground truth curves

        Returns:
            total_loss, loss_dict
        """
        # MAE loss
        mae = torch.mean(torch.abs(pred - target))

        # Monotonicity penalty
        mono_loss = self._monotonicity_loss(pred)

        # Smoothness loss
        smooth_loss = self._smoothness_loss(pred)

        total = mae + self.mono_weight * mono_loss + self.smooth_weight * smooth_loss

        loss_dict = {
            'total': total.item(),
            'mae': mae.item(),
            'mono': mono_loss.item(),
            'smooth': smooth_loss.item(),
        }

        return total, loss_dict

    def _monotonicity_loss(self, pred):
        """Penalize non-monotonic fader curves.

        For each stem (dim 1):
        - param index 0 = fader_prev: should decrease (diff <= 0)
        - param index 1 = fader_next: should increase (diff >= 0)
        """
        fader_prev = pred[:, :, 0, :]  # (B, 4, T)
        fader_next = pred[:, :, 1, :]  # (B, 4, T)

        # Penalize increases in fader_prev
        diff_prev = torch.diff(fader_prev, dim=-1)
        mono_prev = torch.mean(torch.relu(diff_prev))

        # Penalize decreases in fader_next
        diff_next = torch.diff(fader_next, dim=-1)
        mono_next = torch.mean(torch.relu(-diff_next))

        return mono_prev + mono_next

    def _smoothness_loss(self, pred):
        """L1 penalty on second derivative (jitter)."""
        # Second derivative: diff of diff
        first_diff = torch.diff(pred, dim=-1)
        second_diff = torch.diff(first_diff, dim=-1)
        return torch.mean(torch.abs(second_diff))
