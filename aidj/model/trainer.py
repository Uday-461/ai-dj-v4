from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aidj.model.architecture import StemTransitionNet
from aidj.model.dataset import StemTransitionDataset
from aidj.model.losses import TransitionLoss

log = logging.getLogger(__name__)


class Trainer:
    """Training loop for StemTransitionNet."""

    def __init__(
        self,
        model: StemTransitionNet,
        train_dataset: StemTransitionDataset,
        val_dataset: StemTransitionDataset | None = None,
        lr: float = 1e-4,
        batch_size: int = 16,
        mono_weight: float = 0.1,
        smooth_weight: float = 0.05,
        device: str | torch.device | None = None,
        checkpoint_dir: str = "checkpoints",
    ):
        self.device = self._get_device(device)
        self.model = model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=2, pin_memory=True,
            )

        self.criterion = TransitionLoss(mono_weight, smooth_weight)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200,
        )

        self.best_val_mae = float('inf')
        self.epoch = 0

    def _get_device(self, device=None):
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_mae = 0
        n_batches = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(inputs)
            loss, loss_dict = self.criterion(pred, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss_dict['total']
            total_mae += loss_dict['mae']
            n_batches += 1

        self.scheduler.step()

        return {
            'train_loss': total_loss / max(n_batches, 1),
            'train_mae': total_mae / max(n_batches, 1),
            'lr': self.scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0
        total_mae = 0
        n_batches = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            pred = self.model(inputs)
            loss, loss_dict = self.criterion(pred, targets)

            total_loss += loss_dict['total']
            total_mae += loss_dict['mae']
            n_batches += 1

        return {
            'val_loss': total_loss / max(n_batches, 1),
            'val_mae': total_mae / max(n_batches, 1),
        }

    def save_checkpoint(self, filename=None):
        filename = filename or f"checkpoint_epoch{self.epoch:04d}.pt"
        path = self.checkpoint_dir / filename
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_mae': self.best_val_mae,
            'model_config': {
                'n_mels': 128,
                'n_stems': self.model.n_stems,
                'n_params': self.model.n_params,
                'hidden_dim': self.model.hidden_dim,
                'n_transformer_layers': self.model.n_transformer_layers,
                'in_channels': self.model.in_channels,
            },
        }, path)
        log.info(f"Checkpoint saved: {path}")

    def train(self, n_epochs=200, log_interval=1, save_interval=10):
        log.info(f"Training on {self.device} for {n_epochs} epochs")

        for epoch in range(1, n_epochs + 1):
            self.epoch = epoch

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            metrics = {**train_metrics, **val_metrics}

            if epoch % log_interval == 0:
                parts = [f"Epoch {epoch}/{n_epochs}"]
                for k, v in metrics.items():
                    parts.append(f"{k}={v:.6f}")
                log.info(" | ".join(parts))

            # Save best model
            val_mae = val_metrics.get('val_mae', train_metrics['train_mae'])
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self.save_checkpoint("best_model.pt")

            # Periodic checkpoint
            if epoch % save_interval == 0:
                self.save_checkpoint()
