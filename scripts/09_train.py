#!/usr/bin/env python3
"""Train the StemTransitionNet model."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aidj.model.architecture import StemTransitionNet
from aidj.model.dataset import StemTransitionDataset
from aidj.model.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=str, default="data/training/train")
    parser.add_argument("--val-dir", type=str, default="data/training/val")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-transformer-layers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    train_dataset = StemTransitionDataset(args.train_dir)
    val_dataset = StemTransitionDataset(args.val_dir)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    from aidj import config
    model = StemTransitionNet(
        hidden_dim=args.hidden_dim,
        n_transformer_layers=args.n_transformer_layers,
        in_channels=config.MODEL_IN_CHANNELS,
    )

    if args.resume:
        import torch
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed from {args.resume}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer.train(n_epochs=args.epochs)
    print(f"\nTraining complete. Best val MAE: {trainer.best_val_mae:.6f}")


if __name__ == "__main__":
    main()
