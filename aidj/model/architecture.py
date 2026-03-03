from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StemEncoder(nn.Module):
    """CNN encoder for a single stem's spectrogram channels.

    Input: (batch, in_channels, n_mels, n_frames) — prev, next, and optional
           residual spectrograms for one stem
    Output: (batch, hidden_dim, n_frames) — per-frame features
    """

    def __init__(self, in_channels=4, n_mels=128, hidden_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(2, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(2, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # After 3x stride-2 on freq axis: n_mels/8 frequency bins
        freq_bins = n_mels // 8
        self.proj = nn.Conv1d(256 * freq_bins, hidden_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, in_channels, n_mels, T)
        B, C, F, T = x.shape
        h = self.conv(x)  # (B, 256, F//8, T)
        h = h.reshape(B, -1, T)  # (B, 256*F//8, T)
        h = self.proj(h)  # (B, hidden_dim, T)
        return h


class CurveHead(nn.Module):
    """Decoder head that outputs fader + EQ curves for one stem.

    Input: (batch, hidden_dim, n_frames)
    Output: (batch, n_params, n_frames) — 8 curves: fader_prev, fader_next,
            eq_prev_low/mid/high, eq_next_low/mid/high
    """

    def __init__(self, hidden_dim=256, n_params=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, n_params, kernel_size=1),
        )
        self.scale = 2.0  # curves bounded to [0, 2] after sigmoid * scale

    def forward(self, x):
        # x: (B, hidden_dim, T)
        out = self.net(x)  # (B, n_params, T)
        out = torch.sigmoid(out) * self.scale
        return out


class StemTransitionNet(nn.Module):
    """Neural network for predicting per-stem DJ transition curves.

    Architecture:
    1. Per-stem CNN encoders process each stem's spectrogram channels
       (prev, next, prev_residual, next_residual)
    2. Cross-stem Transformer enables stems to coordinate (e.g., bass swap)
    3. Per-stem curve heads predict fader + EQ curves

    Input:  (batch, n_stems*in_channels, n_mels, n_frames) — 4 stems x 4 channels
    Output: (batch, 4, 8, n_frames) — 4 stems x 8 params
    """

    def __init__(self, n_mels=128, n_stems=4, n_params=8, hidden_dim=256,
                 n_transformer_layers=4, n_heads=8, dropout=0.1,
                 in_channels=4):
        super().__init__()
        self.n_stems = n_stems
        self.n_params = n_params
        self.hidden_dim = hidden_dim
        self.n_transformer_layers = n_transformer_layers
        self.in_channels = in_channels

        # Per-stem CNN encoders
        self.stem_encoders = nn.ModuleList([
            StemEncoder(in_channels=in_channels, n_mels=n_mels, hidden_dim=hidden_dim)
            for _ in range(n_stems)
        ])

        # Cross-stem Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_stem_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers,
        )

        # Stem position embeddings (so transformer knows which stem is which)
        self.stem_embeddings = nn.Embedding(n_stems, hidden_dim)

        # Per-stem curve heads
        self.curve_heads = nn.ModuleList([
            CurveHead(hidden_dim, n_params)
            for _ in range(n_stems)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, n_stems*in_channels, n_mels, T) — 4 stems x in_channels

        Returns:
            (B, 4, 8, T) — 4 stems x 8 curve params
        """
        B, _, F, T = x.shape
        C = self.in_channels

        # Split into per-stem groups: each is (B, in_channels, n_mels, T)
        stem_inputs = []
        for s in range(self.n_stems):
            stem_group = x[:, s*C:(s+1)*C, :, :]  # (B, C, F, T)
            stem_inputs.append(stem_group)

        # Encode each stem
        stem_features = []
        for s in range(self.n_stems):
            feat = self.stem_encoders[s](stem_inputs[s])  # (B, hidden, T)
            stem_features.append(feat)

        # Cross-stem attention: reshape for transformer
        # Stack stems along sequence dimension: (B, n_stems * T, hidden)
        stem_embs = self.stem_embeddings.weight  # (n_stems, hidden)

        transformer_input = []
        for s in range(self.n_stems):
            feat = stem_features[s]  # (B, hidden, T)
            feat = feat.transpose(1, 2)  # (B, T, hidden)
            feat = feat + stem_embs[s].unsqueeze(0).unsqueeze(0)  # add stem position
            transformer_input.append(feat)

        transformer_input = torch.cat(transformer_input, dim=1)  # (B, n_stems * T, hidden)

        transformer_output = self.cross_stem_transformer(transformer_input)  # (B, n_stems * T, hidden)

        # Split back into per-stem features
        cross_features = torch.split(transformer_output, T, dim=1)  # list of (B, T, hidden)

        # Predict curves per stem
        outputs = []
        for s in range(self.n_stems):
            feat = cross_features[s].transpose(1, 2)  # (B, hidden, T)
            curves = self.curve_heads[s](feat)  # (B, n_params, T)
            outputs.append(curves)

        # Stack: (B, n_stems, n_params, T)
        output = torch.stack(outputs, dim=1)

        return output

    @classmethod
    def load(cls, path, device=None, **kwargs):
        """Load a trained model from checkpoint."""
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Support both full checkpoint and state_dict only
        if 'model_config' in checkpoint:
            model = cls(**checkpoint['model_config'])
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_state_dict' in checkpoint:
            model = cls(**kwargs)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model = cls(**kwargs)
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        return model
