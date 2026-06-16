"""
Model Definitions for DAC-GAN Adversarial Attack
=================================================
Contains:
  - StrongGenerator: Latent-space perturbation generator
  - Cnn14: PANNs audio classifier (Kong et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


# =====================================================================
# Utility Initialisers
# =====================================================================

def init_layer(layer):
    """Xavier-uniform init for Linear / Conv layers."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Standard BatchNorm init."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


# =====================================================================
# PANNs Cnn14 Classifier
# =====================================================================

class ConvBlock(nn.Module):
    """Double-conv block used in PANNs Cnn14."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x = F.avg_pool2d(x, kernel_size=pool_size) \
                + F.max_pool2d(x, kernel_size=pool_size)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")
        return x


class Cnn14(nn.Module):
    """
    PANNs Cnn14 audio classifier (Kong et al., TASLP 2020).

    Accepts raw waveforms and internally computes log-mel spectrograms.
    The final layer outputs raw logits (no sigmoid/softmax).
    """

    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320,
                 mel_bins=64, fmin=50, fmax=14000, classes_num=527):
        super().__init__()

        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window='hann', center=True,
            pad_mode='reflect', freeze_parameters=True
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size, n_mels=mel_bins,
            fmin=fmin, fmax=fmax, ref=1.0, amin=1e-10, top_db=None,
            freeze_parameters=True
        )
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2
        )
        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Args:
            input: (batch_size, data_length) raw waveform
        Returns:
            logits: (batch_size, classes_num) raw logits
            embedding: (batch_size, 2048) feature embedding
        """
        x = self.spectrogram_extractor(input)   # (B, 1, T, F)
        x = self.logmel_extractor(x)            # (B, 1, T, mel_bins)

        x = x.transpose(1, 3)                  # (B, mel_bins, T, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)                  # (B, 1, T, mel_bins)

        if self.training:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')

        x = torch.mean(x, dim=3)        # (B, 2048, T')
        (x1, _) = torch.max(x, dim=2)   # (B, 2048)
        x2 = torch.mean(x, dim=2)       # (B, 2048)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)

        logits = self.fc_audioset(x)
        return logits, embedding


# =====================================================================
# Adversarial Perturbation Generator
# =====================================================================

class StrongGenerator(nn.Module):
    """
    Latent-space perturbation generator for DAC-GAN adversarial attack.

    Takes clean DAC latent codes z_clean ∈ ℝ^{B×D×T} and target class
    labels, and produces a perturbation δ such that decoding z_clean + δ
    yields audio classified as the target class.

    Architecture:
        z_clean + ClassEmb(target) → FC(D → 2D) → ReLU
        → Conv1d(2D, 1024, k=5) → BN → ReLU
        → Conv1d(1024, 1024, k=3) → BN → ReLU
        → Conv1d(1024, 512, k=3) → BN → ReLU
        → Conv1d(512, 256, k=3) → BN → ReLU
        → Conv1d(256, D, k=1)   [zero-init]
        → clamp(±δ_max)

    The final conv is zero-initialised so δ starts at zero (no audio
    distortion), and the generator gradually learns minimal perturbations.
    """

    def __init__(self, latent_dim, num_classes, delta_clamp=2.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.delta_clamp = delta_clamp

        # Target class conditioning
        self.class_embedding = nn.Embedding(num_classes, latent_dim)
        nn.init.normal_(self.class_embedding.weight, std=0.05)

        # Dimension expansion
        self.fc = nn.Linear(latent_dim, latent_dim * 2)

        # Convolutional backbone
        self.conv1 = nn.Conv1d(latent_dim * 2, 1024, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(1024)

        self.conv2 = nn.Conv1d(1024, 1024, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(1024)

        self.conv3 = nn.Conv1d(1024, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(512)

        self.conv4 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)

        # Output projection (zero-init → starts with δ ≈ 0)
        self.conv5 = nn.Conv1d(256, latent_dim, kernel_size=1)

        # Initialisation
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.conv5.weight)
        nn.init.zeros_(self.conv5.bias)

    def forward(self, z_clean, target_labels):
        """
        Args:
            z_clean:       (B, D, T) clean DAC latent codes
            target_labels: (B,) integer target class indices

        Returns:
            delta: (B, D, T) clamped latent perturbation
        """
        B, D, T = z_clean.shape

        # Condition on target class
        class_emb = self.class_embedding(target_labels)        # (B, D)
        class_emb = class_emb.unsqueeze(2).expand(-1, -1, T)   # (B, D, T)
        x = z_clean + class_emb

        # FC expansion: (B, D, T) → (B, 2D, T)
        x = x.permute(0, 2, 1)    # (B, T, D)
        x = self.fc(x)             # (B, T, 2D)
        x = x.permute(0, 2, 1)    # (B, 2D, T)
        x = F.relu(x)

        # Conv blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Output projection (no skip — delta must be learned from scratch)
        delta = self.conv5(x)

        # Clamp perturbation magnitude
        delta = torch.clamp(delta, -self.delta_clamp, self.delta_clamp)

        return delta
