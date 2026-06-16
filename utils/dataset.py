"""
UrbanSound8K Dataset Loader for DAC-GAN
========================================
Loads audio at the DAC-native 16 kHz sample rate with fixed 4-second duration.
"""

import os
import warnings
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from config.config import AUDIO_DIR, DAC_SR, DURATION


class UrbanSound8kDataset(Dataset):
    """
    UrbanSound8K dataset loader.

    Loads audio files from the fold-based directory structure, resamples
    to the target sample rate, converts stereo to mono, and pads/truncates
    to a fixed duration.

    Args:
        df:        Pandas DataFrame with UrbanSound8K metadata
                   (columns: fold, slice_file_name, classID)
        target_sr: Target sample rate in Hz (default: 16000)
        duration:  Target duration in seconds (default: 4)
    """

    def __init__(self, df, target_sr=DAC_SR, duration=DURATION):
        self.df = df.reset_index(drop=True)
        self.target_sr = target_sr
        self.num_samples = target_sr * duration

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(
            AUDIO_DIR, f"fold{row['fold']}", row['slice_file_name']
        )

        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.target_sr:
                waveform = torchaudio.functional.resample(
                    waveform, sr, self.target_sr
                )
            waveform = waveform.mean(dim=0)  # stereo → mono
        except Exception as e:
            warnings.warn(f"Failed to load {audio_path}: {e}")
            waveform = torch.zeros(self.num_samples)

        # Pad or truncate to fixed length
        if len(waveform) < self.num_samples:
            waveform = F.pad(waveform, (0, self.num_samples - len(waveform)))
        else:
            waveform = waveform[:self.num_samples]

        return waveform.float(), row['classID']
