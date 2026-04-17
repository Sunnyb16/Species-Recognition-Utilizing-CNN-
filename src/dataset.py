import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
import random

from src.config import SR, DURATION


class BirdChunkDataset(Dataset):
    def __init__(
        self,
        samples,
        augment=False,
        aug_params=None,
        sr=SR,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        fmin=20,
        fmax=14000,
    ):
        self.samples = samples
        self.augment = augment
        self.aug_params = aug_params or {}

        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio chunk
        y, _ = librosa.load(sample["chunk_path"], sr=self.sr, mono=True)

        # Augmentation
        if self.augment:
            y = augment_audio(
                y,
                sr=self.sr,
                duration=DURATION,
                **self.aug_params
            )

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        log_mel = log_mel.astype(np.float32)

        spec = torch.from_numpy(log_mel).unsqueeze(0)
        target = torch.tensor(sample["target"], dtype=torch.float32)

        return spec, target


def augment_audio(
    y,
    sr,
    duration=5,
    pitch_prob=0.4,
    pitch_range=(-1.5, 1.5),
    stretch_prob=0.4,
    stretch_range=(0.9, 1.1),
    shift_prob=0.5,
    shift_max_sec=1.0,
    noise_prob=0.5,
    noise_std=0.005,
):
    target_len = sr * duration

    if random.random() < pitch_prob:
        y = librosa.effects.pitch_shift(
            y,
            sr=sr,
            n_steps=random.uniform(*pitch_range)
        )

    if random.random() < stretch_prob:
        y = librosa.effects.time_stretch(
            y,
            rate=random.uniform(*stretch_range)
        )

    if random.random() < shift_prob:
        max_shift = int(shift_max_sec * sr)
        shift = int(random.uniform(-max_shift, max_shift))
        y = np.roll(y, shift)
        if shift > 0:
            y[:shift] = 0
        else:
            y[shift:] = 0

    if random.random() < noise_prob:
        noise = np.random.randn(len(y)) * noise_std
        y = y + noise

    y = pad_crop_audio(y, target_len)
    return y


def pad_crop_audio(y, target_len):
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        start = np.random.randint(0, len(y) - target_len + 1)
        y = y[start:start + target_len]
    return y