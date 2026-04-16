"""
Noise floor estimation using exponential minimum tracking.

Implements Eq. (2) from:
  "When Does Neural Speech Enhancement Help Meeting Transcription?
   An Empirical Study" (Liu, 2026)

The noise floor N(t) is tracked as an exponentially weighted minimum:
    N(t) = gamma * N(t-1) + (1-gamma) * E(t)   if E(t) < eta * N(t-1)
    N(t) = delta * N(t-1)                       otherwise

where gamma=0.98, eta=2.0, delta=1.001 by default.
"""

import numpy as np
from typing import Tuple


def estimate_noise_floor(
    audio: np.ndarray,
    sr: int = 16000,
    frame_len: int = 1600,
    gamma: float = 0.98,
    eta: float = 2.0,
    delta: float = 1.001,
    eps: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate per-frame SNR using exponential minimum noise floor tracking.

    Parameters
    ----------
    audio : np.ndarray
        Input audio signal (1D, float32).
    sr : int
        Sample rate. Default 16000.
    frame_len : int
        Frame length in samples. Default 1600 (100ms at 16kHz).
    gamma : float
        Smoothing factor for noise floor update. Default 0.98.
    eta : float
        Threshold multiplier for noise-frame detection. Default 2.0.
        If E(t) < eta * N(t-1), the frame is treated as noise-only.
    delta : float
        Slow upward drift factor for non-stationary noise. Default 1.001.
    eps : float
        Numerical stability constant. Default 1e-10.

    Returns
    -------
    snr_estimates : np.ndarray
        Per-frame SNR estimates in dB, shape (n_frames,).
    noise_floors : np.ndarray
        Per-frame noise floor power, shape (n_frames,).
    """
    n_frames = len(audio) // frame_len
    snr_estimates = np.zeros(n_frames, dtype=np.float32)
    noise_floors = np.zeros(n_frames, dtype=np.float32)

    # Initialize noise floor from first frame
    first_frame = audio[:frame_len]
    noise_floor = float(np.mean(first_frame ** 2) + eps)

    for i in range(n_frames):
        frame = audio[i * frame_len:(i + 1) * frame_len]
        energy = float(np.mean(frame ** 2) + eps)

        # Update noise floor: decrease quickly, increase slowly
        if energy < eta * noise_floor:
            noise_floor = gamma * noise_floor + (1 - gamma) * energy
        else:
            noise_floor = delta * noise_floor

        noise_floors[i] = noise_floor
        snr_estimates[i] = 10.0 * np.log10(energy / (noise_floor + eps))

    return snr_estimates, noise_floors


class NoiseFloorTracker:
    """Stateful noise floor tracker for streaming/online processing.

    Example
    -------
    >>> tracker = NoiseFloorTracker(sr=16000, frame_len=1600)
    >>> for frame in audio_stream:
    ...     snr_db = tracker.update(frame)
    ...     print(f"Estimated SNR: {snr_db:.1f} dB")
    """

    def __init__(
        self,
        sr: int = 16000,
        frame_len: int = 1600,
        gamma: float = 0.98,
        eta: float = 2.0,
        delta: float = 1.001,
        eps: float = 1e-10,
    ):
        self.sr = sr
        self.frame_len = frame_len
        self.gamma = gamma
        self.eta = eta
        self.delta = delta
        self.eps = eps
        self.noise_floor = None

    def update(self, frame: np.ndarray) -> float:
        """Process one frame, return current SNR estimate in dB."""
        energy = float(np.mean(frame ** 2) + self.eps)

        if self.noise_floor is None:
            self.noise_floor = energy

        if energy < self.eta * self.noise_floor:
            self.noise_floor = self.gamma * self.noise_floor + (1 - self.gamma) * energy
        else:
            self.noise_floor *= self.delta

        return 10.0 * np.log10(energy / (self.noise_floor + self.eps))

    def reset(self):
        """Reset tracker state."""
        self.noise_floor = None


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        import soundfile as sf
        audio, sr = sf.read(sys.argv[1], dtype='float32')
        snr, noise = estimate_noise_floor(audio, sr=sr)
        print(f"Mean SNR estimate: {np.mean(snr):.2f} dB")
        print(f"SNR range: [{np.min(snr):.2f}, {np.max(snr):.2f}] dB")
    else:
        # Synthetic test
        np.random.seed(0)
        fs = 16000
        t = np.linspace(0, 5, 5*fs)
        signal = np.sin(2*np.pi*440*t) * 0.3 * (1 + 0.5*np.sin(2*np.pi*3*t))
        noise = np.random.randn(len(t)) * 0.01
        audio = (signal + noise).astype(np.float32)
        snr, _ = estimate_noise_floor(audio)
        print(f"Synthetic test: mean SNR = {np.mean(snr):.2f} dB")
