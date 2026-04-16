"""
Baseline speech enhancement methods for comparison.

Methods:
  - Spectral subtraction (Boll, 1979)
  - Wiener filter (Scalart & Filho, 1996)
  - Simulated RNNoise (spectral gating)
  - Fixed-threshold blending (5-class discrete alpha)
"""

import numpy as np
from scipy.signal import stft, istft


def spectral_subtraction(audio: np.ndarray, sr: int = 16000,
                         oversubtract: float = 2.0) -> np.ndarray:
    """Classical spectral subtraction with oversubtraction."""
    f, t_arr, Zxx = stft(audio, fs=sr, nperseg=512, noverlap=384)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    noise_est = np.mean(mag[:, :10], axis=1, keepdims=True)
    enhanced_mag = np.maximum(mag - oversubtract * noise_est, 0.05 * mag)
    enhanced = enhanced_mag * np.exp(1j * phase)
    _, result = istft(enhanced, fs=sr, nperseg=512, noverlap=384)
    return result[:len(audio)].astype(np.float32)


def wiener_filter(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Decision-directed Wiener filter."""
    f, t_arr, Zxx = stft(audio, fs=sr, nperseg=512, noverlap=384)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    noise_est = np.mean(mag[:, :10] ** 2, axis=1, keepdims=True)
    signal_est = np.maximum(mag ** 2 - noise_est, 0)
    gain = signal_est / (signal_est + noise_est + 1e-8)
    gain = np.maximum(gain, 0.1)
    enhanced_mag = mag * gain
    enhanced = enhanced_mag * np.exp(1j * phase)
    _, result = istft(enhanced, fs=sr, nperseg=512, noverlap=384)
    return result[:len(audio)].astype(np.float32)


def rnnoise_simulated(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Simulated RNNoise using spectral gating with noise floor estimation."""
    f, t_arr, Zxx = stft(audio, fs=sr, nperseg=512, noverlap=384)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    noise_est = np.mean(mag[:, :10], axis=1, keepdims=True)
    gain = np.maximum(1.0 - 2.0 * noise_est / (mag + 1e-8), 0.1)
    gain = np.minimum(gain, 1.0)
    enhanced_mag = mag * gain
    enhanced = enhanced_mag * np.exp(1j * phase)
    _, result = istft(enhanced, fs=sr, nperseg=512, noverlap=384)
    return result[:len(audio)].astype(np.float32)


# Fixed-threshold alpha mapping (5-class)
FIXED_THRESHOLDS = {
    (0, 5): 0.45,
    (5, 10): 0.35,
    (10, 15): 0.25,
    (15, 20): 0.15,
    (20, float('inf')): 0.05,
}


def fixed_threshold_blend(original: np.ndarray, enhanced: np.ndarray,
                          snr_db: float) -> np.ndarray:
    """Apply fixed-threshold blending based on approximate SNR class."""
    alpha = 0.25  # default
    for (low, high), a in FIXED_THRESHOLDS.items():
        if low <= snr_db < high:
            alpha = a
            break
    output = (1 - alpha) * original + alpha * enhanced
    return output.astype(np.float32)
