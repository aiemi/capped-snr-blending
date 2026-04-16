"""
Capped SNR-Adaptive Blending for Neural Speech Enhancement.

Algorithm from:
  "Capped SNR-Adaptive Blending for Neural Speech Enhancement
   in Meeting Transcription" (Liu, 2026)

Core idea: dynamically mix original and enhanced audio using a sigmoid
function of estimated SNR, with a hard cap on maximum enhancement to
prevent over-processing artifacts.
"""

import numpy as np


def estimate_noise_floor(audio: np.ndarray, sr: int = 16000,
                         frame_len: int = 1600,
                         gamma: float = 0.98,
                         eta: float = 2.0,
                         delta: float = 1.001) -> np.ndarray:
    """
    Estimate per-frame noise floor using exponential minimum tracking.

    Parameters
    ----------
    audio : np.ndarray
        Input audio signal (1D, float32).
    sr : int
        Sample rate (default 16000).
    frame_len : int
        Frame length in samples (default 1600 = 100ms at 16kHz).
    gamma : float
        Smoothing factor for noise floor update (default 0.98).
    eta : float
        Threshold multiplier for noise-frame detection (default 2.0).
    delta : float
        Slow upward drift factor for non-stationary noise (default 1.001).

    Returns
    -------
    snr_estimates : np.ndarray
        Per-frame SNR estimates in dB.
    alphas_raw : np.ndarray
        Per-frame raw energy values (for debugging).
    """
    n_frames = len(audio) // frame_len
    snr_estimates = np.zeros(n_frames)
    energies = np.zeros(n_frames)

    # Initialize noise floor from first frame
    first_frame = audio[:frame_len]
    noise_floor = np.mean(first_frame ** 2) + 1e-10
    eps = 1e-10

    for i in range(n_frames):
        frame = audio[i * frame_len:(i + 1) * frame_len]
        energy = np.mean(frame ** 2) + eps
        energies[i] = energy

        # Update noise floor
        if energy < eta * noise_floor:
            noise_floor = gamma * noise_floor + (1 - gamma) * energy
        else:
            noise_floor = delta * noise_floor

        # Estimate SNR
        snr_estimates[i] = 10 * np.log10(energy / (noise_floor + eps))

    return snr_estimates, energies


def compute_alpha(snr_estimates: np.ndarray,
                  snr_ref: float = 20.0,
                  beta: float = 0.08,
                  alpha_max: float = 0.45) -> np.ndarray:
    """
    Compute per-frame capped blending coefficients.

    Parameters
    ----------
    snr_estimates : np.ndarray
        Per-frame SNR estimates in dB.
    snr_ref : float
        Reference SNR at which alpha_raw = 0.5 (default 20 dB).
    beta : float
        Sigmoid steepness (default 0.08, conservative).
    alpha_max : float
        Maximum blending coefficient (default 0.45).

    Returns
    -------
    alphas : np.ndarray
        Per-frame capped blending coefficients in [0, alpha_max].
    """
    # Sigmoid: high SNR -> low alpha (less enhancement)
    alpha_raw = 1.0 / (1.0 + np.exp(-beta * (snr_ref - snr_estimates)))
    # Cap to prevent over-enhancement
    alphas = np.minimum(alpha_raw, alpha_max)
    return alphas


def capped_snr_adaptive_blend(original: np.ndarray,
                               enhanced: np.ndarray,
                               sr: int = 16000,
                               frame_len: int = 1600,
                               snr_ref: float = 20.0,
                               beta: float = 0.08,
                               alpha_max: float = 0.45,
                               **noise_floor_kwargs) -> np.ndarray:
    """
    Apply capped SNR-adaptive blending between original and enhanced audio.

    Parameters
    ----------
    original : np.ndarray
        Original (possibly noisy) audio signal.
    enhanced : np.ndarray
        Neural-enhanced audio signal (e.g., DeepFilterNet output).
    sr : int
        Sample rate.
    frame_len : int
        Frame length in samples.
    snr_ref : float
        Reference SNR for sigmoid midpoint.
    beta : float
        Sigmoid steepness.
    alpha_max : float
        Maximum blending coefficient (the "cap").
    **noise_floor_kwargs
        Additional kwargs for estimate_noise_floor().

    Returns
    -------
    output : np.ndarray
        Blended output signal.
    """
    # Ensure same length
    min_len = min(len(original), len(enhanced))
    original = original[:min_len]
    enhanced = enhanced[:min_len]

    # Estimate SNR and compute alphas
    snr_estimates, _ = estimate_noise_floor(original, sr, frame_len,
                                             **noise_floor_kwargs)
    alphas = compute_alpha(snr_estimates, snr_ref, beta, alpha_max)

    # Apply frame-wise blending
    output = np.copy(original)
    n_frames = len(alphas)

    for i in range(n_frames):
        start = i * frame_len
        end = min(start + frame_len, min_len)
        alpha = alphas[i]
        output[start:end] = (1 - alpha) * original[start:end] + alpha * enhanced[start:end]

    return output.astype(np.float32)


# ============================================================
# Convenience: process a WAV file
# ============================================================
def process_file(input_path: str, output_path: str,
                 enhance_fn=None, **blend_kwargs):
    """
    Process a single WAV file with capped SNR-adaptive blending.

    Parameters
    ----------
    input_path : str
        Path to input WAV file.
    output_path : str
        Path to save output WAV file.
    enhance_fn : callable, optional
        Enhancement function: audio -> enhanced_audio.
        If None, uses DeepFilterNet.
    **blend_kwargs
        Passed to capped_snr_adaptive_blend().
    """
    import soundfile as sf

    audio, sr = sf.read(input_path, dtype='float32')

    if enhance_fn is None:
        # Default: use DeepFilterNet
        from df.enhance import enhance, init_df
        from df.io import load_audio as df_load_audio
        import tempfile, os
        model, state, _ = init_df()
        df_sr = state.sr()
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(tmp.name, audio, sr)
        audio_tensor, _ = df_load_audio(tmp.name, sr=df_sr)
        os.unlink(tmp.name)
        enhanced_tensor = enhance(model, state, audio_tensor)
        enhanced_48k = enhanced_tensor.squeeze().numpy()
        if df_sr != sr:
            from scipy.signal import resample
            enhanced = resample(enhanced_48k, len(audio)).astype(np.float32)
        else:
            enhanced = enhanced_48k.astype(np.float32)
    else:
        enhanced = enhance_fn(audio)

    output = capped_snr_adaptive_blend(audio, enhanced, sr=sr, **blend_kwargs)
    sf.write(output_path, output, sr)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python snr_adaptive_blending.py input.wav output.wav")
        sys.exit(1)
    process_file(sys.argv[1], sys.argv[2])
