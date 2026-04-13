#!/usr/bin/env python3
"""
Evaluate adaptive noise reduction performance.

Computes RMS-based environment classification, SNR before/after, and
SNR improvement. Uses only NumPy for portability.

Usage:
    python evaluate_nr.py --before noisy.wav --after cleaned.wav

See README.md for methodology details and reference to ITU-T P.862 (PESQ).
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Environment classification thresholds (dBFS)
# ---------------------------------------------------------------------------

ENVIRONMENT_THRESHOLDS = [
    (-50.0, "quiet_room"),
    (-40.0, "home_office"),
    (-30.0, "open_office"),
    (-20.0, "cafe"),
    (  0.0, "outdoor"),       # RMS >= -20 dBFS
]

SUPPRESSION_DB = {
    "quiet_room":   6,
    "home_office": 12,
    "open_office": 18,
    "cafe":        24,
    "outdoor":     30,
}


# ---------------------------------------------------------------------------
# WAV I/O (minimal, 16-bit PCM only)
# ---------------------------------------------------------------------------

def read_wav(path: str) -> tuple:
    """
    Read a 16-bit PCM WAV file.

    Returns:
        (samples, sample_rate) where samples is a float64 numpy array
        normalized to [-1.0, 1.0].
    """
    with open(path, "rb") as f:
        riff = f.read(4)
        if riff != b"RIFF":
            raise ValueError(f"Not a WAV file: {path}")
        f.read(4)  # file size
        wave = f.read(4)
        if wave != b"WAVE":
            raise ValueError(f"Not a WAV file: {path}")

        sample_rate = 16000
        num_channels = 1
        bits_per_sample = 16
        data_bytes = b""

        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack("<I", f.read(4))[0]

            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                audio_format = struct.unpack("<H", fmt_data[0:2])[0]
                if audio_format != 1:
                    raise ValueError("Only PCM format is supported")
                num_channels = struct.unpack("<H", fmt_data[2:4])[0]
                sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
                bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
            elif chunk_id == b"data":
                data_bytes = f.read(chunk_size)
            else:
                f.read(chunk_size)

    if not data_bytes:
        raise ValueError(f"No audio data found in {path}")

    if bits_per_sample != 16:
        raise ValueError(f"Only 16-bit PCM is supported, got {bits_per_sample}-bit")

    samples = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float64)
    samples /= 32768.0

    if num_channels > 1:
        samples = samples[::num_channels]

    return samples, sample_rate


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rms_dbfs(signal: np.ndarray) -> float:
    """
    Compute RMS level in dBFS.

    Input signal is assumed normalized to [-1.0, 1.0].
    """
    rms = np.sqrt(np.mean(signal ** 2) + 1e-10)
    return 20.0 * np.log10(rms + 1e-10)


def classify_environment(rms_dbfs: float) -> str:
    """
    Classify the acoustic environment based on RMS level.

    Thresholds:
        < -50 dBFS  -> quiet_room
        -50 to -40  -> home_office
        -40 to -30  -> open_office
        -30 to -20  -> cafe
        >= -20      -> outdoor
    """
    for threshold, label in ENVIRONMENT_THRESHOLDS:
        if rms_dbfs < threshold:
            return label
    return "outdoor"


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio in dB."""
    sig_power = np.mean(signal ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    return 10.0 * np.log10(sig_power / noise_power)


def compute_segmental_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    frame_length: int = 512,
    hop_length: int = 256,
) -> float:
    """
    Compute segmental SNR (average of per-frame SNR values).

    More perceptually relevant than global SNR because it weights
    quiet and loud frames equally.
    """
    n_frames = (len(signal) - frame_length) // hop_length + 1
    if n_frames <= 0:
        return compute_snr(signal, noise)

    frame_snrs = []
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        s_frame = signal[start:end]
        n_frame = noise[start:end]

        s_power = np.mean(s_frame ** 2) + 1e-10
        n_power = np.mean(n_frame ** 2) + 1e-10
        frame_snr = 10.0 * np.log10(s_power / n_power)

        # Clamp to [-10, 35] dB to avoid outlier frames
        frame_snr = max(-10.0, min(35.0, frame_snr))
        frame_snrs.append(frame_snr)

    return float(np.mean(frame_snrs))


def estimate_noise(signal: np.ndarray, percentile: int = 10) -> np.ndarray:
    """
    Rough noise floor estimation based on low-energy frames.

    Segments the signal into short frames and uses the lowest-energy
    frames as a noise proxy. This is a simplified estimator for
    evaluation purposes.
    """
    frame_len = 512
    hop = 256
    n_frames = (len(signal) - frame_len) // hop + 1

    if n_frames <= 0:
        return signal

    energies = []
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + frame_len]
        energies.append(np.mean(frame ** 2))

    threshold = np.percentile(energies, percentile)
    noise_frames = []
    for i in range(n_frames):
        if energies[i] <= threshold:
            start = i * hop
            noise_frames.append(signal[start:start + frame_len])

    if noise_frames:
        # Build a noise signal by repeating low-energy frames
        noise_template = np.concatenate(noise_frames)
        repeats = len(signal) // len(noise_template) + 1
        noise_signal = np.tile(noise_template, repeats)[:len(signal)]
    else:
        noise_signal = np.zeros_like(signal)

    return noise_signal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate noise reduction: RMS classification, SNR improvement."
    )
    parser.add_argument("--before", required=True, help="Noisy input WAV")
    parser.add_argument("--after", required=True, help="NR-processed WAV")
    parser.add_argument("--clean", default=None, help="Clean reference WAV (optional, improves SNR accuracy)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Expected sample rate (Hz)")
    parser.add_argument("--output", default=None, help="Output JSON path for results")
    args = parser.parse_args()

    # Load audio
    print(f"Loading noisy input: {args.before}")
    noisy, sr_noisy = read_wav(args.before)
    print(f"Loading NR output:   {args.after}")
    cleaned, sr_cleaned = read_wav(args.after)

    if sr_noisy != args.sample_rate or sr_cleaned != args.sample_rate:
        print(f"WARNING: Sample rate mismatch. Expected {args.sample_rate}, "
              f"got before={sr_noisy}, after={sr_cleaned}")

    # Ensure equal length
    min_len = min(len(noisy), len(cleaned))
    noisy = noisy[:min_len]
    cleaned = cleaned[:min_len]

    # RMS and environment classification
    rms_before = compute_rms_dbfs(noisy)
    rms_after = compute_rms_dbfs(cleaned)
    env_class = classify_environment(rms_before)
    expected_suppression = SUPPRESSION_DB.get(env_class, 0)

    results = {
        "noisy_file": args.before,
        "cleaned_file": args.after,
        "sample_rate": args.sample_rate,
        "duration_seconds": round(min_len / args.sample_rate, 2),
        "rms_before_dbfs": round(rms_before, 2),
        "rms_after_dbfs": round(rms_after, 2),
        "rms_reduction_db": round(rms_before - rms_after, 2),
        "classified_environment": env_class,
        "expected_suppression_db": expected_suppression,
    }

    # SNR computation
    if args.clean:
        print(f"Loading clean ref:   {args.clean}")
        clean, _ = read_wav(args.clean)
        clean = clean[:min_len]

        noise_before = noisy - clean
        noise_after = cleaned - clean

        snr_before = compute_segmental_snr(clean, noise_before)
        snr_after = compute_segmental_snr(clean, noise_after)
        snr_improvement = snr_after - snr_before

        results["snr_before_db"] = round(snr_before, 2)
        results["snr_after_db"] = round(snr_after, 2)
        results["snr_improvement_db"] = round(snr_improvement, 2)
    else:
        # Without clean reference, estimate noise from low-energy frames
        noise_est_before = estimate_noise(noisy)
        noise_est_after = estimate_noise(cleaned)

        snr_before = compute_snr(noisy, noise_est_before)
        snr_after = compute_snr(cleaned, noise_est_after)

        results["snr_before_db_estimated"] = round(snr_before, 2)
        results["snr_after_db_estimated"] = round(snr_after, 2)
        results["snr_improvement_db_estimated"] = round(snr_after - snr_before, 2)
        results["note"] = "SNR estimated without clean reference; provide --clean for accurate measurement."

    # Report
    print("\n--- Noise Reduction Evaluation Results ---")
    for key, value in results.items():
        print(f"  {key}: {value}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
