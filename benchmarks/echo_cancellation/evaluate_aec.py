#!/usr/bin/env python3
"""
Evaluate Acoustic Echo Cancellation (AEC) performance.

Computes Echo Return Loss Enhancement (ERLE) and SNR improvement by comparing
a recorded microphone signal (containing echo) against the far-end loopback
reference. Uses only NumPy for portability.

Usage:
    python evaluate_aec.py --input recorded.wav --reference loopback.wav

Metrics:
    ERLE (dB)          Higher is better. Measures echo suppression.
    SNR improvement     Increase in signal-to-noise ratio post-AEC.

See README.md for methodology details and reference to ITU-T P.862 (PESQ).
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# WAV I/O (minimal, 16-bit PCM only — avoids scipy dependency)
# ---------------------------------------------------------------------------

def read_wav(path: str) -> tuple:
    """
    Read a 16-bit PCM WAV file.

    Returns:
        (samples, sample_rate) where samples is a float64 numpy array
        normalized to [-1.0, 1.0].
    """
    with open(path, "rb") as f:
        # RIFF header
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

        # Parse chunks
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
    samples /= 32768.0  # normalize to [-1, 1]

    # If stereo, take first channel
    if num_channels > 1:
        samples = samples[::num_channels]

    return samples, sample_rate


# ---------------------------------------------------------------------------
# AEC: Frequency-Domain Block NLMS Adaptive Filter
# ---------------------------------------------------------------------------

class FrequencyDomainAEC:
    """
    Partitioned-block frequency-domain NLMS adaptive filter for echo
    cancellation. Matches the configuration described in Paper Section 4.1.
    """

    def __init__(
        self,
        block_size: int = 1600,
        filter_blocks: int = 8,
        step_size: float = 0.3,
    ):
        self.block_size = block_size
        self.filter_blocks = filter_blocks
        self.fft_size = 2 * block_size
        self.step_size = step_size

        # Filter coefficients in frequency domain (one per partition)
        self.W = np.zeros((filter_blocks, self.fft_size // 2 + 1), dtype=np.complex128)

        # Reference signal buffer (stores last filter_blocks blocks)
        self.ref_buffer = np.zeros((filter_blocks, self.fft_size // 2 + 1), dtype=np.complex128)

        # Previous input block for overlap-save
        self.prev_input = np.zeros(block_size)
        self.prev_ref = np.zeros(block_size)

    def process_block(self, mic_block: np.ndarray, ref_block: np.ndarray) -> np.ndarray:
        """
        Process one block of audio through the adaptive filter.

        Args:
            mic_block: Microphone signal block (block_size samples, contains echo).
            ref_block: Far-end reference block (block_size samples).

        Returns:
            Echo-cancelled output block (block_size samples).
        """
        # Construct overlap-save input frames
        mic_frame = np.concatenate([self.prev_input, mic_block])
        ref_frame = np.concatenate([self.prev_ref, ref_block])

        # FFT of reference
        Ref = np.fft.rfft(ref_frame)

        # Shift reference buffer (FIFO)
        self.ref_buffer = np.roll(self.ref_buffer, 1, axis=0)
        self.ref_buffer[0] = Ref

        # Compute echo estimate: sum of W[k] * Ref[k] for each partition
        Y = np.zeros(self.fft_size // 2 + 1, dtype=np.complex128)
        for k in range(self.filter_blocks):
            Y += self.W[k] * self.ref_buffer[k]

        y_time = np.fft.irfft(Y)
        echo_estimate = y_time[self.block_size:]  # overlap-save: take second half

        # Error signal
        error = mic_block - echo_estimate

        # Compute normalization factor (power of reference)
        ref_power = np.sum(np.abs(self.ref_buffer) ** 2) + 1e-10

        # NLMS update in frequency domain
        Error_frame = np.concatenate([np.zeros(self.block_size), error])
        E = np.fft.rfft(Error_frame)

        for k in range(self.filter_blocks):
            self.W[k] += (self.step_size / ref_power) * E * np.conj(self.ref_buffer[k])

        # Save state
        self.prev_input = mic_block.copy()
        self.prev_ref = ref_block.copy()

        return error


def run_aec(mic_signal: np.ndarray, ref_signal: np.ndarray, block_size: int = 1600) -> np.ndarray:
    """
    Run AEC on full-length signals.

    Args:
        mic_signal: Recorded microphone signal (with echo).
        ref_signal: Far-end loopback signal.
        block_size: Processing block size in samples.

    Returns:
        Echo-cancelled signal.
    """
    # Ensure equal length
    min_len = min(len(mic_signal), len(ref_signal))
    mic_signal = mic_signal[:min_len]
    ref_signal = ref_signal[:min_len]

    aec = FrequencyDomainAEC(block_size=block_size)
    output = np.zeros(min_len)

    num_blocks = min_len // block_size
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        output[start:end] = aec.process_block(
            mic_signal[start:end],
            ref_signal[start:end],
        )

    return output[:num_blocks * block_size]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_erle(echo_signal: np.ndarray, residual_signal: np.ndarray) -> float:
    """
    Compute Echo Return Loss Enhancement in dB.

    ERLE = 10 * log10( mean(echo^2) / mean(residual^2) )
    """
    echo_power = np.mean(echo_signal ** 2) + 1e-10
    residual_power = np.mean(residual_signal ** 2) + 1e-10
    return 10.0 * np.log10(echo_power / residual_power)


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio in dB."""
    sig_power = np.mean(signal ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    return 10.0 * np.log10(sig_power / noise_power)


def compute_snr_improvement(
    mic_signal: np.ndarray,
    output_signal: np.ndarray,
    clean_signal: np.ndarray,
) -> float:
    """
    Compute SNR improvement after AEC.

    If a clean reference is available, SNR is computed as
    power(clean) / power(signal - clean). Improvement is the
    difference between output SNR and input SNR.
    """
    snr_before = compute_snr(clean_signal, mic_signal - clean_signal)
    snr_after = compute_snr(clean_signal, output_signal - clean_signal)
    return snr_after - snr_before


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AEC performance: compute ERLE and SNR improvement."
    )
    parser.add_argument("--input", required=True, help="Recorded microphone WAV (with echo)")
    parser.add_argument("--reference", required=True, help="Far-end loopback WAV")
    parser.add_argument("--clean", default=None, help="Clean speech reference WAV (for SNR improvement)")
    parser.add_argument("--output", default=None, help="Output JSON path for results")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Expected sample rate (Hz)")
    parser.add_argument("--block-size", type=int, default=1600, help="AEC block size (samples)")
    args = parser.parse_args()

    # Load audio
    print(f"Loading input:     {args.input}")
    mic_signal, sr_mic = read_wav(args.input)
    print(f"Loading reference: {args.reference}")
    ref_signal, sr_ref = read_wav(args.reference)

    if sr_mic != args.sample_rate or sr_ref != args.sample_rate:
        print(f"WARNING: Sample rate mismatch. Expected {args.sample_rate}, "
              f"got input={sr_mic}, reference={sr_ref}")

    # Run AEC
    print(f"Running AEC (block_size={args.block_size})...")
    output_signal = run_aec(mic_signal, ref_signal, block_size=args.block_size)

    # Truncate all signals to output length
    n = len(output_signal)
    mic_signal = mic_signal[:n]
    ref_signal = ref_signal[:n]

    # Compute ERLE (echo = ref component in mic, residual = ref component in output)
    erle = compute_erle(ref_signal, output_signal)

    results = {
        "input_file": args.input,
        "reference_file": args.reference,
        "block_size": args.block_size,
        "sample_rate": args.sample_rate,
        "duration_seconds": round(n / args.sample_rate, 2),
        "erle_db": round(erle, 2),
    }

    # SNR improvement (requires clean reference)
    if args.clean:
        print(f"Loading clean:     {args.clean}")
        clean_signal, _ = read_wav(args.clean)
        clean_signal = clean_signal[:n]
        snr_imp = compute_snr_improvement(mic_signal, output_signal, clean_signal)
        results["snr_improvement_db"] = round(snr_imp, 2)

    # Report
    print("\n--- AEC Evaluation Results ---")
    for key, value in results.items():
        print(f"  {key}: {value}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
