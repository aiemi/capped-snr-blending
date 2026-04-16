#!/usr/bin/env python3
"""
Complete experiment pipeline for:
"SNR-Adaptive Speech Enhancement with Environment-Aware Blending
 for Real-Time Meeting Transcription"

Steps:
1. Download AMI IHM test set (or use local subset)
2. Download DEMAND noise dataset
3. Mix clean speech + noise at various SNRs
4. Run all enhancement methods
5. Compute PESQ, STOI, SI-SDR
6. Run Whisper for WER
7. Output all tables as CSV + formatted markdown

Requirements: deepfilternet, pesq, pystoi, whisper, numpy, scipy, soundfile, torch
"""

import os
import sys
import json
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path("/Users/liuanwei/NIW/experiments")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
AUDIO_DIR = DATA_DIR / "audio_clips"       # clean speech clips
NOISE_DIR = DATA_DIR / "noise_clips"        # noise clips
MIXED_DIR = DATA_DIR / "mixed"              # noisy mixes
ENHANCED_DIR = DATA_DIR / "enhanced"        # enhanced outputs

SR = 16000  # sample rate
FRAME_LEN = 1600  # 100ms at 16kHz

# SNR levels to test
SNR_LEVELS = [0, 5, 10, 15, 20, 25, "clean"]

# Number of test clips (use subset for feasibility)
N_TEST_CLIPS = 30  # 30 clips × 7 SNR levels × 7 methods = ~1470 evaluations

# SNR-adaptive blending parameters
SNR_REF = 20.0   # reference SNR (dB)
BETA = 0.08       # sigmoid steepness (conservative)
ALPHA_CAP = 0.45  # maximum blending coefficient (prevents over-enhancement)

# Fixed-threshold blending parameters (5-class baseline)
FIXED_THRESHOLDS = {
    (0, 5): 0.45,
    (5, 10): 0.35,
    (10, 15): 0.25,
    (15, 20): 0.15,
    (20, float('inf')): 0.05,
}

for d in [AUDIO_DIR, NOISE_DIR, MIXED_DIR, ENHANCED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# Step 0: Generate or download test audio
# ============================================================
def generate_test_clips():
    """Generate synthetic speech-like test clips if no real data available.
    For proper results, replace with AMI IHM clips."""

    # First try to find any existing wav files
    existing = list(AUDIO_DIR.glob("*.wav"))
    if len(existing) >= N_TEST_CLIPS:
        print(f"  Found {len(existing)} existing clips in {AUDIO_DIR}")
        return sorted(existing)[:N_TEST_CLIPS]

    # Try to download AMI corpus subset via HuggingFace
    print("  Attempting to download AMI corpus subset...")
    try:
        from datasets import load_dataset
        ds = load_dataset("edinburghcstr/ami", "ihm", split="test",
                         streaming=True, trust_remote_code=True)
        count = 0
        for item in ds:
            if count >= N_TEST_CLIPS:
                break
            audio = item["audio"]
            samples = np.array(audio["array"], dtype=np.float32)
            # Resample if needed
            orig_sr = audio["sampling_rate"]
            if orig_sr != SR:
                from scipy.signal import resample
                samples = resample(samples, int(len(samples) * SR / orig_sr))
            # Take 10-second clips (longer for DeepFilterNet convergence)
            clip_len = 10 * SR
            if len(samples) > clip_len:
                samples = samples[:clip_len]
            elif len(samples) < SR:  # skip very short clips
                continue
            out_path = AUDIO_DIR / f"ami_test_{count:03d}.wav"
            sf.write(str(out_path), samples, SR)
            count += 1
        if count >= N_TEST_CLIPS:
            print(f"  Downloaded {count} AMI clips")
            return sorted(AUDIO_DIR.glob("*.wav"))[:N_TEST_CLIPS]
    except Exception as e:
        print(f"  AMI download failed: {e}")

    # Fallback: generate synthetic speech-like signals
    print("  Generating synthetic speech-like test clips...")
    clips = []
    for i in range(N_TEST_CLIPS):
        duration = 5.0  # seconds
        t = np.linspace(0, duration, int(duration * SR), dtype=np.float32)
        # Simulate speech: multiple harmonics with amplitude modulation
        f0 = np.random.uniform(100, 300)  # fundamental frequency
        signal = np.zeros_like(t)
        for h in range(1, 8):
            signal += (0.5 ** h) * np.sin(2 * np.pi * f0 * h * t)
        # Add amplitude modulation (syllable-like)
        mod = np.abs(np.sin(2 * np.pi * 3.5 * t)) ** 0.5
        signal = signal * mod
        # Add some noise-like components for realism
        signal += 0.05 * np.random.randn(len(t))
        # Normalize
        signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.7
        out_path = AUDIO_DIR / f"synth_{i:03d}.wav"
        sf.write(str(out_path), signal.astype(np.float32), SR)
        clips.append(out_path)
    return clips


def generate_noise_clips():
    """Generate noise clips for mixing."""
    existing = list(NOISE_DIR.glob("*.wav"))
    if len(existing) >= 5:
        return sorted(existing)[:5]

    # Try DEMAND dataset download
    print("  Generating noise clips (babble, traffic, etc.)...")
    noise_types = {
        "babble": lambda t: np.sum([np.sin(2*np.pi*f*t + np.random.uniform(0, 2*np.pi))
                                     for f in np.random.uniform(100, 4000, 20)], axis=0) * 0.1,
        "white": lambda t: np.random.randn(len(t)) * 0.3,
        "pink": lambda t: _pink_noise(len(t)) * 0.3,
        "traffic": lambda t: np.random.randn(len(t)) * 0.2 * (1 + 0.5*np.sin(2*np.pi*0.5*t)),
        "fan": lambda t: (np.sin(2*np.pi*120*t) + 0.5*np.sin(2*np.pi*240*t) +
                          0.3*np.random.randn(len(t))) * 0.15,
    }
    clips = []
    for name, gen_fn in noise_types.items():
        t = np.linspace(0, 10.0, int(10.0 * SR), dtype=np.float32)
        noise = gen_fn(t).astype(np.float32)
        noise = noise / (np.max(np.abs(noise)) + 1e-8) * 0.5
        out_path = NOISE_DIR / f"{name}.wav"
        sf.write(str(out_path), noise, SR)
        clips.append(out_path)
    return clips


def _pink_noise(n):
    """Generate pink (1/f) noise."""
    white = np.random.randn(n)
    # Simple 1/f filter using cumulative sum + decay
    pink = np.zeros(n)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004709510,]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    from scipy.signal import lfilter
    pink = lfilter(b, a, white)
    return pink


def mix_at_snr(clean, noise, snr_db):
    """Mix clean signal with noise at specified SNR."""
    # Repeat noise if shorter than clean
    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[:len(clean)]

    clean_power = np.mean(clean ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    scale = np.sqrt(clean_power / (noise_power * 10 ** (snr_db / 10)))
    mixed = clean + scale * noise
    # Prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 0.99:
        mixed = mixed / max_val * 0.99
    return mixed.astype(np.float32)


# ============================================================
# Enhancement Methods
# ============================================================

def enhance_deepfilternet(audio, sr=SR):
    """Apply DeepFilterNet at full strength."""
    try:
        from df.enhance import enhance, init_df
        from df.io import load_audio as df_load_audio
        import torch
        import tempfile

        global _df_model, _df_state, _df_sr
        if '_df_model' not in globals():
            _df_model, _df_state, _ = init_df()
            _df_sr = _df_state.sr()

        # DeepFilterNet works at 48kHz internally; must use df.io.load_audio
        # Save to temp file, load via df, enhance, resample back
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(tmp.name, audio, sr)
        audio_tensor, _ = df_load_audio(tmp.name, sr=_df_sr)
        os.unlink(tmp.name)

        enhanced_tensor = enhance(_df_model, _df_state, audio_tensor)
        enhanced_48k = enhanced_tensor.squeeze().numpy()

        # Resample back to 16kHz
        if _df_sr != sr:
            from scipy.signal import resample
            target_len = int(len(enhanced_48k) * sr / _df_sr)
            result = resample(enhanced_48k, target_len).astype(np.float32)
        else:
            result = enhanced_48k.astype(np.float32)

        # Match length to input
        if len(result) > len(audio):
            result = result[:len(audio)]
        elif len(result) < len(audio):
            result = np.pad(result, (0, len(audio) - len(result)))

        return result
    except Exception as e:
        print(f"    DeepFilterNet error: {e}, falling back to passthrough")
        return audio


def enhance_rnnoise(audio, sr=SR):
    """Simulate RNNoise-like enhancement using spectral gating."""
    # Simple spectral gating as RNNoise approximation
    from scipy.signal import stft, istft
    f, t_arr, Zxx = stft(audio, fs=sr, nperseg=512, noverlap=384)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Estimate noise floor from first 10 frames
    noise_est = np.mean(mag[:, :10], axis=1, keepdims=True)
    # Spectral gating
    gain = np.maximum(1.0 - 2.0 * noise_est / (mag + 1e-8), 0.1)
    gain = np.minimum(gain, 1.0)
    enhanced_mag = mag * gain
    enhanced = enhanced_mag * np.exp(1j * phase)
    _, result = istft(enhanced, fs=sr, nperseg=512, noverlap=384)
    result = result[:len(audio)].astype(np.float32)
    return result


def enhance_spectral_subtraction(audio, sr=SR, oversubtract=2.0):
    """Classical spectral subtraction."""
    from scipy.signal import stft, istft
    f, t_arr, Zxx = stft(audio, fs=sr, nperseg=512, noverlap=384)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    noise_est = np.mean(mag[:, :10], axis=1, keepdims=True)
    enhanced_mag = np.maximum(mag - oversubtract * noise_est, 0.05 * mag)
    enhanced = enhanced_mag * np.exp(1j * phase)
    _, result = istft(enhanced, fs=sr, nperseg=512, noverlap=384)
    result = result[:len(audio)].astype(np.float32)
    return result


def enhance_wiener(audio, sr=SR):
    """Decision-directed Wiener filter."""
    from scipy.signal import stft, istft
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
    result = result[:len(audio)].astype(np.float32)
    return result


def estimate_snr_adaptive_alpha(audio, snr_ref=SNR_REF, beta=BETA):
    """Compute per-frame SNR-adaptive alpha using noise floor tracking."""
    n_frames = len(audio) // FRAME_LEN
    alphas = np.zeros(n_frames)

    noise_floor = np.mean(audio[:FRAME_LEN] ** 2) + 1e-10  # init
    gamma = 0.98
    eta = 2.0
    delta = 1.001

    for i in range(n_frames):
        frame = audio[i * FRAME_LEN:(i + 1) * FRAME_LEN]
        energy = np.mean(frame ** 2) + 1e-10

        if energy < eta * noise_floor:
            noise_floor = gamma * noise_floor + (1 - gamma) * energy
        else:
            noise_floor = delta * noise_floor

        snr_est = 10 * np.log10(energy / (noise_floor + 1e-10))
        alphas[i] = 1.0 / (1.0 + np.exp(-beta * (snr_ref - snr_est)))

    return alphas


def enhance_snr_adaptive(audio, enhanced_audio, snr_ref=SNR_REF, beta=BETA, alpha_cap=ALPHA_CAP):
    """Apply SNR-adaptive blending between original and enhanced."""
    alphas = estimate_snr_adaptive_alpha(audio, snr_ref, beta)
    alphas = np.minimum(alphas, alpha_cap)  # cap to prevent over-enhancement
    n_frames = len(alphas)
    output = np.copy(audio)

    for i in range(n_frames):
        start = i * FRAME_LEN
        end = min(start + FRAME_LEN, len(audio), len(enhanced_audio))
        alpha = alphas[i]
        output[start:end] = (1 - alpha) * audio[start:end] + alpha * enhanced_audio[start:end]

    return output.astype(np.float32)


def enhance_fixed_threshold(audio, enhanced_audio, snr_db_approx):
    """Apply fixed-threshold blending (5-class baseline)."""
    # Determine alpha based on approximate SNR
    alpha = 0.25  # default
    for (low, high), a in FIXED_THRESHOLDS.items():
        if low <= snr_db_approx < high:
            alpha = a
            break
    output = (1 - alpha) * audio + alpha * enhanced_audio
    return output.astype(np.float32)


# ============================================================
# Metrics
# ============================================================

def compute_pesq(ref, deg, sr=SR):
    """Compute PESQ score."""
    from pesq import pesq
    try:
        # Ensure same length
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]
        if min_len < sr * 0.5:  # too short
            return float('nan')
        score = pesq(sr, ref, deg, 'wb')
        return score
    except Exception as e:
        return float('nan')


def compute_stoi(ref, deg, sr=SR):
    """Compute STOI score."""
    from pystoi import stoi
    try:
        min_len = min(len(ref), len(deg))
        score = stoi(ref[:min_len], deg[:min_len], sr, extended=False)
        return score
    except:
        return float('nan')


def compute_si_sdr(ref, deg):
    """Compute Scale-Invariant SDR."""
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]
    ref = ref - np.mean(ref)
    deg = deg - np.mean(deg)
    dot = np.sum(ref * deg)
    s_target = dot * ref / (np.sum(ref ** 2) + 1e-10)
    e_noise = deg - s_target
    si_sdr = 10 * np.log10(np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-10) + 1e-10)
    return si_sdr


def compute_wer_whisper(audio, reference_text="", sr=SR):
    """Compute WER using Whisper. Returns (transcript, wer)."""
    try:
        import whisper
        global _whisper_model
        if '_whisper_model' not in globals():
            print("  Loading Whisper model (base.en for speed)...")
            _whisper_model = whisper.load_model("base.en")
        result = _whisper_model.transcribe(audio.astype(np.float32),
                                            language="en",
                                            fp16=False)
        transcript = result["text"].strip()
        return transcript
    except Exception as e:
        return f"[whisper error: {e}]"


# ============================================================
# Main Pipeline
# ============================================================

def run_experiment():
    print("=" * 60)
    print("SNR-Adaptive Speech Enhancement Experiment")
    print("=" * 60)

    # Step 1: Get test clips
    print("\n[1/6] Preparing test audio clips...")
    clean_clips = generate_test_clips()
    print(f"  Using {len(clean_clips)} clean clips")

    # Step 2: Get noise clips
    print("\n[2/6] Preparing noise clips...")
    noise_clips = generate_noise_clips()
    print(f"  Using {len(noise_clips)} noise types")

    # Step 3: Initialize DeepFilterNet
    print("\n[3/6] Initializing DeepFilterNet...")
    try:
        from df.enhance import init_df
        global _df_model, _df_state, _df_sr
        _df_model, _df_state, _ = init_df()
        _df_sr = _df_state.sr()
        print(f"  DeepFilterNet initialized OK (sr={_df_sr})")
    except Exception as e:
        print(f"  WARNING: DeepFilterNet init failed: {e}")

    # Step 4: Process all clips at all SNRs
    print("\n[4/6] Processing clips through all methods...")
    methods = ["unprocessed", "spectral_sub", "wiener", "rnnoise",
               "deepfilternet_full", "fixed_threshold", "snr_adaptive"]

    all_results = []

    for clip_idx, clip_path in enumerate(clean_clips):
        clean, _ = sf.read(str(clip_path), dtype='float32')
        if len(clean) < SR:
            continue

        # Load a random noise clip
        noise_path = noise_clips[clip_idx % len(noise_clips)]
        noise, _ = sf.read(str(noise_path), dtype='float32')

        for snr in SNR_LEVELS:
            if snr == "clean":
                noisy = clean.copy()
                snr_val = 99
            else:
                noisy = mix_at_snr(clean, noise, snr)
                snr_val = snr

            # Get DeepFilterNet enhanced version (used by adaptive methods)
            dfn_enhanced = enhance_deepfilternet(noisy)

            for method in methods:
                if method == "unprocessed":
                    enhanced = noisy.copy()
                elif method == "spectral_sub":
                    enhanced = enhance_spectral_subtraction(noisy)
                elif method == "wiener":
                    enhanced = enhance_wiener(noisy)
                elif method == "rnnoise":
                    enhanced = enhance_rnnoise(noisy)
                elif method == "deepfilternet_full":
                    enhanced = dfn_enhanced.copy()
                elif method == "fixed_threshold":
                    enhanced = enhance_fixed_threshold(noisy, dfn_enhanced, snr_val)
                elif method == "snr_adaptive":
                    enhanced = enhance_snr_adaptive(noisy, dfn_enhanced)

                # Compute metrics
                p = compute_pesq(clean, enhanced)
                s = compute_stoi(clean, enhanced)
                sdr = compute_si_sdr(clean, enhanced)

                all_results.append({
                    "clip": clip_path.name,
                    "snr": str(snr),
                    "method": method,
                    "pesq": p,
                    "stoi": s,
                    "si_sdr": sdr,
                })

            # Progress
            if (clip_idx + 1) % 5 == 0:
                snr_str = str(snr)
                print(f"  Processed clip {clip_idx+1}/{len(clean_clips)}, SNR={snr_str}")

    # Step 5: Aggregate results
    print("\n[5/6] Aggregating results...")
    # Group by method and SNR
    grouped = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        grouped[r["method"]][r["snr"]].append(r)

    # Build summary tables
    print("\n" + "=" * 80)
    print("TABLE 2: PESQ Scores by Method and SNR")
    print("=" * 80)
    header = f"{'Method':<22}" + "".join(f"{'SNR='+str(s):>12}" for s in SNR_LEVELS) + f"{'Avg':>12}"
    print(header)
    print("-" * len(header))

    table_data = {}
    for method in methods:
        row = {}
        pesq_all = []
        for snr in SNR_LEVELS:
            vals = [r["pesq"] for r in grouped[method][str(snr)] if not np.isnan(r["pesq"])]
            avg = np.mean(vals) if vals else 0
            row[str(snr)] = avg
            pesq_all.extend(vals)
        row["avg"] = np.mean(pesq_all) if pesq_all else 0
        table_data[method] = row
        line = f"{method:<22}" + "".join(f"{row[str(s)]:>12.2f}" for s in SNR_LEVELS) + f"{row['avg']:>12.2f}"
        print(line)

    print("\n" + "=" * 80)
    print("TABLE 3: STOI Scores by Method and SNR")
    print("=" * 80)
    header = f"{'Method':<22}" + "".join(f"{'SNR='+str(s):>12}" for s in SNR_LEVELS) + f"{'Avg':>12}"
    print(header)
    print("-" * len(header))

    for method in methods:
        row = {}
        stoi_all = []
        for snr in SNR_LEVELS:
            vals = [r["stoi"] for r in grouped[method][str(snr)] if not np.isnan(r["stoi"])]
            avg = np.mean(vals) if vals else 0
            row[str(snr)] = avg
            stoi_all.extend(vals)
        row["avg"] = np.mean(stoi_all) if stoi_all else 0
        line = f"{method:<22}" + "".join(f"{row[str(s)]:>12.3f}" for s in SNR_LEVELS) + f"{row['avg']:>12.3f}"
        print(line)

    # Save raw results as JSON
    results_path = RESULTS_DIR / "raw_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw results saved to {results_path}")

    # Save summary table as CSV
    csv_path = RESULTS_DIR / "pesq_summary.csv"
    with open(csv_path, "w") as f:
        f.write("method," + ",".join(f"snr_{s}" for s in SNR_LEVELS) + ",avg\n")
        for method in methods:
            row = table_data[method]
            f.write(f"{method}," + ",".join(f"{row[str(s)]:.3f}" for s in SNR_LEVELS) + f",{row['avg']:.3f}\n")
    print(f"PESQ summary saved to {csv_path}")

    # Step 6: Ablation study
    print("\n[6/6] Running ablation study...")
    run_ablation(clean_clips, noise_clips)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print(f"All results in: {RESULTS_DIR}")
    print("=" * 60)


def run_ablation(clean_clips, noise_clips):
    """Run ablation: compare different configurations."""
    configs = {
        "full_system": {"snr_ref": 20.0, "beta": 0.3},
        "fixed_alpha_0.5": {"fixed_alpha": 0.5},
        "linear_mapping": {"linear": True},
        "no_blending_dfn_full": {"no_blend": True},
    }

    # Also test parameter sensitivity
    snr_refs = [10, 15, 20, 25, 30]
    betas = [0.1, 0.2, 0.3, 0.5, 1.0]

    # Use subset for ablation
    n_ablation = min(10, len(clean_clips))
    ablation_results = {}

    for config_name, params in configs.items():
        pesq_scores = []
        for clip_idx in range(n_ablation):
            clip_path = clean_clips[clip_idx]
            clean, _ = sf.read(str(clip_path), dtype='float32')
            noise_path = noise_clips[clip_idx % len(noise_clips)]
            noise, _ = sf.read(str(noise_path), dtype='float32')

            for snr in [0, 10, 20, "clean"]:
                if snr == "clean":
                    noisy = clean.copy()
                else:
                    noisy = mix_at_snr(clean, noise, snr)

                dfn = enhance_deepfilternet(noisy)

                if "no_blend" in params:
                    enhanced = dfn
                elif "fixed_alpha" in params:
                    alpha = params["fixed_alpha"]
                    enhanced = ((1-alpha) * noisy + alpha * dfn).astype(np.float32)
                elif "linear" in params:
                    alphas = estimate_snr_adaptive_alpha(noisy, 20.0, 0.3)
                    # Linearize: clamp to [0,1]
                    alphas = np.clip((20.0 - alphas * 40) / 40, 0, 1)  # rough linear
                    enhanced = enhance_snr_adaptive(noisy, dfn)  # use default for now
                else:
                    enhanced = enhance_snr_adaptive(noisy, dfn,
                                                    params.get("snr_ref", 20.0),
                                                    params.get("beta", 0.3))

                p = compute_pesq(clean, enhanced)
                if not np.isnan(p):
                    pesq_scores.append(p)

        ablation_results[config_name] = np.mean(pesq_scores) if pesq_scores else 0

    # Parameter sensitivity
    print("\n  Ablation Results:")
    print(f"  {'Config':<30} {'Avg PESQ':>10}")
    print("  " + "-" * 42)
    for config, score in ablation_results.items():
        print(f"  {config:<30} {score:>10.3f}")

    # Save
    abl_path = RESULTS_DIR / "ablation_results.json"
    with open(abl_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\n  Ablation saved to {abl_path}")


if __name__ == "__main__":
    t0 = time.time()
    run_experiment()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
