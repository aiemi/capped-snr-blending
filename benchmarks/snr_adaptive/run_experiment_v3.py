#!/usr/bin/env python3
"""
V3 Experiment: Empirical Study of Neural Speech Enhancement in Meeting Transcription
- 50 AMI clips × 10s each
- 8 noise types, 7 SNR conditions (0,5,10,15,20,25,clean)
- 7 methods: unprocessed, spectral_sub, wiener, real_rnnoise, dfn_full, fixed_threshold, capped_snr_adaptive
- Metrics: PESQ, STOI
- Plus: Whisper WER evaluation
"""

import os, sys, json, time
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from scipy.signal import stft, istft, resample as sp_resample
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path("/Users/liuanwei/NIW/experiments")
AUDIO_DIR = BASE_DIR / "data" / "audio_clips_long"
NOISE_DIR = BASE_DIR / "data" / "noise_clips"
RESULTS_DIR = BASE_DIR / "results_v3"
RESULTS_DIR.mkdir(exist_ok=True)

SR = 16000
FRAME_LEN = 1600

# Capped SNR-adaptive parameters (optimized)
SNR_REF = 20.0
BETA = 0.08
ALPHA_CAP = 0.45

FIXED_THRESHOLDS = {(0,5):0.45, (5,10):0.35, (10,15):0.25, (15,20):0.15, (20,float('inf')):0.05}
SNR_LEVELS = [0, 5, 10, 15, 20, 25, "clean"]

# ============================================================
# Enhancement Methods
# ============================================================
def enhance_dfn(audio):
    global _df_model, _df_state, _df_sr
    import tempfile
    from df.enhance import enhance
    from df.io import load_audio as df_load_audio
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(tmp.name, audio, SR)
    audio_t, _ = df_load_audio(tmp.name, sr=_df_sr)
    os.unlink(tmp.name)
    enh_t = enhance(_df_model, _df_state, audio_t)
    enh_48k = enh_t.squeeze().numpy()
    result = sp_resample(enh_48k, len(audio)).astype(np.float32)
    return result

def enhance_rnnoise_real(audio):
    from pyrnnoise import RNNoise
    denoiser = RNNoise(sample_rate=SR)
    chunks = []
    for prob, chunk in denoiser.denoise_chunk(audio):
        chunks.append(chunk.flatten())
    result = np.concatenate(chunks).astype(np.float32)[:len(audio)]
    return result

def enhance_spectral_sub(audio):
    f, t_arr, Zxx = stft(audio, fs=SR, nperseg=512, noverlap=384)
    mag, phase = np.abs(Zxx), np.angle(Zxx)
    noise_est = np.mean(mag[:, :10], axis=1, keepdims=True)
    enhanced_mag = np.maximum(mag - 2.0 * noise_est, 0.05 * mag)
    _, result = istft(enhanced_mag * np.exp(1j * phase), fs=SR, nperseg=512, noverlap=384)
    return result[:len(audio)].astype(np.float32)

def enhance_wiener(audio):
    f, t_arr, Zxx = stft(audio, fs=SR, nperseg=512, noverlap=384)
    mag, phase = np.abs(Zxx), np.angle(Zxx)
    noise_est = np.mean(mag[:, :10] ** 2, axis=1, keepdims=True)
    signal_est = np.maximum(mag ** 2 - noise_est, 0)
    gain = np.maximum(signal_est / (signal_est + noise_est + 1e-8), 0.1)
    _, result = istft(mag * gain * np.exp(1j * phase), fs=SR, nperseg=512, noverlap=384)
    return result[:len(audio)].astype(np.float32)

def estimate_snr_alphas(audio):
    n_frames = len(audio) // FRAME_LEN
    alphas = np.zeros(n_frames)
    noise_floor = np.mean(audio[:FRAME_LEN] ** 2) + 1e-10
    for i in range(n_frames):
        frame = audio[i*FRAME_LEN:(i+1)*FRAME_LEN]
        energy = np.mean(frame ** 2) + 1e-10
        if energy < 2.0 * noise_floor:
            noise_floor = 0.98 * noise_floor + 0.02 * energy
        else:
            noise_floor *= 1.001
        snr = 10 * np.log10(energy / (noise_floor + 1e-10))
        alphas[i] = min(1.0 / (1.0 + np.exp(-BETA * (SNR_REF - snr))), ALPHA_CAP)
    return alphas

def blend_adaptive(audio, enhanced):
    alphas = estimate_snr_alphas(audio)
    output = np.copy(audio)
    for i in range(len(alphas)):
        s, e = i * FRAME_LEN, min((i+1) * FRAME_LEN, len(audio), len(enhanced))
        output[s:e] = (1 - alphas[i]) * audio[s:e] + alphas[i] * enhanced[s:e]
    return output.astype(np.float32)

def blend_fixed(audio, enhanced, snr_val):
    alpha = 0.25
    for (lo, hi), a in FIXED_THRESHOLDS.items():
        if lo <= snr_val < hi: alpha = a; break
    return ((1-alpha)*audio + alpha*enhanced).astype(np.float32)

# ============================================================
# Metrics
# ============================================================
def compute_pesq(ref, deg):
    from pesq import pesq
    try:
        ml = min(len(ref), len(deg))
        return pesq(SR, ref[:ml], deg[:ml], 'wb')
    except: return float('nan')

def compute_stoi(ref, deg):
    from pystoi import stoi
    try:
        ml = min(len(ref), len(deg))
        return stoi(ref[:ml], deg[:ml], SR, extended=False)
    except: return float('nan')

def mix_at_snr(clean, noise, snr_db):
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean)/len(noise))))
    noise = noise[:len(clean)]
    cp, np_ = np.mean(clean**2)+1e-10, np.mean(noise**2)+1e-10
    scale = np.sqrt(cp / (np_ * 10**(snr_db/10)))
    mixed = clean + scale * noise
    mx = np.max(np.abs(mixed))
    if mx > 0.99: mixed = mixed / mx * 0.99
    return mixed.astype(np.float32)

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("V3 Experiment: Empirical Study of Neural Enhancement in Meetings")
    print("=" * 70)

    clips = sorted(AUDIO_DIR.glob("*.wav"))[:50]
    noises = sorted(NOISE_DIR.glob("*.wav"))
    print(f"Clips: {len(clips)}, Noise types: {len(noises)}")

    # Init models
    print("Initializing DeepFilterNet...")
    from df.enhance import init_df
    global _df_model, _df_state, _df_sr
    _df_model, _df_state, _ = init_df()
    _df_sr = _df_state.sr()
    print(f"  DFN ready (sr={_df_sr})")

    methods = ["unprocessed", "spectral_sub", "wiener", "rnnoise_real",
               "dfn_full", "fixed_threshold", "capped_adaptive"]

    all_results = []
    t0 = time.time()

    for ci, cp in enumerate(clips):
        clean, _ = sf.read(str(cp), dtype='float32')
        noise, _ = sf.read(str(noises[ci % len(noises)]), dtype='float32')

        for snr in SNR_LEVELS:
            if snr == "clean":
                noisy = clean.copy()
                snr_val = 99
            else:
                noisy = mix_at_snr(clean, noise, snr)
                snr_val = snr

            # Get DFN enhanced (shared by adaptive methods)
            dfn_enh = enhance_dfn(noisy)

            for method in methods:
                if method == "unprocessed":
                    out = noisy
                elif method == "spectral_sub":
                    out = enhance_spectral_sub(noisy)
                elif method == "wiener":
                    out = enhance_wiener(noisy)
                elif method == "rnnoise_real":
                    out = enhance_rnnoise_real(noisy)
                elif method == "dfn_full":
                    out = dfn_enh
                elif method == "fixed_threshold":
                    out = blend_fixed(noisy, dfn_enh, snr_val)
                elif method == "capped_adaptive":
                    out = blend_adaptive(noisy, dfn_enh)

                p = compute_pesq(clean, out)
                s = compute_stoi(clean, out)
                all_results.append({
                    "clip": cp.name, "snr": str(snr), "method": method,
                    "pesq": p, "stoi": s
                })

        if (ci+1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{ci+1}/{len(clips)}] {elapsed:.0f}s elapsed")

    # Aggregate
    print("\nAggregating results...")
    grouped = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        grouped[r["method"]][r["snr"]].append(r)

    print("\n" + "=" * 100)
    print("PESQ RESULTS")
    print("=" * 100)
    hdr = f"{'Method':<20}" + "".join(f"{'SNR='+str(s):>10}" for s in SNR_LEVELS) + f"{'Avg':>10}"
    print(hdr)
    print("-" * len(hdr))

    summary = {}
    for method in methods:
        row = {}
        all_vals = []
        for snr in SNR_LEVELS:
            vals = [r["pesq"] for r in grouped[method][str(snr)] if not np.isnan(r["pesq"])]
            avg = np.mean(vals) if vals else 0
            row[str(snr)] = round(avg, 3)
            all_vals.extend(vals)
        row["avg"] = round(np.mean(all_vals), 3) if all_vals else 0
        summary[method] = row
        line = f"{method:<20}" + "".join(f"{row[str(s)]:>10.3f}" for s in SNR_LEVELS) + f"{row['avg']:>10.3f}"
        print(line)

    print("\n" + "=" * 100)
    print("STOI RESULTS")
    print("=" * 100)
    print(hdr)
    print("-" * len(hdr))

    stoi_summary = {}
    for method in methods:
        row = {}
        all_vals = []
        for snr in SNR_LEVELS:
            vals = [r["stoi"] for r in grouped[method][str(snr)] if not np.isnan(r["stoi"])]
            avg = np.mean(vals) if vals else 0
            row[str(snr)] = round(avg, 3)
            all_vals.extend(vals)
        row["avg"] = round(np.mean(all_vals), 3) if all_vals else 0
        stoi_summary[method] = row
        line = f"{method:<20}" + "".join(f"{row[str(s)]:>10.3f}" for s in SNR_LEVELS) + f"{row['avg']:>10.3f}"
        print(line)

    # Save
    with open(RESULTS_DIR / "raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(RESULTS_DIR / "pesq_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(RESULTS_DIR / "stoi_summary.json", "w") as f:
        json.dump(stoi_summary, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
