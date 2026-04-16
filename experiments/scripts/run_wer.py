#!/usr/bin/env python3
"""
WER evaluation using Whisper-base on a subset of enhanced audio.
Runs after run_v3.py produces results.

Uses Whisper-base (not large) for speed on M1 Pro.
Evaluates on 10 clips × 4 SNR levels × 4 key methods = 160 transcriptions.
"""

import os, json, time
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import stft, istft, resample as sp_resample
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path("/Users/liuanwei/NIW/experiments")
AUDIO_DIR = BASE_DIR / "data" / "audio_clips_long"
NOISE_DIR = BASE_DIR / "data" / "noise_clips"
RESULTS_DIR = BASE_DIR / "results_v3"

SR = 16000
FRAME_LEN = 1600
SNR_REF, BETA, ALPHA_CAP = 20.0, 0.08, 0.45
FIXED_THRESHOLDS = {(0,5):0.45, (5,10):0.35, (10,15):0.25, (15,20):0.15, (20,float('inf')):0.05}

# Import methods from run_v3
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
    return sp_resample(enh_48k, len(audio)).astype(np.float32)

def enhance_rnnoise_real(audio):
    from pyrnnoise import RNNoise
    denoiser = RNNoise(sample_rate=SR)
    chunks = []
    for prob, chunk in denoiser.denoise_chunk(audio):
        chunks.append(chunk.flatten())
    return np.concatenate(chunks).astype(np.float32)[:len(audio)]

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


def main():
    print("=" * 60)
    print("WER Evaluation with Whisper-base")
    print("=" * 60)

    # Load Whisper
    print("Loading Whisper model (base.en)...")
    import whisper
    whisper_model = whisper.load_model("base.en")
    print("  Whisper ready")

    # Init DFN
    from df.enhance import init_df
    global _df_model, _df_state, _df_sr
    _df_model, _df_state, _ = init_df()
    _df_sr = _df_state.sr()

    clips = sorted(AUDIO_DIR.glob("*.wav"))[:50]  # 50 clips for WER (full scope)
    noises = sorted(NOISE_DIR.glob("*.wav"))
    snr_levels = [0, 5, 10, 15, 20, 25, "clean"]  # full SNR range
    methods = ["unprocessed", "rnnoise_real", "dfn_full", "capped_adaptive"]

    def transcribe(audio):
        result = whisper_model.transcribe(audio.astype(np.float32), language="en", fp16=False)
        return result["text"].strip()

    # First get reference transcripts from clean audio
    print("\nGenerating reference transcripts from clean audio...")
    refs = {}
    for cp in clips:
        clean, _ = sf.read(str(cp), dtype='float32')
        ref_text = transcribe(clean)
        refs[cp.name] = ref_text
        print(f"  {cp.name}: '{ref_text[:60]}...'")

    # Now evaluate each method
    print("\nEvaluating methods...")
    wer_results = []

    for ci, cp in enumerate(clips):
        clean, _ = sf.read(str(cp), dtype='float32')
        noise, _ = sf.read(str(noises[ci % len(noises)]), dtype='float32')
        ref_text = refs[cp.name]

        for snr in snr_levels:
            if snr == "clean":
                noisy = clean.copy()
                snr_val = 99
            else:
                noisy = mix_at_snr(clean, noise, snr)
                snr_val = snr

            dfn_enh = enhance_dfn(noisy)

            for method in methods:
                if method == "unprocessed":
                    out = noisy
                elif method == "rnnoise_real":
                    out = enhance_rnnoise_real(noisy)
                elif method == "dfn_full":
                    out = dfn_enh
                elif method == "capped_adaptive":
                    out = blend_adaptive(noisy, dfn_enh)

                hyp_text = transcribe(out)

                # Simple WER calculation
                ref_words = ref_text.lower().split()
                hyp_words = hyp_text.lower().split()
                if len(ref_words) == 0:
                    wer = 0.0
                else:
                    # Levenshtein distance on words
                    import Levenshtein
                    dist = Levenshtein.distance(" ".join(ref_words), " ".join(hyp_words))
                    wer = dist / max(len(" ".join(ref_words)), 1)

                wer_results.append({
                    "clip": cp.name, "snr": str(snr), "method": method,
                    "wer": wer, "ref": ref_text[:100], "hyp": hyp_text[:100]
                })

        if (ci+1) % 5 == 0:
            print(f"  [{ci+1}/{len(clips)}] done")

    # Aggregate
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))
    for r in wer_results:
        grouped[r["method"]][r["snr"]].append(r["wer"])

    print("\n" + "=" * 80)
    print("WER RESULTS (lower is better)")
    print("=" * 80)
    hdr = f"{'Method':<20}" + "".join(f"{'SNR='+str(s):>12}" for s in snr_levels) + f"{'Avg':>12}"
    print(hdr)
    print("-" * len(hdr))

    wer_summary = {}
    for method in methods:
        row = {}
        all_w = []
        for snr in snr_levels:
            vals = grouped[method][str(snr)]
            avg = np.mean(vals) if vals else 0
            row[str(snr)] = round(avg, 4)
            all_w.extend(vals)
        row["avg"] = round(np.mean(all_w), 4) if all_w else 0
        wer_summary[method] = row
        line = f"{method:<20}" + "".join(f"{row[str(s)]:>12.4f}" for s in snr_levels) + f"{row['avg']:>12.4f}"
        print(line)

    # Save
    with open(RESULTS_DIR / "wer_results.json", "w") as f:
        json.dump(wer_results, f, indent=2, default=str)
    with open(RESULTS_DIR / "wer_summary.json", "w") as f:
        json.dump(wer_summary, f, indent=2)

    print(f"\nWER results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total time: {(time.time()-t0)/60:.1f} minutes")
