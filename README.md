# When Does Neural Speech Enhancement Help Meeting Transcription?

Evaluation code, algorithm implementation, and experimental results for:

> **"When Does Neural Speech Enhancement Help Meeting Transcription? An Empirical Study"**
> Anwei Liu, HELPM AI INC
> *Submitted to IEEE Access, 2026*

## Key Findings

Based on 2,450 PESQ/STOI measurements + 1,400 WER measurements on 50 AMI Meeting Corpus utterances:

1. **RNNoise catastrophically fails on meeting audio** — STOI collapses to 0.101, WER exceeds 1.0 (hallucinations).
2. **DeepFilterNet3 full-strength degrades clean audio** — PESQ drops 33% (4.64 → 3.08), clean-audio WER rises from 0.000 to 0.147.
3. **Capped SNR-adaptive blending is the best overall method** — Avg PESQ 2.19, STOI 0.881, WER 0.354 (beats unprocessed 0.364).

## Repository Structure

```
├── src/
│   ├── snr_adaptive_blending.py   # Core capped adaptive algorithm (Eq. 3-6)
│   ├── noise_floor_tracker.py     # SNR estimation via exp min tracking (Eq. 1-2)
│   ├── noise_generator.py         # 8 noise types (babble, office, cafe, etc.)
│   └── baselines.py               # Spectral subtraction, Wiener, simulated RNNoise
├── benchmarks/
│   └── snr_adaptive/
│       ├── run_experiment_v3.py   # Main PESQ/STOI evaluation (recommended)
│       ├── run_experiment.py      # Legacy entry
│       └── run_wer.py             # Whisper-based WER evaluation
├── results/                       # Raw and aggregated results
│   ├── raw_results.json           # 2,450 individual PESQ/STOI measurements (359KB)
│   ├── pesq_summary.json          # PESQ aggregated by method × SNR
│   ├── stoi_summary.json          # STOI aggregated by method × SNR
│   ├── wer_results.json           # 1,400 WER measurements (Whisper-base.en, 408KB)
│   ├── wer_summary.json           # WER aggregated
│   ├── ablation_results.json      # Ablation study data
│   └── pesq_summary.csv           # CSV version
├── config/
│   └── experiment_config.yaml     # All hyperparameters
├── requirements.txt               # Python dependencies
├── CITATION.bib                   # BibTeX citation
└── LICENSE                        # MIT License
```

## Quick Start

### Installation

```bash
git clone https://github.com/aiemi/capped-snr-blending.git
cd capped-snr-blending
pip install -r requirements.txt
```

Python 3.11 recommended. Installation takes ~5 min.

### Reproducing the Paper Results

```bash
# Full PESQ/STOI evaluation (50 AMI clips × 7 SNR × 7 methods = 2,450 measurements)
python benchmarks/snr_adaptive/run_experiment_v3.py

# WER evaluation (50 AMI clips × 7 SNR × 4 methods = 1,400 measurements)
python benchmarks/snr_adaptive/run_wer.py
```

The first script:
1. Downloads 50 AMI test utterances (via HuggingFace `edinburghcstr/ami`)
2. Generates 8 noise types
3. Mixes at SNRs {0, 5, 10, 15, 20, 25, clean}
4. Runs all 7 methods (unprocessed, spectral subtraction, Wiener, RNNoise, DFN3 full, fixed-threshold, capped adaptive)
5. Computes PESQ and STOI
6. Saves results to `results/`

Runtime: ~12 min on Apple M1 Pro (CPU). ~60 min for WER evaluation.

### Standalone Algorithm Usage

```python
import soundfile as sf
from src.snr_adaptive_blending import capped_snr_adaptive_blend

# Load audio
noisy, sr = sf.read("meeting_noisy.wav")

# Get DeepFilterNet-enhanced version (or any other enhancement model)
from df.enhance import enhance, init_df
model, state, _ = init_df()
# ... run DFN3 to get `enhanced` ...

# Apply capped adaptive blending
output = capped_snr_adaptive_blend(
    original=noisy,
    enhanced=enhanced,
    sr=16000,
    snr_ref=20.0,      # reference SNR (dB)
    beta=0.08,         # sigmoid steepness
    alpha_max=0.45,    # maximum blending coefficient (the "cap")
)
sf.write("meeting_enhanced.wav", output, sr)
```

## Algorithm Summary

**Problem.** Full-strength DeepFilterNet3 degrades meeting audio quality, especially on clean recordings (33% PESQ reduction).

**Solution.** Blend the original and enhanced signals using a capped SNR-adaptive mixing coefficient:

```
α(t) = min( σ(β · (SNR_ref - SNR_hat(t))),  α_max )
y(t) = (1 - α(t)) · x(t) + α(t) · x_enhanced(t)
```

where:
- `SNR_hat(t)` is estimated via exponential minimum noise floor tracking (see `src/noise_floor_tracker.py`)
- `σ` is the sigmoid function
- `SNR_ref = 20` dB, `β = 0.08`, `α_max = 0.45`

**Why the cap matters.** Without the cap, α → 1 in noisy conditions and the system applies full DFN3 — inheriting the same degradation. With α_max = 0.45, the original signal always dominates (at least 55% weight), preventing catastrophic over-enhancement.

## Results Summary

### PESQ (higher is better)

| Method | 0 dB | 10 dB | 20 dB | Clean | Avg |
|--------|------|-------|-------|-------|-----|
| Unprocessed | 1.09 | 1.32 | 2.00 | 4.64 | 2.05 |
| DFN3 (full) | 1.49 | 1.84 | 2.25 | 3.08 | 2.11 |
| RNNoise | 1.26 | 1.37 | 1.44 | 1.49 | 1.41 |
| **Capped adaptive** | **1.14** | **1.48** | **2.25** | **4.60** | **2.19** |

### WER (Whisper-base.en, lower is better) — 50 clips × 7 SNRs

| Method | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | Clean | Avg |
|--------|------|------|-------|-------|-------|-------|-------|-----|
| Unprocessed | 0.624 | 0.534 | 0.462 | 0.315 | 0.325 | 0.287 | 0.000 | 0.364 |
| DFN3 (full) | **0.572** | **0.483** | **0.384** | 0.359 | 0.359 | 0.264 | 0.258 | 0.383 |
| RNNoise | 1.179 | 1.466 | 1.153 | 1.192 | 1.191 | 1.081 | 1.219 | 1.212 |
| **Capped adaptive** | 0.584 | 0.534 | 0.436 | **0.301** | **0.285** | **0.245** | 0.091 | **0.354** |

### Statistical Significance (Wilcoxon, N=350 paired comparisons)

| Comparison | Mean Δ PESQ | p-value |
|-----------|-------------|---------|
| Ours vs. Unprocessed | +0.139 | 3.5 × 10⁻⁴⁴ |
| Ours vs. Fixed-threshold | +0.096 | 2.1 × 10⁻³⁵ |
| Ours vs. RNNoise | +0.773 | 1.3 × 10⁻²⁷ |

## Citation

```bibtex
@article{liu2026capped,
  author  = {Liu, Anwei},
  title   = {When Does Neural Speech Enhancement Help Meeting Transcription? An Empirical Study},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

See [CITATION.bib](CITATION.bib) for a ready-to-copy citation.

## License

MIT License. Copyright (c) 2026 Anwei Liu / HELPM AI INC. See [LICENSE](LICENSE).

## Contact

Anwei Liu — liuanwei2022@gmail.com
HELPM AI INC, Philadelphia, PA 19104, USA
