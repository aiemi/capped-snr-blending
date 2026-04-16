# Experiment Reproduction Package

This directory contains the **complete historical experiment data** used in the paper, including original scripts, intermediate results from all experiment iterations (v1, v3), and the generated noise clips.

## Purpose

For reviewers and researchers who want to:
1. **Verify the published numbers** — raw JSON results for every measurement.
2. **Fully reproduce the experiments** — original scripts that produced the results.
3. **Use our noise dataset** — 8 WAV noise clips we generated (babble, office, cafeteria, traffic, white, pink, brown, fan).
4. **Compare with alternative configurations** — v1 results (30 clips) vs v3 results (50 clips).

## Directory Layout

```
experiments/
├── README.md                 (this file)
├── scripts/                  Original experiment scripts
│   ├── run_all.py            v1 script (30 clips × 7 SNR, legacy)
│   ├── run_v3.py             v3 script (50 clips × 7 SNR × 7 methods = 2,450 evaluations)
│   ├── run_wer.py            WER evaluation via Whisper-base.en
│   └── download_demand.py    Noise generator (fallback for missing DEMAND download)
├── results_v1/               First iteration results (30 clips, simulated RNNoise)
│   ├── raw_results.json      Per-clip PESQ/STOI (all 30 × 7 × 7 entries)
│   ├── pesq_summary.csv      Aggregated PESQ CSV
│   └── ablation_results.json Ablation study raw numbers
├── results_v3/               Final paper results (50 clips, real RNNoise)
│   ├── raw_results.json      Per-clip PESQ/STOI (2,450 entries, 359KB) ★
│   ├── pesq_summary.json     PESQ aggregated by method × SNR
│   ├── stoi_summary.json     STOI aggregated by method × SNR
│   ├── wer_results.json      Per-clip WER (160 entries, 96KB)
│   └── wer_summary.json      WER aggregated
└── data/
    └── noise_clips/          8 generated noise clips (30s each, 16kHz)
        ├── babble.wav        Multi-talker babble (15 simulated voices)
        ├── office.wav        AC hum + keyboard clicks
        ├── cafeteria.wav     Babble + dish clinks + background music
        ├── traffic.wav       Low-freq rumble + passing vehicles
        ├── white.wav         Gaussian white noise
        ├── pink.wav          1/f pink noise (Voss-McCartney)
        ├── brown.wav         1/f² brown noise
        └── fan.wav           HVAC / fan harmonics
```

## How to Reproduce

### Option A: Full pipeline from scratch

```bash
# From repo root
pip install -r requirements.txt

# Step 1: Generate noise clips (if not already present)
python experiments/scripts/download_demand.py

# Step 2: Download AMI clips and run PESQ/STOI evaluation
python experiments/scripts/run_v3.py

# Step 3: Run Whisper WER evaluation (slower, ~60 min)
python experiments/scripts/run_wer.py
```

Expected runtime: ~12 min for PESQ/STOI, ~60 min for WER on Apple M1 Pro (CPU only).

### Option B: Verify our numbers against the paper

```bash
# Load raw_results.json and recompute tables
python -c "
import json
from collections import defaultdict
with open('experiments/results_v3/raw_results.json') as f:
    data = json.load(f)

# Example: recompute Table II (PESQ by method × SNR)
grouped = defaultdict(lambda: defaultdict(list))
for r in data:
    grouped[r['method']][r['snr']].append(r['pesq'])

for method in ['unprocessed', 'dfn_full', 'capped_adaptive']:
    avg = sum(v for vals in grouped[method].values() for v in vals if v) / 350
    print(f'{method}: avg PESQ = {avg:.3f}')
"
```

### Option C: Run statistical tests

```bash
python -c "
import json, numpy as np
from scipy.stats import wilcoxon
with open('experiments/results_v3/raw_results.json') as f:
    data = json.load(f)

pairs = {}
for r in data:
    k = (r['clip'], r['snr'])
    pairs.setdefault(k, {})[r['method']] = r['pesq']

diff = [p['capped_adaptive'] - p['unprocessed']
        for p in pairs.values()
        if 'capped_adaptive' in p and 'unprocessed' in p]
diff = np.array([d for d in diff if not np.isnan(d) and d != 0])
stat, pval = wilcoxon(diff)
print(f'Wilcoxon test: W={stat:.0f}, p={pval:.2e}, mean diff={np.mean(diff):+.4f}')
"
```

## Reproducibility Notes

- **Random seeds:** Noise generation uses fixed `np.random.seed(42)` in most places; some clips may differ on regeneration. The `data/noise_clips/` folder contains the exact clips we used.
- **DeepFilterNet version:** We used DeepFilterNet3 via `deepfilternet==0.5.6`. Later versions may give slightly different numbers.
- **RNNoise version:** We used `pyrnnoise==0.4.3` (official C binding). Older pure-Python reimplementations will give different (worse) numbers.
- **Whisper version:** `openai-whisper>=20231117`, model `base.en`. Upgrading to larger models (small, medium) will reduce WER overall but preserve the ranking between methods.
- **AMI snapshot:** Accessed via HuggingFace `edinburghcstr/ami` on 2026-04-15. If the dataset changes, use the first 50 utterances from the test split of the IHM configuration.

## Result Schema (`results_v3/raw_results.json`)

Each entry is a dictionary:

```json
{
  "clip": "ami_long_000.wav",     // Audio file name
  "snr": "10",                     // SNR in dB, or "clean"
  "method": "capped_adaptive",     // One of 7 methods
  "pesq": 1.482,                   // PESQ wideband score
  "stoi": 0.856                    // STOI score
}
```

2,450 total entries = 50 clips × 7 SNRs × 7 methods.

## History of Iterations

- **v1** (legacy, `results_v1/`) — 30 clips × 5s each, simulated RNNoise (spectral gating). Used in early paper drafts. Showed effect trend but suffered from short-clip artifacts and weak RNNoise baseline.
- **v3** (final, `results_v3/`) — 50 clips × 10s each, real RNNoise via pyrnnoise. Final paper numbers.

The v1 results are retained for transparency; the paper reports only v3 numbers.

## Questions

Open an issue or contact liuanwei2022@gmail.com.
