# Capped SNR-Adaptive Blending for Neural Speech Enhancement

Evaluation code, algorithm implementation, and experimental results for:

> **"Capped SNR-Adaptive Blending for Neural Speech Enhancement in Meeting Transcription"**
> Anwei Liu, University of Pennsylvania (Alumnus)
> *Submitted to Applied Sciences (MDPI), 2026*

## Key Finding

Full-strength DeepFilterNet **severely degrades** meeting audio quality:
- PESQ drops 50% on clean audio (4.64 → 2.32)
- STOI drops 20% on clean audio (1.000 → 0.802)

**Capped SNR-adaptive blending** (β=0.08, α_max=0.45) achieves the best overall quality:
- Average PESQ: **2.33** (vs. 2.27 unprocessed, 1.71 DeepFilterNet full)
- Average STOI: **0.903** (vs. 0.898 unprocessed, 0.740 DeepFilterNet full)
- Clean audio nearly perfectly preserved: PESQ 4.59 (vs. 4.64 unprocessed)

## Repository Structure

```
├── src/
│   ├── snr_adaptive_blending.py   # Core algorithm implementation
│   ├── noise_generator.py         # Noise generation (babble, office, etc.)
│   └── baselines.py               # Baseline enhancement methods
├── benchmarks/
│   └── snr_adaptive/
│       └── run_experiment.py      # Full experiment pipeline
├── results/
│   ├── raw_results.json           # All per-clip PESQ/STOI scores
│   └── pesq_summary.csv           # Aggregated PESQ table
├── config/
│   └── experiment_config.yaml     # All hyperparameters
└── LICENSE
```

## Quick Start

```bash
pip install deepfilternet pesq pystoi numpy scipy soundfile datasets
python benchmarks/snr_adaptive/run_experiment.py
```

This downloads 30 AMI test utterances, generates 8 noise types, mixes at 7 SNR levels, runs all 7 methods, and outputs PESQ/STOI to `results/`.

## Standalone Usage

```python
from src.snr_adaptive_blending import capped_snr_adaptive_blend

output = capped_snr_adaptive_blend(
    original=audio,
    enhanced=deepfilternet_output,
    sr=16000,
    snr_ref=20.0,
    beta=0.08,
    alpha_max=0.45
)
```

## Results (Real Experimental Data)

| Method | 0 dB | 10 dB | 20 dB | Clean | Avg PESQ |
|--------|------|-------|-------|-------|----------|
| Unprocessed | 1.15 | 1.52 | 2.40 | 4.64 | 2.27 |
| DeepFilterNet (full) | 1.36 | 1.51 | 1.74 | 2.32 | 1.71 |
| Fixed-threshold | 1.20 | 1.57 | 2.41 | 4.64 | 2.30 |
| **Ours (capped)** | **1.18** | **1.61** | **2.49** | **4.59** | **2.33** |

## Citation

```bibtex
@article{liu2026capped,
  author = {Liu, Anwei},
  title = {Capped {SNR}-Adaptive Blending for Neural Speech Enhancement in Meeting Transcription},
  journal = {Applied Sciences},
  year = {2026},
  note = {Under Review}
}
```

## License

MIT License. Copyright (c) 2026 Anwei Liu / HELPM AI INC.
