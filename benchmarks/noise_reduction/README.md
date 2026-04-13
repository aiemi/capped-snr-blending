# Noise Reduction Evaluation

## Overview

This directory contains the evaluation script for the Voxclar adaptive noise reduction (NR) module, as described in Paper Section 4.2. The NR pipeline classifies the acoustic environment by RMS level and applies scene-appropriate spectral suppression.

## Environment Classification

The system uses a 500 ms sliding-window RMS estimator to classify the current acoustic scene into one of five categories:

| Environment   | RMS Threshold (dBFS) | Suppression (dB) |
|---------------|---------------------:|------------------:|
| Quiet room    |              < -50   |                 6 |
| Home office   |        -50 to -40    |                12 |
| Open office   |        -40 to -30    |                18 |
| Cafe          |        -30 to -20    |                24 |
| Outdoor       |             >= -20   |                30 |

## Evaluation Methodology

### Test Signals

For each environment class, 20 test clips were constructed:

- **Clean speech**: 10-second segments from LibriSpeech test-clean
- **Noise**: Environment-specific recordings (DEMAND corpus) mixed at the target SNR
- **SNR range**: 0 dB to 20 dB in 5 dB steps

### Metrics

1. **SNR Improvement**: Difference in segmental SNR before and after NR processing.

2. **RMS Classification Accuracy**: Whether the RMS-based classifier correctly identifies the environment.

3. **PESQ (Perceptual Evaluation of Speech Quality)**: Measured per ITU-T P.862, comparing the NR output against the clean reference. A score above 3.0 indicates good quality; above 3.5 indicates excellent quality.

## Usage

```bash
python evaluate_nr.py --before noisy.wav --after cleaned.wav --sample-rate 16000
```

### Arguments

| Flag            | Description                                          |
|-----------------|------------------------------------------------------|
| `--before`      | Path to the noisy input signal                       |
| `--after`       | Path to the NR-processed signal                      |
| `--clean`       | Path to the clean reference signal (optional, for PESQ) |
| `--sample-rate` | Sample rate in Hz (default: 16000)                   |
| `--output`      | Path to write JSON results (optional)                |

## Reference Results (Paper Table 2)

| Environment   | SNR Improvement (dB) | PESQ (before) | PESQ (after) |
|---------------|---------------------:|---------------:|-------------:|
| Quiet room    |                  2.1 |           3.8  |          4.1 |
| Home office   |                  5.4 |           3.2  |          3.7 |
| Open office   |                  8.7 |           2.7  |          3.4 |
| Cafe          |                 11.2 |           2.3  |          3.2 |
| Outdoor       |                 13.8 |           1.9  |          3.0 |
| **Mean**      |              **8.2** |       **2.8**  |      **3.5** |
