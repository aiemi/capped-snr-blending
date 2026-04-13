# Echo Cancellation Evaluation

## Overview

This directory contains the evaluation script for the Voxclar Acoustic Echo Cancellation (AEC) module, as described in Paper Section 4.1. The AEC operates in the frequency domain using a partitioned-block NLMS adaptive filter on the system audio loopback stream.

## Methodology

### Test Signal Construction

Each test case consists of a **30-second speech clip** (sourced from the LibriSpeech test-clean corpus) convolved with a synthetic room impulse response (RIR) to simulate echo at various delays:

| Parameter            | Values tested                       |
|----------------------|-------------------------------------|
| Echo delay           | 20 ms, 50 ms, 100 ms, 200 ms       |
| Echo-to-signal ratio | -6 dB, -3 dB, 0 dB, +3 dB          |
| Room type            | 5 environments (see below)          |

### Acoustic Environments

| Environment    | RT60 (ms) | Description                        |
|----------------|----------:|------------------------------------|
| Anechoic       |        0  | No reflections (baseline)          |
| Small office   |      200  | Carpeted, furnished                |
| Conference     |      400  | Medium room, hard surfaces         |
| Open plan      |      600  | Large open office                  |
| Auditorium     |      900  | Large reverberant space            |

### Metrics

1. **ERLE (Echo Return Loss Enhancement)**: Ratio of echo power before and after cancellation, measured in dB. Higher is better.

   ```
   ERLE = 10 * log10( mean(echo^2) / mean(residual^2) )
   ```

2. **SNR Improvement**: Increase in signal-to-noise ratio after AEC processing.

3. **PESQ (Perceptual Evaluation of Speech Quality)**: Measured per ITU-T P.862. Reported separately as it requires the `pesq` library (not included in the minimal dependency set).

## Usage

```bash
python evaluate_aec.py --input recorded.wav --reference loopback.wav --output results.json
```

### Arguments

| Flag            | Description                                      |
|-----------------|--------------------------------------------------|
| `--input`       | Path to the recorded microphone signal (with echo)|
| `--reference`   | Path to the far-end loopback signal              |
| `--output`      | Path to write JSON results (optional)            |
| `--sample-rate` | Sample rate in Hz (default: 16000)               |
| `--block-size`  | AEC filter block size (default: 1600)            |

## Reference Results (Paper Table 1)

| Environment    | ERLE (dB) | SNR Improvement (dB) | PESQ |
|----------------|----------:|---------------------:|-----:|
| Anechoic       |      48.1 |                 18.2 |  4.1 |
| Small office   |      45.7 |                 16.8 |  3.9 |
| Conference     |      42.3 |                 14.5 |  3.6 |
| Open plan      |      38.9 |                 12.1 |  3.3 |
| Auditorium     |      36.5 |                 10.4 |  3.1 |
| **Mean**       |  **42.3** |             **14.4** |**3.6**|
