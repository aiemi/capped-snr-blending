# Real-Time AI-Assisted Meeting Intelligence: A Dual-Stream Audio Architecture with Context-Aware Response Generation

**Anwei Liu**
HELPM AI INC

---

## Abstract

Modern knowledge workers spend a significant portion of their time in virtual meetings, yet existing tools provide only passive transcription with limited actionable intelligence. We present **Voxclar**, a real-time meeting intelligence system built on a novel dual-stream audio architecture that independently captures microphone input and system audio output, enabling acoustic echo cancellation (AEC), adaptive noise reduction, and accurate speaker-attributed transcription. The system further implements a context-aware question detection pipeline that identifies directed questions in real time and generates suggested responses grounded in meeting context. We evaluate each subsystem independently: our AEC module achieves a mean Echo Return Loss Enhancement (ERLE) of 42.3 dB across five acoustic environments; the adaptive noise reduction pipeline maintains PESQ scores above 3.2 under adverse conditions; and the question detection classifier reaches 94.6% F1-score on a 500-segment benchmark spanning three common meeting scenarios. End-to-end latency from utterance completion to suggested response averages 1.8 seconds on consumer hardware. All benchmark datasets and evaluation scripts are provided in this repository to support reproducibility.

## Repository Structure

```
Voxclar-Meeting-Intelligence/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── config/
│   └── example_config.yaml            # Reference configuration parameters
├── benchmarks/
│   ├── question_detection/
│   │   ├── README.md                  # Dataset description and methodology
│   │   └── generate_benchmark.py      # Script to generate the 500-segment dataset
│   ├── echo_cancellation/
│   │   ├── README.md                  # AEC evaluation methodology
│   │   └── evaluate_aec.py            # ERLE and SNR computation script
│   └── noise_reduction/
│       ├── README.md                  # Adaptive NR evaluation methodology
│       └── evaluate_nr.py             # RMS classification and SNR evaluation
```

## Getting Started

### Prerequisites

- Python 3.9 or later
- NumPy >= 1.21

```bash
pip install numpy
```

### Generate the Question Detection Benchmark

```bash
cd benchmarks/question_detection
python generate_benchmark.py
```

This produces `benchmark_dataset.json` containing 500 labeled meeting segments.

### Run Echo Cancellation Evaluation

```bash
cd benchmarks/echo_cancellation
python evaluate_aec.py --input recorded.wav --reference loopback.wav --output results.json
```

### Run Noise Reduction Evaluation

```bash
cd benchmarks/noise_reduction
python evaluate_nr.py --before noisy.wav --after cleaned.wav --sample-rate 16000
```

## Configuration

See [`config/example_config.yaml`](config/example_config.yaml) for a complete reference of system parameters including AEC filter settings, VAD thresholds, noise reduction scene profiles, ASR configuration, and question detection parameters.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{liu2026realtime,
  title={Real-Time {AI}-Assisted Meeting Intelligence: A Dual-Stream Audio Architecture with Context-Aware Response Generation},
  author={Liu, Anwei},
  journal={arXiv preprint arXiv:2026.XXXXX},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Anwei Liu — HELPM AI INC

For questions about the paper or this repository, please open an issue on GitHub.
