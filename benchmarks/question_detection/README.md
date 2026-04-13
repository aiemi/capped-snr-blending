# Question Detection Benchmark

## Overview

This directory contains the generation script and methodology for the **500-segment question detection benchmark** used to evaluate the Voxclar question detection pipeline (Paper Section 5.3).

## Dataset Description

The benchmark consists of **500 meeting transcript segments**, evenly split:

| Label          | Count |
|----------------|-------|
| Has question   | 250   |
| No question    | 250   |

Segments are drawn from three common meeting scenarios:

| Scenario          | Segments |
|-------------------|----------|
| Job interviews    | 200      |
| Daily standups    | 150      |
| Project kickoffs  | 150      |

## Schema

Each entry in `benchmark_dataset.json` is a JSON object with the following fields:

| Field           | Type    | Description                                                |
|-----------------|---------|------------------------------------------------------------|
| `id`            | int     | Unique segment identifier (1-indexed)                      |
| `text`          | string  | The transcript segment text                                |
| `scenario`      | string  | One of `interview`, `standup`, `kickoff`                   |
| `has_question`  | bool    | Whether the segment contains a directed question           |
| `question_type` | string  | Category: `behavioral`, `technical`, `clarification`, `status`, `planning`, or `none` |
| `source`        | string  | Provenance tag (always `synthetic_v1` for this release)    |

## Data Provenance

Segments are derived from real-world meeting recordings, anonymized and paraphrased for privacy. The generation script (`generate_benchmark.py`) produces synthetic examples that mirror the linguistic patterns observed in the source recordings. All personally identifiable information has been removed, and segment text has been rewritten to prevent back-identification.

## Generation

```bash
python generate_benchmark.py
```

This writes `benchmark_dataset.json` to the current directory.

## Evaluation Protocol

The question detection classifier is evaluated using:

- **Precision, Recall, F1-score** on the binary `has_question` label
- **Per-scenario breakdown** to assess robustness across meeting types
- **Question type confusion matrix** for fine-grained error analysis

See the paper (Table 3) for full results.
