# Dataset

This directory contains the dataset generation pipeline used for the Edge AI classification stage of the intelligent cable-harness monitoring system.

The dataset is designed to remain consistent with the FPGA-based detection stage and the subsequent feature-extraction and Edge AI pipeline.

## Current Dataset

The current dataset is an experimental, reproducible dataset generated from signal scenarios representing different cable-harness conditions.

### Dataset configuration

| Parameter              |       Value |
| ---------------------- | ----------: |
| Number of scenarios    |       1,000 |
| Samples per scenario   |         256 |
| Sampling frequency     |       1 MHz |
| Observation window     |      256 µs |
| Classes                |           5 |
| Feature representation | 20 features |
| Training scenarios     |         800 |
| Test scenarios         |         200 |

The five classes are:

* `normal`
* `micro_cut`
* `noise`
* `transient`
* `abrupt`

Each signal window is converted into a fixed 20-dimensional feature vector combining:

* statistical and amplitude characteristics,
* waveform and temporal characteristics,
* FPGA-derived detection information.

The same feature representation is maintained during training and Edge AI inference.

## Dataset Generation

The dataset is generated programmatically using:

```text
generate_dataset.py
```

The generated files provide the signal data, class labels, scenario identifiers, and FPGA-related metadata required by the feature-extraction and machine-learning pipelines.

The scenario identifiers are also used to perform a scenario-level train/test split, preventing observations from the same scenario from being distributed across both subsets.

## Role in the Edge AI Pipeline

The dataset provides the interface between the deterministic FPGA processing stage and the machine-learning classifier:

```text
Signal Windows
      │
      ▼
Feature Extraction
      │
      ▼
20-Dimensional Feature Vector
      │
      ▼
Random Forest
      │
      ▼
ONNX Runtime
      │
      ▼
Edge AI Prediction
```

This organization allows the machine-learning component to use a compact representation of the signal rather than directly processing the complete raw waveform.

## Research Direction: Dataset Expansion

The current dataset is intended as a **reproducible experimental foundation**, not as a final representation of all real-world cable-harness conditions.

A major next step is to expand the dataset substantially by increasing the number and diversity of generated scenarios and operating conditions.

Future dataset development will progressively include:

* larger numbers of signal scenarios;
* greater variation in signal and fault characteristics;
* different operating conditions;
* additional degradation patterns;
* broader disturbance and noise conditions;
* more diverse cable-harness configurations.

## Toward a Real Experimental Dataset

A further research direction is the development of a **real-world experimental dataset** obtained from physical cable-harness experiments.

Future measurements can investigate factors such as:

* different cable types and lengths;
* connector conditions;
* load conditions;
* degradation levels;
* physical fault conditions;
* environmental disturbances.

The objective is to move from the current reproducible experimental dataset toward a larger dataset combining **controlled generated data with measurements acquired from physical cable-harness experiments**.

This future dataset would provide a stronger basis for evaluating Edge AI models under conditions closer to real industrial operation, while enabling further research into robust fault classification and intelligent cable-harness diagnostics.

## Reproducibility

The dataset generation process is included in this repository so that the experimental data can be regenerated and the complete Edge AI pipeline can be reproduced from signal generation through feature extraction and model training.

