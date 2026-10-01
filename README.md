# Intelligent Edge AI + FPGA Cyber-Physical System for Real-Time Fault Detection

## Overview

This project develops an **Edge AI and Cyber-Physical System (CPS) architecture** for real-time monitoring, fault detection, classification, and system-state management.

The current implementation uses a **cable-harness monitoring scenario** as a concrete experimental case. The broader architecture is designed around the integration of:

* deterministic FPGA signal processing;
* feature-based Edge AI;
* low-latency inference;
* Digital Twin-oriented state representation;
* Industrial IoT communication;
* persistent system monitoring.

The project therefore goes beyond a conventional machine-learning classification experiment by investigating the complete path from a physical signal to an embedded decision and finally to a system-level representation.

---

## Core Architecture

```text
Physical Signal
      │
      ▼
┌──────────────────────┐
│ Real-Time FPGA       │
│ Signal Processing    │
│ & Fault Detection    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Feature Extraction   │
│ 20-Dimensional       │
│ Feature Vector       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Edge AI              │
│ Classification       │
│ Random Forest / DT   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ ONNX Runtime         │
│ Edge Inference       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Digital Twin / CPS   │
│ System State         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ MQTT + FastAPI       │
│ InfluxDB + Grafana   │
└──────────┬───────────┘
           │
           ▼
     Monitoring Layer
```

The architecture separates **deterministic low-level processing** from **higher-level intelligent interpretation and system monitoring**.

---

# 1. Project Objectives

The project investigates a complete intelligent monitoring pipeline capable of:

* real-time FPGA-based signal processing;
* hardware-level event and fault detection;
* extraction of compact signal descriptors;
* machine-learning-based condition classification;
* low-latency Edge AI inference;
* ONNX-based deployment;
* Cyber-Physical System state representation;
* Digital Twin-oriented monitoring;
* Industrial IoT communication;
* persistent time-series monitoring;
* containerized deployment.

### Complete Workflow

```text
Signal Acquisition
      ↓
FPGA Processing
      ↓
Event Detection
      ↓
Feature Extraction
      ↓
Machine Learning
      ↓
Edge Inference
      ↓
Digital Twin
      ↓
IIoT Communication
      ↓
Monitoring & Visualization
```

---

# 2. FPGA Real-Time Processing

The FPGA forms the deterministic real-time processing layer.

### Target Platform

* **AMD/Xilinx Artix-7 XC7A100T**
* 100 MHz system clock
* 40 MHz processing clock
* Fixed-point arithmetic
* Real-time signal processing
* Threshold-based event detection
* Temporal validation

### Processing Chain

```text
Input Signal
     ↓
FIR Filtering
     ↓
Fixed-Point Conversion
     ↓
Unit Delay
     ↓
Difference
     ↓
Absolute Value
     ↓
Gain
     ↓
Threshold Comparison
     ↓
Event Detection
```

The FPGA stage provides deterministic low-level information that is subsequently combined with higher-level feature-based Edge AI.

This creates a hardware/software architecture in which:

* **FPGA** → deterministic signal processing;
* **Edge AI** → higher-level interpretation;
* **CPS layer** → system-state representation.

---

# 3. Feature Extraction

Instead of directly providing the complete raw waveform to the machine-learning model, each signal window is transformed into a compact feature representation.

Each **256-sample signal window** is represented by:

**20 features**

including:

* statistical descriptors;
* amplitude-related characteristics;
* waveform variations;
* temporal characteristics;
* FPGA-derived information.

### Representation

```text
256 Raw Samples
       ↓
Feature Extraction
       ↓
20-Dimensional Feature Vector
```

The feature-extraction layer provides a fixed and reproducible interface between the signal-processing and machine-learning stages.

---

# 4. Edge AI Classification

The current experimental dataset contains:

| Parameter                  | Value |
| -------------------------- | ----: |
| Scenarios                  | 1,000 |
| Features per scenario      |    20 |
| Signal conditions          |     5 |
| Training scenarios         |   800 |
| Independent test scenarios |   200 |

### Signal Conditions

* `normal`
* `micro_cut`
* `noise`
* `transient`
* `abrupt`

A **scenario-level split** is used rather than randomly distributing samples from the same scenario between training and testing.

This provides a cleaner evaluation of the classification pipeline under the current controlled experimental setup.

---

# 5. Model Comparison

Three classification approaches were evaluated.

| Model               | Accuracy | Macro F1 |  Latency | Model Size |
| ------------------- | -------: | -------: | -------: | ---------: |
| Random Forest       |   94.00% |   0.9323 | 7.190 ms |   748.9 KB |
| Decision Tree       |   93.50% |   0.9277 | 0.157 ms |     8.7 KB |
| Logistic Regression |   78.50% |   0.7769 | 0.169 ms |     1.3 KB |

The comparison considers more than classification accuracy. It examines the trade-off between:

* classification performance;
* inference latency;
* model size;
* deployment suitability.

The results indicate that the **Decision Tree achieves performance close to Random Forest while requiring substantially less memory and computational latency**, illustrating an important design trade-off for constrained Edge AI systems.

---

# 6. Random Forest Results

The following results were obtained using the training pipeline described in the technical report.

## Overall Performance

| Metric          |     Result |
| --------------- | ---------: |
| Accuracy        | **94.00%** |
| Macro Precision | **0.9379** |
| Macro Recall    | **0.9352** |
| Macro F1        | **0.9323** |

> **Note:** the metrics in this table are the latest verified results from the current training script; the corresponding confusion-matrix values in the attached technical report reflect an earlier training iteration of the same pipeline.

## Per-Class F1

| Class     | F1-score |
| --------- | -------: |
| normal    |    0.838 |
| micro_cut |    1.000 |
| noise     |    1.000 |
| transient |    0.824 |
| abrupt    |    1.000 |

The main classification difficulty occurs between the `normal` and `transient` classes.

In the evaluated test subset:

* 10 transient samples were classified as normal;
* 2 normal samples were classified as transient.

This provides a concrete direction for further investigation rather than treating the classification result as a final solution.

---

# 7. Edge Deployment

The trained model is exported to **ONNX** and executed using **ONNX Runtime**.

### Deployment Pipeline

```text
Feature Vector
      ↓
ONNX Model
      ↓
Edge Inference
      ↓
Predicted Condition
      ↓
Confidence / System State
```

### ONNX Verification

```text
ONNX Export       : PASSED
ONNX Verification : PASSED
```

The deployed model expects:

```text
Input shape : [None, 20]
Input type  : float32
```

This provides a clear interface between model development and edge deployment.

---

# 8. Edge Inference Benchmark

Inference performance was evaluated using repeated ONNX Runtime executions following a warm-up phase.

| Metric                   |      Result |
| ------------------------ | ----------: |
| Mean latency             | 0.014197 ms |
| Median latency           | 0.012600 ms |
| P95 latency              | 0.023400 ms |
| Maximum observed latency | 0.415100 ms |

These values characterize the **computational inference stage** under the experimental execution environment.

They should **not** be interpreted as complete end-to-end system latency, since acquisition, FPGA processing, communication, database operations, and visualization are separate stages.

---

# 9. Cyber-Physical System & Digital Twin

The Edge AI output is integrated into a lightweight **CPS-oriented representation** of the monitored system.

The Digital Twin representation maintains information such as:

* operating state;
* health score;
* detected events;
* classification results;
* confidence information;
* FPGA-related detections.

### Concept

```text
Physical System
      ↕
Signal Processing
      ↕
Edge Intelligence
      ↕
Digital Representation
```

The current implementation should be considered a **lightweight Digital Twin / CPS-oriented architecture**, rather than a complete industrial-scale Digital Twin platform.

---

# 10. Industrial IoT Layer

The project extends the local Edge AI decision toward system-level monitoring.

### Monitoring Stack

* MQTT — communication
* FastAPI — REST interfaces
* InfluxDB — time-series storage
* Grafana — visualization
* Docker Compose — integrated deployment

### Software Flow

```text
Edge AI
   ↓
 MQTT
   ↓
FastAPI
   ↓
InfluxDB
   ↓
Grafana
```

This transforms an isolated ML prediction into an observable system-level event.

---

# 11. Why the Architecture Is Generalizable

Although the current experimental case focuses on cable-harness monitoring, the underlying methodology is not inherently limited to cable harnesses.

The reusable concept is:

```text
Physical Signal
      ↓
Deterministic Processing
      ↓
Feature Representation
      ↓
Edge Intelligence
      ↓
System State
      ↓
Monitoring
```

This architecture could potentially be adapted to other systems where:

* sensor signals must be processed locally;
* abnormal conditions must be detected;
* decisions must be generated with low computational latency;
* connectivity to higher-level monitoring systems is required.

Adaptation to a new domain would require **domain-specific sensors, datasets, features, models, and validation**.

---

# 12. Potential Application Domains

The following represent **potential research directions**, not applications experimentally validated by the current project.

## Industrial Predictive Maintenance

Potential applications include:

* motor condition monitoring;
* bearing fault detection;
* machine anomaly detection;
* actuator monitoring;
* industrial equipment health monitoring.

The same feature-based Edge AI concept could potentially be adapted to vibration, current, voltage, acoustic, or other sensor signals.

## Power Electronics & Electrical Systems

Potential directions include:

* converter fault detection;
* inverter monitoring;
* electrical transient classification;
* power-quality event detection;
* abnormal operating-condition detection.

The combination of FPGA-based real-time processing and Edge AI is relevant where deterministic signal processing must be combined with higher-level intelligent interpretation.

## Automotive & Electric Vehicles

Potential applications include:

* ECU health monitoring;
* electrical fault diagnosis;
* vehicle sensor anomaly detection;
* battery-system monitoring;
* EV power-electronics monitoring;
* real-time vehicle condition assessment.

The underlying principle remains:

```text
Vehicle Signal
      ↓
Embedded Processing
      ↓
Edge AI
      ↓
Vehicle Health State
```

## Industrial IoT & Smart Manufacturing

Potential applications include:

* machine monitoring;
* production-line anomaly detection;
* distributed equipment diagnostics;
* edge-based condition monitoring;
* predictive maintenance.

The existing MQTT, API, time-series database, and visualization architecture provides a potential foundation for distributed monitoring systems.

## Robotics & Autonomous Systems

Potential directions include:

* sensor anomaly detection;
* actuator monitoring;
* motor fault detection;
* embedded health monitoring;
* abnormal behavior detection.

In such systems, local inference can be useful when decisions need to be generated close to the physical process.

---

# 13. Research-Oriented Extensions

Several research directions emerge naturally from the current implementation.

## Larger Experimental Datasets

The current dataset is **scenario-generated and controlled**.

A stronger experimental evaluation could incorporate measurements covering:

* different hardware configurations;
* operating conditions;
* fault severities;
* noise levels;
* degradation levels;
* environmental conditions;
* real measured fault events.

The objective would not simply be to increase the number of samples, but to increase the **diversity and realism of the experimental conditions**.

## Hardware-Aware Edge AI

Future work could investigate:

* FPGA-based feature extraction;
* hardware/software co-design;
* model compression;
* quantization;
* lightweight classifiers;
* FPGA or embedded accelerator deployment;
* resource/accuracy/latency trade-offs.

## Reliability-Aware Intelligent Monitoring

The architecture could also be extended toward:

* fault-tolerant monitoring;
* uncertainty-aware classification;
* anomaly detection;
* early fault diagnosis;
* multi-stage fault detection.

## Digital Twin Integration

Future versions could develop a richer Digital Twin capable of:

* historical state tracking;
* degradation estimation;
* predictive maintenance;
* what-if analysis;
* synchronization with physical measurements.

---

# 14. Current Limitations

The current results should be interpreted within the scope of the experimental setup.

The principal limitations are:

1. The dataset is scenario-generated rather than a large collection of experimentally measured field data.
2. The current experiments cover a controlled set of signal conditions.
3. The Edge inference benchmark characterizes model execution rather than complete end-to-end system latency.
4. The Digital Twin is intentionally lightweight.
5. Generalization to other physical systems has not yet been experimentally validated.

These limitations define potential next research steps rather than invalidating the current prototype.

---

# 15. Reproducibility

The repository contains the main components required to reproduce the Edge AI experiment.

The training pipeline includes:

* dataset validation;
* scenario-level train/test splitting;
* feature processing;
* model comparison;
* classification metrics;
* confusion matrices;
* feature importance;
* ONNX export;
* ONNX verification.

### Generated Artifacts

```text
model.pkl
model.onnx
model_comparison.png
confusion_matrices.png
feature_importance.png
```

---

# 16. Technology Stack

## Hardware & Digital Design

* AMD/Xilinx Artix-7 FPGA
* VHDL
* MATLAB / Simulink
* HDL-oriented design flow
* Fixed-point DSP

## Machine Learning

* Python
* scikit-learn
* Random Forest
* Decision Tree
* Logistic Regression
* Feature engineering
* Classification evaluation

## Edge Deployment

* ONNX
* ONNX Runtime
* Low-latency inference

## CPS / IIoT

* Digital Twin
* MQTT
* FastAPI
* InfluxDB
* Grafana
* Docker Compose

---

# 17. Project Perspective

This project started from a concrete fault-detection problem, but its main technical value lies in the integration of several layers:

```text
Embedded Hardware
        +
Digital Signal Processing
        +
Machine Learning
        +
Edge AI
        +
Cyber-Physical Systems
        +
Digital Twins
        +
Industrial IoT
```

The cable-harness application provides a controlled environment in which this architecture can be developed and evaluated.

The broader objective is to investigate how **real-time embedded processing and intelligent edge-based decision making** can be combined to create responsive, resource-aware, and monitorable cyber-physical systems.

---

## Project Status

| Component                         | Status                     |
| --------------------------------- | -------------------------- |
| FPGA signal-processing pipeline   | Implemented                |
| Fault/event detection             | Implemented                |
| Feature extraction                | Implemented                |
| ML model comparison               | Implemented                |
| Random Forest classification      | Implemented                |
| ONNX export                       | Verified                   |
| ONNX Runtime inference            | Benchmarked                |
| CPS-oriented state representation | Implemented                |
| MQTT communication layer          | Architecture / integration |
| FastAPI                           | Architecture / integration |
| InfluxDB                          | Architecture / integration |
| Grafana                           | Architecture / integration |
| Docker Compose                    | Integration layer          |
| Large-scale real-world validation | Future work                |

---

## Key Takeaway

The project demonstrates a complete prototype architecture connecting:

**physical signal → deterministic FPGA processing → compact feature representation → Edge AI classification → ONNX inference → CPS/Digital Twin state → IIoT monitoring.**

The current cable-harness experiment serves as a concrete validation case, while the architecture provides a basis for future research into **hardware-aware Edge AI, intelligent fault diagnosis, embedded systems, and cyber-physical monitoring**.
