# Real-Time Intelligent Cable-Harness Test Bench: FPGA, Edge AI, CPS & IIoT

This project develops a prototype intelligent cable-harness monitoring system combining deterministic FPGA-based signal processing, feature-based Edge AI classification, a lightweight Digital Twin, and Industrial IoT monitoring. The complete pipeline connects physical signal acquisition to embedded detection, machine-learning inference, system-state representation, data storage, and visualization.

---

## Core Architecture

```text
Physical Signal Acquisition
            │
            ▼
┌─────────────────────────────┐
│ Real-Time FPGA Processing   │
│ Detection + DSP             │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Feature Extraction          │
│ 20-dimensional vector       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Edge AI Classification      │
│ Random Forest / DT / LR     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ ONNX Runtime Edge Inference │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Lightweight Digital Twin    │
│ System State + Health Score │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ IIoT Monitoring             │
│ MQTT → FastAPI → InfluxDB   │
│ → Grafana                   │
└─────────────────────────────┘
```

---

# Key Results

## FPGA Implementation

| Metric                   |                      Result |
| ------------------------ | --------------------------: |
| Target FPGA              | AMD/Xilinx Artix-7 XC7A100T |
| System clock             |                     100 MHz |
| Processing clock         |                      40 MHz |
| Worst Negative Slack     |                   +0.093 ns |
| Failing timing endpoints |                           0 |
| LUT utilization          |                       1.17% |
| Flip-flop utilization    |                       0.82% |
| DSP48 utilization        |                       0.83% |

The FPGA implementation achieved positive timing margin with low resource utilization, providing a deterministic hardware layer for signal-change detection.

---

## Edge AI Classification

The current training pipeline uses **1,000 scenario-generated samples** represented by **20 extracted features**, with an **800/200 scenario-level train/test split**.

| Model               | Accuracy | Macro F1 |  Latency | Model Size |
| ------------------- | -------: | -------: | -------: | ---------: |
| Random Forest       |   94.00% |   0.9323 | 7.190 ms |   748.9 KB |
| Decision Tree       |   93.50% |   0.9277 | 0.157 ms |     8.7 KB |
| Logistic Regression |   78.50% |   0.7769 | 0.169 ms |     1.3 KB |

### Random Forest Per-Class F1

| Class     | F1-score |
| --------- | -------: |
| normal    |    0.838 |
| micro_cut |    1.000 |
| noise     |    1.000 |
| transient |    0.824 |
| abrupt    |    1.000 |

The results show strong classification performance under the current controlled scenario-based evaluation, while also revealing a specific normal/transient classification challenge.

These classification values reflect the latest run of the training pipeline; the technical report contains an earlier iteration of these metrics.

---

## Edge Inference

| Metric                      |      Result |
| --------------------------- | ----------: |
| Mean ONNX inference latency | 0.014197 ms |
| Median latency              | 0.012600 ms |
| P95 latency                 | 0.023400 ms |
| Maximum observed latency    | 0.415100 ms |

The benchmark characterizes the model inference stage on the tested software environment; it is not a measurement of complete end-to-end system latency.

---

## End-to-End Monitoring Validation

| Parameter                         |  Result |
| --------------------------------- | ------: |
| Controlled events                 |      20 |
| Event classes                     |       5 |
| Events per class                  |       4 |
| FPGA detections                   |       6 |
| Initial Digital Twin health score |    100% |
| Final Digital Twin health score   |   75.2% |
| Final system status               | WARNING |

The controlled experiment verified the information path from Edge AI through MQTT, Digital Twin/state management, FastAPI, InfluxDB, and Grafana.

---

# Development / Workflow Flow

```text
MATLAB / Simulink
        │
        ▼
Reference Signal Processing
        │
        ├──────────────► FPGA / VHDL Implementation
        │                         │
        │                         ▼
        │                   Vivado Verification
        │
        ▼
Feature Extraction
        │
        ▼
Python / scikit-learn
        │
        ├── Random Forest
        ├── Decision Tree
        └── Logistic Regression
        │
        ▼
Model Evaluation
        │
        ▼
ONNX Export
        │
        ▼
ONNX Runtime Verification
        │
        ▼
Digital Twin / CPS State
        │
        ▼
MQTT / FastAPI
        │
        ├──► InfluxDB
        │
        └──► Grafana
        │
        ▼
Docker Compose Integration
        │
        ▼
End-to-End Validation
```

---

# Why This Architecture Is Transferable

Although the demonstrated application is cable-harness monitoring, the underlying processing method is not inherently limited to cable harnesses.

The combination of deterministic signal processing, compact feature extraction, Edge AI inference, system-state representation, and IIoT monitoring could be adapted to other signal-driven cyber-physical systems.

Such adaptation would require the sensing interface, feature set, training data, classification models, and validation procedures to be redesigned for the target system rather than assuming that the current results transfer directly.

---

# Potential Applications

## 1. Industrial Predictive Maintenance

The same architecture could be adapted for condition monitoring and fault diagnosis of industrial equipment such as motors, actuators, rotating machinery, or other sensor-equipped assets.

Potential future directions include:

* vibration-based fault detection;
* current/voltage anomaly detection;
* machine-condition classification;
* early fault diagnosis;
* edge-based predictive maintenance.

The existing separation between deterministic processing and Edge AI could provide a basis for investigating low-latency local diagnostics, while the Digital Twin and IIoT layers could be redesigned for equipment health and maintenance information.

## 2. Automotive, Electric Vehicles & Power Electronics

The methodology could also be adapted to embedded monitoring of automotive and electrical systems.

Potential directions include:

* ECU condition monitoring;
* electrical fault diagnosis;
* EV power-electronics monitoring;
* converter/inverter fault detection;
* sensor anomaly detection;
* real-time vehicle-system health monitoring.

These applications would require domain-specific signals, datasets, features, and validation under realistic operating conditions.

---

# Current Limitations

### Dataset Scope

The current ML dataset is generated from predefined scenarios rather than a large collection of experimentally measured field faults. Therefore, the classification results characterize the current experimental methodology rather than industrial field performance.

### Physical Validation

Broader validation with different cable types, conductor lengths, load conditions, degradation levels, connector conditions, and environmental disturbances remains future work.

### FPGA Scope

The current FPGA implementation focuses on deterministic event detection. The timestamp mechanism remains available at the reference/Simulink level but was not retained as a final external FPGA output.

### Digital Twin Scope

The implemented Digital Twin is a lightweight software representation of the monitored condition. It does not model complete physical behavior or provide closed-loop physical control.

### IIoT Scalability

The monitoring infrastructure was validated locally using Docker Compose. Larger data volumes, longer operating periods, network interruptions, multiple devices, and distributed deployment remain to be evaluated.

### Generalization

The proposed extensions to other application domains are research directions; they have not been experimentally validated within this project.

---

# Technology Stack

## Hardware & Digital Design

* AMD/Xilinx Artix-7 XC7A100T
* VHDL
* MATLAB / Simulink
* Fixed-point signal processing
* Vivado

## Machine Learning

* Python
* scikit-learn
* Random Forest
* Decision Tree
* Logistic Regression
* Feature extraction
* Classification evaluation

## Edge AI

* ONNX
* ONNX Runtime
* Model benchmarking
* Low-latency inference

## CPS / Digital Twin

* Lightweight Digital Twin
* System-state representation
* Health-score management
* FPGA event information
* Classification confidence

## Industrial IoT

* MQTT / Mosquitto
* FastAPI
* InfluxDB
* Grafana
* Docker Compose

---

# Project Status

| Component                                   | Status                         |
| ------------------------------------------- | ------------------------------ |
| Signal-processing reference model           | Implemented                    |
| FPGA signal-processing and detection        | Implemented / Verified         |
| FPGA timing closure                         | Verified                       |
| FPGA resource evaluation                    | Verified                       |
| Feature extraction                          | Implemented                    |
| ML model comparison                         | Implemented / Verified         |
| Random Forest classification                | Implemented / Verified         |
| Decision Tree classification                | Implemented / Verified         |
| Logistic Regression classification          | Implemented / Verified         |
| ONNX export                                 | Implemented / Verified         |
| ONNX inference benchmark                    | Verified                       |
| Lightweight Digital Twin                    | Implemented                    |
| MQTT communication                          | Implemented / Verified locally |
| FastAPI interface                           | Implemented / Verified locally |
| InfluxDB storage                            | Implemented / Verified locally |
| Grafana monitoring                          | Implemented / Verified locally |
| Docker Compose integration                  | Implemented / Verified locally |
| Controlled end-to-end validation            | Verified                       |
| Large-scale physical dataset                | Future work                    |
| Broader experimental fault validation       | Future work                    |
| Advanced Edge AI / model compression        | Future work                    |
| Hardware-aware AI acceleration              | Future work                    |
| Richer Digital Twin / degradation modelling | Future work                    |
| Multi-device / large-scale IIoT deployment  | Future work                    |

---

# Project Perspective

The project demonstrates a prototype architecture in which **embedded hardware, digital signal processing, machine learning, Edge AI, CPS state representation, and IIoT monitoring** are connected into one workflow.

The cable-harness test bench serves as the current experimental case, while the architecture provides a basis for future investigation of intelligent, low-latency monitoring and fault-diagnosis systems in other signal-driven cyber-physical applications.

