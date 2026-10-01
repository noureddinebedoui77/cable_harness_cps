
"""
benchmark_edge_vs_cloud.py

Research-oriented benchmark comparing local Edge AI inference with
a simulated Cloud inference configuration.

The Edge benchmark measures ONNX Runtime inference on the machine
where this script is executed. It must not be reported as Raspberry
Pi 5 performance unless the benchmark is actually run on a Raspberry Pi 5.

The Cloud configuration is a simulation based on fixed assumptions:
    Network latency  = 20 ms
    Cloud processing = 5 ms

The benchmark does not modify the trained model, dataset, FPGA
parameters, or feature-extraction pipeline.
"""

import pickle
import time
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).parent.parent / "data"
AI_DIR = Path(__file__).parent


# ============================================================================
# BENCHMARK PARAMETERS
# ============================================================================

WARMUP_RUNS = 1000
BENCHMARK_RUNS = 10000

# Simulated Cloud assumptions.
CLOUD_NETWORK_LATENCY_MS = 20.0
CLOUD_PROCESSING_MS = 5.0


# ============================================================================
# EDGE BENCHMARK
# ============================================================================

def benchmark_edge():
    """
    Benchmark ONNX Runtime inference on the current host machine.

    Only the ONNX inference stage is timed. Feature extraction is
    performed before the benchmark so that the measured latency
    represents model inference rather than feature-engineering overhead.
    """

    import onnxruntime as ort

    X_signals = np.load(
        DATA_DIR / "X_signals.npy"
    )

    with open(
        DATA_DIR / "fpga_infos.pkl",
        "rb"
    ) as file:
        fpga_infos = pickle.load(file)

    model_path = AI_DIR / "model.onnx"

    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}\n"
            "Run train_model.py first."
        )

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    from extract_features import extract_features

    # Prepare one representative input.
    sample = X_signals[0:1]

    features = extract_features(
        sample,
        [fpga_infos[0]]
    ).astype(np.float32)

    # Warm-up the inference runtime before measurement.
    for _ in range(WARMUP_RUNS):
        session.run(
            None,
            {input_name: features}
        )

    # Measure individual inference times at nanosecond resolution.
    latencies_ns = np.empty(
        BENCHMARK_RUNS,
        dtype=np.int64
    )

    for i in range(BENCHMARK_RUNS):

        start_ns = time.perf_counter_ns()

        session.run(
            None,
            {input_name: features}
        )

        end_ns = time.perf_counter_ns()

        latencies_ns[i] = end_ns - start_ns

    latencies_ms = latencies_ns / 1_000_000.0

    mean_ms = np.mean(latencies_ms)
    median_ms = np.median(latencies_ms)
    std_ms = np.std(latencies_ms)

    min_ms = np.min(latencies_ms)
    max_ms = np.max(latencies_ms)

    p95_ms = np.percentile(latencies_ms, 95)
    p99_ms = np.percentile(latencies_ms, 99)

    throughput_hz = 1000.0 / mean_ms

    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "throughput_hz": throughput_hz,
        "runs": BENCHMARK_RUNS,
        "warmup_runs": WARMUP_RUNS,
    }


# ============================================================================
# CLOUD SIMULATION
# ============================================================================

def benchmark_cloud_simulation():
    """
    Simulate an end-to-end Cloud inference path.

    The simulation assumes:
        Network latency  = 20 ms
        Cloud processing = 5 ms

    These values are predefined assumptions, not measurements of
    a real Cloud service.
    """

    network_latency = CLOUD_NETWORK_LATENCY_MS
    cloud_processing = CLOUD_PROCESSING_MS

    total_latency = (
        network_latency +
        cloud_processing
    )

    return {
        "mean_ms": total_latency,
        "network_ms": network_latency,
        "processing_ms": cloud_processing,
        "throughput_hz": 1000.0 / total_latency,
    }


# ============================================================================
# RESULTS
# ============================================================================

def print_comparison(edge, cloud):
    """Display the Edge benchmark and simulated Cloud comparison."""

    print("=" * 70)
    print("EDGE AI vs CLOUD BENCHMARK")
    print("=" * 70)

    print("\n--- EDGE AI (ONNX Runtime on Current Host) ---")

    print(
        f"  Warm-up runs       : "
        f"{edge['warmup_runs']}"
    )

    print(
        f"  Benchmark runs     : "
        f"{edge['runs']}"
    )

    print(
        f"  Mean latency       : "
        f"{edge['mean_ms']:.6f} ms"
    )

    print(
        f"  Median latency     : "
        f"{edge['median_ms']:.6f} ms"
    )

    print(
        f"  Standard deviation : "
        f"{edge['std_ms']:.6f} ms"
    )

    print(
        f"  Min latency        : "
        f"{edge['min_ms']:.6f} ms"
    )

    print(
        f"  Max latency        : "
        f"{edge['max_ms']:.6f} ms"
    )

    print(
        f"  P95 latency        : "
        f"{edge['p95_ms']:.6f} ms"
    )

    print(
        f"  P99 latency        : "
        f"{edge['p99_ms']:.6f} ms"
    )

    print(
        f"  Throughput         : "
        f"{edge['throughput_hz']:.2f} inferences/sec"
    )

    print("\n--- CLOUD (SIMULATED) ---")

    print(
        f"  Network latency    : "
        f"{cloud['network_ms']:.1f} ms"
    )

    print(
        f"  Cloud processing   : "
        f"{cloud['processing_ms']:.1f} ms"
    )

    print(
        f"  Total latency      : "
        f"{cloud['mean_ms']:.1f} ms"
    )

    print(
        f"  Throughput         : "
        f"{cloud['throughput_hz']:.2f} inferences/sec"
    )

    print("\n--- COMPARISON ---")

    speedup = (
        cloud["mean_ms"] /
        edge["mean_ms"]
    )

    latency_saved = (
        cloud["mean_ms"] -
        edge["mean_ms"]
    )

    print(
        f"  Simulated Cloud / Edge latency ratio : "
        f"{speedup:.2f}x"
    )

    print(
        f"  Latency difference                   : "
        f"{latency_saved:.3f} ms"
    )

    print(
        "  Network transmission of the raw "
        "256-sample signal is avoided for local inference."
    )

    print("\n--- RESEARCH INTERPRETATION ---")

    print(
        "  The ONNX model demonstrates low-latency "
        "local inference on the current host."
    )

    print(
        "  The Cloud comparison is a simulation based "
        "on predefined network and processing assumptions."
    )

    print(
        "  Raspberry Pi 5 performance must be measured "
        "separately on an actual Raspberry Pi 5."
    )

    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RESEARCH-ORIENTED EDGE AI BENCHMARK")
    print("=" * 70)

    print("\nBenchmark configuration:")

    print(
        f"  Warm-up runs   : {WARMUP_RUNS}"
    )

    print(
        f"  Measured runs  : {BENCHMARK_RUNS}"
    )

    print(
        "  Timer          : perf_counter_ns()"
    )

    print(
        "  Edge platform  : Current host machine"
    )

    print(
        "  Cloud          : Simulation"
    )

    print()

    edge_results = benchmark_edge()

    cloud_results = benchmark_cloud_simulation()

    print_comparison(
        edge_results,
        cloud_results
    )

