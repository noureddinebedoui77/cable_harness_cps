
"""
extract_features.py

Feature extraction pipeline for the Edge AI cable-harness monitoring system.

The model uses exactly 20 features in the fixed order defined by
FEATURE_NAMES. This order must remain unchanged because the trained
ONNX model was built using the same feature ordering.

Project configuration:
    Sampling rate : 1 MHz
    Window size   : 256 samples
    Features      : 20
    Classes       : 5

FPGA parameters:
    Gain      : 50.0
    Threshold : 9.5
"""

from pathlib import Path
import pickle

import numpy as np


# ============================================================================
# PROJECT PARAMETERS
# ============================================================================

SAMPLE_RATE_HZ = 1_000_000
WINDOW_SIZE = 256
FEATURE_COUNT = 20

FPGA_GAIN = 50.0
FPGA_THRESHOLD = 9.5

DT = 1.0 / SAMPLE_RATE_HZ


# ============================================================================
# CANONICAL FEATURE ORDER
# ============================================================================

FEATURE_NAMES = [
    "rms",
    "mean",
    "std",
    "peak_to_peak",
    "max_slope",
    "min_slope",
    "max_abs_slope",
    "zero_crossings",
    "signal_energy",
    "crest_factor",
    "event_duration",
    "threshold_crossings",
    "max_deviation",
    "time_to_peak",
    "slope_change",
    "fpga_triggered",
    "fpga_event_count",
    "fpga_max_duration",
    "fpga_max_diff",
    "fpga_trigger_time",
]


# ============================================================================
# BASIC SIGNAL FEATURES
# ============================================================================

def compute_basic_features(signal):
    """
    Compute the first 15 signal-domain features.

    Returns
    -------
    list of float
        Exactly 15 features.
    """

    x = np.asarray(signal, dtype=np.float64).reshape(-1)

    if x.size == 0:
        raise ValueError("Signal is empty.")

    mean_value = float(np.mean(x))
    std_value = float(np.std(x))

    rms = float(
        np.sqrt(np.mean(np.square(x)))
    )

    peak_to_peak = float(
        np.max(x) - np.min(x)
    )

    # First-order difference used for slope-related features.
    if len(x) >= 2:
        slope = np.diff(x)

        max_slope = float(np.max(slope))
        min_slope = float(np.min(slope))
        max_abs_slope = float(np.max(np.abs(slope)))
    else:
        slope = np.array([0.0], dtype=np.float64)

        max_slope = 0.0
        min_slope = 0.0
        max_abs_slope = 0.0

    # Count sign changes between consecutive samples.
    signs = np.sign(x)

    zero_crossings = (
        int(np.sum(signs[:-1] * signs[1:] < 0))
        if len(x) >= 2
        else 0
    )

    signal_energy = float(
        np.sum(np.square(x))
    )

    crest_factor = (
        float(np.max(np.abs(x)) / rms)
        if rms > 0
        else 0.0
    )

    # Duration of the complete analysis window.
    event_duration = float(len(x) * DT)

    # Match the FPGA detection condition:
    # |diff| * FPGA_GAIN > FPGA_THRESHOLD
    threshold = FPGA_THRESHOLD / FPGA_GAIN

    threshold_crossings = (
        int(np.sum(np.abs(slope) > threshold))
        if len(x) >= 2
        else 0
    )

    max_deviation = float(
        np.max(np.abs(x - mean_value))
    )

    peak_index = int(
        np.argmax(np.abs(x))
    )

    time_to_peak = float(
        peak_index * DT
    )

    if len(slope) >= 2:
        slope_change_values = np.diff(slope)

        slope_change = float(
            np.max(np.abs(slope_change_values))
        )
    else:
        slope_change = 0.0

    return [
        rms,
        mean_value,
        std_value,
        peak_to_peak,
        max_slope,
        min_slope,
        max_abs_slope,
        float(zero_crossings),
        signal_energy,
        crest_factor,
        event_duration,
        float(threshold_crossings),
        max_deviation,
        time_to_peak,
        slope_change,
    ]


# ============================================================================
# FPGA FEATURES
# ============================================================================

def compute_fpga_features(fpga_info, signal):
    """
    Compute the final five FPGA-domain features.

    Returns
    -------
    list of float
        Exactly 5 features.
    """

    if not isinstance(fpga_info, dict):
        fpga_info = {}

    fpga_triggered = bool(
        fpga_info.get("fpga_triggered", False)
    )

    fpga_event_count = int(
        fpga_info.get("fpga_event_count", 0)
    )

    fpga_max_duration = int(
        fpga_info.get(
            "fpga_max_event_duration",
            fpga_info.get("fpga_max_duration", 0)
        )
    )

    # Accept historical field names for compatibility with fpga_infos.pkl.
    fpga_max_diff = fpga_info.get(
        "fpga_max_diff",
        fpga_info.get(
            "max_diff",
            fpga_info.get("max_slope", 0.0)
        )
    )

    try:
        fpga_max_diff = float(fpga_max_diff)
    except (TypeError, ValueError):
        fpga_max_diff = 0.0

    # Stored as a sample index, not a Unix timestamp.
    fpga_trigger_index = fpga_info.get(
        "fpga_first_trigger_index",
        fpga_info.get("fpga_first_trigger", -1)
    )

    try:
        fpga_trigger_index = int(fpga_trigger_index)
    except (TypeError, ValueError):
        fpga_trigger_index = -1

    return [
        float(fpga_triggered),
        float(fpga_event_count),
        float(fpga_max_duration),
        float(fpga_max_diff),
        float(fpga_trigger_index),
    ]


# ============================================================================
# SINGLE-SIGNAL EXTRACTION
# ============================================================================

def extract_features_from_signal(signal, fpga_info=None):
    """
    Extract exactly 20 features from one signal.

    Parameters
    ----------
    signal : array-like
        One 256-sample signal.

    fpga_info : dict, optional
        FPGA metadata associated with the signal.

    Returns
    -------
    numpy.ndarray
        Feature vector with shape (20,).
    """

    x = np.asarray(signal, dtype=np.float64).reshape(-1)

    if x.size != WINDOW_SIZE:
        raise ValueError(
            "Signal window size mismatch.\n"
            f"Expected : {WINDOW_SIZE}\n"
            f"Received : {x.size}"
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "Signal contains NaN or Inf values."
        )

    if fpga_info is None:
        fpga_info = {}

    basic_features = compute_basic_features(x)
    fpga_features = compute_fpga_features(fpga_info, x)

    features = basic_features + fpga_features

    if len(features) != FEATURE_COUNT:
        raise RuntimeError(
            "Feature extraction produced the wrong number of features.\n"
            f"Expected : {FEATURE_COUNT}\n"
            f"Received : {len(features)}"
        )

    result = np.asarray(features, dtype=np.float32)

    if not np.all(np.isfinite(result)):
        raise ValueError(
            "Extracted features contain NaN or Inf."
        )

    return result


# ============================================================================
# BATCH EXTRACTION
# ============================================================================

def extract_features(X_signals, fpga_infos=None):
    """
    Main public feature-extraction interface.

    Parameters
    ----------
    X_signals : array-like
        One signal with shape (256,) or multiple signals with shape (N, 256).

    fpga_infos : list, dict, or None
        FPGA metadata corresponding to the input signal(s).

    Returns
    -------
    numpy.ndarray
        Shape (20,) for one signal or (N, 20) for multiple signals.

    Notes
    -----
    The interface is compatible with the existing mqtt_publisher.py
    pipeline.
    """

    X = np.asarray(X_signals, dtype=np.float64)

    # Single signal
    if X.ndim == 1:

        if X.size != WINDOW_SIZE:
            raise ValueError(
                "Signal window size mismatch.\n"
                f"Expected : {WINDOW_SIZE}\n"
                f"Received : {X.size}"
            )

        if fpga_infos is None:
            info = {}
        elif isinstance(fpga_infos, dict):
            info = fpga_infos
        elif len(fpga_infos) > 0:
            info = fpga_infos[0]
        else:
            info = {}

        return extract_features_from_signal(X, info)

    # Batch of signals
    if X.ndim != 2:
        raise ValueError(
            "X_signals must be 1D or 2D.\n"
            f"Received shape: {X.shape}"
        )

    if X.shape[1] != WINDOW_SIZE:
        raise ValueError(
            "Signal window size mismatch.\n"
            f"Expected : {WINDOW_SIZE}\n"
            f"Received : {X.shape[1]}"
        )

    n_samples = X.shape[0]

    # Normalize FPGA metadata to one entry per signal.
    if fpga_infos is None:
        infos = [{} for _ in range(n_samples)]

    elif isinstance(fpga_infos, dict):
        infos = [fpga_infos for _ in range(n_samples)]

    else:
        if len(fpga_infos) != n_samples:
            raise ValueError(
                "Number of FPGA metadata entries does not "
                "match number of signals.\n"
                f"Signals : {n_samples}\n"
                f"FPGA    : {len(fpga_infos)}"
            )

        infos = fpga_infos

    feature_matrix = np.empty(
        (n_samples, FEATURE_COUNT),
        dtype=np.float32
    )

    for index in range(n_samples):
        feature_matrix[index] = extract_features_from_signal(
            X[index],
            infos[index]
        )

    return feature_matrix


# ============================================================================
# SAVE FEATURE NAMES
# ============================================================================

def save_feature_names(output_file):
    """
    Save the canonical feature names used by the project.
    """

    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_file, "wb") as file:
        pickle.dump(FEATURE_NAMES, file)


# ============================================================================
# SELF TEST
# ============================================================================

def self_test():
    """
    Run a local interface and consistency test.

    The test does not modify the dataset or trained model.
    """

    print("=" * 70)
    print("FEATURE EXTRACTION SELF TEST")
    print("=" * 70)

    signal = np.zeros(
        WINDOW_SIZE,
        dtype=np.float64
    )

    # Small synthetic variation for interface validation.
    signal[100] = 1.0
    signal[101] = 0.5

    fpga_info = {
        "fpga_triggered": True,
        "fpga_event_count": 1,
        "fpga_max_event_duration": 2,
        "fpga_first_trigger_index": 100,
        "fpga_max_diff": 0.5,
    }

    features = extract_features(
        signal,
        [fpga_info]
    )

    print()
    print(f"Feature shape: {features.shape}")

    print()
    print("Features:")

    for index, (name, value) in enumerate(
        zip(FEATURE_NAMES, features)
    ):
        print(
            f"{index:02d}  "
            f"{name:<24} "
            f"{float(value):.8f}"
        )

    if features.shape != (FEATURE_COUNT,):
        raise RuntimeError(
            "Self-test failed: wrong feature shape."
        )

    if not np.all(np.isfinite(features)):
        raise RuntimeError(
            "Self-test failed: NaN/Inf detected."
        )

    print()
    print("=" * 70)
    print("SELF TEST PASSED")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    self_test()

