
"""
generate_dataset.py

Generate a synthetic dataset aligned with the FPGA-based cable-harness
monitoring system developed in Project 1.

The dataset uses:
- 1 MHz sampling frequency
- 256 samples per observation window
- FPGA-compatible fixdt(1,16,12) representation
- The same FPGA gain and detection threshold
- Five signal classes:
    Normal, Micro-cut, Noise, Transient, Abrupt discontinuity

FPGA trigger information is computed directly from the quantized signals.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


# ============================================================================
# SYSTEM PARAMETERS
# ============================================================================

SAMPLE_RATE = 1_000_000
DT = 1.0 / SAMPLE_RATE

# Signal frequency used in Project 1.
SIGNAL_FREQ_HZ = 638.32 / (2 * np.pi)  # ≈ 101.59 Hz

SIGNAL_AMP = 1.0
N_SAMPLES = 256
DURATION = N_SAMPLES * DT

# FPGA fixed-point format: fixdt(1,16,12)
FIXDT_BITS = 16
FIXDT_FRAC = 12
FIXDT_SCALE = 2 ** FIXDT_FRAC

FIXDT_MIN = -(2 ** (FIXDT_BITS - 1))
FIXDT_MAX = (2 ** (FIXDT_BITS - 1)) - 1


# ============================================================================
# FPGA DETECTION PARAMETERS
# ============================================================================

FPGA_GAIN = 50.0
FPGA_THRESHOLD = 9.5

# |diff| * 50 > 9.5  ->  |diff| > 0.19
FPGA_DIFF_THRESHOLD = FPGA_THRESHOLD / FPGA_GAIN


# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================

OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# FIXED-POINT QUANTIZATION
# ============================================================================

def quantize_fixdt_16_12(value):
    """
    Quantize a value using the FPGA fixdt(1,16,12) representation.

    The value is scaled by 2^12, rounded, saturated to the signed
    16-bit range, and converted back to its fixed-point real value.
    """

    int_val = np.round(value * FIXDT_SCALE)
    int_val = np.clip(int_val, FIXDT_MIN, FIXDT_MAX)

    return int_val / FIXDT_SCALE


def quantize_signal(signal):
    """Apply fixdt(1,16,12) quantization to a signal vector."""

    return np.array([quantize_fixdt_16_12(v) for v in signal])


# ============================================================================
# SIGNAL GENERATORS
# ============================================================================

def generate_normal(n_samples=N_SAMPLES, noise_level=0.005):
    """
    Generate a normal operating signal with low measurement noise.
    """

    t = np.linspace(0, DURATION, n_samples)

    signal = SIGNAL_AMP * np.sin(
        2 * np.pi * SIGNAL_FREQ_HZ * t
    )

    signal += np.random.normal(0, noise_level, n_samples)

    return quantize_signal(signal)


def generate_micro_cut(n_samples=N_SAMPLES):
    """
    Generate a short discontinuity by forcing a random signal segment to zero.

    The micro-cut duration is randomly selected between 1 and 14 samples.
    """

    signal = generate_normal(n_samples)

    cut_start = np.random.randint(
        n_samples // 4,
        3 * n_samples // 4
    )

    cut_duration = np.random.randint(1, 15)

    signal[cut_start:cut_start + cut_duration] = 0

    return quantize_signal(signal)


def generate_noise(n_samples=N_SAMPLES):
    """
    Generate a noisy/interference scenario with Gaussian noise and
    zero to two impulsive disturbances.
    """

    signal = generate_normal(
        n_samples,
        noise_level=0.02
    )

    n_impulses = np.random.randint(0, 3)

    for _ in range(n_impulses):

        pos = np.random.randint(10, n_samples - 10)
        amplitude = np.random.uniform(0.2, 0.6)

        signal[pos] += (
            np.random.choice([-1, 1]) * amplitude
        )

    return quantize_signal(signal)


def generate_transient(n_samples=N_SAMPLES):
    """
    Generate a transient disturbance using a damped high-frequency
    oscillatory component superimposed on the normal signal.
    """

    signal = generate_normal(n_samples)

    trans_start = np.random.randint(
        n_samples // 4,
        3 * n_samples // 4
    )

    trans_duration = np.random.randint(5, 40)
    trans_freq = np.random.uniform(500, 5000)
    amplitude = np.random.uniform(0.4, 1.2)

    t_trans = np.linspace(
        0,
        trans_duration * DT,
        trans_duration
    )

    transient = amplitude * np.sin(
        2 * np.pi * trans_freq * t_trans
    )

    envelope = np.exp(
        -np.linspace(0, 3, trans_duration)
    )

    transient *= envelope

    signal[
        trans_start:trans_start + trans_duration
    ] += transient

    return quantize_signal(signal)


def generate_abrupt(n_samples=N_SAMPLES):
    """
    Generate an abrupt discontinuity using one of three mechanisms:
    offset change, amplitude drop, or amplitude rise.
    """

    signal = generate_normal(n_samples)

    abrupt_point = np.random.randint(
        n_samples // 3,
        2 * n_samples // 3
    )

    change_type = np.random.choice([
        'offset',
        'amplitude_drop',
        'amplitude_rise'
    ])

    if change_type == 'offset':

        offset = (
            np.random.uniform(0.3, 1.0)
            * np.random.choice([-1, 1])
        )

        signal[abrupt_point:] += offset

    elif change_type == 'amplitude_drop':

        signal[abrupt_point:] *= np.random.uniform(0.2, 0.5)

    else:

        signal[abrupt_point:] *= np.random.uniform(1.5, 2.5)

    return quantize_signal(signal)


# ============================================================================
# FPGA TRIGGER COMPUTATION
# ============================================================================

def compute_fpga_trigger(signal):
    """
    Compute the FPGA detection response from a quantized signal.

    Detection condition:
        |diff| * FPGA_GAIN > FPGA_THRESHOLD

    The function returns the binary detection state and event-level
    statistics.
    """

    diff = np.diff(signal)
    abs_diff = np.abs(diff)

    scaled_diff = abs_diff * FPGA_GAIN
    triggered = scaled_diff > FPGA_THRESHOLD

    event_count = 0
    event_durations = []
    event_start_indices = []

    in_event = False
    current_duration = 0
    current_start = 0

    for i, trig in enumerate(triggered):

        if trig and not in_event:

            in_event = True
            current_start = i
            current_duration = 1

        elif trig and in_event:

            current_duration += 1

        elif not trig and in_event:

            in_event = False

            event_count += 1
            event_durations.append(current_duration)
            event_start_indices.append(current_start)

    # Handle an event extending to the final sample.
    if in_event:

        event_count += 1
        event_durations.append(current_duration)
        event_start_indices.append(current_start)

    return {
        'fpga_triggered': np.any(triggered),

        'fpga_event_count': event_count,

        'fpga_max_event_duration': (
            max(event_durations)
            if event_durations
            else 0
        ),

        'fpga_first_trigger_index': (
            event_start_indices[0]
            if event_start_indices
            else -1
        ),

        'fpga_max_diff': (
            np.max(abs_diff)
            if len(abs_diff) > 0
            else 0
        ),

        'fpga_max_scaled_diff': (
            np.max(scaled_diff)
            if len(scaled_diff) > 0
            else 0
        ),

        'fpga_triggered_samples': np.sum(triggered)
    }


# ============================================================================
# DATASET GENERATION
# ============================================================================

def create_dataset(n_per_class=200):
    """
    Generate the complete five-class dataset.

    Each observation contains:
    - a quantized signal waveform,
    - a class label,
    - FPGA detection metadata,
    - a unique scenario identifier.

    With the default configuration:
        5 classes × 200 scenarios = 1000 scenarios
        256 samples per scenario
    """

    generators = {
        'normal': generate_normal,
        'micro_cut': generate_micro_cut,
        'noise': generate_noise,
        'transient': generate_transient,
        'abrupt': generate_abrupt
    }

    labels = list(generators.keys())

    all_signals = []
    all_labels = []
    all_fpga_info = []
    all_scenario_ids = []

    print("=" * 60)
    print("GENERATION OF THE FPGA-ALIGNED DATASET")
    print("=" * 60)

    print(
        f"FPGA: {SAMPLE_RATE/1e6:.0f} MHz | "
        f"fixdt(1,16,12)"
    )

    print(
        f"Signal: {SIGNAL_FREQ_HZ:.1f} Hz "
        f"({DURATION*1e6:.0f} µs = "
        f"{SIGNAL_FREQ_HZ*DURATION:.3f} cycle)"
    )

    print(
        f"FPGA threshold: "
        f"|diff|*{FPGA_GAIN} > {FPGA_THRESHOLD}"
    )

    print()

    scenario_id = 0

    for label_idx, (label_name, generator) in enumerate(
        generators.items()
    ):

        print(
            f"  → {label_name:12s} : "
            f"{n_per_class} scenarios"
        )

        for _ in range(n_per_class):

            # Use a deterministic seed for each scenario.
            np.random.seed(scenario_id)

            signal = generator()
            fpga_info = compute_fpga_trigger(signal)

            all_signals.append(signal)
            all_labels.append(label_idx)
            all_fpga_info.append(fpga_info)
            all_scenario_ids.append(scenario_id)

            scenario_id += 1

    np.random.seed(None)

    X = np.array(all_signals)
    y = np.array(all_labels)

    print(
        f"\nDataset created: "
        f"{X.shape[0]} scenarios × "
        f"{X.shape[1]} samples"
    )

    return (
        X,
        y,
        labels,
        all_fpga_info,
        all_scenario_ids
    )


# ============================================================================
# DATASET VISUALIZATION
# ============================================================================

def visualize_samples(X, y, labels, fpga_infos):
    """
    Visualize three representative observations from each class.

    FPGA detection information is displayed with each waveform.
    """

    fig, axes = plt.subplots(
        5,
        3,
        figsize=(14, 14),
        sharex=True
    )

    fig.suptitle(
        f'Signals quantized using fixdt(1,16,12) | '
        f'{SAMPLE_RATE/1e6:.0f} MHz | '
        f'{DURATION*1e6:.0f} µs\n'
        f'Signal: {SIGNAL_FREQ_HZ:.1f} Hz '
        f'({SIGNAL_FREQ_HZ*DURATION:.3f} cycle/window)',
        fontsize=11,
        fontweight='bold'
    )

    colors = [
        '#2ecc71',
        '#e74c3c',
        '#f39c12',
        '#9b59b6',
        '#3498db'
    ]

    for class_idx, label_name in enumerate(labels):

        class_indices = np.where(
            y == class_idx
        )[0]

        selected = np.random.choice(
            class_indices,
            3,
            replace=False
        )

        for plot_idx, sample_idx in enumerate(selected):

            ax = axes[class_idx, plot_idx]

            t = np.linspace(
                0,
                DURATION * 1e6,
                N_SAMPLES
            )

            signal = X[sample_idx]
            fpga = fpga_infos[sample_idx]

            ax.plot(
                t,
                signal,
                color=colors[class_idx],
                linewidth=1.2
            )

            ax.set_ylim(-2.5, 2.5)
            ax.grid(True, alpha=0.3)

            triggered_text = (
                f"FPGA={'ON' if fpga['fpga_triggered'] else 'OFF'} "
                f"N={fpga['fpga_event_count']} "
                f"Dur={fpga['fpga_max_event_duration']}µs"
            )

            if plot_idx == 0:

                ax.set_ylabel(
                    f"{label_name.upper()}\n"
                    f"{triggered_text}",
                    fontsize=8,
                    fontweight='bold',
                    color=colors[class_idx]
                )

            else:

                ax.set_title(
                    triggered_text,
                    fontsize=7,
                    style='italic'
                )

            if class_idx == 4:

                ax.set_xlabel(
                    'Time (µs)',
                    fontsize=9
                )

    plt.tight_layout()

    viz_dir = OUTPUT_DIR / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    plt.savefig(
        viz_dir / 'dataset_visualization.png',
        dpi=150
    )

    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    # Generate the dataset.
    X, y, labels, fpga_infos, scenario_ids = create_dataset(
        n_per_class=200
    )

    # Save signal and label arrays.
    np.save(
        OUTPUT_DIR / 'X_signals.npy',
        X
    )

    np.save(
        OUTPUT_DIR / 'y_labels.npy',
        y
    )

    np.save(
        OUTPUT_DIR / 'scenario_ids.npy',
        np.array(scenario_ids)
    )

    # Save class labels and FPGA metadata.
    with open(
        OUTPUT_DIR / 'labels.pkl',
        'wb'
    ) as f:

        pickle.dump(
            labels,
            f
        )

    with open(
        OUTPUT_DIR / 'fpga_infos.pkl',
        'wb'
    ) as f:

        pickle.dump(
            fpga_infos,
            f
        )

    # Generate representative signal plots.
    visualize_samples(
        X,
        y,
        labels,
        fpga_infos
    )

    # =========================================================================
    # FPGA DETECTION STATISTICS
    # =========================================================================

    print("\n" + "=" * 60)
    print("FPGA DETECTION STATISTICS BY CLASS")
    print("=" * 60)

    for class_idx, label_name in enumerate(labels):

        class_indices = np.where(
            y == class_idx
        )[0]

        triggered_count = sum(
            1
            for i in class_indices
            if fpga_infos[i]['fpga_triggered']
        )

        print(
            f"  {label_name:12s} : "
            f"{triggered_count}/{len(class_indices)} "
            f"trigger the FPGA "
            f"({100 * triggered_count / len(class_indices):.0f}%)"
        )

    # =========================================================================
    # GENERATED FILES
    # =========================================================================

    print("\n" + "=" * 60)
    print("GENERATED FILES")
    print("=" * 60)

    print(
        f"  X_signals.npy     : {X.shape}"
    )

    print(
        f"  y_labels.npy      : {y.shape}"
    )

    print(
        f"  scenario_ids.npy  : "
        f"{len(scenario_ids)} scenarios"
    )

    print(
        f"  fpga_infos.pkl    : "
        f"{len(fpga_infos)} FPGA metadata records"
    )

    print(
        f"  labels.pkl        : "
        f"{labels}"
    )

    print("\nDataset generation completed successfully.")

