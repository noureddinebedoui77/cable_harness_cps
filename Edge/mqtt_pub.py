
"""
mqtt_publisher.py

Edge AI inference and MQTT publisher for the cable-harness monitoring system.

Pipeline:
    Project 1 signal
        -> Feature extraction
        -> ONNX Runtime inference
        -> FPGA metadata
        -> Digital Twin
        -> MQTT
        -> InfluxDB / Grafana

Project 1 configuration:
    Sampling rate : 1 MHz
    Window size   : 256 samples
    Features      : 20
    Scenarios     : 1000
    Classes       : 5
"""

import importlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np


# ============================================================================
# PROJECT PARAMETERS
# ============================================================================

PROJECT_SAMPLE_RATE_HZ = 1_000_000
PROJECT_WINDOW_SIZE = 256
PROJECT_FEATURE_COUNT = 20
PROJECT_SCENARIO_COUNT = 1000

PROJECT_CLASSES = [
    "normal",
    "micro_cut",
    "noise",
    "transient",
    "abrupt",
]


# ============================================================================
# MQTT CONFIGURATION
# ============================================================================

# Docker: set MQTT_BROKER=mosquitto.
# Local execution: defaults to localhost.

MQTT_BROKER = os.getenv(
    "MQTT_BROKER",
    "localhost"
)

MQTT_PORT = int(
    os.getenv(
        "MQTT_PORT",
        "1883"
    )
)

MQTT_TOPIC_EVENTS = os.getenv(
    "MQTT_EVENTS_TOPIC",
    "cable/01/events"
)

MQTT_TOPIC_STATUS = os.getenv(
    "MQTT_STATUS_TOPIC",
    "cable/01/status"
)

MQTT_TOPIC_DIAGNOSTICS = os.getenv(
    "MQTT_DIAGNOSTICS_TOPIC",
    "cable/01/diagnostics"
)

DEVICE_ID = "CABLE_TEST_01"


# ============================================================================
# DEMONSTRATION CONFIGURATION
# ============================================================================

DEMO_EVENT_COUNT = 20
DEMO_INTERVAL_SECONDS = 1.0
ONNX_WARMUP_RUNS = 3


# ============================================================================
# MQTT DEPENDENCY
# ============================================================================

try:
    import paho.mqtt.client as mqtt

    MQTT_AVAILABLE = True

except ImportError:

    MQTT_AVAILABLE = False

    print("WARNING: paho-mqtt is not installed.")
    print("Install with:")
    print("pip install paho-mqtt")


# ============================================================================
# MQTT CLIENT
# ============================================================================

def create_mqtt_client():

    if not MQTT_AVAILABLE:
        return None

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2
    )

    client.connect(
        MQTT_BROKER,
        MQTT_PORT,
        60
    )

    client.loop_start()

    print(
        f"Connected to MQTT broker: "
        f"{MQTT_BROKER}:{MQTT_PORT}"
    )

    return client


# ============================================================================
# MQTT EVENT
# ============================================================================

def publish_event(client, event_data):

    if client is None:

        print("[MQTT SIMULATION]")
        print(
            json.dumps(
                event_data,
                indent=2
            )
        )

        return False

    payload = json.dumps(
        event_data,
        ensure_ascii=False
    )

    result = client.publish(
        MQTT_TOPIC_EVENTS,
        payload,
        qos=1
    )

    if result.rc == mqtt.MQTT_ERR_SUCCESS:

        print(
            f"[MQTT] Published: "
            f"{event_data['event_type']}"
        )

        return True

    print(
        f"[MQTT] Publication failed: "
        f"{result.rc}"
    )

    return False


# ============================================================================
# MQTT STATUS
# ============================================================================

def publish_status(
    client,
    health_score,
    status
):

    if client is None:
        return

    timestamp_us = int(
        time.time() * 1_000_000
    )

    payload = json.dumps({
        "device_id": DEVICE_ID,
        "timestamp_us": timestamp_us,
        "health_score": float(health_score),
        "status": str(status),
    })

    client.publish(
        MQTT_TOPIC_STATUS,
        payload,
        qos=1
    )


# ============================================================================
# DIGITAL TWIN
# ============================================================================

def load_digital_twin(project_root):

    digital_twin_dir = project_root / "digital_twin"

    if not digital_twin_dir.exists():

        raise FileNotFoundError(
            "Digital Twin directory not found:\n"
            f"{digital_twin_dir}"
        )

    path_string = str(digital_twin_dir)

    if path_string not in sys.path:
        sys.path.insert(0, path_string)

    try:

        module = importlib.import_module("twin")

        if not hasattr(module, "DigitalTwin"):

            raise ImportError(
                "digital_twin/twin.py does not "
                "define DigitalTwin."
            )

        return module.DigitalTwin(DEVICE_ID)

    except Exception as exc:

        raise ImportError(
            "Unable to load DigitalTwin "
            "from digital_twin/twin.py"
        ) from exc


# ============================================================================
# FEATURE EXTRACTOR LOADER
# ============================================================================

def load_feature_extractor(project_root):
    """
    Load the feature-extraction implementation.

    Supported interfaces:
        1. extract_features(signal, fpga_info)
        2. extract_features(X, fpga_infos)
        3. extract_features_from_signal(signal, fpga_info)
        4. FeatureExtractor class
    """

    ai_dir = project_root / "ai"

    if not ai_dir.exists():

        raise FileNotFoundError(
            f"AI directory not found:\n{ai_dir}"
        )

    ai_path = str(ai_dir.resolve())

    if ai_path not in sys.path:
        sys.path.insert(0, ai_path)

    # Avoid importing a cached implementation from another location.
    sys.modules.pop(
        "extract_features",
        None
    )

    try:

        module = importlib.import_module(
            "extract_features"
        )

    except Exception as exc:

        raise ImportError(
            "Unable to import ai/extract_features.py"
        ) from exc

    if hasattr(module, "extract_features"):

        function = module.extract_features

        if callable(function):

            print(
                "Feature extractor loaded: "
                "extract_features()"
            )

            return _build_feature_wrapper(
                function,
                "extract_features"
            )

    if hasattr(
        module,
        "extract_features_from_signal"
    ):

        function = module.extract_features_from_signal

        if callable(function):

            print(
                "Feature extractor loaded: "
                "extract_features_from_signal()"
            )

            return _build_feature_wrapper(
                function,
                "extract_features_from_signal"
            )

    if hasattr(module, "FeatureExtractor"):

        extractor_class = module.FeatureExtractor
        extractor = extractor_class()

        for method_name in [
            "extract_features",
            "extract",
            "transform",
        ]:

            if hasattr(extractor, method_name):

                method = getattr(
                    extractor,
                    method_name
                )

                if callable(method):

                    print(
                        "Feature extractor loaded: "
                        f"FeatureExtractor.{method_name}()"
                    )

                    return _build_feature_wrapper(
                        method,
                        f"FeatureExtractor.{method_name}"
                    )

    available = [
        name
        for name in dir(module)
        if not name.startswith("_")
    ]

    raise ImportError(
        "ai/extract_features.py was imported "
        "successfully, but no supported feature "
        "extraction interface was found.\n\n"
        "Expected one of:\n"
        "  - extract_features()\n"
        "  - extract_features_from_signal()\n"
        "  - FeatureExtractor class\n\n"
        f"Available symbols:\n{available}"
    )


# ============================================================================
# FEATURE EXTRACTION WRAPPER
# ============================================================================

def _build_feature_wrapper(
    function,
    function_name
):

    def wrapper(signal, fpga_info):

        signal_array = np.asarray(signal)

        # Support the main project interface as well as compatible
        # alternative signatures.
        attempts = [
            (
                "signal, [fpga_info]",
                lambda: function(
                    signal_array,
                    [fpga_info]
                )
            ),
            (
                "signal, fpga_info",
                lambda: function(
                    signal_array,
                    fpga_info
                )
            ),
            (
                "signal",
                lambda: function(
                    signal_array
                )
            ),
        ]

        errors = []

        for description, call in attempts:

            try:

                result = call()

                if result is not None:
                    return result

            except (
                TypeError,
                ValueError,
                KeyError,
                AttributeError,
            ) as exc:

                errors.append(
                    f"{description}: {exc}"
                )

        raise RuntimeError(
            f"Feature extraction failed using "
            f"{function_name}.\n\n"
            + "\n".join(errors)
        )

    return wrapper


# ============================================================================
# LOAD DATASET
# ============================================================================

def load_dataset(project_root):

    data_dir = project_root / "data"

    signals_file = data_dir / "X_signals.npy"
    fpga_file = data_dir / "fpga_infos.pkl"
    labels_file = data_dir / "labels.pkl"

    for file in [
        signals_file,
        fpga_file,
        labels_file,
    ]:

        if not file.exists():

            raise FileNotFoundError(
                f"Missing required file:\n{file}"
            )

    X_signals = np.load(signals_file)

    with open(fpga_file, "rb") as file:
        fpga_infos = pickle.load(file)

    with open(labels_file, "rb") as file:
        labels = pickle.load(file)

    return X_signals, fpga_infos, labels


# ============================================================================
# CLASS NORMALIZATION
# ============================================================================

def normalize_class_name(value):

    if isinstance(value, bytes):

        value = value.decode(
            "utf-8",
            errors="ignore"
        )

    return str(value).strip().lower()


# ============================================================================
# FPGA GROUND TRUTH
# ============================================================================

def extract_ground_truth_from_fpga_info(fpga_info):

    if not isinstance(fpga_info, dict):
        return None

    possible_keys = [
        "label",
        "class",
        "event_type",
        "fault_type",
        "ground_truth",
        "ground_truth_label",
        "target",
    ]

    for key in possible_keys:

        if key not in fpga_info:
            continue

        value = fpga_info[key]

        if value is None:
            continue

        value = normalize_class_name(value)

        if value in PROJECT_CLASSES:
            return value

    return None


# ============================================================================
# FIND LABELS
# ============================================================================

def find_per_sample_labels(
    fpga_infos,
    labels
):

    n_samples = len(fpga_infos)

    try:

        if len(labels) == n_samples:

            normalized = [
                normalize_class_name(x)
                for x in labels
            ]

            if all(
                x in PROJECT_CLASSES
                for x in normalized
            ):

                return normalized

    except TypeError:
        pass

    detected = []

    for info in fpga_infos:

        detected.append(
            extract_ground_truth_from_fpga_info(info)
        )

    if len(detected) == n_samples:

        if all(
            value in PROJECT_CLASSES
            for value in detected
        ):

            return detected

    return None


# ============================================================================
# ONNX MODEL INFORMATION
# ============================================================================

def print_onnx_model_information(session):

    print()
    print("ONNX model information:")
    print("-" * 60)

    print("Inputs:")

    for inp in session.get_inputs():

        print(f"  name : {inp.name}")
        print(f"  shape: {inp.shape}")
        print(f"  type : {inp.type}")

    print()
    print("Outputs:")

    for i, output in enumerate(
        session.get_outputs()
    ):

        print(f"  [{i}] name : {output.name}")
        print(f"      shape: {output.shape}")
        print(f"      type : {output.type}")

    print("-" * 60)


# ============================================================================
# PROBABILITY VECTOR
# ============================================================================

def extract_probability_vector(
    output,
    class_count
):

    if isinstance(output, dict):

        probabilities = np.zeros(
            class_count,
            dtype=np.float32
        )

        found = False

        for key, value in output.items():

            try:

                index = int(key)

                if 0 <= index < class_count:

                    probabilities[index] = float(value)
                    found = True

            except (ValueError, TypeError):

                key_name = normalize_class_name(key)

                if key_name in PROJECT_CLASSES:

                    index = PROJECT_CLASSES.index(
                        key_name
                    )

                    probabilities[index] = float(value)
                    found = True

        if found:
            return probabilities

        return None

    try:

        array = np.asarray(
            output,
            dtype=np.float32
        )

    except (ValueError, TypeError):

        return None

    if array.size == 0:
        return None

    if array.dtype == object:
        return None

    if array.ndim >= 2:
        array = array[0]

    array = array.reshape(-1)

    if len(array) != class_count:
        return None

    if np.any(array < 0):
        return None

    total = float(np.sum(array))

    if total <= 0:
        return None

    if not np.isclose(
        total,
        1.0,
        atol=1e-3
    ):

        array = array / total

    return array


# ============================================================================
# PREDICTION DECODER
# ============================================================================

def decode_model_outputs(
    outputs,
    class_names
):

    class_count = len(class_names)

    probability_vector = None
    predicted_index = None

    # Prefer probability outputs when available.
    for output in outputs:

        probabilities = extract_probability_vector(
            output,
            class_count
        )

        if probabilities is not None:

            probability_vector = probabilities
            break

    # Look for an explicit predicted label.
    for output in outputs:

        if isinstance(output, dict):
            continue

        try:

            array = np.asarray(output)

            if array.size == 0:
                continue

            value = array.reshape(-1)[0]

        except Exception:

            continue

        if isinstance(value, bytes):

            value = value.decode(
                "utf-8",
                errors="ignore"
            )

        if isinstance(value, str):

            name = normalize_class_name(value)

            if name in class_names:

                predicted_index = class_names.index(name)
                break

        try:

            numeric = float(value)

            if (
                numeric.is_integer()
                and 0 <= int(numeric) < class_count
            ):

                predicted_index = int(numeric)
                break

        except (ValueError, TypeError):

            pass

    if probability_vector is not None:

        predicted_index = int(
            np.argmax(probability_vector)
        )

        confidence = float(
            probability_vector[predicted_index]
        )

        return (
            predicted_index,
            confidence,
            probability_vector
        )

    if predicted_index is not None:

        return (
            predicted_index,
            1.0,
            None
        )

    raise RuntimeError(
        "Unable to decode ONNX outputs."
    )


# ============================================================================
# SAFE FEATURE EXTRACTION
# ============================================================================

def compute_features(
    extract_features,
    signal,
    fpga_info
):

    raw_features = extract_features(
        signal,
        fpga_info
    )

    features = np.asarray(
        raw_features,
        dtype=np.float32
    )

    # Normalize supported single-vector shapes.
    if features.ndim == 1:

        features = features.reshape(1, -1)

    elif features.ndim == 2:

        if (
            features.shape[0] != 1
            and features.shape[1] == 1
        ):

            features = features.reshape(1, -1)

    else:

        raise ValueError(
            "Unexpected feature shape: "
            f"{features.shape}"
        )

    if features.shape[0] != 1:

        raise ValueError(
            "Expected one feature vector. "
            f"Received: {features.shape}"
        )

    if features.shape[1] != PROJECT_FEATURE_COUNT:

        raise ValueError(
            "FEATURE COUNT MISMATCH.\n"
            f"Expected : {PROJECT_FEATURE_COUNT}\n"
            f"Received : {features.shape[1]}\n\n"
            "The MQTT pipeline cannot safely "
            "run inference with a different "
            "feature definition."
        )

    if not np.isfinite(features).all():

        raise ValueError(
            "Feature vector contains NaN or Inf."
        )

    return np.ascontiguousarray(
        features,
        dtype=np.float32
    )


# ============================================================================
# DEMONSTRATION INDICES
# ============================================================================

def select_demo_indices(
    X_signals,
    fpga_infos,
    sample_labels
):

    n_samples = len(X_signals)

    if n_samples == 0:

        raise ValueError(
            "Dataset contains no samples."
        )

    selected = []

    if sample_labels is not None:

        for class_name in PROJECT_CLASSES:

            for index, label in enumerate(
                sample_labels
            ):

                if (
                    label == class_name
                    and index not in selected
                ):

                    selected.append(index)
                    break

    for index in np.linspace(
        0,
        n_samples - 1,
        min(
            DEMO_EVENT_COUNT,
            n_samples
        ),
        dtype=int
    ):

        index = int(index)

        if index not in selected:
            selected.append(index)

        if len(selected) >= DEMO_EVENT_COUNT:
            break

    return selected[:DEMO_EVENT_COUNT]


# ============================================================================
# FPGA METADATA
# ============================================================================

def get_fpga_metadata(fpga_info):

    if not isinstance(fpga_info, dict):
        fpga_info = {}

    triggered = bool(
        fpga_info.get(
            "fpga_triggered",
            False
        )
    )

    event_count = int(
        fpga_info.get(
            "fpga_event_count",
            0
        )
    )

    max_duration = int(
        fpga_info.get(
            "fpga_max_event_duration",
            fpga_info.get(
                "fpga_max_duration",
                0
            )
        )
    )

    trigger_index = int(
        fpga_info.get(
            "fpga_first_trigger_index",
            -1
        )
    )

    return (
        triggered,
        event_count,
        max_duration,
        trigger_index
    )


# ============================================================================
# SIGNAL RMS
# ============================================================================

def compute_signal_rms(signal):

    array = np.asarray(
        signal,
        dtype=np.float64
    )

    return float(
        np.sqrt(
            np.mean(
                np.square(array)
            )
        )
    )


# ============================================================================
# ONNX WARM-UP
# ============================================================================

def warmup_onnx(
    session,
    input_name,
    features
):

    print(
        f"\nONNX warm-up: "
        f"{ONNX_WARMUP_RUNS} runs"
    )

    for _ in range(ONNX_WARMUP_RUNS):

        session.run(
            None,
            {
                input_name: features
            }
        )

    print("ONNX warm-up completed.")


# ============================================================================
# MAIN EDGE LOOP
# ============================================================================

def simulate_edge_loop(
    client,
    model_path="ai/model.onnx"
):

    print()
    print("=" * 60)
    print("EDGE AI + MQTT PUBLISHER")
    print("=" * 60)

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    # Feature extraction.
    extract_features = load_feature_extractor(
        project_root
    )

    # ONNX Runtime.
    try:

        import onnxruntime as ort

    except ImportError as exc:

        raise ImportError(
            "onnxruntime is not installed.\n"
            "Run:\n"
            "pip install onnxruntime"
        ) from exc

    model_file = project_root / model_path

    if not model_file.exists():

        raise FileNotFoundError(
            f"ONNX model not found:\n"
            f"{model_file}"
        )

    session = ort.InferenceSession(
        str(model_file)
    )

    inputs = session.get_inputs()

    print()
    print(
        f"ONNX model loaded:\n"
        f"{model_file}"
    )

    print_onnx_model_information(session)

    # Dataset.
    (
        X_signals,
        fpga_infos,
        labels
    ) = load_dataset(project_root)

    print()
    print("Dataset:")
    print(f"  Signals : {X_signals.shape}")
    print(f"  FPGA    : {len(fpga_infos)}")

    if len(X_signals) != len(fpga_infos):

        raise ValueError(
            "Signals and FPGA metadata "
            "have different lengths."
        )

    if X_signals.ndim != 2:

        raise ValueError(
            f"Unexpected signal shape: "
            f"{X_signals.shape}"
        )

    if X_signals.shape[1] != PROJECT_WINDOW_SIZE:

        raise ValueError(
            "Project 1 window size mismatch.\n"
            f"Expected : {PROJECT_WINDOW_SIZE}\n"
            f"Received : {X_signals.shape[1]}"
        )

    # Ground truth.
    sample_labels = find_per_sample_labels(
        fpga_infos,
        labels
    )

    if sample_labels is None:
        print("\nGround truth: not available.")
    else:
        print("\nGround truth: available.")

    # Select representative demonstration samples.
    selected_indices = select_demo_indices(
        X_signals,
        fpga_infos,
        sample_labels
    )

    print("\nSelected demonstration indices:")
    print(selected_indices)

    # Digital Twin.
    twin = load_digital_twin(project_root)

    # Verify feature/model compatibility before starting the loop.
    test_index = selected_indices[0]

    test_signal = X_signals[
        test_index:test_index + 1
    ]

    test_fpga_info = fpga_infos[test_index]

    print()
    print("Testing feature extraction...")

    test_features = compute_features(
        extract_features,
        test_signal,
        test_fpga_info
    )

    print(
        f"Feature vector : "
        f"{test_features.shape}"
    )

    model_input_shape = inputs[0].shape

    if (
        len(model_input_shape) == 2
        and isinstance(model_input_shape[1], int)
    ):

        if model_input_shape[1] != PROJECT_FEATURE_COUNT:

            raise ValueError(
                "ONNX feature dimension mismatch.\n"
                f"Model    : {model_input_shape[1]}\n"
                f"Project  : {PROJECT_FEATURE_COUNT}"
            )

    print("Feature/model compatibility: PASSED")

    warmup_onnx(
        session,
        inputs[0].name,
        test_features
    )

    # Process demonstration events.
    print()
    print("=" * 60)
    print("STARTING EDGE AI + MQTT")
    print("=" * 60)

    published_count = 0

    for event_number, index in enumerate(
        selected_indices,
        start=1
    ):

        print()
        print("=" * 60)
        print(
            f"EVENT {event_number}/"
            f"{len(selected_indices)}"
        )
        print(f"Dataset index : {index}")

        signal = X_signals[
            index:index + 1
        ]

        fpga_info = fpga_infos[index]

        if sample_labels is not None:
            ground_truth = sample_labels[index]
        else:
            ground_truth = "not stored in dataset"

        # Feature extraction.
        features = compute_features(
            extract_features,
            signal,
            fpga_info
        )

        # ONNX inference latency.
        start_time = time.perf_counter()

        outputs = session.run(
            None,
            {
                inputs[0].name: features
            }
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        # Decode model output.
        (
            pred_idx,
            confidence,
            probability_vector
        ) = decode_model_outputs(
            outputs,
            PROJECT_CLASSES
        )

        if not (
            0 <= pred_idx < len(PROJECT_CLASSES)
        ):

            raise RuntimeError(
                f"Invalid prediction index: "
                f"{pred_idx}"
            )

        event_type = PROJECT_CLASSES[pred_idx]

        confidence = float(
            np.clip(
                confidence,
                0.0,
                1.0
            )
        )

        # FPGA metadata.
        (
            fpga_triggered,
            fpga_event_count,
            fpga_max_duration,
            fpga_trigger_index
        ) = get_fpga_metadata(fpga_info)

        # Wall-clock timestamp for the MQTT event.
        timestamp_us = int(
            time.time() * 1_000_000
        )

        # Update Digital Twin state.
        twin.update(
            event_type,
            confidence,
            fpga_info
        )

        signal_rms = compute_signal_rms(signal)

        # MQTT event payload.
        event_data = {
            "device_id": DEVICE_ID,
            "timestamp_us": timestamp_us,
            "event_type": event_type,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 4),
            "fpga_triggered": fpga_triggered,
            "fpga_event_count": fpga_event_count,
            "fpga_max_duration": fpga_max_duration,
            "fpga_first_trigger_index": fpga_trigger_index,
            "health_score": round(
                float(twin.state.health_score),
                1
            ),
            "status": str(twin.state.status),
            "signal_rms": round(signal_rms, 4),
            "sample_rate_hz": PROJECT_SAMPLE_RATE_HZ,
            "window_size": PROJECT_WINDOW_SIZE,
            "feature_count": PROJECT_FEATURE_COUNT,
            "ground_truth": ground_truth,
        }

        # Include class probabilities when available.
        if probability_vector is not None:

            probability_vector = np.asarray(
                probability_vector,
                dtype=np.float32
            ).reshape(-1)

            event_data["class_probabilities"] = {
                class_name: round(
                    float(
                        probability_vector[class_index]
                    ),
                    4
                )
                for class_index, class_name in enumerate(
                    PROJECT_CLASSES
                )
            }

        # Console output.
        print(f"Ground truth : {ground_truth}")
        print(f"Prediction   : {event_type}")
        print(f"Confidence   : {confidence * 100:.2f}%")
        print(f"Latency      : {latency_ms:.4f} ms")
        print(
            f"FPGA         : "
            f"{'YES' if fpga_triggered else 'NO'}"
        )
        print(
            f"Health       : "
            f"{twin.state.health_score:.1f}"
        )
        print(
            f"Status       : "
            f"{twin.state.status}"
        )

        # Publish event and status.
        if publish_event(client, event_data):
            published_count += 1

        publish_status(
            client,
            twin.state.health_score,
            twin.state.status
        )

        twin.display()

        if event_number < len(selected_indices):
            time.sleep(DEMO_INTERVAL_SECONDS)

    print()
    print("=" * 60)
    print("EDGE AI + MQTT LOOP COMPLETED")
    print("=" * 60)
    print(
        f"Published events: "
        f"{published_count}"
    )
    print("=" * 60)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    client = None

    try:

        client = create_mqtt_client()

        simulate_edge_loop(client)

    except KeyboardInterrupt:

        print(
            "\nMQTT publisher stopped by user."
        )

    except Exception as exc:

        print()
        print("=" * 60)
        print("ERROR")
        print("=" * 60)
        print(str(exc))
        print("=" * 60)

        raise

    finally:

        if client is not None:

            try:

                client.loop_stop()
                client.disconnect()

                print(
                    "Disconnected from MQTT broker."
                )

            except Exception:
                pass
