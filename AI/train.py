
"""
train_model.py

Research-oriented model training and evaluation with scenario-level splitting.

Pipeline:
    X_signals
        ↓
    extract_features()
        ↓
    20 features
        ↓
    GroupShuffleSplit (80/20)
        ↓
    Random Forest / Decision Tree / Logistic Regression
        ↓
    Model selection based on Macro-F1
        ↓
    model.pkl + model.onnx
"""

import numpy as np
import pickle
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
AI_DIR = ROOT_DIR / "ai"
RESULTS_DIR = AI_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

FEATURE_COUNT = 20
SCENARIO_COUNT = 1000
RANDOM_STATE = 42

CLASS_NAMES = [
    "normal",
    "micro_cut",
    "noise",
    "transient",
    "abrupt",
]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():

    X_signals = np.load(
        DATA_DIR / "X_signals.npy"
    )

    y = np.load(
        DATA_DIR / "y_labels.npy"
    )

    scenario_ids = np.load(
        DATA_DIR / "scenario_ids.npy"
    )

    with open(
        DATA_DIR / "labels.pkl",
        "rb"
    ) as f:
        labels = pickle.load(f)

    with open(
        DATA_DIR / "fpga_infos.pkl",
        "rb"
    ) as f:
        fpga_infos = pickle.load(f)

    # Use the same feature-extraction interface as the rest of the pipeline.
    from extract_features import extract_features

    X_features = extract_features(
        X_signals,
        fpga_infos
    )

    X_features = np.asarray(
        X_features,
        dtype=np.float32
    )

    print(
        f"Data: "
        f"{X_features.shape[0]} scenarios, "
        f"{X_features.shape[1]} features"
    )

    return (
        X_features,
        y,
        labels,
        scenario_ids
    )


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_data(
    X,
    y,
    labels,
    scenario_ids
):

    if X.ndim != 2:
        raise ValueError(
            f"X must be 2D. Received: {X.shape}"
        )

    if X.shape[1] != FEATURE_COUNT:
        raise ValueError(
            f"Incorrect number of features: "
            f"{X.shape[1]} instead of {FEATURE_COUNT}"
        )

    if X.shape[0] != SCENARIO_COUNT:
        raise ValueError(
            f"Incorrect number of scenarios: "
            f"{X.shape[0]} instead of {SCENARIO_COUNT}"
        )

    if len(y) != len(X):
        raise ValueError(
            "X and y have different lengths."
        )

    if len(scenario_ids) != len(X):
        raise ValueError(
            "X and scenario_ids have different lengths."
        )

    if len(labels) != len(CLASS_NAMES):
        raise ValueError(
            f"Incorrect number of classes: "
            f"{len(labels)}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "X contains NaN or Inf values."
        )

    print("\nData validation: PASSED")


# ============================================================================
# SCENARIO-LEVEL SPLIT
# ============================================================================

def split_by_scenario(
    X,
    y,
    scenario_ids,
    test_size=0.20
):

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        gss.split(
            X,
            y,
            groups=scenario_ids
        )
    )

    return train_idx, test_idx


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_and_evaluate(
    X,
    y,
    labels,
    scenario_ids
):

    train_idx, test_idx = split_by_scenario(
        X,
        y,
        scenario_ids
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    print(
        f"\nScenario-level split: "
        f"{len(train_idx)} train, "
        f"{len(test_idx)} test"
    )

    models = {

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=1
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=RANDOM_STATE
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    }

    results = {}

    print(
        "\n" + "=" * 60
    )
    print(
        "MODEL COMPARISON "
        "(scenario-level split)"
    )
    print(
        "=" * 60
    )

    for name, model in models.items():

        print(
            f"\n--- {name} ---"
        )

        # --------------------------------------------------------------------
        # TRAINING
        # --------------------------------------------------------------------

        start = time.perf_counter()

        model.fit(
            X_train,
            y_train
        )

        train_time = (
            time.perf_counter() - start
        )

        # --------------------------------------------------------------------
        # PREDICTION
        # --------------------------------------------------------------------

        y_pred = model.predict(
            X_test
        )

        # --------------------------------------------------------------------
        # SINGLE-SAMPLE LATENCY
        # --------------------------------------------------------------------

        warmup = 20
        repetitions = 500

        for _ in range(warmup):
            model.predict(
                X_test[:1]
            )

        start = time.perf_counter()

        for _ in range(repetitions):
            model.predict(
                X_test[:1]
            )

        latency_ms = (
            (
                time.perf_counter() - start
            )
            / repetitions
            * 1000
        )

        # --------------------------------------------------------------------
        # CLASSIFICATION METRICS
        # --------------------------------------------------------------------

        acc = accuracy_score(
            y_test,
            y_pred
        )

        macro_prec = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        macro_rec = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        f1_per_class = f1_score(
            y_test,
            y_pred,
            average=None,
            labels=np.arange(len(labels)),
            zero_division=0
        )

        # --------------------------------------------------------------------
        # CONFUSION MATRIX
        # --------------------------------------------------------------------

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=np.arange(len(labels))
        )

        fn_per_class = (
            cm.sum(axis=1)
            - np.diag(cm)
        )

        fp_per_class = (
            cm.sum(axis=0)
            - np.diag(cm)
        )

        # --------------------------------------------------------------------
        # MODEL SIZE
        # --------------------------------------------------------------------

        temp_path = (
            AI_DIR / "temp_model.pkl"
        )

        joblib.dump(
            model,
            temp_path
        )

        size_kb = (
            temp_path.stat().st_size
            / 1024
        )

        temp_path.unlink(
            missing_ok=True
        )

        # --------------------------------------------------------------------
        # STORE RESULTS
        # --------------------------------------------------------------------

        results[name] = {

            "model": model,

            "accuracy": acc,

            "macro_precision":
                macro_prec,

            "macro_recall":
                macro_rec,

            "macro_f1":
                macro_f1,

            "f1_per_class":
                f1_per_class,

            "train_time":
                train_time,

            "latency_ms":
                latency_ms,

            "size_kb":
                size_kb,

            "y_test":
                y_test,

            "y_pred":
                y_pred,

            "confusion_matrix":
                cm,

            "fn_per_class":
                fn_per_class,

            "fp_per_class":
                fp_per_class
        }

        # --------------------------------------------------------------------
        # DISPLAY RESULTS
        # --------------------------------------------------------------------

        print(
            f"  Accuracy        : {acc:.4f}"
        )

        print(
            f"  Macro Precision : "
            f"{macro_prec:.4f}"
        )

        print(
            f"  Macro Recall    : "
            f"{macro_rec:.4f}"
        )

        print(
            f"  Macro F1        : "
            f"{macro_f1:.4f}"
        )

        print(
            f"  Latency         : "
            f"{latency_ms:.3f} ms"
        )

        print(
            f"  Size            : "
            f"{size_kb:.1f} KB"
        )

        print(
            "  F1 per class    :"
        )

        for cls, score in zip(
            labels,
            f1_per_class
        ):
            print(
                f"    {cls:12s} : "
                f"{score:.3f}"
            )

        print(
            f"  False negatives : "
            f"{fn_per_class}"
        )

    return results, labels


# ============================================================================
# MODEL COMPARISON PLOT
# ============================================================================

def plot_comparison(
    results,
    labels
):

    names = list(
        results.keys()
    )

    metrics = [
        ("accuracy", "Accuracy"),
        (
            "macro_precision",
            "Macro Precision"
        ),
        (
            "macro_recall",
            "Macro Recall"
        ),
        (
            "macro_f1",
            "Macro F1"
        )
    ]

    plt.figure(
        figsize=(10, 6)
    )

    x = np.arange(
        len(names)
    )

    width = 0.2

    for i, (
        metric,
        title
    ) in enumerate(metrics):

        values = [
            results[name][metric]
            for name in names
        ]

        plt.bar(
            x + i * width,
            values,
            width,
            label=title
        )

        for j, value in enumerate(
            values
        ):
            plt.text(
                x[j] + i * width,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                fontsize=9
            )

    plt.xticks(
        x + width * 1.5,
        names
    )

    plt.ylim(
        0,
        1.05
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Model Comparison "
        "(scenario-level split)"
    )

    plt.legend()

    plt.tight_layout()

    output = (
        RESULTS_DIR
        / "model_comparison.png"
    )

    plt.savefig(
        output,
        dpi=150
    )

    plt.close()

    print(
        f"\nPlot saved: "
        f"{output}"
    )


# ============================================================================
# CONFUSION MATRICES
# ============================================================================

def plot_confusion_matrices(
    results,
    labels
):

    names = list(
        results.keys()
    )

    fig, axes = plt.subplots(
        1,
        len(names),
        figsize=(
            5 * len(names),
            4
        )
    )

    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(
        axes,
        names
    ):

        cm = results[name][
            "confusion_matrix"
        ]

        image = ax.imshow(
            cm
        )

        ax.set_title(
            f"{name}\n"
            f"Macro F1 = "
            f"{results[name]['macro_f1']:.3f}"
        )

        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "True"
        )

        ax.set_xticks(
            range(len(labels))
        )

        ax.set_yticks(
            range(len(labels))
        )

        ax.set_xticklabels(
            labels,
            rotation=45,
            ha="right"
        )

        ax.set_yticklabels(
            labels
        )

        for i in range(
            len(labels)
        ):

            for j in range(
                len(labels)
            ):

                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center"
                )

    fig.colorbar(
        image,
        ax=axes
    )

    plt.tight_layout()

    output = (
        RESULTS_DIR
        / "confusion_matrices.png"
    )

    plt.savefig(
        output,
        dpi=150
    )

    plt.close()

    print(
        f"Plot saved: "
        f"{output}"
    )


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def plot_feature_importance(
    model
):

    if not hasattr(
        model,
        "feature_importances_"
    ):

        print(
            "\nFeature importance: "
            "not available for this model."
        )

        return

    feature_file = (
        DATA_DIR
        / "feature_names.pkl"
    )

    if feature_file.exists():

        with open(
            feature_file,
            "rb"
        ) as f:

            feature_names = pickle.load(
                f
            )

    else:

        feature_names = [
            f"feature_{i}"
            for i in range(
                FEATURE_COUNT
            )
        ]

    feature_names = list(
        feature_names
    )

    if len(feature_names) != FEATURE_COUNT:

        raise ValueError(
            "Incorrect number of feature names.\n"
            f"Expected: {FEATURE_COUNT}\n"
            f"Received: {len(feature_names)}"
        )

    importances = np.asarray(
        model.feature_importances_,
        dtype=float
    )

    if len(importances) != FEATURE_COUNT:

        raise ValueError(
            "Incorrect number of feature importances.\n"
            f"Expected: {FEATURE_COUNT}\n"
            f"Received: {len(importances)}"
        )

    order = np.argsort(
        importances
    )[::-1]

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        range(FEATURE_COUNT),
        importances[order]
    )

    plt.xticks(
        range(FEATURE_COUNT),
        [
            feature_names[i]
            for i in order
        ],
        rotation=45,
        ha="right"
    )

    plt.ylabel(
        "Importance"
    )

    plt.title(
        "Feature Importance — "
        "Random Forest"
    )

    plt.tight_layout()

    output = (
        RESULTS_DIR
        / "feature_importance.png"
    )

    plt.savefig(
        output,
        dpi=150
    )

    plt.close()

    print(
        f"Plot saved: "
        f"{output}"
    )


# ============================================================================
# ONNX EXPORT
# ============================================================================

def export_onnx(
    model,
    filename="model.onnx"
):

    print(
        "\n" + "=" * 60
    )

    print(
        "ONNX EXPORT"
    )

    print(
        "=" * 60
    )

    try:

        from skl2onnx import (
            convert_sklearn
        )

        from skl2onnx.common.data_types import (
            FloatTensorType
        )

    except ImportError:

        raise ImportError(
            "ONNX dependencies are missing.\n"
            "Install with:\n"
            "pip install skl2onnx onnxruntime"
        )

    initial_type = [
        (
            "float_input",
            FloatTensorType(
                [None, FEATURE_COUNT]
            )
        )
    ]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type
    )

    filepath = (
        AI_DIR / filename
    )

    with open(
        filepath,
        "wb"
    ) as f:

        f.write(
            onnx_model.SerializeToString()
        )

    print(
        f"ONNX model : {filepath}"
    )

    print(
        f"Size       : "
        f"{filepath.stat().st_size / 1024:.2f} KB"
    )

    return filepath


# ============================================================================
# ONNX VERIFICATION
# ============================================================================

def verify_onnx(
    model_file,
    X
):

    import onnxruntime as ort

    print(
        "\n" + "=" * 60
    )

    print(
        "ONNX VERIFICATION"
    )

    print(
        "=" * 60
    )

    session = ort.InferenceSession(
        str(model_file)
    )

    input_meta = (
        session.get_inputs()[0]
    )

    print(
        "\nInput:"
    )

    print(
        f"  name : {input_meta.name}"
    )

    print(
        f"  shape: {input_meta.shape}"
    )

    print(
        f"  type : {input_meta.type}"
    )

    print(
        "\nOutputs:"
    )

    for i, output in enumerate(
        session.get_outputs()
    ):

        print(
            f"  [{i}] {output.name}"
        )

        print(
            f"      shape: {output.shape}"
        )

        print(
            f"      type : {output.type}"
        )

    # Test inference using one feature vector.
    sample = X[:1].astype(
        np.float32
    )

    outputs = session.run(
        None,
        {
            input_meta.name:
                sample
        }
    )

    prediction = np.asarray(
        outputs[0]
    ).reshape(-1)[0]

    print(
        "\nTest inference: PASSED"
    )

    print(
        f"Prediction index: "
        f"{prediction}"
    )

    # Measure single-sample ONNX inference latency.
    for _ in range(100):

        session.run(
            None,
            {
                input_meta.name:
                    sample
            }
        )

    repetitions = 1000

    start = time.perf_counter()

    for _ in range(repetitions):

        session.run(
            None,
            {
                input_meta.name:
                    sample
            }
        )

    latency_ms = (
        (
            time.perf_counter() - start
        )
        / repetitions
        * 1000
    )

    print(
        f"ONNX latency: "
        f"{latency_ms:.4f} ms"
    )

    print(
        "\nONNX verification: PASSED"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print(
        "=" * 60
    )

    print(
        "EDGE AI TRAINING - "
        "RESEARCH VERSION"
    )

    print(
        "=" * 60
    )

    # ------------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------------

    X, y, labels, scenario_ids = (
        load_data()
    )

    # ------------------------------------------------------------------------
    # Validate data
    # ------------------------------------------------------------------------

    validate_data(
        X,
        y,
        labels,
        scenario_ids
    )

    # ------------------------------------------------------------------------
    # Train and evaluate candidate models
    # ------------------------------------------------------------------------

    results, labels = (
        train_and_evaluate(
            X,
            y,
            labels,
            scenario_ids
        )
    )

    # ------------------------------------------------------------------------
    # Generate evaluation plots
    # ------------------------------------------------------------------------

    plot_comparison(
        results,
        labels
    )

    plot_confusion_matrices(
        results,
        labels
    )

    # ------------------------------------------------------------------------
    # Select the model with the highest Macro-F1
    # ------------------------------------------------------------------------

    best_name = max(
        results,
        key=lambda k:
            results[k]["macro_f1"]
    )

    best_model = results[
        best_name
    ]["model"]

    print(
        "\n" + "=" * 60
    )

    print(
        "SELECTED MODEL"
    )

    print(
        "=" * 60
    )

    print(
        f"Model    : {best_name}"
    )

    print(
        f"Accuracy : "
        f"{results[best_name]['accuracy']:.4f}"
    )

    print(
        f"Macro F1 : "
        f"{results[best_name]['macro_f1']:.4f}"
    )

    # ------------------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------------------

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            results[best_name]["y_test"],
            results[best_name]["y_pred"],
            labels=np.arange(
                len(labels)
            ),
            target_names=labels,
            zero_division=0
        )
    )

    # ------------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------------

    plot_feature_importance(
        best_model
    )

    # ------------------------------------------------------------------------
    # Save the selected scikit-learn model
    # ------------------------------------------------------------------------

    model_pkl = (
        AI_DIR / "model.pkl"
    )

    joblib.dump(
        best_model,
        model_pkl
    )

    print(
        f"Scikit-learn model saved: "
        f"{model_pkl}"
    )

    # ------------------------------------------------------------------------
    # Export to ONNX
    # ------------------------------------------------------------------------

    onnx_file = export_onnx(
        best_model
    )

    # ------------------------------------------------------------------------
    # Verify ONNX model and measure inference latency
    # ------------------------------------------------------------------------

    verify_onnx(
        onnx_file,
        X
    )

    # ------------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "TRAINING COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"  {model_pkl}"
    )

    print(
        f"  {onnx_file}"
    )

    print(
        f"  {RESULTS_DIR / 'model_comparison.png'}"
    )

    print(
        f"  {RESULTS_DIR / 'confusion_matrices.png'}"
    )

    print(
        f"  {RESULTS_DIR / 'feature_importance.png'}"
    )


if __name__ == "__main__":
    main()

