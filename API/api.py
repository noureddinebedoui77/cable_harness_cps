
"""
FastAPI REST API for the cable-harness Cyber-Physical System.

Architecture:

    Edge AI / FPGA
          |
          v
        MQTT
          |
          v
      FastAPI API
          |
          +----> Events
          +----> Predictions
          +----> Diagnostics
          |
          v
       Monitoring

The API subscribes to MQTT topics and exposes the latest
system state and monitoring history through REST endpoints.
"""

import json
import os
import threading
import time
from contextlib import asynccontextmanager

import paho.mqtt.client as mqtt
from fastapi import FastAPI, HTTPException


# ============================================================================
# Configuration
# ============================================================================

# Use "mosquitto" when running inside Docker.
# Local execution defaults to localhost.
MQTT_BROKER = os.getenv(
    "MQTT_BROKER",
    "localhost",
)

MQTT_PORT = int(
    os.getenv(
        "MQTT_PORT",
        "1883",
    )
)

MQTT_TOPIC_EVENTS = os.getenv(
    "MQTT_EVENTS_TOPIC",
    "cable/01/events",
)

MQTT_TOPIC_STATUS = os.getenv(
    "MQTT_STATUS_TOPIC",
    "cable/01/status",
)

MQTT_TOPIC_DIAGNOSTICS = os.getenv(
    "MQTT_DIAGNOSTICS_TOPIC",
    "cable/01/diagnostics",
)

DEVICE_ID = "CABLE_TEST_01"
MAX_HISTORY = 100


# ============================================================================
# Runtime storage
# ============================================================================

events_history = []
predictions_history = []
diagnostics_history = []

system_state = {
    "device_id": DEVICE_ID,
    "status": "offline",
    "health_score": 100.0,
    "device_status": "UNKNOWN",
    "total_events": 0,
    "fpga_events": 0,
    "last_event": None,
    "last_confidence": 0.0,
    "last_timestamp_us": None,
}

data_lock = threading.Lock()
mqtt_client = None


# ============================================================================
# Helpers
# ============================================================================

def add_to_history(history, item):
    """Append an item while keeping only the latest MAX_HISTORY entries."""

    history.append(item)

    if len(history) > MAX_HISTORY:
        del history[:-MAX_HISTORY]


# ============================================================================
# MQTT callbacks
# ============================================================================

def on_connect(
    client,
    userdata,
    flags,
    reason_code,
    properties,
):
    """Handle the MQTT broker connection."""

    print(
        f"[FastAPI MQTT] Connected to "
        f"{MQTT_BROKER}:{MQTT_PORT}"
    )

    client.subscribe(
        MQTT_TOPIC_EVENTS,
        qos=1,
    )

    client.subscribe(
        MQTT_TOPIC_STATUS,
        qos=1,
    )

    client.subscribe(
        MQTT_TOPIC_DIAGNOSTICS,
        qos=1,
    )

    print("[FastAPI MQTT] Subscribed to:")
    print(f"  - {MQTT_TOPIC_EVENTS}")
    print(f"  - {MQTT_TOPIC_STATUS}")
    print(f"  - {MQTT_TOPIC_DIAGNOSTICS}")


def on_disconnect(
    client,
    userdata,
    disconnect_flags,
    reason_code,
    properties,
):
    """Handle MQTT disconnection."""

    print(
        f"[FastAPI MQTT] Disconnected. "
        f"Reason: {reason_code}"
    )

    with data_lock:
        system_state["status"] = "offline"


def on_message(
    client,
    userdata,
    message,
):
    """Process incoming MQTT messages."""

    try:
        payload = json.loads(
            message.payload.decode("utf-8")
        )

    except Exception as exc:
        print(
            f"[FastAPI MQTT] Invalid JSON: {exc}"
        )
        return

    topic = message.topic

    # ------------------------------------------------------------------------
    # Event messages
    # ------------------------------------------------------------------------

    if topic == MQTT_TOPIC_EVENTS:

        with data_lock:
            add_to_history(
                events_history,
                payload,
            )

            system_state["status"] = "online"

            system_state["total_events"] = len(
                events_history
            )

            system_state["last_event"] = payload.get(
                "event_type"
            )

            system_state["last_confidence"] = float(
                payload.get(
                    "confidence",
                    0.0,
                )
            )

            system_state["last_timestamp_us"] = (
                payload.get("timestamp_us")
            )

            # health_score may legitimately be None for /test-event.
            health_score = payload.get("health_score")

            if health_score is not None:
                system_state["health_score"] = float(
                    health_score
                )

            system_state["device_status"] = payload.get(
                "status",
                "UNKNOWN",
            )

            if payload.get(
                "fpga_triggered",
                False,
            ):
                system_state["fpga_events"] += 1

            prediction = {
                "device_id": payload.get(
                    "device_id",
                    DEVICE_ID,
                ),
                "timestamp_us": payload.get(
                    "timestamp_us"
                ),
                "model": "Edge AI",
                "prediction": payload.get(
                    "event_type"
                ),
                "confidence": payload.get(
                    "confidence",
                    0.0,
                ),
                "latency_ms": payload.get(
                    "latency_ms"
                ),
                "class_probabilities": payload.get(
                    "class_probabilities"
                ),
            }

            add_to_history(
                predictions_history,
                prediction,
            )

        print(
            "[FastAPI MQTT] Event received: "
            f"{payload.get('event_type')} "
            f"| confidence="
            f"{payload.get('confidence')}"
        )

    # ------------------------------------------------------------------------
    # Status messages
    # ------------------------------------------------------------------------

    elif topic == MQTT_TOPIC_STATUS:

        with data_lock:
            system_state["status"] = "online"

            if "health_score" in payload:
                health_score = payload.get(
                    "health_score"
                )

                if health_score is not None:
                    system_state["health_score"] = float(
                        health_score
                    )

            if "status" in payload:
                system_state["device_status"] = (
                    payload["status"]
                )

            if "timestamp_us" in payload:
                system_state["last_timestamp_us"] = (
                    payload["timestamp_us"]
                )

        print(
            "[FastAPI MQTT] Status received: "
            f"{payload.get('status')} "
            f"| health="
            f"{payload.get('health_score')}"
        )

    # ------------------------------------------------------------------------
    # Diagnostics messages
    # ------------------------------------------------------------------------

    elif topic == MQTT_TOPIC_DIAGNOSTICS:

        with data_lock:
            add_to_history(
                diagnostics_history,
                payload,
            )

        print(
            "[FastAPI MQTT] Diagnostics received."
        )


# ============================================================================
# MQTT client
# ============================================================================

def create_mqtt_client():
    """Create, configure, and start the MQTT client."""

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2
    )

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    client.connect(
        MQTT_BROKER,
        MQTT_PORT,
        60,
    )

    client.loop_start()

    return client


def shutdown_mqtt():
    """Stop and disconnect the MQTT client."""

    global mqtt_client

    if mqtt_client is not None:
        try:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()

        except Exception:
            pass

        mqtt_client = None


# ============================================================================
# FastAPI lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app):
    """Initialize and shut down the MQTT subscriber."""

    global mqtt_client

    print()
    print("=" * 60)
    print("STARTING CABLE HARNESS CPS API")
    print("=" * 60)

    print(
        f"[FastAPI] MQTT broker: "
        f"{MQTT_BROKER}:{MQTT_PORT}"
    )

    try:
        mqtt_client = create_mqtt_client()

        with data_lock:
            system_state["status"] = "online"

        print(
            "[FastAPI] MQTT subscriber started."
        )

    except Exception as exc:
        print(
            "[FastAPI] MQTT connection failed:"
        )
        print(str(exc))

        with data_lock:
            system_state["status"] = "offline"

    yield

    print()
    print(
        "[FastAPI] Shutting down MQTT subscriber..."
    )

    shutdown_mqtt()

    print(
        "[FastAPI] Shutdown complete."
    )


# ============================================================================
# FastAPI application
# ============================================================================

app = FastAPI(
    title="Cable Harness CPS API",
    description=(
        "REST API for Edge AI, FPGA, and "
        "Digital Twin monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Root endpoint
# ============================================================================

@app.get("/")
def root():
    """Return basic API information."""

    return {
        "system": "Cable Harness CPS",
        "status": "online",
    }


# ============================================================================
# Status endpoint
# ============================================================================

@app.get("/status")
def status():
    """Return the current system state."""

    with data_lock:
        return dict(system_state)


# ============================================================================
# Events endpoint
# ============================================================================

@app.get("/events")
def events():
    """Return the stored event history."""

    with data_lock:
        return {
            "device_id": DEVICE_ID,
            "total_events": len(events_history),
            "events": list(events_history),
        }


# ============================================================================
# Predictions endpoint
# ============================================================================

@app.get("/predictions")
def predictions():
    """Return the stored Edge AI prediction history."""

    with data_lock:
        return {
            "device_id": DEVICE_ID,
            "total_predictions": len(
                predictions_history
            ),
            "predictions": list(
                predictions_history
            ),
        }


# ============================================================================
# Diagnostics endpoint
# ============================================================================

@app.get("/diagnostics")
def diagnostics():
    """Return the stored diagnostics history."""

    with data_lock:
        return {
            "device_id": DEVICE_ID,
            "total_diagnostics": len(
                diagnostics_history
            ),
            "diagnostics": list(
                diagnostics_history
            ),
        }


# ============================================================================
# Test event endpoint
# ============================================================================

@app.post("/test-event")
def test_event(
    event_type: str = "micro_cut",
    confidence: float = 0.98,
    fpga_triggered: bool = False,
):
    """
    Publish a synthetic test event through MQTT.

    This endpoint is intended for validating the MQTT-to-API
    monitoring pipeline without requiring a live Edge AI event.
    """

    valid_event_types = {
        "normal",
        "micro_cut",
        "noise",
        "transient",
        "abrupt",
    }

    if event_type not in valid_event_types:
        raise HTTPException(
            status_code=400,
            detail="Invalid event_type.",
        )

    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="confidence must be between 0 and 1.",
        )

    if mqtt_client is None:
        raise HTTPException(
            status_code=503,
            detail="MQTT client is not connected.",
        )

    timestamp_us = int(
        time.time() * 1_000_000
    )

    event_data = {
        "device_id": DEVICE_ID,

        "timestamp_us": timestamp_us,

        "event_type": event_type,

        "confidence": confidence,

        "latency_ms": None,

        "fpga_triggered": fpga_triggered,

        "fpga_event_count": (
            1 if fpga_triggered else 0
        ),

        "fpga_max_duration": 0,

        "fpga_first_trigger_index": -1,

        "health_score": None,

        "status": "TEST",

        "signal_rms": None,

        "sample_rate_hz": 1_000_000,

        "window_size": 256,

        "feature_count": 20,

        "ground_truth": "API_TEST",
    }

    result = mqtt_client.publish(
        MQTT_TOPIC_EVENTS,
        json.dumps(event_data),
        qos=1,
    )

    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        raise HTTPException(
            status_code=500,
            detail=(
                "MQTT publication failed: "
                f"{result.rc}"
            ),
        )

    return {
        "status": "test_event_published",
        "topic": MQTT_TOPIC_EVENTS,
        "event": event_data,
    }
