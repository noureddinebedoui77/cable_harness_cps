
"""
mqtt_subscriber.py

MQTT -> InfluxDB subscriber.

Architecture:

    Edge AI / FPGA
          |
          v
        MQTT
          |
          v
    MQTT Subscriber
          |
          v
      InfluxDB
          |
          v
       Grafana
"""

import json

import paho.mqtt.client as mqtt

from influxdb_writer import InfluxDBWriter


# ============================================================================
# MQTT configuration
# ============================================================================

MQTT_BROKER = "localhost"
MQTT_PORT = 1883

MQTT_TOPIC_EVENTS = "cable/01/events"
MQTT_TOPIC_STATUS = "cable/01/status"


# ============================================================================
# InfluxDB
# ============================================================================

writer = InfluxDBWriter()


# ============================================================================
# MQTT callbacks
# ============================================================================

def on_connect(client, userdata, flags, reason_code, properties):
    """Handle the MQTT broker connection."""

    print()
    print("=" * 60)
    print("MQTT SUBSCRIBER")
    print("=" * 60)

    if reason_code == 0:
        print(
            f"Connected to MQTT broker: "
            f"{MQTT_BROKER}:{MQTT_PORT}"
        )

        client.subscribe(MQTT_TOPIC_EVENTS, qos=1)
        client.subscribe(MQTT_TOPIC_STATUS, qos=1)

        print(f"Subscribed to: {MQTT_TOPIC_EVENTS}")
        print(f"Subscribed to: {MQTT_TOPIC_STATUS}")
        print("=" * 60)

    else:
        print(
            f"MQTT connection failed: {reason_code}"
        )


def on_message(client, userdata, msg):
    """Process incoming MQTT messages."""

    print()
    print("-" * 60)
    print("MQTT message received")
    print(f"Topic: {msg.topic}")

    try:
        payload = json.loads(
            msg.payload.decode("utf-8")
        )

    except Exception as e:
        print(f"Invalid JSON: {e}")
        return

    # ------------------------------------------------------------------------
    # Event messages
    # ------------------------------------------------------------------------

    if msg.topic == MQTT_TOPIC_EVENTS:
        print(
            f"Event: "
            f"{payload.get('event_type', 'unknown')}"
        )

        print(
            f"Health: "
            f"{payload.get('health_score', 'N/A')}"
        )

        print(
            f"Confidence: "
            f"{payload.get('confidence', 'N/A')}"
        )

        success = writer.write_event(payload)

        if success:
            print("MQTT event written to InfluxDB.")
        else:
            print("Failed to write event to InfluxDB.")

    # ------------------------------------------------------------------------
    # Status messages
    # ------------------------------------------------------------------------

    elif msg.topic == MQTT_TOPIC_STATUS:
        print(
            f"Status: "
            f"{payload.get('status', 'N/A')}"
        )

        print(
            f"Health: "
            f"{payload.get('health_score', 'N/A')}"
        )

        print(
            "Status message received "
            "(not stored in cable_events)."
        )


# ============================================================================
# Main
# ============================================================================

def main():
    """Start the MQTT subscriber."""

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2
    )

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        print(
            f"Connecting to MQTT: "
            f"{MQTT_BROKER}:{MQTT_PORT}"
        )

        client.connect(
            MQTT_BROKER,
            MQTT_PORT,
            60
        )

        print("Subscriber started.")
        print("Waiting for MQTT events...")

        client.loop_forever()

    except KeyboardInterrupt:
        print("\nSubscriber stopped.")

    except Exception as e:
        print(f"\nMQTT error: {e}")

    finally:
        try:
            client.disconnect()
        except Exception:
            pass

        writer.close()


# ============================================================================
# Start
# ============================================================================

if __name__ == "__main__":
    main()

