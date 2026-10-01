
"""
influxdb_writer.py

Writes MQTT Edge AI / FPGA events into InfluxDB.

Architecture:
    FPGA / Edge AI
          |
          v
        MQTT
          |
          v
    InfluxDB Writer
          |
          v
      InfluxDB
          |
          v
       Grafana

Project 1 parameters preserved:
    Sample rate       : 1 MHz
    Window size       : 256 samples
    FPGA trigger      : preserved
    FPGA event count  : preserved
    FPGA max duration : preserved
    FPGA metadata     : preserved

InfluxDB measurement:
    cable_events

Tags:
    device_id
    event_type
    status
    fpga_triggered

Fields:
    confidence
    latency_ms
    health_score
    fpga_event_count
    fpga_max_duration
    signal_rms
    sample_rate_hz
    window_size
    fpga_timestamp_us
"""

from datetime import datetime, timezone

try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
    from influxdb_client.rest import ApiException

    INFLUX_AVAILABLE = True

except ImportError:
    INFLUX_AVAILABLE = False

    print("influxdb-client is not installed.")
    print("Install it with: pip install influxdb-client")


# ============================================================================
# InfluxDB configuration
# ============================================================================

INFLUX_URL = "http://localhost:8086"

# Replace this value with a valid InfluxDB API token.
INFLUX_TOKEN = "cable-harness-token"

INFLUX_ORG = "cable-harness"
INFLUX_BUCKET = "cable_harness"


# ============================================================================
# InfluxDB writer
# ============================================================================

class InfluxDBWriter:
    """Write cable-harness monitoring events into InfluxDB."""

    def __init__(self):
        self.client = None
        self.write_api = None

        if not INFLUX_AVAILABLE:
            print("InfluxDB client unavailable.")
            return

        if (
            not INFLUX_TOKEN
            or INFLUX_TOKEN == "REPLACE_WITH_YOUR_INFLUXDB_TOKEN"
        ):
            print("\nINFLUX_TOKEN is not configured.")
            print("Add your InfluxDB API token to INFLUX_TOKEN.")
            return

        try:
            self.client = InfluxDBClient(
                url=INFLUX_URL,
                token=INFLUX_TOKEN,
                org=INFLUX_ORG,
            )

            self.write_api = self.client.write_api(
                write_options=SYNCHRONOUS
            )

            print(f"Connected to InfluxDB: {INFLUX_URL}")
            print(f"   Organization : {INFLUX_ORG}")
            print(f"   Bucket       : {INFLUX_BUCKET}")

        except Exception as e:
            print("\nUnable to create the InfluxDB connection.")
            print(f"Error: {e}")

    # ------------------------------------------------------------------------
    # Write event
    # ------------------------------------------------------------------------

    def write_event(self, event_data: dict):
        """
        Write one FPGA / Edge AI event to InfluxDB.

        Project 1 parameters are preserved in the stored event.
        """

        if self.write_api is None:
            print("Write skipped: InfluxDB is unavailable.")
            return False

        # Required event data
        device_id = str(event_data["device_id"])
        event_type = str(event_data["event_type"])
        status = str(event_data["status"])
        fpga_triggered = bool(event_data["fpga_triggered"])

        confidence = float(event_data["confidence"])
        latency_ms = float(event_data["latency_ms"])
        health_score = float(event_data["health_score"])

        fpga_event_count = int(event_data["fpga_event_count"])
        fpga_max_duration = int(event_data["fpga_max_duration"])

        signal_rms = float(event_data["signal_rms"])

        # Project 1 parameters
        sample_rate_hz = int(
            event_data.get("sample_rate_hz", 1_000_000)
        )

        window_size = int(
            event_data.get("window_size", 256)
        )

        # Preserve the original FPGA timestamp separately.
        fpga_timestamp_us = int(
            event_data.get("timestamp_us", 0)
        )

        # InfluxDB uses a real UTC timestamp for its _time field.
        # The original FPGA timestamp is preserved as a separate field.
        influx_timestamp = datetime.now(timezone.utc)

        point = (
            Point("cable_events")

            # Tags
            .tag("device_id", device_id)
            .tag("event_type", event_type)
            .tag("status", status)
            .tag("fpga_triggered", str(fpga_triggered))

            # Edge AI fields
            .field("confidence", confidence)
            .field("latency_ms", latency_ms)
            .field("health_score", health_score)

            # FPGA event fields
            .field("fpga_event_count", fpga_event_count)
            .field("fpga_max_duration", fpga_max_duration)

            # Signal field
            .field("signal_rms", signal_rms)

            # Project 1 parameters
            .field("sample_rate_hz", sample_rate_hz)
            .field("window_size", window_size)

            # Original FPGA timestamp
            .field("fpga_timestamp_us", fpga_timestamp_us)

            # InfluxDB timestamp
            .time(influx_timestamp)
        )

        try:
            self.write_api.write(
                bucket=INFLUX_BUCKET,
                record=point,
            )

            print(
                f"[InfluxDB] Event written: "
                f"{event_type} "
                f"(health={health_score}%)"
            )

            return True

        except ApiException as e:
            print("\nInfluxDB API error.")
            print(f"HTTP status : {e.status}")
            print(f"Reason      : {e.reason}")

            if e.status == 401:
                print("\n401 Unauthorized.")
                print(
                    "The InfluxDB token is invalid or does not "
                    "have the required permissions."
                )
                print(
                    f"Verify that the token has WRITE access "
                    f"to bucket: {INFLUX_BUCKET}"
                )
                print(f"Expected organization: {INFLUX_ORG}")

            elif e.status == 404:
                print("\nBucket or organization not found.")

            return False

        except Exception as e:
            print("\nError while writing to InfluxDB:")
            print(e)
            return False

    # ------------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------------

    def close(self):
        """Close the InfluxDB write API and client."""

        if self.write_api is not None:
            try:
                self.write_api.close()
            except Exception:
                pass

        if self.client is not None:
            try:
                self.client.close()
                print("InfluxDB connection closed.")
            except Exception:
                pass


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("INFLUXDB WRITER TEST")
    print("=" * 60 + "\n")

    writer = InfluxDBWriter()

    # Test event aligned with the Project 1 configuration.
    test_event = {
        "device_id": "CABLE_TEST_01",

        "timestamp_us": int(
            datetime.now(timezone.utc).timestamp() * 1_000_000
        ),

        "event_type": "micro_cut",
        "confidence": 0.97,
        "latency_ms": 4.8,

        "fpga_triggered": True,
        "fpga_event_count": 1,
        "fpga_max_duration": 5,

        "health_score": 94.0,
        "status": "WARNING",
        "signal_rms": 0.71,

        # Project 1 parameters
        "sample_rate_hz": 1_000_000,
        "window_size": 256,
    }

    success = writer.write_event(test_event)

    if success:
        print("\n" + "=" * 60)
        print("TEST PASSED")
        print("The event was written to InfluxDB.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("TEST FAILED")
        print("Check the InfluxDB token and permissions.")
        print("=" * 60)

    writer.close()

