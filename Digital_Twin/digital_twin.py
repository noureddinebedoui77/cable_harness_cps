
"""
twin.py

Lightweight Digital Twin for intelligent cable-harness monitoring.

The Digital Twin maintains a software representation of the physical
cable-harness state.

Inputs:
    - Edge AI event classification
    - AI confidence
    - FPGA detection metadata

Outputs:
    - Health score
    - Cable status
    - Event statistics
    - FPGA detection statistics
"""

from dataclasses import dataclass, field
from typing import Dict


# ============================================================================
# Cable state
# ============================================================================

@dataclass
class CableState:
    """Current state of a monitored cable harness."""

    device_id: str

    health_score: float = 100.0
    status: str = "NORMAL"

    last_event: str = "normal"
    last_confidence: float = 0.0

    total_events: int = 0
    fpga_detections: int = 0

    event_counts: Dict[str, int] = field(
        default_factory=lambda: {
            "normal": 0,
            "micro_cut": 0,
            "noise": 0,
            "transient": 0,
            "abrupt": 0,
        }
    )


# ============================================================================
# Digital Twin
# ============================================================================

class DigitalTwin:
    """Maintain and update the software state of a cable harness."""

    def __init__(self, device_id: str):
        self.state = CableState(
            device_id=device_id
        )

    @staticmethod
    def event_penalty(event_type: str) -> float:
        """
        Return the health-score penalty associated with an event type.
        """

        penalties = {
            "normal": 0.0,
            "noise": 0.8,
            "transient": 1.0,
            "micro_cut": 1.4,
            "abrupt": 3.0,
        }

        return penalties.get(event_type, 0.0)

    def update(
        self,
        event_type: str,
        confidence: float,
        fpga_info=None,
    ):
        """Update the Digital Twin using an Edge AI event."""

        event_type = str(
            event_type
        ).strip().lower()

        confidence = max(
            0.0,
            min(
                1.0,
                float(confidence),
            ),
        )

        if event_type not in self.state.event_counts:
            self.state.event_counts[event_type] = 0

        # Update event statistics.
        self.state.total_events += 1
        self.state.event_counts[event_type] += 1
        self.state.last_event = event_type
        self.state.last_confidence = confidence

        # Update FPGA detection statistics.
        if isinstance(fpga_info, dict):
            fpga_triggered = bool(
                fpga_info.get(
                    "fpga_triggered",
                    False,
                )
            )

            if fpga_triggered:
                self.state.fpga_detections += 1

        # Apply confidence-weighted health degradation.
        penalty = self.event_penalty(event_type)

        effective_penalty = (
            penalty * confidence
        )

        self.state.health_score -= effective_penalty

        # Keep the health score within [0, 100].
        self.state.health_score = max(
            0.0,
            min(
                100.0,
                self.state.health_score,
            ),
        )

        self.update_status()

    def update_status(self):
        """Update the cable status from the current health score."""

        score = self.state.health_score

        if score >= 80.0:
            self.state.status = "NORMAL"

        elif score >= 60.0:
            self.state.status = "WARNING"

        else:
            self.state.status = "CRITICAL"

    def reset(self):
        """Reset the Digital Twin to its initial state."""

        device_id = self.state.device_id

        self.state = CableState(
            device_id=device_id
        )

    def display(self):
        """Display the current Digital Twin state."""

        print()
        print("=" * 50)
        print("DIGITAL TWIN")
        print("=" * 50)

        print(
            f"Device       : "
            f"{self.state.device_id}"
        )

        print(
            f"Health Score : "
            f"{self.state.health_score:.1f}%"
        )

        print(
            f"Status       : "
            f"{self.state.status}"
        )

        print(
            f"Last Event   : "
            f"{self.state.last_event}"
        )

        print(
            f"Confidence   : "
            f"{self.state.last_confidence * 100:.2f}%"
        )

        print(
            f"Total Events : "
            f"{self.state.total_events}"
        )

        print(
            f"FPGA Events  : "
            f"{self.state.fpga_detections}"
        )

        print()
        print("Event Statistics:")

        for event_type, count in (
            self.state.event_counts.items()
        ):
            print(
                f"  {event_type:10s}: "
                f"{count}"
            )

        print("=" * 50)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DIGITAL TWIN TEST")
    print("=" * 60)

    twin = DigitalTwin("CABLE_TEST_01")

    print("\nInitial state:")
    twin.display()

    print("\n--- Normal event ---")
    twin.update(
        event_type="normal",
        confidence=0.99,
        fpga_info={
            "fpga_triggered": False,
            "fpga_event_count": 0,
        },
    )
    twin.display()

    print("\n--- Micro-cut event ---")
    twin.update(
        event_type="micro_cut",
        confidence=0.98,
        fpga_info={
            "fpga_triggered": False,
            "fpga_event_count": 0,
        },
    )
    twin.display()

    print("\n--- FPGA abrupt event ---")
    twin.update(
        event_type="abrupt",
        confidence=0.97,
        fpga_info={
            "fpga_triggered": True,
            "fpga_event_count": 1,
        },
    )
    twin.display()

    print("\n" + "=" * 60)
    print("DIGITAL TWIN TEST COMPLETED")
    print("=" * 60)

