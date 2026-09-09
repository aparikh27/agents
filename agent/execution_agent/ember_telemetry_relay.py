"""Relays fault/battery/thermal/connection telemetry pushed by EMBER into
AgentCore's synchronous Message/Coordinator pipeline, so the Planner can
react to hardware state changes without polling for them.

Periodic Motion/Manipulator samples are NOT relayed here -- those are
pulled on demand via EmberBridgeClient.get_latest_telemetry() (see
EmberRobotDriver.get_distance_to_front) rather than pushed as Messages,
since a Message per encoder tick would flood the Coordinator for state a
synchronous getter already serves as a last-known-value read.

The Planner agent must implement handling for action="telemetry_alert" to
actually do anything with these; this relay only delivers them.
"""

from __future__ import annotations

import logging
import threading

from agents.master_planner.coordinator import Coordinator
from agents.messaging import Message, MessageStatus, MessageType
from agents.messaging.ember_wire import TelemetryFrame, TelemetrySubsystem

logger = logging.getLogger(__name__)

_ALERT_SUBSYSTEMS = frozenset(
    {
        TelemetrySubsystem.FAULT,
        TelemetrySubsystem.BATTERY,
        TelemetrySubsystem.THERMAL,
        TelemetrySubsystem.CONNECTION,
    }
)


class EmberTelemetryRelay:
    """Pass `self.on_telemetry` as EmberBridgeClient(on_telemetry=...).

    agents/master_planner/coordinator.py's Coordinator.dispatch is a plain
    dict lookup plus a direct synchronous call -- it documents no
    thread-safety guarantee. EmberBridgeClient's telemetry callback fires
    on its own background event-loop thread, which can race a request
    thread (e.g. FastAPI) also calling coordinator.dispatch(). This relay
    serializes its own dispatches behind one lock; it cannot protect
    against some other caller hitting the Coordinator concurrently and
    unlocked -- if that matters for a given deployment, the fix belongs in
    Coordinator itself, out of scope for this bridge.
    """

    def __init__(self, coordinator: Coordinator, planner_agent_name: str = "Planner"):
        self._coordinator = coordinator
        self._planner_agent_name = planner_agent_name
        self._lock = threading.Lock()

    def on_telemetry(self, frame: TelemetryFrame) -> None:
        if frame.subsystem not in _ALERT_SUBSYSTEMS:
            return
        if self._planner_agent_name not in self._coordinator.all_agents:
            return

        message = Message(
            sender="EmberBridge",
            receiver=self._planner_agent_name,
            action="telemetry_alert",
            payload={
                "subsystem": frame.subsystem.name,
                "value_a": frame.value_a,
                "value_b": frame.value_b,
                "code": frame.code,
                "text": frame.text,
            },
            status=MessageStatus.SUCCESS,
            message_type=MessageType.EVENT,
        )

        with self._lock:
            try:
                self._coordinator.dispatch(message)
            except Exception:
                logger.exception(
                    "EmberTelemetryRelay: dispatch to %s failed", self._planner_agent_name
                )
