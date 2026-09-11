"""Integration tests for the telemetry direction of the bridge: real EMBER
EventBus events, published from the C++ side via pipeline_test_server.cpp's
stdin INJECT_* control commands, flowing over the real socket into
EmberBridgeClient and being relayed into the real Coordinator/agent
pipeline by EmberTelemetryRelay.

Note what this does and doesn't claim: PlannerAgent (agents/agent/planner_agent/planner.py)
has no case for action="telemetry_alert" today, so Coordinator.dispatch
returns an ERROR Message for "unsupported action" -- EmberTelemetryRelay's
job is delivery, not business logic (see its docstring and
design-decisions/05-ember-agentcore-bridge.md). These tests assert the
Message actually reaches Planner.handle_message with the right shape, not
that the Planner "does" anything with it yet.
"""

from __future__ import annotations

import time

import pytest


@pytest.mark.integration
class TestEmberTelemetryIntegration:
    def _spy_on_planner(self, ember_coordinator):
        planner = ember_coordinator.all_agents["Planner"]
        received = []
        original_handle_message = planner.handle_message

        def spy(msg):
            received.append(msg)
            return original_handle_message(msg)

        planner.handle_message = spy
        return received

    def _wait_for(self, received, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not received:
            time.sleep(0.02)
        return received

    def test_hardware_fault_reaches_planner(self, ember_server, ember_telemetry_relay, ember_coordinator):
        received = self._spy_on_planner(ember_coordinator)

        ember_server.inject_fault(0x1001, "grip_motor", "stalled under load")
        self._wait_for(received)

        assert len(received) == 1
        msg = received[0]
        assert msg.sender == "EmberBridge"
        assert msg.receiver == "Planner"
        assert msg.action == "telemetry_alert"
        assert msg.payload["subsystem"] == "FAULT"
        assert msg.payload["code"] == 0x1001
        assert "grip_motor" in msg.payload["text"]
        assert "stalled under load" in msg.payload["text"]

    def test_battery_low_reaches_planner(self, ember_server, ember_telemetry_relay, ember_coordinator):
        received = self._spy_on_planner(ember_coordinator)

        ember_server.inject_battery_low(percentage=12, voltage=10.4)
        self._wait_for(received)

        assert len(received) == 1
        msg = received[0]
        assert msg.action == "telemetry_alert"
        assert msg.payload["subsystem"] == "BATTERY"
        assert msg.payload["value_a"] == pytest.approx(12.0)
        assert msg.payload["value_b"] == pytest.approx(10.4, abs=1e-3)

    def test_thermal_warning_reaches_planner(self, ember_server, ember_telemetry_relay, ember_coordinator):
        received = self._spy_on_planner(ember_coordinator)

        ember_server.inject_thermal("drive_motor", 87.5)
        self._wait_for(received)

        assert len(received) == 1
        msg = received[0]
        assert msg.payload["subsystem"] == "THERMAL"
        assert msg.payload["value_a"] == pytest.approx(87.5, abs=1e-3)

    def test_multiple_alerts_all_arrive_in_order(self, ember_server, ember_telemetry_relay, ember_coordinator):
        received = self._spy_on_planner(ember_coordinator)

        ember_server.inject_battery_low(percentage=40, voltage=11.0)
        ember_server.inject_thermal("drive_motor", 60.0)
        ember_server.inject_fault(0x2, "wheel_encoder", "no signal")

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(received) < 3:
            time.sleep(0.02)

        assert len(received) == 3
        assert [m.payload["subsystem"] for m in received] == ["BATTERY", "THERMAL", "FAULT"]

    def test_telemetry_relay_is_a_noop_when_planner_not_registered(self, ember_server, ember_bridge_client):
        # No ember_coordinator/ember_telemetry_relay fixture pulled in here:
        # exercises EmberTelemetryRelay directly against a Coordinator with
        # no agents registered, confirming on_telemetry() degrades safely
        # (see its "planner_agent_name not in all_agents" guard) rather than
        # raising when nothing is listening yet.
        from agents.agent.execution_agent.ember_telemetry_relay import EmberTelemetryRelay
        from agents.master_planner.coordinator import Coordinator

        empty_coordinator = Coordinator()
        relay = EmberTelemetryRelay(empty_coordinator, planner_agent_name="Planner")
        ember_bridge_client.set_telemetry_callback(relay.on_telemetry)

        ember_server.inject_fault(1, "x", "y")
        time.sleep(0.3)  # nothing to wait_for(): asserting silence, not an event

        assert empty_coordinator.all_agents == {}
