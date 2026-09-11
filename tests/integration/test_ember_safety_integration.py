"""Integration tests for the bridge's real-time-isolation guarantees (see
"Safety & Real-Time Isolation Guarantees" in
design-decisions/05-ember-agentcore-bridge.md), exercised against the real
compiled EMBER process rather than asserted only by reading the code.
"""

from __future__ import annotations

import pytest

from agents.agent.execution_agent.ember_bridge_client import EmberBridgeClient
from agents.agent.execution_agent.ember_execution_agent import EmberRobotDriver
from agents.messaging.ember_wire import AckResult, CommandOp
from agents.tests.integration.ember_test_server import find_free_port


@pytest.mark.integration
class TestEmberWatchdogIntegration:
    @pytest.fixture
    def ember_bridge_client_no_heartbeat(self, ember_server):
        """A client with heartbeats effectively disabled, so nothing keeps
        refreshing EMBER's watchdog deadline after the one command this
        test sends -- isolates the watchdog's own autonomous-stop behavior
        from the client's normal keep-alive traffic."""
        client = EmberBridgeClient(
            host="127.0.0.1",
            port=ember_server.port,
            command_timeout_s=3.0,
            heartbeat_interval_s=10_000.0,
        )
        client.start()
        assert client.wait_connected(5.0)
        try:
            yield client
        finally:
            client.stop()

    def test_watchdog_autonomously_stops_robot_when_agentcore_goes_silent(
        self, ember_server, ember_bridge_client_no_heartbeat
    ):
        client = ember_bridge_client_no_heartbeat

        result = client.send_command(CommandOp.MOVE_FORWARD, param=1.0)
        assert result == AckResult.ACCEPTED

        # No further command or heartbeat follows. ember_server is
        # configured (conftest.py) with a 300ms watchdog deadline -- EMBER's
        # own Scheduler-driven watchdog task must self-issue a stop within
        # that window, entirely independent of AgentCore continuing to run.
        stop_line = ember_server.wait_for_line(
            lambda line: line.startswith("CMD_RECEIVED") and "topic=cmd/motion/stop" in line,
            timeout=2.0,
        )
        assert stop_line is not None, (
            "EMBER did not autonomously stop after the watchdog deadline elapsed:\n"
            + "\n".join(ember_server.all_lines())
        )

    def test_heartbeats_prevent_the_watchdog_from_tripping(self, ember_server, ember_bridge_client):
        """Sanity check on the other side of the same behavior: with normal
        heartbeats (the default 100ms interval), an in-flight move must NOT
        be autonomously stopped just for taking a while."""
        result = ember_bridge_client.send_command(CommandOp.MOVE_FORWARD, param=1.0)
        assert result == AckResult.ACCEPTED

        stop_line = ember_server.wait_for_line(
            lambda line: line.startswith("CMD_RECEIVED") and "topic=cmd/motion/stop" in line,
            timeout=1.0,
        )
        assert stop_line is None, "watchdog tripped even though heartbeats were still arriving"

        # Explicitly stop so the fixture teardown doesn't race a lingering
        # "in flight" motion state against the next test in the session.
        ember_bridge_client.send_command(CommandOp.STOP)


@pytest.mark.integration
class TestEmberRobotDriverDisconnectedBehavior:
    def test_send_command_returns_rejected_when_never_connected(self):
        # No ember_server/toolchain dependency at all: nothing is listening
        # on this port, so the client never reaches AckResult resolution via
        # the network -- exercises EmberBridgeClient.send_command's
        # "not connected" short-circuit directly.
        client = EmberBridgeClient(host="127.0.0.1", port=find_free_port(), command_timeout_s=0.5)
        client.start()
        try:
            result = client.send_command(CommandOp.STOP)
            assert result == AckResult.REJECTED
        finally:
            client.stop()

    def test_robot_driver_raises_runtime_error_when_disconnected(self):
        # EmberRobotDriver deliberately raises (not swallows) on a
        # REJECTED/FAULTED result -- see ember_execution_agent.py's
        # docstring -- so it propagates through ExecutorAgent.handle_message
        # into Coordinator.dispatch's existing exception-to-ERROR-Message
        # handling with no framework code needing to know the bridge exists.
        client = EmberBridgeClient(host="127.0.0.1", port=find_free_port(), command_timeout_s=0.5)
        client.start()
        driver = EmberRobotDriver(client)
        try:
            with pytest.raises(RuntimeError, match="EMBER rejected command MOVE_FORWARD"):
                driver.move_forward(0.5)
        finally:
            client.stop()

    def test_get_distance_to_front_defaults_to_zero_without_telemetry(self):
        # No Motion-subsystem telemetry producer exists yet (see the ADR's
        # "Future Considerations") -- pins the documented last-known-value
        # default so a future MotionSubsystem implementation changes this
        # deliberately, not by accident.
        client = EmberBridgeClient(host="127.0.0.1", port=find_free_port(), command_timeout_s=0.5)
        driver = EmberRobotDriver(client)
        assert driver.get_distance_to_front() == 0.0
