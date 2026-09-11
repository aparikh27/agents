"""Fixtures for the EMBER <-> AgentCore bridge integration suite.

These tests exercise the real compiled C++ runtime (edge/ built as the
ember_pipeline_test_server subprocess -- see ember_test_server.py) talking
over a real loopback socket to the real Python bridge client, driving the
same five-agent Coordinator wiring the top-level e2e suite uses
(agents/tests/e2e/test_full_pipeline.py) -- except the Executor drives a
real EMBER process instead of a MockRobotDriver.

Fixtures here build on, rather than duplicate, agents/tests/conftest.py:
`mocker`, `mock_whisper_model`, `mock_yolo_model`, `mock_qwen_model`,
`mock_world_model`, `temp_sqlite_db`, and `message_factory` all come from
there via pytest's normal conftest cascading and don't need
reimporting/redefining here.
"""

from __future__ import annotations

import pytest

from agents.agent.audio_agent.whisper_audio import WhisperAudioAgent
from agents.agent.execution_agent.ember_bridge_client import EmberBridgeClient
from agents.agent.execution_agent.ember_execution_agent import EmberRobotDriver
from agents.agent.execution_agent.ember_telemetry_relay import EmberTelemetryRelay
from agents.agent.execution_agent.webot_execution import WebotsExecutorAgent
from agents.agent.memory_agent.memory_engine_agent import MemoryEngineAgent
from agents.agent.planner_agent.qwen_planner import QwenPlannerAgent
from agents.agent.vision_agent.yolo_vision import YOLOVisionAgent
from agents.master_planner.coordinator import Coordinator
from agents.tests.integration.ember_test_server import (
    EmberTestServerProcess,
    find_free_port,
    find_or_build_server_binary,
)


@pytest.fixture(scope="session")
def ember_server_binary():
    """Session-scoped: locate/build once and reuse the same binary path
    for every test in this session (each test still gets its own
    subprocess+port via the `ember_server` fixture below)."""
    binary = find_or_build_server_binary()
    if binary is None:
        pytest.skip(
            "No C++ toolchain (g++) and no prebuilt ember_pipeline_test_server "
            "found. Set EMBER_TEST_SERVER_BIN to a prebuilt binary, build "
            "edge/'s ember_pipeline_test_server CMake target, or install a "
            "C++20 compiler (g++/clang++) to run EMBER bridge integration tests."
        )
    return binary


@pytest.fixture
def ember_server(ember_server_binary):
    """A fresh EMBER bridge subprocess, on its own port, per test."""
    port = find_free_port()
    server = EmberTestServerProcess(ember_server_binary, port=port, watchdog_ms=300, actuation_ms=15)
    try:
        ready = server.wait_ready(timeout=10.0)
        assert ready, (
            "ember_pipeline_test_server did not print READY within 10s:\n"
            + "\n".join(server.all_lines())
        )
        yield server
    finally:
        server.stop()


@pytest.fixture
def ember_bridge_client(ember_server):
    """A connected EmberBridgeClient against the per-test server above."""
    client = EmberBridgeClient(host="127.0.0.1", port=ember_server.port, command_timeout_s=3.0)
    client.start()
    connected = client.wait_connected(5.0)
    assert connected, "EmberBridgeClient failed to connect to ember_pipeline_test_server"
    try:
        yield client
    finally:
        client.stop()


@pytest.fixture
def ember_robot_driver(ember_bridge_client):
    return EmberRobotDriver(ember_bridge_client, camera_width_px=640)


@pytest.fixture
def ember_coordinator(
    ember_robot_driver, mock_world_model, temp_sqlite_db, mock_whisper_model, mock_yolo_model, mock_qwen_model
):
    """Same five-agent wiring as agents/tests/conftest.py's
    `configured_coordinator`, except WebotsExecutorAgent drives a real
    EMBER bridge subprocess (via EmberRobotDriver) instead of
    MockRobotDriver -- everything upstream of the Executor (Audio, Vision,
    Planner, Memory) is unchanged.

    mock_whisper_model/mock_yolo_model/mock_qwen_model are listed as
    dependencies (and otherwise unused here) for the same reason
    `configured_coordinator` in agents/tests/conftest.py lists them: pytest
    resolves fixtures in dependency order, and WhisperAudioAgent()/
    YOLOVisionAgent()/QwenPlannerAgent() below must not construct their
    real models before these patches are in place -- dropping this would
    let YOLOVisionAgent load a real model and then choke on the mocked
    cv2.imread() output at inference time instead of using the mock.
    """
    coordinator = Coordinator()

    coordinator.add_agent(WhisperAudioAgent())
    coordinator.add_agent(YOLOVisionAgent())
    coordinator.add_agent(QwenPlannerAgent())
    coordinator.add_agent(WebotsExecutorAgent(robot_driver=ember_robot_driver, world=mock_world_model))
    coordinator.add_agent(MemoryEngineAgent(db_path=temp_sqlite_db))

    return coordinator


@pytest.fixture
def ember_telemetry_relay(ember_bridge_client, ember_coordinator):
    """Wires EmberTelemetryRelay into the already-connected bridge client
    via set_telemetry_callback -- see that method's docstring for why this
    can't just be an EmberBridgeClient(on_telemetry=...) constructor arg
    here: the relay needs `ember_coordinator`, which itself needs the
    client to already exist (via ember_robot_driver)."""
    relay = EmberTelemetryRelay(ember_coordinator, planner_agent_name="Planner")
    ember_bridge_client.set_telemetry_callback(relay.on_telemetry)
    return relay
