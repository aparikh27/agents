"""End-to-end integration: the real five-agent AgentCore/BAI pipeline
(Audio -> Vision -> Planner -> Executor -> Memory) driving the real,
compiled EMBER C++ runtime over a real loopback socket.

This is the cross-language counterpart to
agents/tests/e2e/test_full_pipeline.py: same Coordinator/pipeline wiring
and the same Whisper/YOLO/Qwen model mocks, but the Executor's robot_driver
is EmberRobotDriver talking to a real ember_pipeline_test_server
subprocess (edge/tests/integration/pipeline_test_server.cpp) instead of
MockRobotDriver -- so a passing test here demonstrates a Message actually
crossing the process boundary, getting framed/checksummed, routed through
EMBER's Coordinator/Subscriber/ThreadSafeQueue onto a worker thread, acked,
and the ack flowing back into ExecutorAgent's synchronous call stack.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.messaging import MessageStatus

from agents.tests.integration.ember_test_server import EmberTestServerProcess


@pytest.mark.integration
class TestFullPipelineEmberIntegration:
    def test_pick_up_plan_drives_real_ember_runtime(
        self,
        ember_server,
        ember_coordinator,
        mock_whisper_model,
        mock_yolo_model,
        mock_qwen_model,
        tmp_path,
        mocker,
    ):
        audio_file = tmp_path / "input_audio.wav"
        audio_file.write_bytes(b"dummy audio content")
        image_file = tmp_path / "input_image.jpg"
        image_file.write_bytes(b"dummy image content")
        mocker.patch("cv2.imread", return_value=MagicMock())

        mock_whisper_model.transcribe.return_value = {"text": "pick up the red ball", "language": "en"}
        mock_qwen_model.return_value = {
            "choices": [{"text": '[{"action": "pick_up", "target": "red ball"}]'}]
        }

        initial_payload = {"audio_path": str(audio_file), "image_path": str(image_file)}
        result = ember_coordinator.run_premade_pipeline("full_pipeline", initial_payload)

        assert result.status == MessageStatus.SUCCESS
        assert result.sender == "Executor"
        assert result.payload["completed_steps"][0] == {"action": "pick_up", "target": "red ball"}

        # The pick_up path issues lower_arm -> grab_item -> raise_arm
        # (WebotsExecutorAgent._execute_single_step, PICKUP_ALIASES branch).
        # Assert each one actually reached the real EMBER process, in order,
        # not just that ExecutorAgent believes it called robot_driver.
        for op in ("lower_arm", "grab_item", "raise_arm"):
            line = ember_server.wait_for_line(
                lambda l, op=op: l.startswith("CMD_RECEIVED") and f"op={op}" in l, timeout=3.0
            )
            assert line is not None, f"EMBER never reported receiving op={op}:\n" + "\n".join(
                ember_server.all_lines()
            )

        completed = ember_server.wait_for_line(lambda l: l.startswith("CMD_COMPLETED"), timeout=3.0)
        assert completed is not None

    def test_get_object_plan_fails_cleanly_without_motion_telemetry(
        self,
        ember_server,
        ember_coordinator,
        mock_whisper_model,
        mock_yolo_model,
        mock_qwen_model,
        tmp_path,
        mocker,
    ):
        """Documents current, honest behavior rather than papering over it:
        EmberRobotDriver.get_distance_to_front() polls the last received
        Motion TelemetryFrame (see ember_execution_agent.py) and defaults to
        0.0 when none has arrived yet. No MotionSubsystem/HAL producer of
        Motion telemetry exists yet (see design-decisions/05-ember-agentcore-bridge.md's
        "Future Considerations") -- pipeline_test_server.cpp's stand-in
        worker never emits one -- so `_align_and_approach` always sees
        distance <= 0.0 and `_get_object` fails deterministically. This test
        pins that contract: it must start failing (and be updated) the day
        a real MotionSubsystem starts publishing Motion telemetry."""
        audio_file = tmp_path / "input.wav"
        audio_file.write_bytes(b"audio")
        image_file = tmp_path / "input.jpg"
        image_file.write_bytes(b"image")
        mocker.patch("cv2.imread", return_value=MagicMock())

        mock_whisper_model.transcribe.return_value = {"text": "get the red ball", "language": "en"}
        mock_qwen_model.return_value = {
            "choices": [{"text": '[{"action": "get_object", "target": "red ball"}]'}]
        }

        initial_payload = {"audio_path": str(audio_file), "image_path": str(image_file)}
        result = ember_coordinator.run_premade_pipeline("full_pipeline", initial_payload)

        assert result.status == MessageStatus.ERROR
        assert result.sender == "Executor"
        assert "Could not reach/retrieve target" in result.error

        # It still had to attempt alignment first, which does reach real
        # EMBER -- specifically a `stop`, not a `turn`: the default
        # mock_world_model's "red ball" box (300,100,340,140) is already
        # pixel-centered for camera_width_px=640 (screen_center=320, and
        # the box's own x-center is (300+340)/2=320), so
        # `_align_and_approach`'s pixel-error check passes on its very
        # first iteration and calls robot.stop() before ever calling
        # robot.turn() -- only then does get_distance_to_front() return
        # 0.0 (see the sibling default-telemetry test) and _get_object fail.
        stop_line = ember_server.wait_for_line(lambda l: l.startswith("CMD_RECEIVED") and "op=stop" in l, timeout=3.0)
        assert stop_line is not None

    def test_pipeline_recovers_after_ember_process_restart(
        self,
        ember_server_binary,
        ember_server,
        ember_bridge_client,
        ember_coordinator,
        mock_whisper_model,
        mock_yolo_model,
        mock_qwen_model,
        tmp_path,
        mocker,
    ):
        """Runs the full pipeline once, kills the EMBER process out from
        under the (still-running) AgentCore side, restarts a new one on the
        same port, and runs the pipeline again through the exact same
        Coordinator/agents/EmberBridgeClient -- nothing here retries the
        connection itself; EmberBridgeClient's own exponential-backoff
        reconnect (agents/agent/execution_agent/ember_bridge_client.py) is
        what has to make this work. This is the scenario that exposed the
        one-shot outbound_queue_.shutdown() bug during development (see the
        ADR) -- a single-connection test could never have caught it."""
        audio_file = tmp_path / "input.wav"
        audio_file.write_bytes(b"audio")
        image_file = tmp_path / "input.jpg"
        image_file.write_bytes(b"image")
        mocker.patch("cv2.imread", return_value=MagicMock())

        mock_whisper_model.transcribe.return_value = {"text": "pick up the red ball", "language": "en"}
        mock_qwen_model.return_value = {
            "choices": [{"text": '[{"action": "pick_up", "target": "red ball"}]'}]
        }
        initial_payload = {"audio_path": str(audio_file), "image_path": str(image_file)}

        first = ember_coordinator.run_premade_pipeline("full_pipeline", initial_payload)
        assert first.status == MessageStatus.SUCCESS
        assert ember_server.wait_for_line(lambda l: l.startswith("CMD_RECEIVED"), timeout=3.0) is not None

        port = ember_server.port
        ember_server.stop()

        new_server = EmberTestServerProcess(ember_server_binary, port=port, watchdog_ms=300, actuation_ms=15)
        try:
            assert new_server.wait_ready(timeout=10.0)
            reconnected = ember_bridge_client.wait_connected(10.0)
            assert reconnected, "EmberBridgeClient never reconnected to the restarted EMBER process"

            second = ember_coordinator.run_premade_pipeline("full_pipeline", initial_payload)
            assert second.status == MessageStatus.SUCCESS

            line = new_server.wait_for_line(lambda l: l.startswith("CMD_RECEIVED") and "op=grab_item" in l, timeout=3.0)
            assert line is not None, "restarted EMBER process never saw the second pipeline run's commands"
        finally:
            new_server.stop()
