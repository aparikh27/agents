import pytest
from unittest.mock import MagicMock
from agents.messaging import Message, MessageType, MessageStatus
from agents.master_planner.coordinator import Coordinator
from agents.master_planner.pipeline import PipelineStep
from agents.agent.audio_agent.whisper_audio import WhisperAudioAgent
from agents.agent.vision_agent.yolo_vision import YOLOVisionAgent
from agents.agent.planner_agent.qwen_planner import QwenPlannerAgent
from agents.agent.execution_agent.webot_execution import WebotsExecutorAgent
from agents.agent.memory_agent.memory_engine_agent import MemoryEngineAgent


@pytest.mark.e2e
class TestFullPipelineE2E:

    def test_full_pipeline_success(self, configured_coordinator, mock_whisper_model, mock_yolo_model, mock_qwen_model, mock_robot_driver, tmp_path, mocker):
        # Create dummy physical files for audio and vision payload testing
        audio_file = tmp_path / "input_audio.wav"
        audio_file.write_bytes(b"dummy audio content")

        image_file = tmp_path / "input_image.jpg"
        image_file.write_bytes(b"dummy image content")

        mocker.patch("cv2.imread", return_value=MagicMock())

        # Configure mock model outputs
        mock_whisper_model.transcribe.return_value = {
            "text": "pick up red ball",
            "language": "en"
        }

        mock_qwen_model.return_value = {
            "choices": [
                {
                    "text": '[{"action": "get_object", "target": "red ball"}]'
                }
            ]
        }

        initial_payload = {
            "audio_path": str(audio_file),
            "image_path": str(image_file),
        }

        # Run premade pipeline: Audio -> Vision -> Planner -> Executor
        result = configured_coordinator.run_premade_pipeline("full_pipeline", initial_payload)

        # Assert final status and payload
        assert result.status == MessageStatus.SUCCESS
        assert result.sender == "Executor"
        assert result.payload["total_steps"] == 1
        assert result.payload["completed_steps"][0] == {"action": "get_object", "target": "red ball"}

        # Verify hardware calls occurred on mock robot driver
        assert mock_robot_driver.grabbed is True
        assert len(mock_robot_driver.moves) == 2

    def test_pipeline_message_lineage_tracing(self, configured_coordinator, mock_whisper_model, mock_yolo_model, mock_qwen_model, tmp_path, mocker):
        audio_file = tmp_path / "input.wav"
        audio_file.write_bytes(b"audio data")
        image_file = tmp_path / "input.jpg"
        image_file.write_bytes(b"image data")
        mocker.patch("cv2.imread", return_value=MagicMock())

        dispatched_messages = []
        response_messages = []

        original_dispatch = configured_coordinator.dispatch

        def tracking_dispatch(msg: Message) -> Message:
            dispatched_messages.append(msg)
            resp = original_dispatch(msg)
            response_messages.append(resp)
            return resp

        configured_coordinator.dispatch = tracking_dispatch

        initial_payload = {"audio_path": str(audio_file), "image_path": str(image_file)}
        final_response = configured_coordinator.run_premade_pipeline("full_pipeline", initial_payload)

        assert final_response.status == MessageStatus.SUCCESS
        assert len(dispatched_messages) == 4  # Audio, Vision, Planner, Executor
        assert len(response_messages) == 4

        # Step 1: Audio request parent_id should be None
        audio_req = dispatched_messages[0]
        audio_resp = response_messages[0]
        assert audio_req.receiver == "Audio"
        assert audio_req.parent_id is None
        assert audio_resp.parent_id == audio_req.request_id

        # Step 2: Vision request parent_id should be Audio response's request_id
        vision_req = dispatched_messages[1]
        vision_resp = response_messages[1]
        assert vision_req.receiver == "Vision"
        assert vision_req.parent_id == audio_resp.request_id
        assert vision_resp.parent_id == vision_req.request_id

        # Step 3: Planner request parent_id should be Vision response's request_id
        planner_req = dispatched_messages[2]
        planner_resp = response_messages[2]
        assert planner_req.receiver == "Planner"
        assert planner_req.parent_id == vision_resp.request_id
        assert planner_resp.parent_id == planner_req.request_id

        # Step 4: Executor request parent_id should be Planner response's request_id
        executor_req = dispatched_messages[3]
        executor_resp = response_messages[3]
        assert executor_req.receiver == "Executor"
        assert executor_req.parent_id == planner_resp.request_id
        assert executor_resp.parent_id == executor_req.request_id

    def test_pipeline_payload_schema_propagation(self, configured_coordinator, mock_whisper_model, mock_yolo_model, mock_qwen_model, tmp_path, mocker):
        audio_file = tmp_path / "input.wav"
        audio_file.write_bytes(b"audio data")
        image_file = tmp_path / "input.jpg"
        image_file.write_bytes(b"image data")
        mocker.patch("cv2.imread", return_value=MagicMock())

        received_payloads = {}

        # Intercept handle_message calls to inspect payloads received by each agent
        for name, agent in configured_coordinator.all_agents.items():
            orig_handle = agent.handle_message

            def make_wrapper(agent_name, original_fn):
                def wrapper(msg: Message) -> Message:
                    received_payloads[agent_name] = dict(msg.payload)
                    return original_fn(msg)
                return wrapper

            agent.handle_message = make_wrapper(name, orig_handle)

        initial_payload = {"audio_path": str(audio_file), "image_path": str(image_file)}
        configured_coordinator.run_premade_pipeline("full_pipeline", initial_payload)

        # Verify Audio received initial payload
        assert received_payloads["Audio"]["audio_path"] == str(audio_file)

        # Verify Vision received Audio's transcribed text in accumulated payload
        assert "text" in received_payloads["Vision"]
        assert received_payloads["Vision"]["text"] == "pick up red ball"

        # Verify Planner received text and vision summary in accumulated payload
        assert "text" in received_payloads["Planner"]
        assert "summary" in received_payloads["Planner"]

        # Verify Executor received plan generated by Planner
        assert "plan" in received_payloads["Executor"]

    def test_pipeline_failure_propagation(self, configured_coordinator, mock_whisper_model):
        # Provide invalid audio_path (non-existent file) to force Audio step failure
        initial_payload = {"audio_path": "/invalid/nonexistent_audio.wav"}

        result = configured_coordinator.run_premade_pipeline("full_pipeline", initial_payload)

        assert result.status == MessageStatus.ERROR
        assert result.sender == "Audio"
        assert "Audio file not found" in result.error

    def test_custom_pipeline_execution(self, configured_coordinator, mock_qwen_model, mock_robot_driver):
        custom_steps = [
            PipelineStep(receiver="Planner", action="create_plan"),
            PipelineStep(receiver="Executor", action="execute"),
        ]

        initial_payload = {"task": "pick up red ball"}

        result = configured_coordinator.run_custom_pipeline("planner_to_executor", custom_steps, initial_payload)

        assert result.status == MessageStatus.SUCCESS
        assert result.sender == "Executor"
        assert result.payload["total_steps"] == 1
        assert mock_robot_driver.grabbed is True
