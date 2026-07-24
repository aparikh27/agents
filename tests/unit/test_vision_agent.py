import pytest
from unittest.mock import MagicMock
from messaging import Message, MessageType, MessageStatus
from agent.vision_agent.vision import VisionAgent
from agent.vision_agent.yolo_vision import YOLOVisionAgent


class DummyVisionAgent(VisionAgent):
    """Concrete dummy class for testing abstract VisionAgent routing."""

    def _detect_objects(self, message: Message, payload: dict) -> Message:
        return self.create_response(request=message, payload={"objects": ["cup"]})

    def _analyze_scene(self, message: Message, payload: dict) -> Message:
        return self.create_response(request=message, payload={"summary": "A room with a cup."})


@pytest.mark.unit
class TestVisionAgentBase:

    def test_vision_agent_routing_detect_objects(self, message_factory):
        agent = DummyVisionAgent()
        msg = message_factory(receiver="Vision", action="detect_objects")
        resp = agent.handle_message(msg)
        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload == {"objects": ["cup"]}

    def test_vision_agent_routing_analyze_scene(self, message_factory):
        agent = DummyVisionAgent()
        msg = message_factory(receiver="Vision", action="analyze_scene")
        resp = agent.handle_message(msg)
        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload == {"summary": "A room with a cup."}

    def test_vision_agent_unsupported_action(self, message_factory):
        agent = DummyVisionAgent()
        msg = message_factory(receiver="Vision", action="invalid_action")
        resp = agent.handle_message(msg)
        assert resp.status == MessageStatus.ERROR
        assert "VisionAgent does not support action 'invalid_action'." in resp.error


@pytest.mark.unit
class TestYOLOVisionAgent:

    def test_detect_objects_missing_image_path(self, mock_yolo_model, message_factory):
        agent = YOLOVisionAgent()
        msg = message_factory(receiver="Vision", action="detect_objects", payload={})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload["image_path"] is None
        assert resp.payload["count"] == 0
        assert "warning" in resp.payload

    def test_detect_objects_nonexistent_file(self, mock_yolo_model, message_factory):
        agent = YOLOVisionAgent()
        msg = message_factory(receiver="Vision", action="detect_objects", payload={"image_path": "/missing/image.jpg"})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Image file not found" in resp.error

    def test_detect_objects_corrupted_image(self, mock_yolo_model, message_factory, mocker, tmp_path):
        img_file = tmp_path / "corrupt.jpg"
        img_file.write_bytes(b"invalid image bytes")

        mocker.patch("cv2.imread", return_value=None)

        agent = YOLOVisionAgent()
        msg = message_factory(receiver="Vision", action="detect_objects", payload={"image_path": str(img_file)})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Failed to read image" in resp.error

    def test_detect_objects_success(self, mock_yolo_model, message_factory, mocker, tmp_path):
        img_file = tmp_path / "valid.jpg"
        img_file.write_bytes(b"dummy image data")

        mock_frame = MagicMock()
        mocker.patch("cv2.imread", return_value=mock_frame)

        agent = YOLOVisionAgent()
        msg = message_factory(receiver="Vision", action="detect_objects", payload={"image_path": str(img_file)})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload["image_path"] == str(img_file)
        assert resp.payload["objects"] == ["red ball"]
        assert resp.payload["count"] == 1
        assert len(resp.payload["detections"]) == 1
        assert resp.payload["detections"][0]["class_name"] == "red ball"
        assert resp.payload["detections"][0]["confidence"] == 0.95

    def test_analyze_scene_missing_image_path(self, mock_yolo_model, message_factory):
        agent = YOLOVisionAgent()
        msg = message_factory(receiver="Vision", action="analyze_scene", payload={})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload["image_path"] is None
        assert "No image provided" in resp.payload["summary"]

    def test_analyze_scene_success(self, mock_yolo_model, message_factory, mocker, tmp_path):
        img_file = tmp_path / "valid.jpg"
        img_file.write_bytes(b"dummy image data")

        mock_frame = MagicMock()
        mocker.patch("cv2.imread", return_value=mock_frame)

        agent = YOLOVisionAgent()
        msg = message_factory(receiver="Vision", action="analyze_scene", payload={"image_path": str(img_file)})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload["summary"] == "Scene contains 1 red ball."
        assert len(resp.payload["detections"]) == 1

    def test_summarize_scene_formatting(self):
        agent = YOLOVisionAgent()
        
        # Test empty detections
        assert agent._summarize_scene([]) == "No recognizable objects detected in the scene."

        # Test single item
        dets_single = [{"class_name": "cup"}]
        assert agent._summarize_scene(dets_single) == "Scene contains 1 cup."

        # Test multiple items of same class
        dets_plural = [{"class_name": "chair"}, {"class_name": "chair"}]
        assert agent._summarize_scene(dets_plural) == "Scene contains 2 chairs."

        # Test distinct classes
        dets_multiple = [{"class_name": "chair"}, {"class_name": "chair"}, {"class_name": "table"}]
        assert agent._summarize_scene(dets_multiple) == "Scene contains 2 chairs, and 1 table."
