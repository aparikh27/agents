import sys
from unittest.mock import MagicMock, patch
import pytest
import os

# Dynamic stubbing of heavy third-party dependencies if not installed
for mod_name in ["whisper", "cv2", "ultralytics", "llama_cpp"]:
    if mod_name not in sys.modules:
        try:
            __import__(mod_name)
        except ImportError:
            sys.modules[mod_name] = MagicMock()

from messaging import Message, MessageType, MessageStatus
from master_planner.coordinator import Coordinator
from agent.audio_agent.whisper_audio import WhisperAudioAgent
from agent.vision_agent.yolo_vision import YOLOVisionAgent
from agent.planner_agent.qwen_planner import QwenPlannerAgent
from agent.execution_agent.webot_execution import WebotsExecutorAgent, RobotDriver, WorldModel
from agent.memory_agent.memory_engine_agent import MemoryEngineAgent


# ----------------------------------------------------------------------
# Mock Objects & Protocols
# ----------------------------------------------------------------------

class MockRobotDriver:
    """Mock implementation of the RobotDriver protocol."""

    def __init__(self):
        self.camera_width = 640
        self.distance_to_front = 1.0
        self.arm_position = "raised"
        self.grabbed = False
        self.stop_called = False
        self.turns = []
        self.moves = []

    def get_camera_width(self) -> int:
        return self.camera_width

    def get_distance_to_front(self) -> float:
        return self.distance_to_front

    def move_forward(self, distance: float) -> None:
        self.moves.append(distance)

    def turn(self, angle: float) -> None:
        self.turns.append(angle)

    def stop(self) -> None:
        self.stop_called = True

    def raise_arm(self) -> None:
        self.arm_position = "raised"

    def lower_arm(self) -> None:
        self.arm_position = "lowered"

    def grab_item(self) -> None:
        self.grabbed = True

    def release_item(self) -> None:
        self.grabbed = False


class DummyWorldObject:
    def __init__(self, track_id: int, class_name: str, visible: bool = True, box=(300.0, 100.0, 340.0, 140.0)):
        self.track_id = track_id
        self.class_name = class_name
        self.visible = visible
        self.box = box


class MockWorldModel:
    """Mock implementation of the WorldModel protocol."""

    def __init__(self, objects: list[DummyWorldObject] | None = None):
        self.objects = objects if objects is not None else [
            DummyWorldObject(track_id=1, class_name="red ball", visible=True, box=(300.0, 100.0, 340.0, 140.0)),
            DummyWorldObject(track_id=2, class_name="keys", visible=True, box=(100.0, 50.0, 150.0, 80.0)),
        ]

    def get_visible_objects(self) -> list[DummyWorldObject]:
        return [obj for obj in self.objects if obj.visible]

    def get_object_by_track_id(self, track_id: int) -> DummyWorldObject | None:
        for obj in self.objects:
            if obj.track_id == track_id:
                return obj
        return None


# ----------------------------------------------------------------------
# Pytest Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def mock_whisper_model(mocker):
    """Mocks whisper.load_model and returns a mock Whisper model."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": "pick up red ball",
        "language": "en",
    }
    mocker.patch("whisper.load_model", return_value=mock_model)
    return mock_model


@pytest.fixture
def mock_yolo_model(mocker):
    """Mocks ultralytics.YOLO and returns a mock YOLO model with sample detections."""
    mock_box = MagicMock()
    mock_box.xyxy = [[300.0, 100.0, 340.0, 140.0]]
    mock_box.cls = [0]
    mock_box.conf = [0.95]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "red ball"}

    mock_yolo_instance = MagicMock()
    mock_yolo_instance.return_value = [mock_result]

    mocker.patch("agent.vision_agent.yolo_vision.YOLO", return_value=mock_yolo_instance)
    return mock_yolo_instance


@pytest.fixture
def mock_qwen_model(mocker):
    """Mocks llama_cpp.Llama and returns a mock Qwen LLM instance."""
    mock_llama_instance = MagicMock()
    mock_llama_instance.return_value = {
        "choices": [
            {
                "text": '[{"action": "pick_up", "target": "red ball"}]'
            }
        ]
    }
    mocker.patch("agent.planner_agent.qwen_planner.Llama", return_value=mock_llama_instance)
    return mock_llama_instance


@pytest.fixture
def mock_robot_driver():
    """Provides a fresh MockRobotDriver instance."""
    return MockRobotDriver()


@pytest.fixture
def mock_world_model():
    """Provides a fresh MockWorldModel instance."""
    return MockWorldModel()


@pytest.fixture
def temp_sqlite_db(tmp_path):
    """Provides an isolated SQLite database file path per test."""
    db_file = tmp_path / "test_robot_memory.db"
    return str(db_file)


@pytest.fixture
def message_factory():
    """Factory fixture for quickly building Message objects in tests."""
    def _create_message(
        sender: str = "test_sender",
        receiver: str = "test_receiver",
        action: str = "test_action",
        payload: dict | None = None,
        status: MessageStatus = MessageStatus.PENDING,
        message_type: MessageType = MessageType.REQUEST,
        parent_id: str | None = None,
        error: str | None = None,
    ) -> Message:
        return Message(
            sender=sender,
            receiver=receiver,
            action=action,
            payload=payload if payload is not None else {},
            status=status,
            message_type=message_type,
            parent_id=parent_id,
            error=error,
        )

    return _create_message


@pytest.fixture
def configured_coordinator(mock_whisper_model, mock_yolo_model, mock_qwen_model, mock_robot_driver, mock_world_model, temp_sqlite_db):
    """Returns a fully populated Coordinator with all 5 mocked concrete agents registered."""
    coord = Coordinator()

    audio_agent = WhisperAudioAgent()
    vision_agent = YOLOVisionAgent()
    planner_agent = QwenPlannerAgent()
    executor_agent = WebotsExecutorAgent(robot_driver=mock_robot_driver, world=mock_world_model)
    memory_agent = MemoryEngineAgent(db_path=temp_sqlite_db)

    coord.add_agent(audio_agent)
    coord.add_agent(vision_agent)
    coord.add_agent(planner_agent)
    coord.add_agent(executor_agent)
    coord.add_agent(memory_agent)

    return coord
