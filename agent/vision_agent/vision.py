from abc import abstractmethod
from typing import Any
from agent.base_agent import Agent
from messaging import Message, MessageStatus


class VisionAgent(Agent):
    """Abstract Vision Agent defining the required interface for all vision implementations."""

    def __init__(self, name: str = "Vision"):
        super().__init__(name=name)

    def handle_message(self, message: Message) -> Message:
        action = message.action
        payload = message.payload

        if action == "detect_objects":
            return self._detect_objects(message, payload)
        elif action == "analyze_scene":
            return self._analyze_scene(message, payload)
        else:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=f"VisionAgent does not support action '{action}'.",
            )

    @abstractmethod
    def _detect_objects(self, message: Message, payload: dict[str, Any]) -> Message:
        """Subclasses MUST implement object detection logic here."""
        pass

    @abstractmethod
    def _analyze_scene(self, message: Message, payload: dict[str, Any]) -> Message:
        """Subclasses MUST implement scene analysis logic here."""
        pass