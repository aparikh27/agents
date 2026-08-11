from abc import abstractmethod
from typing import Any
from agents.agent.base_agent import Agent
from agents.messaging import Message, MessageStatus


class AudioAgent(Agent):
    """Abstract Audio Agent defining the required interface for all audio implementations."""

    def __init__(self, name: str = "Audio"):
        super().__init__(name=name)

    def handle_message(self, message: Message) -> Message:
        """Routes incoming audio tasks based on action type."""
        action = message.action
        payload = message.payload

        if action == "transcribe":
            return self._transcribe(message, payload)
        elif action == "text_to_speech":
            return self._text_to_speech(message, payload)
        else:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=f"AudioAgent does not support action '{action}'.",
            )

    @abstractmethod
    def _transcribe(self, message: Message, payload: dict[str, Any]) -> Message:
        """Subclasses MUST implement speech-to-text logic here."""
        pass

    @abstractmethod
    def _text_to_speech(self, message: Message, payload: dict[str, Any]) -> Message:
        """Subclasses MUST implement text-to-speech logic here."""
        pass