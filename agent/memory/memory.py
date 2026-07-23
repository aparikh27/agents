from abc import abstractmethod
from typing import Any
from agent.base_agent import Agent
from messaging import Message, MessageStatus


class MemoryAgent(Agent):
    """Abstract Memory Agent defining the required interface for all memory implementations."""

    def __init__(self, name: str = "Memory"):
        super().__init__(name=name)

    def handle_message(self, message: Message) -> Message:
        """Routes incoming memory tasks based on action type."""
        action = message.action
        payload = message.payload

        if action == "store":
            return self._store(message, payload)
        elif action == "retrieve":
            return self._retrieve(message, payload)
        elif action == "clear":
            return self._clear(message, payload)
        else:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=f"MemoryAgent does not support action '{action}'.",
            )

    @abstractmethod
    def _store(self, message: Message, payload: dict[str, Any]) -> Message:
        """Subclasses MUST implement logic to save information to memory."""
        pass

    @abstractmethod
    def _retrieve(self, message: Message, payload: dict[str, Any]) -> Message:
        """Subclasses MUST implement logic to fetch information from memory."""
        pass

    @abstractmethod
    def _clear(self, message: Message, payload: dict[str, Any]) -> Message:
        """Subclasses MUST implement logic to wipe or reset memory."""
        pass