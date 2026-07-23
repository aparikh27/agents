from abc import abstractmethod
from typing import Any
from agent.base_agent import Agent
from messaging import Message, MessageStatus


class ExecutorAgent(Agent):
    """Abstract Executor Agent defining the required interface for all robot execution implementations."""

    def __init__(self, name: str = "Executor"):
        super().__init__(name=name)

    def handle_message(self, message: Message) -> Message:
        """Routes incoming execution tasks based on action type."""
        action = message.action
        payload = message.payload

        if action in ("execute", "execute_plan", "run_step"):
            return self._execute(message, payload)
        elif action in ("stop", "cancel", "halt"):
            return self._stop(message, payload)
        else:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=f"ExecutorAgent does not support action '{action}'.",
            )

    @abstractmethod
    def _execute(self, message: Message, payload: dict[str, Any]) -> Message:
        """Subclasses MUST implement robot plan/step execution logic here."""
        pass

    @abstractmethod
    def _stop(self, message: Message, payload: dict[str, Any]) -> Message:
        """Subclasses MUST implement hardware stop/halt/cancellation logic here."""
        pass