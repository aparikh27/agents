from abc import abstractmethod
from typing import Any
from agent.base_agent import Agent
from messaging import Message, MessageStatus


class PlannerAgent(Agent):
    """Abstract Planner Agent defining the required interface for all planning implementations."""

    def __init__(self, name: str = "Planner"):
        super().__init__(name=name)

    def handle_message(self, message: Message) -> Message:
        """Routes incoming planning tasks based on action type."""
        action = message.action
        payload = message.payload

        if action in ("create_plan", "plan"):
            return self._create_plan(message, payload)
        elif action in ("update_plan", "replan", "modify_plan"):
            return self._update_plan(message, payload)
        else:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=f"PlannerAgent does not support action '{action}'.",
            )

    @abstractmethod
    def _create_plan(self, message: Message, payload: dict[str, Any]) -> Message:
        """Subclasses MUST implement plan creation logic here."""
        pass

    @abstractmethod
    def _update_plan(self, message: Message, payload: dict[str, Any]) -> Message:
        """Subclasses MUST implement plan update logic here."""
        pass
