import time
from typing import Any
from agents.messaging import Message, MessageStatus
from agents.agent.memory_agent.memory import MemoryAgent

from agents.agent.memory_agent.MemoryEngine.memory import MemoryItem
from agents.agent.memory_agent.MemoryEngine.memory_manager import MemoryManager


class MemoryEngineAgent(MemoryAgent):
    """Adapter agent wrapping MemoryManager (Short-Term + Long-Term SQLite Memory)."""

    def __init__(self, capacity: int = 100, db_path: str = "robot_memory.db"):
        super().__init__()  # Passes name="Memory" up to BaseAgent
        self.memory_manager = MemoryManager(capacity=capacity, db_path=db_path)
        print(f"[MemoryManagerAgent] Initialized with short-term capacity={capacity}, db='{db_path}'")

    def _store(self, message: Message, payload: dict[str, Any]) -> Message:
        """Stores or modifies a MemoryItem.
        
        Expected payload keys:
        - key: str
        - value: Any
        - modify: bool (optional, set True if modifying an existing memory)
        """
        key = payload.get("key")
        value = payload.get("value")
        is_modify = payload.get("modify", False)

        if not key or value is None:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error="Payload missing required parameters: 'key' and 'value'",
            )

        try:
            item = MemoryItem(key=key, value=value, timestamp=payload.get("timestamp") or time.time())

            if is_modify:
                success = self.memory_manager.modify(item)
                action_name = "modify"
            else:
                success = self.memory_manager.add(item)
                action_name = "add"

            if not success:
                return self.create_response(
                    request=message,
                    status=MessageStatus.ERROR,
                    error=f"MemoryManager failed to {action_name} memory for key '{key}'.",
                )

            return self.create_response(
                request=message,
                payload={"key": key, "status": f"{action_name}_success"},
            )
        except Exception as e:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=f"Failed to store in MemoryManager: {e}",
            )

    def _retrieve(self, message: Message, payload: dict[str, Any]) -> Message:
        """Retrieves a MemoryItem by key (checks short-term first, then falls back to long-term).
        
        Expected payload keys:
        - key: str
        """
        key = payload.get("key")

        if not key:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error="Payload missing required parameter: 'key'",
            )

        try:
            item = self.memory_manager.get(key)

            if item is None:
                return self.create_response(
                    request=message,
                    payload={"key": key, "found": False, "item": None},
                )

            # Unpack MemoryItem into payload dictionary
            item_data = {
                "key": item.key if hasattr(item, "key") else key,
                "value": item.value if hasattr(item, "value") else item,
            }

            return self.create_response(
                request=message,
                payload={"key": key, "found": True, "item": item_data},
            )
        except Exception as e:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=f"Failed to retrieve from MemoryManager: {e}",
            )

    def _clear(self, message: Message, payload: dict[str, Any]) -> Message:
        """Deletes a MemoryItem by key from both short-term and long-term memory.
        
        Expected payload keys:
        - key: str
        """
        key = payload.get("key")

        if not key:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error="Payload missing required parameter: 'key'",
            )

        try:
            deleted = self.memory_manager.delete(key)

            return self.create_response(
                request=message,
                payload={"key": key, "deleted": deleted},
            )
        except Exception as e:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=f"Failed to delete key '{key}' from MemoryManager: {e}",
            )