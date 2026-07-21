from messaging import *
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def handle_message(self, msg: Message):
        pass

    def create_response(
        self,
        request: Message,
        payload: dict | None = None,
        status: MessageStatus = MessageStatus.SUCCESS,
        error: str | None = None,
    ) -> Message:
        """Helper method to easily build a valid response Message back to the sender."""
        return Message(
            sender=self.name,
            receiver=request.sender,
            action=f"{request.action}_response",
            payload=payload or {},
            status=status,
            message_type=MessageType.RESPONSE,       
            parent_id=request.parent_id,
            error=error,
        )
