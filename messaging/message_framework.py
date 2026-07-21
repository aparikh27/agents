from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import uuid
import time


class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"


class MessageStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    ERROR = "error"
    STALE = "stale"


@dataclass
class Message:

    sender: str

    receiver: str

    action: str

    payload: dict[str, Any]

    status: MessageStatus

    message_type: MessageType


    request_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    timestamp: float = field(
        default_factory=time.time
    )

    parent_id: str | None = None

    error: str | None = None