# messaging/helpers.py
from dataclasses import asdict
import json
from messaging.message import Message, MessageType, MessageStatus

def message_to_dict(msg: Message) -> dict:
    """Converts a Message object into a clean Python dictionary."""
    data = asdict(msg)
    # Convert Enums to raw strings for JSON compatibility
    data["message_type"] = msg.message_type.value
    data["status"] = msg.status.value
    return data

def message_from_dict(data: dict) -> Message:
    """Reconstructs a Message object from a dictionary."""
    return Message(
        sender=data["sender"],
        receiver=data["receiver"],
        action=data["action"],
        payload=data.get("payload", {}),
        message_type=MessageType(data.get("message_type", "request")),
        status=MessageStatus(data.get("status", "pending")),
        request_id=data.get("request_id"),
        timestamp=data.get("timestamp"),
        parent_id=data.get("parent_id"),
        error=data.get("error")
    )