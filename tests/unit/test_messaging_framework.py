import pytest
from agents.messaging import Message, MessageType, MessageStatus
from agents.messaging.helpers import message_to_dict, message_from_dict


@pytest.mark.unit
class TestMessagingFramework:

    def test_message_creation_defaults(self):
        msg = Message(
            sender="agent_a",
            receiver="agent_b",
            action="do_something",
            payload={"key": "val"},
            status=MessageStatus.PENDING,
            message_type=MessageType.REQUEST,
        )

        assert msg.sender == "agent_a"
        assert msg.receiver == "agent_b"
        assert msg.action == "do_something"
        assert msg.payload == {"key": "val"}
        assert msg.status == MessageStatus.PENDING
        assert msg.message_type == MessageType.REQUEST
        assert msg.request_id is not None
        assert isinstance(msg.timestamp, float)
        assert msg.parent_id is None
        assert msg.error is None

    def test_message_to_dict_conversion(self):
        msg = Message(
            sender="agent_a",
            receiver="agent_b",
            action="do_something",
            payload={"key": "val"},
            status=MessageStatus.SUCCESS,
            message_type=MessageType.RESPONSE,
            request_id="req-123",
            timestamp=100.0,
            parent_id="parent-000",
            error=None,
        )

        data = message_to_dict(msg)
        assert data["sender"] == "agent_a"
        assert data["receiver"] == "agent_b"
        assert data["action"] == "do_something"
        assert data["payload"] == {"key": "val"}
        assert data["status"] == "success"  # Raw string value of Enum
        assert data["message_type"] == "response"  # Raw string value of Enum
        assert data["request_id"] == "req-123"
        assert data["timestamp"] == 100.0
        assert data["parent_id"] == "parent-000"

    def test_message_from_dict_reconstruction(self):
        data = {
            "sender": "agent_x",
            "receiver": "agent_y",
            "action": "transcribe",
            "payload": {"audio": "test"},
            "message_type": "request",
            "status": "pending",
            "request_id": "req-999",
            "timestamp": 200.0,
            "parent_id": "parent-111",
            "error": "some_err",
        }

        msg = message_from_dict(data)
        assert msg.sender == "agent_x"
        assert msg.receiver == "agent_y"
        assert msg.action == "transcribe"
        assert msg.payload == {"audio": "test"}
        assert msg.message_type == MessageType.REQUEST
        assert msg.status == MessageStatus.PENDING
        assert msg.request_id == "req-999"
        assert msg.timestamp == 200.0
        assert msg.parent_id == "parent-111"
        assert msg.error == "some_err"

    def test_message_dict_roundtrip(self):
        original = Message(
            sender="coordinator",
            receiver="Executor",
            action="execute",
            payload={"plan": [{"action": "pick_up", "target": "ball"}]},
            status=MessageStatus.SUCCESS,
            message_type=MessageType.RESPONSE,
            parent_id="req-001",
        )

        as_dict = message_to_dict(original)
        reconstructed = message_from_dict(as_dict)

        assert reconstructed.sender == original.sender
        assert reconstructed.receiver == original.receiver
        assert reconstructed.action == original.action
        assert reconstructed.payload == original.payload
        assert reconstructed.status == original.status
        assert reconstructed.message_type == original.message_type
        assert reconstructed.request_id == original.request_id
        assert reconstructed.parent_id == original.parent_id
