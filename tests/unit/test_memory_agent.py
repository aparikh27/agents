import pytest
from unittest.mock import MagicMock
from agents.messaging import Message, MessageType, MessageStatus
from agents.agent.memory_agent.memory import MemoryAgent
from agents.agent.memory_agent.memory_engine_agent import MemoryEngineAgent


class DummyMemoryAgent(MemoryAgent):
    """Concrete dummy class for testing abstract MemoryAgent routing."""

    def _store(self, message: Message, payload: dict) -> Message:
        return self.create_response(request=message, payload={"status": "stored"})

    def _retrieve(self, message: Message, payload: dict) -> Message:
        return self.create_response(request=message, payload={"found": True})

    def _clear(self, message: Message, payload: dict) -> Message:
        return self.create_response(request=message, payload={"deleted": True})


@pytest.mark.unit
class TestMemoryAgentBase:

    def test_memory_agent_routing_store(self, message_factory):
        agent = DummyMemoryAgent()
        msg = message_factory(receiver="Memory", action="store")
        resp = agent.handle_message(msg)
        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload == {"status": "stored"}

    def test_memory_agent_routing_retrieve(self, message_factory):
        agent = DummyMemoryAgent()
        msg = message_factory(receiver="Memory", action="retrieve")
        resp = agent.handle_message(msg)
        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload == {"found": True}

    def test_memory_agent_routing_clear(self, message_factory):
        agent = DummyMemoryAgent()
        msg = message_factory(receiver="Memory", action="clear")
        resp = agent.handle_message(msg)
        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload == {"deleted": True}

    def test_memory_agent_unsupported_action(self, message_factory):
        agent = DummyMemoryAgent()
        msg = message_factory(receiver="Memory", action="invalid_action")
        resp = agent.handle_message(msg)
        assert resp.status == MessageStatus.ERROR
        assert "MemoryAgent does not support action 'invalid_action'." in resp.error


@pytest.mark.unit
class TestMemoryEngineAgent:

    def test_store_missing_parameters(self, temp_sqlite_db, message_factory):
        agent = MemoryEngineAgent(db_path=temp_sqlite_db)
        
        # Missing value
        msg1 = message_factory(receiver="Memory", action="store", payload={"key": "k1"})
        resp1 = agent.handle_message(msg1)
        assert resp1.status == MessageStatus.ERROR

        # Missing key
        msg2 = message_factory(receiver="Memory", action="store", payload={"value": "v1"})
        resp2 = agent.handle_message(msg2)
        assert resp2.status == MessageStatus.ERROR

    def test_store_and_retrieve_success(self, temp_sqlite_db, message_factory):
        agent = MemoryEngineAgent(db_path=temp_sqlite_db)

        # Store item
        store_msg = message_factory(
            receiver="Memory",
            action="store",
            payload={"key": "last_seen_location", "value": "kitchen_table"}
        )
        store_resp = agent.handle_message(store_msg)
        assert store_resp.status == MessageStatus.SUCCESS
        assert store_resp.payload == {"key": "last_seen_location", "status": "add_success"}

        # Retrieve item
        ret_msg = message_factory(
            receiver="Memory",
            action="retrieve",
            payload={"key": "last_seen_location"}
        )
        ret_resp = agent.handle_message(ret_msg)
        assert ret_resp.status == MessageStatus.SUCCESS
        assert ret_resp.payload["found"] is True
        assert ret_resp.payload["item"]["key"] == "last_seen_location"
        assert ret_resp.payload["item"]["value"] == "kitchen_table"

    def test_store_modify_existing(self, temp_sqlite_db, message_factory):
        agent = MemoryEngineAgent(db_path=temp_sqlite_db)

        # Initial add
        agent.handle_message(message_factory(
            receiver="Memory", action="store", payload={"key": "target", "value": "ball"}
        ))

        # Modify item
        mod_msg = message_factory(
            receiver="Memory",
            action="store",
            payload={"key": "target", "value": "cup", "modify": True}
        )
        mod_resp = agent.handle_message(mod_msg)
        assert mod_resp.status == MessageStatus.SUCCESS
        assert mod_resp.payload == {"key": "target", "status": "modify_success"}

        # Verify modified value
        ret_msg = message_factory(receiver="Memory", action="retrieve", payload={"key": "target"})
        ret_resp = agent.handle_message(ret_msg)
        assert ret_resp.payload["item"]["value"] == "cup"

    def test_retrieve_missing_key(self, temp_sqlite_db, message_factory):
        agent = MemoryEngineAgent(db_path=temp_sqlite_db)
        msg = message_factory(receiver="Memory", action="retrieve", payload={})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Payload missing required parameter: 'key'" in resp.error

    def test_retrieve_nonexistent_key(self, temp_sqlite_db, message_factory):
        agent = MemoryEngineAgent(db_path=temp_sqlite_db)
        msg = message_factory(receiver="Memory", action="retrieve", payload={"key": "unknown_key"})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload["found"] is False
        assert resp.payload["item"] is None

    def test_clear_missing_key(self, temp_sqlite_db, message_factory):
        agent = MemoryEngineAgent(db_path=temp_sqlite_db)
        msg = message_factory(receiver="Memory", action="clear", payload={})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Payload missing required parameter: 'key'" in resp.error

    def test_clear_existing_and_nonexistent_key(self, temp_sqlite_db, message_factory):
        agent = MemoryEngineAgent(db_path=temp_sqlite_db)

        # Store item
        agent.handle_message(message_factory(
            receiver="Memory", action="store", payload={"key": "temp_key", "value": "val"}
        ))

        # Clear existing key
        clear_msg = message_factory(receiver="Memory", action="clear", payload={"key": "temp_key"})
        clear_resp = agent.handle_message(clear_msg)
        assert clear_resp.status == MessageStatus.SUCCESS
        assert clear_resp.payload == {"key": "temp_key", "deleted": True}

        # Clear non-existent key
        clear_nonexist = agent.handle_message(message_factory(receiver="Memory", action="clear", payload={"key": "temp_key"}))
        assert clear_nonexist.status == MessageStatus.SUCCESS
        assert clear_nonexist.payload["deleted"] is False

    def test_memory_engine_exception_handling(self, temp_sqlite_db, message_factory, mocker):
        agent = MemoryEngineAgent(db_path=temp_sqlite_db)
        mocker.patch.object(agent.memory_manager, "add", side_effect=Exception("Database lock error"))

        msg = message_factory(receiver="Memory", action="store", payload={"key": "k", "value": "v"})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Failed to store in MemoryManager: Database lock error" in resp.error
