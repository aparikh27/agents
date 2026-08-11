import pytest
from unittest.mock import MagicMock
from agents.messaging import Message, MessageType, MessageStatus
from agents.agent.execution_agent.execution import ExecutorAgent
from agents.agent.execution_agent.webot_execution import WebotsExecutorAgent
from agents.tests.conftest import DummyWorldObject


class DummyExecutorAgent(ExecutorAgent):
    """Concrete dummy class for testing abstract ExecutorAgent routing."""

    def _execute(self, message: Message, payload: dict) -> Message:
        return self.create_response(request=message, payload={"completed_steps": []})

    def _stop(self, message: Message, payload: dict) -> Message:
        return self.create_response(request=message, payload={"status": "stopped"})


@pytest.mark.unit
class TestExecutorAgentBase:

    def test_executor_agent_routing_execute(self, message_factory):
        agent = DummyExecutorAgent()
        for action_name in ["execute", "execute_plan", "run_step"]:
            msg = message_factory(receiver="Executor", action=action_name)
            resp = agent.handle_message(msg)
            assert resp.status == MessageStatus.SUCCESS
            assert resp.payload == {"completed_steps": []}

    def test_executor_agent_routing_stop(self, message_factory):
        agent = DummyExecutorAgent()
        for action_name in ["stop", "cancel", "halt"]:
            msg = message_factory(receiver="Executor", action=action_name)
            resp = agent.handle_message(msg)
            assert resp.status == MessageStatus.SUCCESS
            assert resp.payload == {"status": "stopped"}

    def test_executor_agent_unsupported_action(self, message_factory):
        agent = DummyExecutorAgent()
        msg = message_factory(receiver="Executor", action="invalid_action")
        resp = agent.handle_message(msg)
        assert resp.status == MessageStatus.ERROR
        assert "ExecutorAgent does not support action 'invalid_action'." in resp.error


@pytest.mark.unit
class TestWebotsExecutorAgent:

    def test_execute_invalid_payload(self, mock_robot_driver, mock_world_model, message_factory):
        agent = WebotsExecutorAgent(robot_driver=mock_robot_driver, world=mock_world_model)
        msg = message_factory(receiver="Executor", action="execute", payload={"plan": "not_a_list"})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Missing or invalid payload" in resp.error

    def test_execute_empty_plan(self, mock_robot_driver, mock_world_model, message_factory):
        agent = WebotsExecutorAgent(robot_driver=mock_robot_driver, world=mock_world_model)
        msg = message_factory(receiver="Executor", action="execute", payload={"plan": []})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload == {"completed_steps": [], "total_steps": 0}

    def test_execute_invalid_step_element(self, mock_robot_driver, mock_world_model, message_factory):
        agent = WebotsExecutorAgent(robot_driver=mock_robot_driver, world=mock_world_model)
        msg = message_factory(receiver="Executor", action="execute", payload={"plan": ["invalid_step_string"]})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Step 0 is not a valid step object" in resp.error
        assert mock_robot_driver.stop_called is True

    def test_execute_detect_object_found(self, mock_robot_driver, mock_world_model, message_factory):
        agent = WebotsExecutorAgent(robot_driver=mock_robot_driver, world=mock_world_model)
        msg = message_factory(
            receiver="Executor",
            action="execute",
            payload={"action": "detect_object", "target": "red ball"}
        )
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload["total_steps"] == 1
        assert resp.payload["completed_steps"][0]["action"] == "detect_object"

    def test_execute_detect_object_not_found(self, mock_robot_driver, mock_world_model, message_factory):
        agent = WebotsExecutorAgent(robot_driver=mock_robot_driver, world=mock_world_model)
        msg = message_factory(
            receiver="Executor",
            action="execute",
            payload={"action": "detect_object", "target": "nonexistent_object"}
        )
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Target 'nonexistent_object' not found" in resp.error
        assert len(mock_robot_driver.turns) == 36  # 360 degree scan (36 * 10 degrees)

    def test_execute_pick_up_action(self, mock_robot_driver, mock_world_model, message_factory):
        agent = WebotsExecutorAgent(robot_driver=mock_robot_driver, world=mock_world_model)
        msg = message_factory(
            receiver="Executor",
            action="execute",
            payload={"action": "pick_up", "target": "red ball"}
        )
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.SUCCESS
        assert mock_robot_driver.grabbed is True
        assert mock_robot_driver.arm_position == "raised"

    def test_execute_get_object_success(self, mock_robot_driver, mock_world_model, message_factory):
        agent = WebotsExecutorAgent(robot_driver=mock_robot_driver, world=mock_world_model)
        msg = message_factory(
            receiver="Executor",
            action="execute",
            payload={"action": "get_object", "target": "red ball"}
        )
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.SUCCESS
        assert mock_robot_driver.grabbed is True
        assert 180 in mock_robot_driver.turns
        assert len(mock_robot_driver.moves) == 2  # moved forward and returned

    def test_execute_put_object_success(self, mock_robot_driver, mock_world_model, message_factory):
        agent = WebotsExecutorAgent(robot_driver=mock_robot_driver, world=mock_world_model)
        mock_robot_driver.grabbed = True

        msg = message_factory(
            receiver="Executor",
            action="execute",
            payload={"action": "put_object", "target": "red ball"}
        )
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.SUCCESS
        assert mock_robot_driver.grabbed is False

    def test_execute_unknown_action(self, mock_robot_driver, mock_world_model, message_factory):
        agent = WebotsExecutorAgent(robot_driver=mock_robot_driver, world=mock_world_model)
        msg = message_factory(
            receiver="Executor",
            action="execute",
            payload={"action": "dance", "target": "floor"}
        )
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Unknown action: 'dance'" in resp.error

    def test_execute_multi_step_plan_halt_on_failure(self, mock_robot_driver, mock_world_model, message_factory):
        agent = WebotsExecutorAgent(robot_driver=mock_robot_driver, world=mock_world_model)
        plan = [
            {"action": "detect_object", "target": "red ball"},
            {"action": "unknown_action", "target": "target"},
            {"action": "pick_up", "target": "red ball"},
        ]

        msg = message_factory(receiver="Executor", action="execute", payload={"plan": plan})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert resp.payload["failed_step_index"] == 1
        assert len(resp.payload["completed_steps"]) == 1
        assert mock_robot_driver.stop_called is True

    def test_stop_command_success(self, mock_robot_driver, mock_world_model, message_factory):
        agent = WebotsExecutorAgent(robot_driver=mock_robot_driver, world=mock_world_model)
        msg = message_factory(receiver="Executor", action="stop")
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload == {"status": "stopped"}
        assert mock_robot_driver.stop_called is True

    def test_stop_command_hardware_exception(self, mock_robot_driver, mock_world_model, message_factory, mocker):
        mocker.patch.object(mock_robot_driver, "stop", side_effect=RuntimeError("Bus connection lost"))

        agent = WebotsExecutorAgent(robot_driver=mock_robot_driver, world=mock_world_model)
        msg = message_factory(receiver="Executor", action="stop")
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Failed to stop robot: Bus connection lost" in resp.error
