import pytest
from unittest.mock import MagicMock
from agents.messaging import Message, MessageType, MessageStatus
from agents.agent.planner_agent.planner import PlannerAgent
from agents.agent.planner_agent.qwen_planner import QwenPlannerAgent


class DummyPlannerAgent(PlannerAgent):
    """Concrete dummy class for testing abstract PlannerAgent routing."""

    def _create_plan(self, message: Message, payload: dict) -> Message:
        return self.create_response(request=message, payload={"plan": [{"action": "move_to"}]})

    def _update_plan(self, message: Message, payload: dict) -> Message:
        return self.create_response(request=message, payload={"plan": [{"action": "stop"}]})


@pytest.mark.unit
class TestPlannerAgentBase:

    def test_planner_agent_routing_create_plan(self, message_factory):
        agent = DummyPlannerAgent()
        for action_name in ["create_plan", "plan"]:
            msg = message_factory(receiver="Planner", action=action_name)
            resp = agent.handle_message(msg)
            assert resp.status == MessageStatus.SUCCESS
            assert resp.payload == {"plan": [{"action": "move_to"}]}

    def test_planner_agent_routing_update_plan(self, message_factory):
        agent = DummyPlannerAgent()
        for action_name in ["update_plan", "replan", "modify_plan"]:
            msg = message_factory(receiver="Planner", action=action_name)
            resp = agent.handle_message(msg)
            assert resp.status == MessageStatus.SUCCESS
            assert resp.payload == {"plan": [{"action": "stop"}]}

    def test_planner_agent_unsupported_action(self, message_factory):
        agent = DummyPlannerAgent()
        msg = message_factory(receiver="Planner", action="unknown_planner_action")
        resp = agent.handle_message(msg)
        assert resp.status == MessageStatus.ERROR
        assert "PlannerAgent does not support action 'unknown_planner_action'." in resp.error


@pytest.mark.unit
class TestQwenPlannerAgent:

    def test_create_plan_missing_task_parameters(self, mock_qwen_model, message_factory):
        agent = QwenPlannerAgent()
        msg = message_factory(receiver="Planner", action="create_plan", payload={})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Missing required payload parameter" in resp.error

    def test_create_plan_inference_failure(self, mock_qwen_model, message_factory):
        mock_qwen_model.side_effect = RuntimeError("Out of memory during decoding")

        agent = QwenPlannerAgent()
        msg = message_factory(receiver="Planner", action="create_plan", payload={"task": "find my keys"})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Qwen model inference failed" in resp.error

    def test_create_plan_invalid_json_output(self, mock_qwen_model, message_factory):
        mock_qwen_model.return_value = {"choices": [{"text": "I am an AI and cannot help with that."}]}

        agent = QwenPlannerAgent()
        msg = message_factory(receiver="Planner", action="create_plan", payload={"task": "find my keys"})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Failed to parse model output as JSON plan" in resp.error

    def test_create_plan_non_list_json_output(self, mock_qwen_model, message_factory):
        mock_qwen_model.return_value = {"choices": [{"text": '{"action": "pick_up"}'}]}

        agent = QwenPlannerAgent()
        msg = message_factory(receiver="Planner", action="create_plan", payload={"task": "pick up cup"})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Expected a JSON array" in resp.error

    def test_create_plan_success_with_markdown_fences(self, mock_qwen_model, message_factory):
        mock_qwen_model.return_value = {
            "choices": [
                {
                    "text": "```json\n[{\"action\": \"detect_object\", \"target\": \"keys\"}]\n```"
                }
            ]
        }

        agent = QwenPlannerAgent()
        msg = message_factory(receiver="Planner", action="create_plan", payload={"task": "find my keys"})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload["task"] == "find my keys"
        assert resp.payload["plan"] == [{"action": "detect_object", "target": "keys"}]

    def test_update_plan_missing_plan_parameter(self, mock_qwen_model, message_factory):
        agent = QwenPlannerAgent()
        msg = message_factory(receiver="Planner", action="update_plan", payload={"context": "avoid obstacle"})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Missing required payload parameter: 'plan'" in resp.error

    def test_update_plan_missing_context_parameter(self, mock_qwen_model, message_factory):
        agent = QwenPlannerAgent()
        msg = message_factory(receiver="Planner", action="update_plan", payload={"plan": [{"action": "move_to"}]})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Missing required payload parameter: 'context'" in resp.error

    def test_update_plan_non_serializable_plan(self, mock_qwen_model, message_factory):
        agent = QwenPlannerAgent()
        # set object in plan list that is not JSON serializable
        msg = message_factory(
            receiver="Planner",
            action="update_plan",
            payload={"plan": [object()], "context": "update context"}
        )
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "is not JSON-serializable" in resp.error

    def test_update_plan_success(self, mock_qwen_model, message_factory):
        mock_qwen_model.return_value = {
            "choices": [
                {
                    "text": '[{"action": "go_to", "target": "table"}, {"action": "put_object", "target": "cup"}]'
                }
            ]
        }

        agent = QwenPlannerAgent()
        existing_plan = [{"action": "go_to", "target": "table"}]
        msg = message_factory(
            receiver="Planner",
            action="update_plan",
            payload={"plan": existing_plan, "context": "place cup on table"}
        )
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload["previous_plan"] == existing_plan
        assert len(resp.payload["plan"]) == 2
        assert resp.payload["plan"][1]["action"] == "put_object"
