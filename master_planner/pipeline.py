from dataclasses import dataclass
from typing import Any
from agents.messaging import Message, MessageType, MessageStatus

@dataclass
class PipelineStep:
    """Represents a single step in a pipeline sequence."""
    receiver: str
    action: str


class PipelineManager:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.pipelines: dict[str, list[PipelineStep]] = {}
        self._register_default_pipelines()

    def create_custom_pipeline(self, name: str, steps: list[PipelineStep]):
        self.pipelines[name] = steps

    def _register_default_pipelines(self):
        """Pre-packaged default pathways with corrected agent names."""
        self.pipelines["full_pipeline"] = [
            PipelineStep(receiver="Audio", action="transcribe"),
            PipelineStep(receiver="Vision", action="analyze_scene"),
            PipelineStep(receiver="Planner", action="create_plan"),  # Fixed: "Planner"
            PipelineStep(receiver="Executor", action="execute"),     # Fixed: "Executor" & action="execute"
        ]

    def run_pipeline(self, name: str, initial_payload: dict[str, Any]) -> Message:
        if name not in self.pipelines:
            return Message(
                sender="pipeline",
                receiver="client",
                action="run_pipeline",
                payload={},
                status=MessageStatus.ERROR,
                message_type=MessageType.RESPONSE,
                error=f"Pipeline '{name}' not found.",
            )

        steps = self.pipelines[name]
        current_payload = dict(initial_payload)
        last_response = None

        try:
            for step in steps:
                msg = Message(
                    sender="pipeline",
                    receiver=step.receiver,
                    action=step.action,
                    payload=current_payload,
                    status=MessageStatus.PENDING,
                    message_type=MessageType.REQUEST,
                    parent_id=last_response.request_id if last_response else None,
                )

                last_response = self.coordinator.dispatch(msg)

                if last_response.status == MessageStatus.ERROR:
                    print(f"❌ Pipeline '{name}' halted at step '{step.receiver}': {last_response.error}")
                    return last_response

                if isinstance(last_response.payload, dict):
                    current_payload.update(last_response.payload)

            return last_response
        except Exception as e:
            print(f"❌ Pipeline '{name}' encountered an unhandled error: {e}")
            return Message(
                sender="pipeline",
                receiver="client",
                action="run_pipeline",
                payload={},
                status=MessageStatus.ERROR,
                message_type=MessageType.RESPONSE,
                parent_id=last_response.request_id if last_response else None,
                error=f"Pipeline '{name}' crashed with unhandled exception: {str(e)}",
            )