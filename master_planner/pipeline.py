from dataclasses import dataclass
from typing import Any
from messaging import Message, MessageType, MessageStatus


@dataclass
class PipelineStep:
    """Represents a single step in a pipeline sequence."""
    receiver: str
    action: str


class PipelineManager:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        # Store pipelines as lists of PipelineStep objects
        self.pipelines: dict[str, list[PipelineStep]] = {}

        # Register default pipelines on startup
        self._register_default_pipelines()

    def create_custom_pipeline(self, name: str, steps: list[PipelineStep]):
        """Allows users to register custom agent sequences using PipelineStep objects.
        
        Example:
            steps = [
                PipelineStep(receiver="Audio", action="transcribe"),
                PipelineStep(receiver="Vision", action="detect")
            ]
        """
        self.pipelines[name] = steps

    def _register_default_pipelines(self):
        """Pre-packaged default pathways."""
        self.pipelines["full_pipeline"] = [
            PipelineStep(receiver="Audio", action="transcribe"),
            PipelineStep(receiver="Vision", action="analyze"),
            PipelineStep(receiver="Planning", action="create_plan"),
            PipelineStep(receiver="Execution", action="run"),
        ]

    def run_pipeline(self, name: str, initial_payload: dict[str, Any]) -> Message:
        """Executes a pipeline sequence step-by-step using Coordinator dispatch."""
        if name not in self.pipelines:
            raise KeyError(f"Pipeline '{name}' not found.")

        steps = self.pipelines[name]
        current_payload = initial_payload
        last_response = None

        for step in steps:
            # Construct a request for the current agent step using step attributes
            msg = Message(
                sender="pipeline",
                receiver=step.receiver,
                action=step.action,
                payload=current_payload,
                status=MessageStatus.PENDING,
                message_type=MessageType.REQUEST,
                parent_id=last_response.request_id if last_response else None,
            )

            # Pass through Coordinator dispatch!
            last_response = self.coordinator.dispatch(msg)

            # Stop pipeline execution if any step returns an error
            if last_response.status == MessageStatus.ERROR:
                print(f"❌ Pipeline '{name}' halted at step '{step.receiver}': {last_response.error}")
                return last_response

            # Pass output payload into next agent's input
            current_payload = last_response.payload

        return last_response