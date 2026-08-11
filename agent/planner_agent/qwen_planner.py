from typing import Any
import json

from llama_cpp import Llama

from agents.messaging import Message, MessageStatus
from agents.agent.planner_agent.planner import PlannerAgent


class QwenPlannerAgent(PlannerAgent):
    """Concrete implementation of PlannerAgent using a local Qwen GGUF model
    (via llama_cpp) to generate step-by-step robot action plans."""

    def __init__(self, model_path: str = "backend/models/qwen2.5-1.5b-instruct-q5_k_m.gguf"):
        super().__init__()  # Passes name="Planner" up to BaseAgent
        self.model_path = model_path
        self.model = None

        self.system_instruction = (
            "You are the brain behind a robot. Given an input command, you must output a "
            "JSON array containing step-by-step action objects. Do not include any "
            "conversational text, explanations, or markdown code blocks.\n"
            "CRITICAL RULE: If the input command is gibberish, background noise, unrelated "
            "chatter, or does not contain an active request for the robot, you MUST ignore "
            "it entirely and output a completely empty JSON array: []."
        )

    def _load_model(self):
        """Lazily loads and caches the Qwen GGUF model (mirrors legacy QWENBRAIN._load_model)."""
        if self.model is None:
            print(f"🧠 [QwenPlannerAgent] Loading Qwen model from '{self.model_path}'...")
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=512,
                n_threads=4,
                verbose=False,
            )
        return self.model

    def _build_prompt(self, task_data: str) -> str:
        """Builds the few-shot ChatML-style prompt (ported verbatim from legacy QWENBRAIN)."""
        return (
            f"<|im_start|>system\n{self.system_instruction}<|im_end|>\n"
            f"<|im_start|>user\nTask Input: look around for my car keys<|im_end|>\n"
            f"<|im_start|>assistant\n" + '[{"action": "detect_object", "target": "keys"}]' + "<|im_end|>\n"
            f"<|im_start|>user\nTask Input: Pick up the red ball and bring it to me<|im_end|>\n"
            f"<|im_start|>assistant\n" + '[{"action": "detect_object", "target": "red ball"}, {"action": "pick_up", "target": "red ball"}, {"action": "get_object", "target": "red ball"}]' + "<|im_end|>\n"
            f"<|im_start|>user\nTask Input: {task_data}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _run_inference(self, task_data: str) -> str:
        """Runs the model on a built prompt and returns the raw text completion."""
        model = self._load_model()
        prompt = self._build_prompt(task_data)

        output = model(
            prompt,
            max_tokens=150,
            temperature=0.1,
            stop=["<|im_end|>", "<|im_start|>"],
        )

        return output["choices"][0]["text"].strip()

    def _parse_plan(self, raw_text: str) -> list:
        """Parses the model's raw text output into a native Python list of step dicts.
        Tolerates minor formatting noise (e.g. stray markdown fences) before failing."""
        cleaned = raw_text.strip()

        # Strip accidental markdown code fences, in case the model ignores instructions
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        parsed = json.loads(cleaned)  # raises json.JSONDecodeError on failure

        if not isinstance(parsed, list):
            raise ValueError(f"Expected a JSON array, got {type(parsed).__name__}")

        return parsed

    def _create_plan(self, message: Message, payload: dict[str, Any]) -> Message:
        """Extracts the task/command, runs it through the Qwen model, and returns
        the parsed plan as a native Python list under payload['plan']."""
        task_data = payload.get("task") or payload.get("command") or payload.get("text")

        if not task_data:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error="Missing required payload parameter: provide 'task', 'command', or 'text'",
            )

        try:
            raw_text = self._run_inference(task_data)
        except Exception as exc:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=f"Qwen model inference failed: {exc}",
            )

        try:
            plan = self._parse_plan(raw_text)
        except (json.JSONDecodeError, ValueError) as exc:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=f"Failed to parse model output as JSON plan: {exc}",
            )

        return self.create_response(
            request=message,
            payload={
                "task": task_data,
                "plan": plan,
            },
        )

    def _update_plan(self, message: Message, payload: dict[str, Any]) -> Message:
        """Re-runs an existing plan through the LLM alongside new context/feedback,
        producing an updated plan. Simple implementation: folds the existing plan
        and update context into a single task string and re-plans from scratch."""
        existing_plan = payload.get("plan")
        update_context = payload.get("context") or payload.get("update") or payload.get("command")

        if existing_plan is None:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error="Missing required payload parameter: 'plan' (existing plan to update)",
            )

        if not update_context:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error="Missing required payload parameter: 'context' (or 'update'/'command') describing the requested change",
            )

        try:
            existing_plan_str = json.dumps(existing_plan)
        except (TypeError, ValueError) as exc:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=f"Existing 'plan' is not JSON-serializable: {exc}",
            )

        combined_task = (
            f"Existing plan: {existing_plan_str}. "
            f"Update instruction: {update_context}"
        )

        try:
            raw_text = self._run_inference(combined_task)
        except Exception as exc:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=f"Qwen model inference failed: {exc}",
            )

        try:
            updated_plan = self._parse_plan(raw_text)
        except (json.JSONDecodeError, ValueError) as exc:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=f"Failed to parse model output as updated JSON plan: {exc}",
            )

        return self.create_response(
            request=message,
            payload={
                "previous_plan": existing_plan,
                "plan": updated_plan,
            },
        )