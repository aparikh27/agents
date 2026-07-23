from typing import Any, Protocol, runtime_checkable

from messaging import Message, MessageStatus
from agent.execution import ExecutorAgent


# ----------------------------------------------------------------------
# Local structural contracts (Protocols), NOT imports.
#
# This agent doesn't import RobotInterface or World from anywhere — it just
# declares the shape of object it needs. Your BAI-side RobotInterface and
# World classes don't need to inherit from these; as long as they implement
# these methods, they satisfy the contract (structural typing / duck typing).
# This keeps the generic framework (execution.py, base_agent.py, messaging)
# completely free of any robotics-specific concept, per your design.
# ----------------------------------------------------------------------

@runtime_checkable
class RobotDriver(Protocol):
    def get_camera_width(self) -> int: ...
    def get_distance_to_front(self) -> float: ...
    def move_forward(self, distance: float) -> None: ...
    def turn(self, angle: float) -> None: ...
    def stop(self) -> None: ...
    def raise_arm(self) -> None: ...
    def lower_arm(self) -> None: ...
    def grab_item(self) -> None: ...
    def release_item(self) -> None: ...


@runtime_checkable
class WorldModel(Protocol):
    def get_visible_objects(self) -> list[Any]: ...
    def get_object_by_track_id(self, track_id: int) -> Any: ...
    # Expected attributes on objects returned above: `.visible` (bool),
    # `.class_name` (str), `.box` (tuple of 4 floats: x1, y1, x2, y2)


# Alias sets, ported from your legacy RobotExecutor.execute_command
DETECT_ALIASES = {
    "detect_item", "detect_object", "find_object", "find_item",
    "search_object", "search_item", "locate_object", "locate_item",
}
PICKUP_ALIASES = {
    "pick_up", "pickup", "grab_item", "grab_object", "lift_item", "lift_object",
}
GET_ALIASES = {
    "get_object", "get_item", "retrieve_object", "retrieve_item",
    "fetch_object", "fetch_item", "move_to", "go_to",
}
PUT_ALIASES = {
    "put_object", "put_item", "drop_item", "drop_object",
    "place_item", "place_object", "deposit_item", "deposit_object",
}


class WebotsExecutorAgent(ExecutorAgent):
    """BAI-specific concrete implementation of ExecutorAgent. Drives a robot
    through sequential action steps. Depends only on injected `robot_driver`
    and `world` objects matching the RobotDriver / WorldModel protocols above
    — it never imports or assumes a specific robot/world implementation."""

    def __init__(self, robot_driver: RobotDriver, world: WorldModel):
        super().__init__()  # Passes name="Executor" up to BaseAgent
        self.robot = robot_driver
        self.world = world

    # ------------------------------------------------------------------
    # Object resolution / servoing (ported from legacy RobotExecutor)
    # ------------------------------------------------------------------

    def _resolve_object(self, target_item: str):
        if target_item is None:
            return None
        target_text = str(target_item).strip()
        if not target_text:
            return None

        if target_text.isdigit():
            obj = self.world.get_object_by_track_id(int(target_text))
            if obj and obj.visible:
                return obj

        normalized_target = target_text.lower()
        for obj in self.world.get_visible_objects():
            if obj.class_name.lower() == normalized_target:
                return obj
        return None

    def _align_and_approach(self, target_item: str) -> float:
        obj = self._resolve_object(target_item)
        if obj is None:
            return 0.0

        camera_width = self.robot.get_camera_width()
        screen_center = camera_width / 2
        pixel_tolerance = 20

        while True:
            obj = self._resolve_object(target_item)
            if not obj:
                self.robot.stop()
                return 0.0

            obj_x_center = (obj.box[0] + obj.box[2]) / 2
            error_pixels = obj_x_center - screen_center

            if abs(error_pixels) <= pixel_tolerance:
                self.robot.stop()
                break

            turn_step = 3.0 if error_pixels > 0 else -3.0
            self.robot.turn(turn_step)

        return self.robot.get_distance_to_front()

    def _get_object(self, target_item: str) -> bool:
        distance = self._align_and_approach(target_item)
        if distance <= 0.0:
            return False

        self.robot.move_forward(distance)
        self.robot.lower_arm()
        self.robot.grab_item()
        self.robot.raise_arm()

        self.robot.turn(180)
        self.robot.move_forward(distance)
        self.robot.turn(180)
        return True

    def _put_object(self, target_item: str) -> bool:
        distance = self._align_and_approach(target_item)
        if distance <= 0.0:
            return False

        self.robot.move_forward(distance)
        self.robot.lower_arm()
        self.robot.release_item()
        self.robot.raise_arm()

        self.robot.turn(180)
        self.robot.move_forward(distance)
        self.robot.turn(180)
        return True

    def _scan_360_for_object(self, target_item: str) -> bool:
        step_angle = 10.0
        total_steps = 36
        for _ in range(total_steps):
            obj = self._resolve_object(target_item)
            if obj is not None:
                self.robot.stop()
                return True
            self.robot.turn(step_angle)

        self.robot.stop()
        return False

    def execute_command(self, command: str, target_item: str) -> bool:
        """Legacy alias-based dispatch, kept for direct-call compatibility."""
        if not command:
            return False

        normalized_cmd = str(command).strip().lower().replace(" ", "_")

        if normalized_cmd in DETECT_ALIASES:
            return self._scan_360_for_object(target_item)
        elif normalized_cmd in PICKUP_ALIASES:
            self.robot.lower_arm()
            self.robot.grab_item()
            self.robot.raise_arm()
            return True
        elif normalized_cmd in GET_ALIASES:
            return self._get_object(target_item)
        elif normalized_cmd in PUT_ALIASES:
            return self._put_object(target_item)

        return False

    # ------------------------------------------------------------------
    # Messaging-layer dispatch
    # ------------------------------------------------------------------

    def _execute_single_step(self, action: str, target: str) -> tuple[bool, str | None]:
        """Executes one action step, returning (success, error_message)."""
        if not action:
            return False, "Missing required parameter: 'action'"

        normalized_cmd = str(action).strip().lower().replace(" ", "_")

        if normalized_cmd in DETECT_ALIASES:
            found = self._scan_360_for_object(target)
            if not found:
                return False, f"Target '{target}' not found after full 360° scan"
            return True, None

        elif normalized_cmd in PICKUP_ALIASES:
            self.robot.lower_arm()
            self.robot.grab_item()
            self.robot.raise_arm()
            return True, None

        elif normalized_cmd in GET_ALIASES:
            success = self._get_object(target)
            if not success:
                return False, f"Could not reach/retrieve target '{target}' (lost alignment or out of range)"
            return True, None

        elif normalized_cmd in PUT_ALIASES:
            success = self._put_object(target)
            if not success:
                return False, f"Could not reach/place target '{target}' (lost alignment or out of range)"
            return True, None

        return False, f"Unknown action: '{action}'"

    def _normalize_steps(self, payload: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Normalizes any of the three supported payload shapes into a list
        of {'action': ..., 'target': ...} step dicts."""
        plan = payload.get("plan")
        if plan is not None:
            if not isinstance(plan, list):
                return None
            return plan

        step = payload.get("step")
        if step is not None:
            if not isinstance(step, dict):
                return None
            return [step]

        action = payload.get("action")
        if action is not None:
            return [{"action": action, "target": payload.get("target")}]

        return None

    def _execute(self, message: Message, payload: dict[str, Any]) -> Message:
        """Executes an incoming plan (list of steps), a single step dict, or a
        direct action/target pair. Steps run sequentially; execution halts
        immediately on the first failure."""
        steps = self._normalize_steps(payload)

        if steps is None:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=(
                    "Missing or invalid payload: provide a 'plan' (list of step dicts), "
                    "a 'step' (dict), or 'action'/'target' directly"
                ),
            )

        if len(steps) == 0:
            return self.create_response(
                request=message,
                payload={"completed_steps": [], "total_steps": 0},
            )

        completed_steps = []

        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                self.robot.stop()
                return self.create_response(
                    request=message,
                    status=MessageStatus.ERROR,
                    error=f"Step {index} is not a valid step object: {step!r}",
                    payload={
                        "completed_steps": completed_steps,
                        "failed_step_index": index,
                        "total_steps": len(steps),
                    },
                )

            step_action = step.get("action")
            step_target = step.get("target")

            success, err = self._execute_single_step(step_action, step_target)

            if not success:
                self.robot.stop()
                return self.create_response(
                    request=message,
                    status=MessageStatus.ERROR,
                    error=f"Execution halted at step {index} ({step_action!r} -> {step_target!r}): {err}",
                    payload={
                        "completed_steps": completed_steps,
                        "failed_step_index": index,
                        "failed_step": step,
                        "total_steps": len(steps),
                    },
                )

            completed_steps.append({"action": step_action, "target": step_target})

        return self.create_response(
            request=message,
            payload={
                "completed_steps": completed_steps,
                "total_steps": len(steps),
            },
        )

    def _stop(self, message: Message, payload: dict[str, Any]) -> Message:
        """Immediately halts the robot and reports status."""
        try:
            self.robot.stop()
        except Exception as exc:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=f"Failed to stop robot: {exc}",
            )

        return self.create_response(
            request=message,
            payload={"status": "stopped"},
        )