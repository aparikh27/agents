"""EMBER-backed RobotDriver.

WebotsExecutorAgent (agents/agent/execution_agent/webot_execution.py) only
depends on the structural RobotDriver Protocol it declares locally -- it
never imports Webots. That means no new ExecutorAgent subclass is needed to
drive real EMBER-controlled hardware: EmberRobotDriver below satisfies the
same Protocol, so

    WebotsExecutorAgent(robot_driver=EmberRobotDriver(bridge), world=world_model)

runs the exact same plan-execution logic (_get_object, _put_object,
_align_and_approach, ...) against real hardware, unchanged.
"""

from __future__ import annotations

from agents.agent.execution_agent.ember_bridge_client import EmberBridgeClient
from agents.messaging.ember_wire import AckResult, CommandOp, TelemetrySubsystem


class EmberRobotDriver:
    def __init__(self, bridge: EmberBridgeClient, camera_width_px: int = 640):
        self._bridge = bridge
        # Camera resolution is a vision-agent/perception concern -- the
        # Vision Agent reads the camera directly, never through this
        # bridge -- so it's a plain injected constant here, not a query to
        # EMBER telemetry.
        self._camera_width_px = camera_width_px

    def get_camera_width(self) -> int:
        return self._camera_width_px

    def get_distance_to_front(self) -> float:
        frame = self._bridge.get_latest_telemetry(TelemetrySubsystem.MOTION)
        return frame.value_a if frame is not None else 0.0

    def move_forward(self, distance: float) -> None:
        self._send(CommandOp.MOVE_FORWARD, distance)

    def turn(self, angle: float) -> None:
        self._send(CommandOp.TURN, angle)

    def stop(self) -> None:
        self._send(CommandOp.STOP)

    def raise_arm(self) -> None:
        self._send(CommandOp.RAISE_ARM)

    def lower_arm(self) -> None:
        self._send(CommandOp.LOWER_ARM)

    def grab_item(self) -> None:
        self._send(CommandOp.GRAB_ITEM)

    def release_item(self) -> None:
        self._send(CommandOp.RELEASE_ITEM)

    def _send(self, op: CommandOp, param: float = 0.0) -> None:
        result = self._bridge.send_command(op, param)
        if result in (AckResult.REJECTED, AckResult.FAULTED):
            # Uncaught here on purpose: it propagates up through
            # ExecutorAgent.handle_message (execution.py has no try/except
            # around _execute) into Coordinator.dispatch's existing
            # try/except, which already turns any agent exception into an
            # ERROR Message -- see agents/master_planner/coordinator.py.
            raise RuntimeError(f"EMBER {result.name.lower()} command {op.name}")
