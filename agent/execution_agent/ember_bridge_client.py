"""Thread-safe synchronous facade over an asyncio TCP connection to EMBER's
EmberBridgeAdapter (edge/bridge/ember_bridge_adapter.hpp).

AgentCore's agent framework (agents/agent/base_agent.py, execution.py,
webot_execution.py) is entirely synchronous -- no asyncio anywhere in
agents/ (see the codebase survey in design-decisions/05-ember-agentcore-bridge.md).
Rather than making the whole agent framework async just for this one I/O
boundary, this class runs its own asyncio event loop on a dedicated
background thread and exposes plain blocking methods (send_command,
get_latest_telemetry) that any RobotDriver implementation can call directly
from ExecutorAgent's synchronous call stack via
asyncio.run_coroutine_threadsafe.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

from agents.messaging.ember_wire import (
    AckFrame,
    AckResult,
    CommandFrame,
    CommandOp,
    MsgType,
    StreamFrameReader,
    TelemetryFrame,
    TelemetrySubsystem,
    fnv1a_32,
    pack_frame,
)

logger = logging.getLogger(__name__)


@dataclass
class _PendingCommand:
    request_id: str
    future: asyncio.Future


class EmberBridgeClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5601,
        command_timeout_s: float = 2.0,
        reconnect_backoff_s: tuple[float, float] = (0.25, 5.0),
        heartbeat_interval_s: float = 0.1,
        on_telemetry: Callable[[TelemetryFrame], None] | None = None,
    ):
        self._host = host
        self._port = port
        self._command_timeout_s = command_timeout_s
        self._reconnect_min, self._reconnect_max = reconnect_backoff_s
        self._heartbeat_interval_s = heartbeat_interval_s
        self._on_telemetry = on_telemetry

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = threading.Event()
        self._stopping = threading.Event()

        self._sequence_lock = threading.Lock()
        self._sequence = 0

        # _pending is mutated only from the event-loop thread (writes in
        # _send_command_async, pops in _resolve_pending/_fail_all_pending,
        # and the timeout-cleanup pop below, which is itself marshalled
        # onto the loop via call_soon_threadsafe) -- so, deliberately, no
        # lock guards it; every access is confined to one thread.
        self._pending: dict[int, _PendingCommand] = {}

        self._telemetry_lock = threading.Lock()
        self._latest_telemetry: dict[int, TelemetryFrame] = {}

    # ---- lifecycle -------------------------------------------------

    def start(self) -> None:
        """Starts the background event-loop thread and begins connecting.
        Non-blocking; use wait_connected() to block until the first
        connection succeeds (e.g. during AgentCore startup)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_loop, name="ember-bridge-client", daemon=True)
        self._thread.start()

    def wait_connected(self, timeout_s: float | None = None) -> bool:
        return self._connected.wait(timeout_s)

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def set_telemetry_callback(self, callback: Callable[[TelemetryFrame], None] | None) -> None:
        """Sets/replaces the telemetry callback after construction.

        Real wiring is naturally circular: an EmberRobotDriver needs this
        client to exist before ExecutorAgent/Coordinator can be built, but a
        telemetry relay (see ember_telemetry_relay.py) needs the
        already-built Coordinator to dispatch into. Constructing the client
        with on_telemetry=None and calling this once the Coordinator exists
        breaks that cycle. Plain attribute assignment is enough here: the
        callback is only ever read from the event-loop thread in
        _handle_decoded, and CPython attribute writes don't tear, so no lock
        is needed -- a callback swapped in mid-flight takes effect on the
        next decoded frame with no other ordering guarantee."""
        self._on_telemetry = callback

    def stop(self) -> None:
        # Setting the flag is enough: _reader_task polls it on a bounded
        # wait_for() below, so _connection_supervisor's coroutine returns
        # on its own within one poll interval and run_until_complete exits
        # normally. Calling loop.stop() directly here instead would raise
        # "Event loop stopped before Future completed." — stop() competes
        # with run_until_complete's own control of the loop.
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connection_supervisor())
        finally:
            self._loop.close()

    # ---- connection supervisor --------------------------------------

    async def _connection_supervisor(self) -> None:
        backoff = self._reconnect_min
        while not self._stopping.is_set():
            try:
                reader, writer = await asyncio.open_connection(self._host, self._port)
                self._writer = writer
                self._connected.set()
                backoff = self._reconnect_min
                logger.info("EmberBridgeClient connected to %s:%s", self._host, self._port)

                await asyncio.gather(
                    self._reader_task(reader),
                    self._heartbeat_task(writer),
                )
            except (ConnectionError, OSError, asyncio.IncompleteReadError) as exc:
                logger.warning("EmberBridgeClient connection lost/failed: %s", exc)
            finally:
                self._connected.clear()
                self._writer = None
                self._fail_all_pending("EMBER bridge connection lost")

            if self._stopping.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._reconnect_max)

    async def _reader_task(self, reader: asyncio.StreamReader) -> None:
        framer = StreamFrameReader()
        while not self._stopping.is_set():
            try:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=0.5)
            except asyncio.TimeoutError:
                continue  # just a poll interval so _stopping gets rechecked, not an error
            if not chunk:
                raise ConnectionError("EMBER closed the bridge connection")
            framer.feed(chunk)
            while (decoded := framer.try_extract()) is not None:
                self._handle_decoded(decoded)

    async def _heartbeat_task(self, writer: asyncio.StreamWriter) -> None:
        # Keeps EMBER's per-connection watchdog (edge/bridge/ember_bridge_adapter.hpp)
        # from tripping during idle periods with no in-flight motion command.
        while not self._stopping.is_set():
            writer.write(pack_frame(int(MsgType.HEARTBEAT), b""))
            await writer.drain()
            await asyncio.sleep(self._heartbeat_interval_s)

    def _handle_decoded(self, decoded) -> None:
        if decoded.msg_type == int(MsgType.ACK):
            ack = AckFrame.from_bytes(decoded.payload)
            if ack is not None:
                self._resolve_pending(ack)
        elif decoded.msg_type == int(MsgType.TELEMETRY):
            telem = TelemetryFrame.from_bytes(decoded.payload)
            if telem is not None:
                with self._telemetry_lock:
                    self._latest_telemetry[int(telem.subsystem)] = telem
                if self._on_telemetry is not None:
                    try:
                        self._on_telemetry(telem)
                    except Exception:
                        logger.exception("EmberBridgeClient: on_telemetry callback raised")

    def _resolve_pending(self, ack: AckFrame) -> None:
        pending = self._pending.pop(ack.ack_sequence, None)
        if pending is not None and not pending.future.done():
            pending.future.set_result(ack)

    def _fail_all_pending(self, reason: str) -> None:
        items = list(self._pending.items())
        self._pending.clear()
        for _sequence, pending in items:
            if not pending.future.done():
                pending.future.set_exception(ConnectionError(reason))

    # ---- public synchronous API (called from ExecutorAgent's thread) -

    def send_command(self, op: CommandOp, param: float = 0.0, request_id: str = "") -> AckResult:
        """Blocks the calling thread until EMBER acks the command or
        command_timeout_s elapses. Returns AckResult.REJECTED on
        timeout/disconnect rather than raising a network exception --
        callers (see EmberRobotDriver) turn that into a RuntimeError that
        propagates up through ExecutorAgent.handle_message into
        Coordinator.dispatch's existing exception-to-ERROR-Message handling,
        so no framework code needs to know the bridge exists."""
        if self._loop is None or not self._connected.is_set():
            return AckResult.REJECTED

        with self._sequence_lock:
            self._sequence += 1
            sequence = self._sequence

        future = asyncio.run_coroutine_threadsafe(
            self._send_command_async(sequence, op, param, request_id), self._loop
        )
        try:
            return future.result(timeout=self._command_timeout_s)
        except Exception:
            # Don't leave an orphaned entry in _pending if the ack never
            # arrives (or arrives after we've already given up) -- marshal
            # the cleanup onto the loop thread, the only thread allowed to
            # touch _pending.
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._pending.pop, sequence, None)
            return AckResult.REJECTED

    async def _send_command_async(
        self, sequence: int, op: CommandOp, param: float, request_id: str
    ) -> AckResult:
        if self._writer is None:
            raise ConnectionError("not connected")

        frame = CommandFrame(
            sequence=sequence,
            op=op,
            timestamp_ns=time.monotonic_ns(),
            param=param,
            request_id_hash=fnv1a_32(request_id) if request_id else 0,
        )

        ack_future = self._loop.create_future()
        self._pending[sequence] = _PendingCommand(request_id=request_id, future=ack_future)

        self._writer.write(pack_frame(int(MsgType.COMMAND), frame.to_bytes()))
        await self._writer.drain()

        ack = await ack_future
        return ack.result

    def get_latest_telemetry(self, subsystem: TelemetrySubsystem) -> TelemetryFrame | None:
        """Non-blocking: returns the most recently received TelemetryFrame
        for `subsystem`, or None if none has arrived yet. Never awaits a
        network round trip -- a synchronous RobotDriver method (e.g.
        get_distance_to_front) polls the last-known value instead, the same
        pattern real embedded/ROS sensor drivers use."""
        with self._telemetry_lock:
            return self._latest_telemetry.get(int(subsystem))
