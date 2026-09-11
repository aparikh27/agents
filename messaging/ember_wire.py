"""Wire-format mirror of EMBER's C++ bridge protocol.

Every layout here must match, byte-for-byte, the C++ side:
  - Framing + Fletcher-16 checksum:  edge/serialization/serializer.hpp,
    edge/serialization/checksum.hpp, edge/serialization/frame_codec.hpp
  - Payload structs:                  edge/bridge/bridge_frames.hpp

`struct.Struct(">...")` (standard size, no padding) is used instead of
`ctypes.Structure` deliberately: ctypes struct layout follows the host
compiler's alignment/padding rules, which do not match EMBER's explicit,
padding-free field-by-field byte layout. `struct` with a `">"` prefix packs
exactly the bytes specified, in order, with no hidden padding -- the same
guarantee the C++ side gets from writing each field by hand instead of
reinterpret_cast-ing a struct.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

MAGIC = b"EM"
_HEADER_FMT = struct.Struct(">BH")  # msg_type, payload length
_CHECKSUM_FMT = struct.Struct(">H")


def fletcher16(data: bytes) -> int:
    """Identical algorithm to ember::serialization::calculate_fletcher16."""
    sum1 = 0
    sum2 = 0
    for byte in data:
        sum1 = (sum1 + byte) % 255
        sum2 = (sum2 + sum1) % 255
    return (sum2 << 8) | sum1


def fnv1a_32(text: str) -> int:
    """32-bit FNV-1a over the UTF-8 encoding of `text`.

    Used to compress an AgentCore Message.request_id (a UUID4 string) into
    the fixed-width request_id_hash carried on CommandFrame/AckFrame, so
    EMBER can log/correlate a command's origin without parsing UUID text.
    The reverse (hash -> request_id) mapping is not needed on the wire:
    EmberBridgeClient keeps its own sequence-number -> request_id table for
    that, client-side, at send time.
    """
    h = 0x811C9DC5
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def pack_frame(msg_type: int, payload: bytes) -> bytes:
    """Equivalent of ember::serialization::Serializer::pack."""
    if len(payload) > 0xFFFF:
        raise ValueError("payload exceeds uint16_t length field")
    header = MAGIC + _HEADER_FMT.pack(msg_type, len(payload))
    body = header + payload
    checksum = fletcher16(body)
    return body + _CHECKSUM_FMT.pack(checksum)


@dataclass
class DecodedFrame:
    msg_type: int
    payload: bytes


def unpack_frame(packet: bytes) -> DecodedFrame | None:
    """Equivalent of ember::serialization::FrameCodec::unpack.

    Returns None (never raises) on any structural/integrity failure, since a
    malformed packet arriving over the socket is an expected condition from
    an external process, not a programming error -- same contract as the
    C++ side.
    """
    if len(packet) < 7:  # magic(2) + type(1) + len(2) + checksum(2), empty payload
        return None
    if packet[0:2] != MAGIC:
        return None

    msg_type, length = _HEADER_FMT.unpack_from(packet, 2)
    expected_size = 5 + length + 2
    if len(packet) != expected_size:
        return None

    check_region = packet[: 5 + length]
    (expected_checksum,) = _CHECKSUM_FMT.unpack_from(packet, 5 + length)
    if expected_checksum != fletcher16(check_region):
        return None

    return DecodedFrame(msg_type=msg_type, payload=packet[5 : 5 + length])


class StreamFrameReader:
    """Python twin of ember::serialization::StreamFrameReader.

    A TCP stream has no message boundaries, so incoming bytes are buffered
    here and complete, checksum-verified frames are pulled out as they
    become available. Resyncs past a single corrupt byte instead of
    discarding the whole buffer on one bad frame.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> None:
        self._buf.extend(data)

    def try_extract(self) -> DecodedFrame | None:
        while True:
            if len(self._buf) < 5:
                return None
            if self._buf[0:2] != MAGIC:
                del self._buf[0:1]
                continue

            _msg_type, length = _HEADER_FMT.unpack_from(self._buf, 2)
            total_size = 5 + length + 2
            if len(self._buf) < total_size:
                return None

            packet = bytes(self._buf[:total_size])
            decoded = unpack_frame(packet)
            if decoded is None:
                del self._buf[0:1]
                continue

            del self._buf[:total_size]
            return decoded

    @property
    def buffered_bytes(self) -> int:
        return len(self._buf)


class MsgType(IntEnum):
    COMMAND = 0x01
    TELEMETRY = 0x02
    ACK = 0x03
    HEARTBEAT = 0x04


class CommandOp(IntEnum):
    MOVE_FORWARD = 0x01
    TURN = 0x02
    STOP = 0x03
    RAISE_ARM = 0x04
    LOWER_ARM = 0x05
    GRAB_ITEM = 0x06
    RELEASE_ITEM = 0x07


class TelemetrySubsystem(IntEnum):
    MOTION = 0x01
    MANIPULATOR = 0x02
    BATTERY = 0x03
    THERMAL = 0x04
    FAULT = 0x05
    TASK_STATE = 0x06
    CONNECTION = 0x07


class AckResult(IntEnum):
    ACCEPTED = 0x00
    COMPLETED = 0x01
    REJECTED = 0x02
    FAULTED = 0x03


_COMMAND_FMT = struct.Struct(">IQBfI")  # sequence, timestamp_ns, op, param, request_id_hash
_TELEMETRY_FIXED_FMT = struct.Struct(">IQBffHH")  # ... code, text_len (text bytes follow)
_ACK_FMT = struct.Struct(">IBI")  # ack_sequence, result, request_id_hash


@dataclass
class CommandFrame:
    sequence: int
    op: CommandOp
    timestamp_ns: int = 0
    param: float = 0.0
    request_id_hash: int = 0

    SIZE = _COMMAND_FMT.size  # 21, must equal CommandFrame::kSize in bridge_frames.hpp

    def to_bytes(self) -> bytes:
        return _COMMAND_FMT.pack(
            self.sequence, self.timestamp_ns, int(self.op), self.param, self.request_id_hash
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> CommandFrame | None:
        if len(data) != cls.SIZE:
            return None
        sequence, timestamp_ns, op, param, request_id_hash = _COMMAND_FMT.unpack(data)
        return cls(
            sequence=sequence,
            op=CommandOp(op),
            timestamp_ns=timestamp_ns,
            param=param,
            request_id_hash=request_id_hash,
        )


@dataclass
class TelemetryFrame:
    sequence: int
    subsystem: TelemetrySubsystem
    timestamp_ns: int = 0
    value_a: float = 0.0
    value_b: float = 0.0
    code: int = 0
    text: str = ""

    FIXED_SIZE = _TELEMETRY_FIXED_FMT.size  # 25, must equal TelemetryFrame::kFixedSize

    def to_bytes(self) -> bytes:
        text_bytes = self.text.encode("utf-8")[:0xFFFF]
        header = _TELEMETRY_FIXED_FMT.pack(
            self.sequence,
            self.timestamp_ns,
            int(self.subsystem),
            self.value_a,
            self.value_b,
            self.code,
            len(text_bytes),
        )
        return header + text_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> TelemetryFrame | None:
        if len(data) < cls.FIXED_SIZE:
            return None
        sequence, timestamp_ns, subsystem, value_a, value_b, code, text_len = (
            _TELEMETRY_FIXED_FMT.unpack_from(data, 0)
        )
        if len(data) != cls.FIXED_SIZE + text_len:
            return None
        text = data[cls.FIXED_SIZE : cls.FIXED_SIZE + text_len].decode("utf-8", errors="replace")
        return cls(
            sequence=sequence,
            subsystem=TelemetrySubsystem(subsystem),
            timestamp_ns=timestamp_ns,
            value_a=value_a,
            value_b=value_b,
            code=code,
            text=text,
        )


@dataclass
class AckFrame:
    ack_sequence: int
    result: AckResult
    request_id_hash: int = 0

    SIZE = _ACK_FMT.size  # 9, must equal AckFrame::kSize

    def to_bytes(self) -> bytes:
        return _ACK_FMT.pack(self.ack_sequence, int(self.result), self.request_id_hash)

    @classmethod
    def from_bytes(cls, data: bytes) -> AckFrame | None:
        if len(data) != cls.SIZE:
            return None
        ack_sequence, result, request_id_hash = _ACK_FMT.unpack(data)
        return cls(ack_sequence=ack_sequence, result=AckResult(result), request_id_hash=request_id_hash)
