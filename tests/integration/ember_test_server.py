"""Support module for launching and driving the real, compiled EMBER bridge
test server (edge/tests/integration/pipeline_test_server.cpp) as a
subprocess from pytest.

Not a test file itself (pytest.ini's python_files = test_*.py won't
collect it) -- imported by agents/tests/integration/conftest.py.

Locating/building the binary tries, in order:
  1. $EMBER_TEST_SERVER_BIN, if set (CI can point this at a prebuilt binary).
  2. A CMake build output under edge/ (build/, cmake-build-*/, out/build/),
     if one has already been built via edge/CMakeLists.txt's
     ember_pipeline_test_server target.
  3. A direct g++ compile+link, cached under the system temp dir and
     rebuilt only when a source file is newer than the cached binary --
     this is the same command line used to validate the bridge during
     development (see design-decisions/05-ember-agentcore-bridge.md), kept
     here so the integration suite is runnable without CMake.

If none of that produces a binary (no C++ toolchain available at all),
find_or_build_server_binary() returns None and the ember_server_binary
fixture in conftest.py skips every test that depends on it -- mirroring
how agents/tests/conftest.py stubs out whisper/cv2/ultralytics/llama_cpp
when they aren't installed, rather than failing the whole suite.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

EDGE_DIR = Path(__file__).resolve().parents[3] / "edge"

_CACHE_DIR = Path(tempfile.gettempdir()) / "ember_pipeline_test_server_cache"

_GXX_SOURCES = [
    "run/log.cpp",
    "run/time.cpp",
    "run/runtime.cpp",
    "messages/coordinator.cpp",
    "messages/publisher.cpp",
    "messages/subscriber.cpp",
    "schedule/task.cpp",
    "schedule/scheduler.cpp",
    "bridge/tcp_socket.cpp",
    "bridge/ember_bridge_adapter.cpp",
    "tests/integration/pipeline_test_server.cpp",
]


def find_free_port() -> int:
    """Binds an ephemeral port and immediately releases it. Small race
    between release and the C++ server's own bind() -- acceptable for test
    infra; a collision just makes that one test's ember_server fixture
    fail to reach READY, which is a clear, non-silent failure."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _candidate_cmake_binaries():
    binary_names = ("ember_pipeline_test_server.exe", "ember_pipeline_test_server")
    for build_dir in ("build", "cmake-build-debug", "cmake-build-release", "out/build"):
        for sub in ("", "Debug", "Release", "RelWithDebInfo"):
            base = EDGE_DIR / build_dir / sub if sub else EDGE_DIR / build_dir
            for name in binary_names:
                yield base / name


def _direct_gxx_build() -> Path | None:
    gxx = shutil.which("g++")
    if gxx is None:
        return None

    binary_name = "ember_pipeline_test_server.exe" if os.name == "nt" else "ember_pipeline_test_server"
    out_path = _CACHE_DIR / binary_name
    source_paths = [EDGE_DIR / s for s in _GXX_SOURCES]

    if out_path.exists():
        newest_source_mtime = max(p.stat().st_mtime for p in source_paths if p.exists())
        if out_path.stat().st_mtime >= newest_source_mtime:
            return out_path  # cache hit, nothing changed since the last build

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # -static: a dev box can easily have more than one MinGW/MSYS2 toolchain
    # on PATH at once (e.g. both /mingw64/bin and a UCRT64 install), each
    # with its own incompatible libstdc++/libgcc/libwinpthread DLLs. Which
    # one a dynamically-linked binary picks up at launch depends on
    # whatever's first on PATH *at that moment* -- which differs between a
    # binary launched by hand and one launched via subprocess.Popen from a
    # different parent process/environment, and a mismatch fails the
    # process at startup with no useful error (observed during development
    # as this fixture's own subprocess printing nothing at all). Statically
    # linking the runtime removes the ambiguity entirely.
    cmd = [gxx, "-std=c++20", "-O1", "-static", "-I", str(EDGE_DIR), *[str(p) for p in source_paths], "-o", str(out_path)]
    cmd += ["-lws2_32", "-lpthread"] if os.name == "nt" else ["-lpthread"]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to build ember_pipeline_test_server via g++:\n"
            f"command: {' '.join(cmd)}\n\nstderr:\n{result.stderr}"
        )
    return out_path


def find_or_build_server_binary() -> Path | None:
    env_path = os.environ.get("EMBER_TEST_SERVER_BIN")
    if env_path:
        candidate = Path(env_path)
        return candidate if candidate.is_file() else None

    for candidate in _candidate_cmake_binaries():
        if candidate.is_file():
            return candidate

    return _direct_gxx_build()


class EmberTestServerProcess:
    """Wraps the compiled pipeline_test_server subprocess: parses its
    structured stdout into a queryable line history, and sends it
    stdin-based test control commands (INJECT_*, QUIT) -- see the protocol
    documented at the top of pipeline_test_server.cpp."""

    def __init__(self, binary_path: Path, port: int, watchdog_ms: int = 300, actuation_ms: int = 15):
        self.port = port
        self._proc = subprocess.Popen(
            [str(binary_path), str(port), str(watchdog_ms), str(actuation_ms)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._all_lines: list[str] = []
        self._lines_lock = threading.Lock()
        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()

    def _read_stdout(self) -> None:
        assert self._proc.stdout is not None
        for raw_line in self._proc.stdout:
            with self._lines_lock:
                self._all_lines.append(raw_line.rstrip("\n"))

    def all_lines(self) -> list[str]:
        with self._lines_lock:
            return list(self._all_lines)

    def wait_for_line(self, predicate, timeout: float = 5.0) -> str | None:
        """Blocks until a line satisfying `predicate` has been seen --
        checking history already buffered before this call as well as
        anything that arrives afterward -- or returns None on timeout."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lines_lock:
                for line in self._all_lines:
                    if predicate(line):
                        return line
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.02)

    def wait_ready(self, timeout: float = 10.0) -> bool:
        return self.wait_for_line(lambda line: line.startswith("READY"), timeout=timeout) is not None

    def _send_control(self, line: str) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def inject_battery_low(self, percentage: int, voltage: float) -> None:
        self._send_control(f"INJECT_BATTERY_LOW {percentage} {voltage}")

    def inject_fault(self, code: int, component: str, description: str) -> None:
        self._send_control(f"INJECT_FAULT {code} {component} {description}")

    def inject_thermal(self, component: str, temperature: float) -> None:
        self._send_control(f"INJECT_THERMAL {component} {temperature}")

    def stop(self, timeout: float = 5.0) -> None:
        if self._proc.poll() is not None:
            return
        try:
            self._send_control("QUIT")
            self._proc.wait(timeout=timeout)
        except Exception:
            self._proc.kill()
            self._proc.wait(timeout=timeout)
