#!/usr/bin/env python3
"""Foxy IPC Python API.

This is the supported way for custom Python VR experiences to talk to Foxy.

The transport is a local Unix-domain socket using a simple packet protocol:

    uint32_be header_length
    header JSON bytes
    optional binary payload of header["payload_len"] bytes

Typical loop:

    from foxy_api import FoxyClient

    client = FoxyClient()
    client.connect()

    while True:
        state = client.get_state()
        jpeg = render_stereo_frame(state)
        client.send_frame(jpeg, eye_width=960, eye_height=960, render_views=...)
"""

from __future__ import annotations

import json
import math
import os
import socket
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


DEFAULT_SOCKET_PATH = os.environ.get("FOXY_IPC", "/tmp/foxy_ipc.sock")


class FoxyIPCError(RuntimeError):
    pass


def _send_packet(sock: socket.socket, header: Dict[str, Any], payload: bytes = b"") -> None:
    if payload:
        header = dict(header)
        header["payload_len"] = len(payload)
    else:
        header = dict(header)
        header["payload_len"] = 0

    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack("!I", len(raw)))
    sock.sendall(raw)
    if payload:
        sock.sendall(payload)


def _recvall(sock: socket.socket, n: int) -> bytes:
    chunks = []
    left = n
    while left:
        chunk = sock.recv(left)
        if not chunk:
            raise FoxyIPCError("socket closed")
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)


def _read_packet(sock: socket.socket) -> Tuple[Dict[str, Any], bytes]:
    length_raw = _recvall(sock, 4)
    length = struct.unpack("!I", length_raw)[0]
    if length <= 0 or length > 16 * 1024 * 1024:
        raise FoxyIPCError(f"invalid header length: {length}")

    header = json.loads(_recvall(sock, length).decode("utf-8"))
    payload_len = int(header.get("payload_len", 0) or 0)
    payload = _recvall(sock, payload_len) if payload_len else b""
    return header, payload


def _buttons(inp: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (((inp.get("gamepad") or {}).get("buttons")) or [])


def _axes(inp: Dict[str, Any]) -> List[float]:
    return (((inp.get("gamepad") or {}).get("axes")) or [])


def _semantic(inp: Dict[str, Any]) -> Dict[str, Any]:
    return (((inp.get("gamepad") or {}).get("semantic")) or {})


def button(inp: Optional[Dict[str, Any]], index: int, key: Optional[str] = None) -> Dict[str, Any]:
    """Return a button state.

    If key is provided, semantic mappings like "trigger", "grip", "primary",
    "secondary", and "thumbstick" are checked first. Raw Gamepad indices are
    still available as fallback.
    """
    if not inp:
        return {"pressed": False, "touched": False, "value": 0.0}

    if key:
        sem = _semantic(inp)
        value = sem.get(key)
        if isinstance(value, dict):
            return {
                "pressed": bool(value.get("pressed", False)),
                "touched": bool(value.get("touched", False)),
                "value": float(value.get("value", 0.0) or 0.0),
            }

    btns = _buttons(inp)
    if 0 <= index < len(btns):
        b = btns[index]
        return {
            "pressed": bool(b.get("pressed", False)),
            "touched": bool(b.get("touched", False)),
            "value": float(b.get("value", 0.0) or 0.0),
        }
    return {"pressed": False, "touched": False, "value": 0.0}


def axis(inp: Optional[Dict[str, Any]], index: int, default: float = 0.0) -> float:
    if not inp:
        return default
    axes = _axes(inp)
    if 0 <= index < len(axes):
        try:
            return float(axes[index])
        except Exception:
            return default
    return default


def thumbstick(inp: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    """Best-effort Quest Touch Plus thumbstick mapping.

    WebXR Gamepad mappings vary a bit. Foxy sends both raw axes and semantic
    guesses. This helper prefers semantic values and falls back to common axes.
    """
    if not inp:
        return 0.0, 0.0
    sem = _semantic(inp)
    x = sem.get("thumbstickX", None)
    y = sem.get("thumbstickY", None)
    try:
        if x is not None and y is not None:
            return float(x), float(y)
    except Exception:
        pass
    # Common browser layouts: axes[2]/axes[3] or axes[0]/axes[1].
    ax = _axes(inp)
    if len(ax) >= 4:
        return float(ax[2]), float(ax[3])
    if len(ax) >= 2:
        return float(ax[0]), float(ax[1])
    return 0.0, 0.0


def input_by_hand(state: Dict[str, Any], handedness: str) -> Optional[Dict[str, Any]]:
    for inp in state.get("inputs", []):
        if inp.get("handedness") == handedness:
            return inp
    return None


def _flat_vec(payload: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(payload, dict):
        return None
    try:
        x = float(payload.get("x", 0.0))
        z = float(payload.get("z", 0.0))
    except Exception:
        return None
    mag = (x * x + z * z) ** 0.5
    if not math.isfinite(mag) or mag < 1e-5:
        return None
    return x / mag, z / mag


def _rotate_xz(vec: Tuple[float, float], yaw: float) -> Tuple[float, float]:
    x, z = vec
    c, s = math.cos(yaw), math.sin(yaw)
    return x * c - z * s, x * s + z * c


def head_yaw(state: Dict[str, Any]) -> float:
    """Best-effort headset yaw in radians from Foxy tracking state.

    0 means looking down -Z in the current XR reference space. Newer Foxy pages
    send browser-computed `headingYaw`, which avoids WebXR matrix layout mistakes.
    """
    def yaw_from_forward(fx: float, fz: float) -> Optional[float]:
        mag = (fx * fx + fz * fz) ** 0.5
        if mag < 1e-5:
            return None
        return math.atan2(fx / mag, -fz / mag)

    try:
        pose = ((state.get("viewer") or {}).get("pose") or {})
        heading = pose.get("headingYaw")
        if isinstance(heading, (int, float)) and math.isfinite(float(heading)):
            return float(heading)
        flat = _flat_vec(pose.get("flatForward"))
        if flat is not None:
            return math.atan2(flat[0], -flat[1])
    except Exception:
        pass

    try:
        pose = ((state.get("viewer") or {}).get("pose") or {})
        matrix = pose.get("matrix")
        if isinstance(matrix, list) and len(matrix) == 16:
            yaw = yaw_from_forward(-float(matrix[8]), -float(matrix[10]))
            if yaw is not None:
                return yaw
    except Exception:
        pass

    try:
        views = state.get("views") or {}
        view_data = views.get("left") or views.get("right") or next(iter(views.values()), None)
        vm = view_data.get("viewMatrix") if view_data else None
        if isinstance(vm, list) and len(vm) == 16:
            m = np.array(vm, dtype=np.float32).reshape((4, 4), order="F")
            pose = np.linalg.inv(m)
            yaw = yaw_from_forward(-float(pose[0, 2]), -float(pose[2, 2]))
            if yaw is not None:
                return yaw
    except Exception:
        pass

    return 0.0


def head_basis(state: Dict[str, Any], smooth_yaw: float = 0.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return `(right_xz, forward_xz)` for stable head-relative movement.

    This is safer than converting to yaw and back because pitch/roll are removed
    by the browser and the right vector is rebuilt on the ground plane.
    `smooth_yaw` may be used by experiences that implement Foxy-style artificial
    turn by rotating the rendered world by -smooth_yaw. The helper applies the
    inverse sign so stick movement stays aligned with what the user sees.
    """
    try:
        pose = ((state.get("viewer") or {}).get("pose") or {})
        forward = _flat_vec(pose.get("flatForward"))
        if forward is not None:
            fx, fz = forward
            right = (-fz, fx)
            return _rotate_xz(right, -smooth_yaw), _rotate_xz(forward, -smooth_yaw)
    except Exception:
        pass

    yaw = head_yaw(state)
    forward = (math.sin(yaw), -math.cos(yaw))
    right = (math.cos(yaw), math.sin(yaw))
    return _rotate_xz(right, -smooth_yaw), _rotate_xz(forward, -smooth_yaw)


@dataclass
class FoxyClient:
    """Synchronous IPC client for Foxy VR experiences."""

    socket_path: str = DEFAULT_SOCKET_PATH
    timeout: float = 2.0
    sock: Optional[socket.socket] = None

    def connect(self) -> None:
        self.close()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.socket_path)
        self.sock = sock
        _send_packet(sock, {"type": "hello", "client": "foxy-python-api", "time": time.time()})
        header, _ = _read_packet(sock)
        if header.get("type") != "hello-ok":
            raise FoxyIPCError(f"unexpected hello response: {header}")

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def _require(self) -> socket.socket:
        if self.sock is None:
            raise FoxyIPCError("not connected")
        return self.sock

    def get_state(self) -> Dict[str, Any]:
        """Get the latest Quest tracking, controller state, and server stats."""
        sock = self._require()
        _send_packet(sock, {"type": "get_state", "time": time.time()})
        header, _ = _read_packet(sock)
        if header.get("type") != "state":
            raise FoxyIPCError(f"unexpected state response: {header}")
        return header

    def send_frame(
        self,
        jpeg_sbs: bytes,
        *,
        eye_width: int,
        eye_height: int,
        render_views: Optional[Dict[str, Any]] = None,
        encoding: str = "jpeg-sbs",
        app_name: str = "foxy-experience",
        frame_id: Optional[int] = None,
    ) -> None:
        """Send one side-by-side stereo frame to the Quest.

        jpeg_sbs must contain a single JPEG image where the left eye is the left
        half and the right eye is the right half.

        render_views is optional but recommended for client-side reprojection:

            {
              "left": {
                "view": [...16 floats...],
                "projection": [...16 floats...],
                "viewProjection": [...16 floats...]
              },
              "right": { ... }
            }
        """
        sock = self._require()
        header = {
            "type": "frame",
            "encoding": encoding,
            "eyeWidth": int(eye_width),
            "eyeHeight": int(eye_height),
            "appName": app_name,
            "serverTimeMs": time.time() * 1000.0,
        }
        if frame_id is not None:
            header["frame"] = int(frame_id)
        if render_views is not None:
            header["renderViews"] = render_views
            header["reprojection"] = "client-rotational-timewarp-v1"
        _send_packet(sock, header, jpeg_sbs)

    def ping(self) -> float:
        sock = self._require()
        t0 = time.time()
        _send_packet(sock, {"type": "ping", "time": t0})
        header, _ = _read_packet(sock)
        if header.get("type") != "pong":
            raise FoxyIPCError(f"unexpected ping response: {header}")
        return (time.time() - t0) * 1000.0
