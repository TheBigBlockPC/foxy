#!/usr/bin/env python3
from __future__ import annotations

import argparse
import array
import asyncio
import io
import json
import math
import os
import pathlib
import ssl
import struct
import sys
import tempfile
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import web
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
except Exception as e:
    RTCPeerConnection = None
    RTCSessionDescription = None
    AIORTC_IMPORT_ERROR = e
else:
    AIORTC_IMPORT_ERROR = None

try:
    import moderngl
except Exception as e:
    moderngl = None
    MODERNGL_IMPORT_ERROR = e
else:
    MODERNGL_IMPORT_ERROR = None

try:
    import mss
except Exception:
    mss = None


ROOT = pathlib.Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
DEFAULT_IPC_PATH = os.environ.get("FOXY_IPC", "/tmp/foxy_ipc.sock")
DEFAULT_HOTSPOT_DOMAIN = "foxy.local"
DEFAULT_HOTSPOT_ADDRESS = "10.42.0.1"
RTC_VIDEO_MAGIC = b"FXV1"


def now_ms() -> float:
    return time.time() * 1000.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mat_from_webxr(vals: Optional[List[float]]) -> Optional[np.ndarray]:
    if not vals or len(vals) != 16:
        return None
    return np.array(vals, dtype=np.float32).reshape((4, 4), order="F")


def mat_to_gl_bytes(m: np.ndarray) -> bytes:
    return m.T.astype("f4", copy=False).tobytes()


def mat_to_webgl_list(m: np.ndarray) -> List[float]:
    return [float(x) for x in m.T.reshape(16)]


def perspective(fovy_rad: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fovy_rad / 2.0)
    out = np.zeros((4, 4), dtype=np.float32)
    out[0, 0] = f / aspect
    out[1, 1] = f
    out[2, 2] = (far + near) / (near - far)
    out[2, 3] = (2.0 * far * near) / (near - far)
    out[3, 2] = -1.0
    return out


def look_at(eye, target, up=(0, 1, 0)) -> np.ndarray:
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-8)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-8)
    u = np.cross(s, f)
    out = np.identity(4, dtype=np.float32)
    out[0, 0:3] = s
    out[1, 0:3] = u
    out[2, 0:3] = -f
    out[0, 3] = -np.dot(s, eye)
    out[1, 3] = -np.dot(u, eye)
    out[2, 3] = np.dot(f, eye)
    return out


def translate(x: float, y: float, z: float) -> np.ndarray:
    m = np.identity(4, dtype=np.float32)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scale(x: float, y: float, z: float) -> np.ndarray:
    m = np.identity(4, dtype=np.float32)
    m[0, 0] = x
    m[1, 1] = y
    m[2, 2] = z
    return m


def rotate_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = np.identity(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


def rotate_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = np.identity(4, dtype=np.float32)
    m[1, 1] = c
    m[1, 2] = -s
    m[2, 1] = s
    m[2, 2] = c
    return m


def rotate_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = np.identity(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return m


def make_cube_vertices() -> np.ndarray:
    p = [
        (-1,-1,1),(1,-1,1),(1,1,1),(-1,-1,1),(1,1,1),(-1,1,1),
        (1,-1,-1),(-1,-1,-1),(-1,1,-1),(1,-1,-1),(-1,1,-1),(1,1,-1),
        (-1,-1,-1),(-1,-1,1),(-1,1,1),(-1,-1,-1),(-1,1,1),(-1,1,-1),
        (1,-1,1),(1,-1,-1),(1,1,-1),(1,-1,1),(1,1,-1),(1,1,1),
        (-1,1,1),(1,1,1),(1,1,-1),(-1,1,1),(1,1,-1),(-1,1,-1),
        (-1,-1,-1),(1,-1,-1),(1,-1,1),(-1,-1,-1),(1,-1,1),(-1,-1,1),
    ]
    colors = [
        (0.9,0.25,0.20),(0.9,0.25,0.20),(0.9,0.25,0.20),(0.9,0.25,0.20),(0.9,0.25,0.20),(0.9,0.25,0.20),
        (0.18,0.55,0.95),(0.18,0.55,0.95),(0.18,0.55,0.95),(0.18,0.55,0.95),(0.18,0.55,0.95),(0.18,0.55,0.95),
        (0.25,0.9,0.45),(0.25,0.9,0.45),(0.25,0.9,0.45),(0.25,0.9,0.45),(0.25,0.9,0.45),(0.25,0.9,0.45),
        (0.95,0.85,0.25),(0.95,0.85,0.25),(0.95,0.85,0.25),(0.95,0.85,0.25),(0.95,0.85,0.25),(0.95,0.85,0.25),
        (0.65,0.35,0.95),(0.65,0.35,0.95),(0.65,0.35,0.95),(0.65,0.35,0.95),(0.65,0.35,0.95),(0.65,0.35,0.95),
        (0.85,0.85,0.85),(0.85,0.85,0.85),(0.85,0.85,0.85),(0.85,0.85,0.85),(0.85,0.85,0.85),(0.85,0.85,0.85),
    ]
    return np.array([(*pos, *col) for pos, col in zip(p, colors)], dtype=np.float32)


def make_grid_vertices(size=8, step=1.0) -> np.ndarray:
    verts = []
    color_major = (0.45, 0.55, 0.75)
    color_minor = (0.18, 0.22, 0.30)
    for i in range(-size, size + 1):
        c = color_major if i == 0 else color_minor
        x = i * step
        verts.append((x, 0.0, -size * step, *c))
        verts.append((x, 0.0, size * step, *c))
        z = i * step
        verts.append((-size * step, 0.0, z, *c))
        verts.append((size * step, 0.0, z, *c))
    return np.array(verts, dtype=np.float32)


def make_quad_vertices() -> np.ndarray:
    return np.array([
        -1, -1, 0,  0, 1,
         1, -1, 0,  1, 1,
        -1,  1, 0,  0, 0,
         1,  1, 0,  1, 0,
    ], dtype=np.float32)


def input_button(inp: Dict[str, Any], idx: int, semantic_key: Optional[str] = None) -> Dict[str, Any]:
    if semantic_key:
        try:
            sem = ((inp.get("gamepad") or {}).get("semantic") or {})
            b = sem.get(semantic_key)
            if isinstance(b, dict):
                return {"pressed": bool(b.get("pressed", False)), "touched": bool(b.get("touched", False)), "value": float(b.get("value", 0.0))}
        except Exception:
            pass
    try:
        b = ((inp.get("gamepad") or {}).get("buttons") or [])[idx]
        return {"pressed": bool(b.get("pressed", False)), "touched": bool(b.get("touched", False)), "value": float(b.get("value", 0.0))}
    except Exception:
        return {"pressed": False, "touched": False, "value": 0.0}


def input_axis(inp: Dict[str, Any], idx: int, default: float = 0.0) -> float:
    try:
        return float(((inp.get("gamepad") or {}).get("axes") or [])[idx])
    except Exception:
        return default


def input_thumbstick(inp: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    if not inp:
        return 0.0, 0.0
    try:
        sem = ((inp.get("gamepad") or {}).get("semantic") or {})
        sx, sy = sem.get("thumbstickX"), sem.get("thumbstickY")
        if sx is not None and sy is not None:
            return float(sx), float(sy)
    except Exception:
        pass
    axes = ((inp.get("gamepad") or {}).get("axes") or [])
    try:
        if len(axes) >= 4:
            return float(axes[2]), float(axes[3])
        if len(axes) >= 2:
            return float(axes[0]), float(axes[1])
    except Exception:
        pass
    return 0.0, 0.0


def input_by_hand(inputs: List[Dict[str, Any]], hand: str) -> Optional[Dict[str, Any]]:
    for inp in inputs:
        if inp.get("handedness") == hand:
            return inp
    return None


def pose_position(pose: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float, float]]:
    if not pose:
        return None
    pos = pose.get("position") or {}
    try:
        return float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))
    except Exception:
        return None


def pose_forward_ray(pose: Optional[Dict[str, Any]], length: float = 1.0) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    if not pose:
        return None
    matrix = pose.get("matrix")
    if isinstance(matrix, list) and len(matrix) == 16:
        try:
            ox, oy, oz = float(matrix[12]), float(matrix[13]), float(matrix[14])
            fx, fy, fz = -float(matrix[8]), -float(matrix[9]), -float(matrix[10])
            n = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
            fx, fy, fz = fx / n, fy / n, fz / n
            return (ox, oy, oz), (ox + fx * length, oy + fy * length, oz + fz * length)
        except Exception:
            pass
    return None


def _yaw_from_forward_xz(fx: float, fz: float) -> Optional[float]:
    """Return yaw where 0 means -Z forward, using an X/Z forward vector."""
    mag = math.sqrt(fx * fx + fz * fz)
    if mag < 1e-5:
        return None
    fx /= mag
    fz /= mag
    return math.atan2(fx, -fz)


def _flat_vec(payload: Any) -> Optional[Tuple[float, float]]:
    """Read and normalize an {x,z} vector from WebXR payload data."""
    if not isinstance(payload, dict):
        return None
    try:
        x = float(payload.get("x", 0.0))
        z = float(payload.get("z", 0.0))
    except Exception:
        return None
    n = math.sqrt(x * x + z * z)
    if not math.isfinite(n) or n < 1e-5:
        return None
    return x / n, z / n


def _rotate_xz(vec: Tuple[float, float], yaw: float) -> Tuple[float, float]:
    """Rotate an X/Z direction in XR/world coordinates.

    Positive yaw rotates the canonical -Z forward vector toward +X. The hub
    renders artificial smooth-turn by rotating the world by -player_yaw, so
    locomotion must pass -player_yaw when converting head-relative input into
    hub-world movement.
    """
    x, z = vec
    c, s = math.cos(yaw), math.sin(yaw)
    return x * c - z * s, x * s + z * c


def head_basis_from_tracking(tracking: Dict[str, Any], smooth_yaw: float = 0.0) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Return (flat_right, flat_forward, head_yaw) for locomotion.

    New clients send flatForward computed from the WebXR orientation quaternion
    in the browser. Prefer that over server-side matrix decoding because it
    avoids row/column confusion and removes pitch/roll from walking. Fallbacks
    are kept for older pages.
    """
    yaw = head_yaw_from_tracking(tracking)

    try:
        pose = ((tracking.get("viewer") or {}).get("pose") or {})
        forward = _flat_vec(pose.get("flatForward"))
        if forward is not None:
            # Rebuild right from the flattened forward vector so head roll never
            # makes strafing diagonal or inverted.
            fx, fz = forward
            right = (-fz, fx)
            return _rotate_xz(right, -smooth_yaw), _rotate_xz(forward, -smooth_yaw), math.atan2(fx, -fz)
    except Exception:
        pass

    # Legacy fallback: synthesize a basis from the yaw estimate.
    forward = (math.sin(yaw), -math.cos(yaw))
    right = (math.cos(yaw), math.sin(yaw))
    return _rotate_xz(right, -smooth_yaw), _rotate_xz(forward, -smooth_yaw), yaw


def head_yaw_from_tracking(tracking: Dict[str, Any]) -> float:
    """Best-effort headset yaw from WebXR tracking data.

    0 means looking down -Z in the XR reference space; positive yaw turns the
    forward vector toward +X. Prefer browser-computed headingYaw/flatForward,
    then fall back to WebXR transform matrices for older clients.
    """
    try:
        pose = ((tracking.get("viewer") or {}).get("pose") or {})
        heading = pose.get("headingYaw")
        if isinstance(heading, (int, float)) and math.isfinite(float(heading)):
            return float(heading)
        flat = _flat_vec(pose.get("flatForward"))
        if flat is not None:
            fx, fz = flat
            return math.atan2(fx, -fz)
    except Exception:
        pass

    try:
        pose = ((tracking.get("viewer") or {}).get("pose") or {})
        matrix = pose.get("matrix")
        if isinstance(matrix, list) and len(matrix) == 16:
            yaw = _yaw_from_forward_xz(-float(matrix[8]), -float(matrix[10]))
            if yaw is not None:
                return yaw
    except Exception:
        pass

    try:
        views = tracking.get("views") or {}
        view_data = views.get("left") or views.get("right") or next(iter(views.values()), None)
        if view_data:
            view = mat_from_webxr(view_data.get("viewMatrix"))
            if view is not None:
                pose = np.linalg.inv(view)
                yaw = _yaw_from_forward_xz(-float(pose[0, 2]), -float(pose[2, 2]))
                if yaw is not None:
                    return yaw
    except Exception:
        pass

    return 0.0


@dataclass
class TrackingState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    views: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    viewer: Dict[str, Any] = field(default_factory=dict)
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    client_time_ms: float = 0.0
    recv_time_ms: float = 0.0
    packets: int = 0

    async def update(self, payload: Dict[str, Any]) -> None:
        async with self.lock:
            self.views = {v.get("eye", f"eye{idx}"): v for idx, v in enumerate(payload.get("views", []))}
            self.viewer = payload.get("viewer", {})
            self.inputs = payload.get("inputs", [])
            ev = payload.get("events", [])
            if ev:
                self.events = (self.events + ev)[-64:]
            self.client_time_ms = float(payload.get("clientTimeMs", 0.0))
            self.recv_time_ms = now_ms()
            self.packets += 1

    async def snapshot(self) -> Dict[str, Any]:
        async with self.lock:
            return {
                "views": dict(self.views),
                "viewer": dict(self.viewer),
                "inputs": list(self.inputs),
                "events": list(self.events),
                "client_time_ms": self.client_time_ms,
                "recv_time_ms": self.recv_time_ms,
                "packets": self.packets,
            }

    async def reset(self) -> None:
        async with self.lock:
            self.views = {}
            self.viewer = {}
            self.inputs = []
            self.events = []


@dataclass
class ExternalFrameState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    frame_bytes: Optional[bytes] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    last_ms: float = 0.0
    client_name: str = ""

    async def set_frame(self, meta: Dict[str, Any], frame: bytes) -> None:
        async with self.lock:
            self.frame_bytes = frame
            self.meta = dict(meta)
            self.last_ms = now_ms()
            self.client_name = str(meta.get("appName") or meta.get("client") or "ipc-experience")

    async def get_active(self, timeout_ms: float = 600.0) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        async with self.lock:
            if self.frame_bytes is None:
                return None
            if now_ms() - self.last_ms > timeout_ms:
                return None
            return self.frame_bytes, dict(self.meta)


class DesktopCapture:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and mss is not None
        self.sct = None
        self.last_capture = 0.0
        self.last_image: Optional[Image.Image] = None
        self.error = None

    def capture(self, size=(1024, 576)) -> Image.Image:
        if not self.enabled:
            return self.placeholder(size, "Desktop capture disabled")
        if time.time() - self.last_capture < 0.20 and self.last_image is not None:
            return self.last_image
        try:
            if self.sct is None:
                self.sct = mss.mss()
            mon = self.sct.monitors[1] if len(self.sct.monitors) > 1 else self.sct.monitors[0]
            grab = self.sct.grab(mon)
            img = Image.frombytes("RGB", grab.size, grab.rgb).resize(size, Image.Resampling.BILINEAR)
            self.last_capture = time.time()
            self.last_image = img
            return img
        except Exception as e:
            self.error = str(e)
            return self.placeholder(size, "Desktop capture unavailable\n" + str(e)[:90])

    @staticmethod
    def placeholder(size, text: str) -> Image.Image:
        img = Image.new("RGB", size, (16, 21, 34))
        d = ImageDraw.Draw(img)
        d.rectangle((18, 18, size[0]-18, size[1]-18), outline=(90, 110, 160), width=3)
        d.text((42, 44), "Desktop Panel", fill=(235, 240, 255))
        d.text((42, 96), text, fill=(190, 205, 230))
        return img


class HubRenderer:
    def __init__(self, eye_width: int, eye_height: int, desktop_enabled: bool = True):
        if moderngl is None:
            raise RuntimeError(f"ModernGL import failed: {MODERNGL_IMPORT_ERROR}")
        self.eye_width = eye_width
        self.eye_height = eye_height
        self.desktop = DesktopCapture(desktop_enabled)

        backend_pref = os.environ.get("FOXY_GL_BACKEND", "").strip()
        backends = [backend_pref] if backend_pref else ["egl", "x11", ""]
        errors = []
        self.ctx = None
        for backend in backends:
            try:
                if backend:
                    log(f"Trying ModernGL standalone context backend={backend!r}")
                    self.ctx = moderngl.create_standalone_context(backend=backend)
                else:
                    log("Trying ModernGL default standalone context")
                    self.ctx = moderngl.create_standalone_context()
                log(f"ModernGL context created. Version code: {self.ctx.version_code}")
                break
            except Exception as e:
                errors.append(f"{backend or 'default'}: {e}")
        if self.ctx is None:
            raise RuntimeError("Could not create ModernGL context:\n" + "\n".join(errors))

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

        self.color_tex = self.ctx.texture((eye_width, eye_height), 4)
        self.depth = self.ctx.depth_renderbuffer((eye_width, eye_height))
        self.fbo = self.ctx.framebuffer(color_attachments=[self.color_tex], depth_attachment=self.depth)

        self.color_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec3 in_color;
                uniform mat4 u_mvp;
                uniform float u_brightness;
                out vec3 v_color;
                void main() {
                    v_color = in_color * u_brightness;
                    gl_Position = u_mvp * vec4(in_pos, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_color;
                out vec4 f_color;
                void main() { f_color = vec4(v_color, 1.0); }
            """,
        )
        self.tex_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec2 in_uv;
                uniform mat4 u_mvp;
                out vec2 v_uv;
                void main() {
                    v_uv = in_uv;
                    gl_Position = u_mvp * vec4(in_pos, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D u_tex;
                in vec2 v_uv;
                out vec4 f_color;
                void main() { f_color = texture(u_tex, v_uv); }
            """,
        )

        self.cube_vbo = self.ctx.buffer(make_cube_vertices().tobytes())
        self.cube_vao = self.ctx.vertex_array(self.color_prog, [(self.cube_vbo, "3f 3f", "in_pos", "in_color")])
        self.grid_vbo = self.ctx.buffer(make_grid_vertices().tobytes())
        self.grid_vao = self.ctx.vertex_array(self.color_prog, [(self.grid_vbo, "3f 3f", "in_pos", "in_color")])
        self.line_vbo = self.ctx.buffer(reserve=4096)
        self.line_vao = self.ctx.vertex_array(self.color_prog, [(self.line_vbo, "3f 3f", "in_pos", "in_color")])
        self.quad_vbo = self.ctx.buffer(make_quad_vertices().tobytes())
        self.quad_vao = self.ctx.vertex_array(self.tex_prog, [(self.quad_vbo, "3f 2f", "in_pos", "in_uv")])

        self.hub_tex = self.ctx.texture((1024, 512), 3)
        self.desktop_tex = self.ctx.texture((1024, 576), 3)
        for tex in (self.hub_tex, self.desktop_tex):
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = False
            tex.repeat_y = False

        self.start_time = time.time()
        self.last_frame_time = time.time()
        self.frame_no = 0
        self.last_hub_update = 0.0

        # Hub locomotion state. This moves/turns the hub world relative to the
        # Quest tracking space, so the browser still handles real head pose.
        self.player_x = 0.0
        self.player_z = 0.0
        self.player_yaw = 0.0
        self.move_speed = 1.65
        self.turn_speed = 1.8

    def update_hub_locomotion(self, tracking: Dict[str, Any]) -> None:
        """Use Quest sticks to move around the hub scene.

        Left stick: walk/strafe.
        Right stick X: smooth turn.
        This intentionally keeps vertical movement fixed to avoid nausea.
        """
        now = time.time()
        dt = max(0.0, min(0.05, now - self.last_frame_time))
        self.last_frame_time = now

        inputs = tracking.get("inputs", [])
        left = input_by_hand(inputs, "left")
        right = input_by_hand(inputs, "right")

        lx, ly = input_thumbstick(left)
        rx, _ = input_thumbstick(right)

        # Dead zones.
        if abs(lx) < 0.12:
            lx = 0.0
        if abs(ly) < 0.12:
            ly = 0.0
        if abs(rx) < 0.12:
            rx = 0.0

        # Quest Browser exposes right-stick X with the opposite sign from the
        # hub's positive-yaw convention. Flip only the camera/smooth-turn input;
        # leave the already-correct head-relative movement basis untouched.
        self.player_yaw -= rx * self.turn_speed * dt

        # WebXR/gamepad Y convention is runtime-dependent. In Quest Browser,
        # pushing forward is commonly negative Y. Use -ly for forward. Movement
        # must be head-relative: the virtual smooth-turn yaw alone is not enough
        # when the user physically turns their head.
        forward = -ly
        strafe = lx

        # Use explicit flat head basis vectors rather than converting to yaw and
        # back. This avoids angle wrap/sign mistakes and stays stable when the
        # headset is pitched or rolled. The helper applies the artificial
        # smooth-turn using the inverse sign, because the renderer implements
        # smooth-turn by rotating the world by -player_yaw.
        right_basis, forward_basis, head_yaw = head_basis_from_tracking(tracking, self.player_yaw)
        world_dx = (strafe * right_basis[0] + forward * forward_basis[0]) * self.move_speed * dt
        world_dz = (strafe * right_basis[1] + forward * forward_basis[1]) * self.move_speed * dt

        self.player_x += world_dx
        self.player_z += world_dz

        # Keep the prototype hub bounded so it is easy to recover.
        self.player_x = max(-7.0, min(7.0, self.player_x))
        self.player_z = max(-7.0, min(7.0, self.player_z))

    def world_model(self) -> np.ndarray:
        """Transform hub objects into the player's locomoted local space."""
        return rotate_y(-self.player_yaw) @ translate(-self.player_x, 0.0, -self.player_z)

    def _view_projection_for_eye(self, tracking: Dict[str, Any], eye: str) -> Tuple[np.ndarray, np.ndarray]:
        view_data = tracking.get("views", {}).get(eye)
        if view_data:
            proj = mat_from_webxr(view_data.get("projectionMatrix"))
            view = mat_from_webxr(view_data.get("viewMatrix"))
            if proj is not None and view is not None:
                return view, proj
        aspect = self.eye_width / max(1.0, self.eye_height)
        proj = perspective(math.radians(88), aspect, 0.05, 100.0)
        ipd = 0.064
        x = -ipd / 2.0 if eye == "left" else ipd / 2.0
        view = look_at((x, 1.55, 2.8), (0, 1.25, -2.2), (0, 1, 0))
        return view, proj

    def update_hub_texture(self, tracking: Dict[str, Any]) -> None:
        if time.time() - self.last_hub_update < 0.25:
            return
        self.last_hub_update = time.time()
        img = Image.new("RGB", (1024, 512), (13, 17, 29))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((20, 20, 1004, 492), radius=28, fill=(18, 24, 40), outline=(80, 100, 150), width=4)
        d.text((48, 42), "FOXY HUB", fill=(250, 252, 255))
        d.text((48, 88), "Default scene active. Start an IPC experience to take over rendering.", fill=(205, 220, 250))
        d.text((48, 136), f"Tracking packets: {tracking.get('packets', 0)}", fill=(170, 190, 230))
        d.text((48, 176), f"Controllers: {len(tracking.get('inputs', []))}", fill=(170, 190, 230))
        head_yaw = math.degrees(head_yaw_from_tracking(tracking))
        d.text((48, 216), f"Hub locomotion: x={self.player_x:+.2f} z={self.player_z:+.2f} turn={math.degrees(self.player_yaw):+.0f}° head={head_yaw:+.0f}°", fill=(170, 190, 230))
        d.text((48, 252), "Left stick: head-relative walk/strafe  |  Right stick: turn", fill=(255, 210, 120))
        d.text((48, 300), "IPC socket:", fill=(255, 210, 120))
        d.text((48, 336), DEFAULT_IPC_PATH, fill=(235, 240, 255))
        d.text((48, 386), "Run example:", fill=(255, 210, 120))
        d.text((48, 424), "python examples/opengl_experience.py", fill=(235, 240, 255))
        d.text((48, 462), "Desktop panel is shown on the right if capture is available.", fill=(170, 190, 230))
        self.hub_tex.write(img.tobytes())

        desktop = self.desktop.capture((1024, 576))
        self.desktop_tex.write(desktop.convert("RGB").tobytes())

    def draw_textured_panel(self, tex, vp: np.ndarray, model: np.ndarray) -> None:
        self.tex_prog["u_mvp"].write(mat_to_gl_bytes(vp @ model))
        tex.use(0)
        self.tex_prog["u_tex"].value = 0
        self.quad_vao.render(mode=moderngl.TRIANGLE_STRIP)

    def _draw_eye(self, eye: str, tracking: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        self.update_hub_texture(tracking)
        world = self.world_model()
        t = time.time() - self.start_time
        view, proj = self._view_projection_for_eye(tracking, eye)
        vp = proj @ view

        self.fbo.use()
        self.ctx.viewport = (0, 0, self.eye_width, self.eye_height)
        self.ctx.clear(0.020, 0.026, 0.045, 1.0, depth=1.0)

        grid_model = world @ translate(0.0, -0.02, -3.0)
        self.color_prog["u_mvp"].write(mat_to_gl_bytes(vp @ grid_model))
        self.color_prog["u_brightness"].value = 1.0
        self.grid_vao.render(mode=moderngl.LINES)

        # Hub panel and desktop panel.
        self.draw_textured_panel(self.hub_tex, vp, world @ translate(-1.25, 1.55, -2.8) @ rotate_y(0.22) @ scale(0.92, 0.46, 1))
        self.draw_textured_panel(self.desktop_tex, vp, world @ translate(1.25, 1.42, -2.95) @ rotate_y(-0.23) @ scale(0.86, 0.49, 1))

        # Logo-ish rotating core cube.
        model = world @ translate(0.0, 0.95, -2.4) @ rotate_y(t * 0.55) @ rotate_x(t * 0.25) @ scale(0.22, 0.22, 0.22)
        self.color_prog["u_mvp"].write(mat_to_gl_bytes(vp @ model))
        self.color_prog["u_brightness"].value = 1.25
        self.cube_vao.render()

        # Controller markers and real target-ray directions.
        line_vertices = []
        for idx, inp in enumerate(tracking.get("inputs", [])):
            grip = inp.get("grip") or inp.get("ray")
            if not grip:
                continue
            pos = grip.get("position") or {}
            x = float(pos.get("x", 0.0))
            y = float(pos.get("y", 1.2))
            z = float(pos.get("z", -1.0))
            trigger = input_button(inp, 0)
            grip_btn = input_button(inp, 1)
            scale_boost = 1.0 + 0.55 * trigger["value"] + 0.35 * grip_btn["value"]
            model = translate(x, y, z) @ rotate_y(t * 1.5 + idx) @ scale(0.075 * scale_boost, 0.075 * scale_boost, 0.075 * scale_boost)
            self.color_prog["u_mvp"].write(mat_to_gl_bytes(vp @ model))
            self.color_prog["u_brightness"].value = 1.15 + trigger["value"] * 0.7
            self.cube_vao.render()

            handed = inp.get("handedness", "")
            color = (0.25, 0.85, 1.0) if handed == "right" else (1.0, 0.55, 0.25)
            ray_points = pose_forward_ray(inp.get("ray"), length=0.75 + trigger["value"] * 0.55)
            if ray_points:
                (rx, ry, rz), (ex, ey, ez) = ray_points
                line_vertices.extend([(rx, ry, rz, *color), (ex, ey, ez, *color)])

        if line_vertices:
            arr = np.array(line_vertices, dtype=np.float32)
            if arr.nbytes <= 4096:
                self.line_vbo.write(arr.tobytes())
                self.color_prog["u_mvp"].write(mat_to_gl_bytes(vp))
                self.color_prog["u_brightness"].value = 1.0
                self.line_vao.render(mode=moderngl.LINES, vertices=len(line_vertices))

        data = self.fbo.read(components=4, alignment=1)
        img = Image.frombytes("RGBA", (self.eye_width, self.eye_height), data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return img, {
            "view": mat_to_webgl_list(view),
            "projection": mat_to_webgl_list(proj),
            "viewProjection": mat_to_webgl_list(vp),
        }

    def render_stereo_jpeg(self, tracking: Dict[str, Any], quality: int = 72) -> Tuple[bytes, Dict[str, Any]]:
        self.update_hub_locomotion(tracking)
        left, left_render_view = self._draw_eye("left", tracking)
        right, right_render_view = self._draw_eye("right", tracking)
        combined = Image.new("RGB", (self.eye_width * 2, self.eye_height))
        combined.paste(left.convert("RGB"), (0, 0))
        combined.paste(right.convert("RGB"), (self.eye_width, 0))

        buf = io.BytesIO()
        combined.save(buf, format="JPEG", quality=quality, optimize=False)
        self.frame_no += 1
        age = max(0.0, now_ms() - float(tracking.get("recv_time_ms", 0.0))) if tracking.get("recv_time_ms") else 0.0
        return buf.getvalue(), {
            "frame": self.frame_no,
            "serverTimeMs": now_ms(),
            "eyeWidth": self.eye_width,
            "eyeHeight": self.eye_height,
            "encoding": "jpeg-sbs",
            "trackingPackets": tracking.get("packets", 0),
            "poseAgeMs": age,
            "renderViews": {"left": left_render_view, "right": right_render_view},
            "reprojection": "client-rotational-timewarp-v1",
            "mode": "hub",
            "appName": "Foxy Hub",
        }


class AudioToneGenerator:
    def __init__(self, sample_rate: int = 48000, channels: int = 2, chunk_ms: int = 40, demo_enabled: bool = False):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_ms = chunk_ms
        self.phase = 0.0
        # Browser audio must be unlocked by a user gesture. The built-in demo
        # tone is separate so IPC audio can be tested without a competing tone.
        self.playback_enabled = False
        self.demo_enabled = demo_enabled

    def generate_chunk(self) -> bytes:
        n = int(self.sample_rate * self.chunk_ms / 1000)
        freq = 220.0
        amp = 0.08
        out = array.array("h")
        for _ in range(n):
            sample = int(math.sin(self.phase) * 32767 * amp)
            self.phase += (2.0 * math.pi * freq) / self.sample_rate
            if self.phase > math.tau:
                self.phase -= math.tau
            out.append(sample)
            out.append(sample)
        return out.tobytes()


class MicChunkBuffer:
    def __init__(self, max_chunks: int = 256):
        self.max_chunks = max_chunks
        self.chunks: Deque[Tuple[Dict[str, Any], bytes]] = deque(maxlen=max_chunks)
        self.lock = asyncio.Lock()
        self.event = asyncio.Event()

    async def push(self, meta: Dict[str, Any], payload: bytes) -> None:
        item_meta = dict(meta)
        item_meta["bytes"] = len(payload)
        item_meta["serverTimeMs"] = now_ms()
        async with self.lock:
            self.chunks.append((item_meta, payload))
            self.event.set()

    async def pop(self, timeout_ms: float = 0.0) -> Optional[Tuple[Dict[str, Any], bytes]]:
        deadline = time.monotonic() + max(0.0, timeout_ms) / 1000.0
        while True:
            async with self.lock:
                if self.chunks:
                    item = self.chunks.popleft()
                    if not self.chunks:
                        self.event.clear()
                    return item
                self.event.clear()

            remaining = deadline - time.monotonic()
            if timeout_ms <= 0 or remaining <= 0:
                return None
            try:
                await asyncio.wait_for(self.event.wait(), remaining)
            except asyncio.TimeoutError:
                return None


async def ipc_read_packet(reader: asyncio.StreamReader) -> Tuple[Dict[str, Any], bytes]:
    raw_len = await reader.readexactly(4)
    n = struct.unpack("!I", raw_len)[0]
    if n <= 0 or n > 16 * 1024 * 1024:
        raise RuntimeError(f"bad IPC header length {n}")
    header = json.loads((await reader.readexactly(n)).decode("utf-8"))
    payload_len = int(header.get("payload_len", 0) or 0)
    payload = await reader.readexactly(payload_len) if payload_len else b""
    return header, payload


async def ipc_write_packet(writer: asyncio.StreamWriter, header: Dict[str, Any], payload: bytes = b"") -> None:
    h = dict(header)
    h["payload_len"] = len(payload)
    raw = json.dumps(h, separators=(",", ":")).encode("utf-8")
    writer.write(struct.pack("!I", len(raw)) + raw + payload)
    await writer.drain()


def encode_raw_sbs_frame(header: Dict[str, Any], payload: bytes, default_quality: int) -> Tuple[bytes, Dict[str, Any]]:
    eye_w = int(header.get("eyeWidth") or 0)
    eye_h = int(header.get("eyeHeight") or 0)
    width = int(header.get("width") or (eye_w * 2))
    height = int(header.get("height") or eye_h)
    if eye_w <= 0 or eye_h <= 0:
        raise RuntimeError("raw_frame requires positive eyeWidth and eyeHeight")
    if width != eye_w * 2 or height != eye_h:
        raise RuntimeError("raw_frame width/height must be side-by-side: width=eyeWidth*2 and height=eyeHeight")
    if width <= 0 or height <= 0:
        raise RuntimeError("raw_frame requires positive width and height")

    fmt = str(header.get("pixelFormat") or header.get("format") or "rgb").lower().replace("-", "").replace("_", "")
    raw_modes = {
        "rgb": ("RGB", "RGB", 3),
        "rgba": ("RGBA", "RGBA", 4),
        "bgr": ("RGB", "BGR", 3),
        "bgra": ("RGBA", "BGRA", 4),
        "gray": ("L", "L", 1),
        "grey": ("L", "L", 1),
        "l": ("L", "L", 1),
    }
    if fmt not in raw_modes:
        raise RuntimeError(f"unsupported raw_frame pixelFormat {fmt!r}; use rgb, rgba, bgr, bgra, or gray")
    mode, raw_mode, channels = raw_modes[fmt]
    expected = width * height * channels
    if len(payload) != expected:
        raise RuntimeError(f"raw_frame payload has {len(payload)} bytes, expected {expected} for {width}x{height} {fmt}")

    if raw_mode == mode:
        img = Image.frombytes(mode, (width, height), payload)
    else:
        img = Image.frombytes(mode, (width, height), payload, "raw", raw_mode)
    if img.mode != "RGB":
        img = img.convert("RGB")

    quality = int(header.get("jpegQuality") or default_quality)
    quality = max(30, min(95, quality))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=False)

    meta = dict(header)
    meta["type"] = "frame"
    meta["encoding"] = "jpeg-sbs"
    meta["sourceEncoding"] = "raw-sbs"
    meta["pixelFormat"] = fmt
    meta["width"] = width
    meta["height"] = height
    meta["eyeWidth"] = eye_w
    meta["eyeHeight"] = eye_h
    meta["serverTimeMs"] = now_ms()
    return buf.getvalue(), meta


class FoxyServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.tracking = TrackingState()
        self.external_frame = ExternalFrameState()
        self.clients: set[web.WebSocketResponse] = set()
        self.client_health: Dict[web.WebSocketResponse, Dict[str, Any]] = {}
        self.client_locks: Dict[web.WebSocketResponse, asyncio.Lock] = {}
        self.rtc_peers: Dict[web.WebSocketResponse, Any] = {}
        self.rtc_video_channels: Dict[web.WebSocketResponse, Any] = {}
        self.ws_video_fallback: set[web.WebSocketResponse] = set()
        self.stream_seq = 0
        self.stream_epoch = 0
        self.renderer: Optional[HubRenderer] = None
        self.audio = AudioToneGenerator(demo_enabled=bool(getattr(args, "demo_audio", False)))
        self.frame_task: Optional[asyncio.Task] = None
        self.audio_task: Optional[asyncio.Task] = None
        self.ipc_server: Optional[asyncio.AbstractServer] = None
        self.hotspot: Optional[FoxyHotspotProcess] = None
        self.pending_binary: Optional[Dict[str, Any]] = None
        self.mic_chunks = MicChunkBuffer()
        self.stats = {
            "frames_sent": 0,
            "frames_dropped_server": 0,
            "frames_skipped_not_ready": 0,
            "client_decode_errors": 0,
            "client_stale_drops": 0,
            "stream_resyncs": 0,
            "clients": 0,
            "last_render_ms": 0.0,
            "last_frame_bytes": 0,
            "audio_chunks_sent": 0,
            "ipc_audio_chunks_sent": 0,
            "ipc_audio_bytes_sent": 0,
            "mic_chunks_received": 0,
            "mic_bytes_received": 0,
            "mic_chunks_served_ipc": 0,
            "last_controller_summary": "",
            "ipc_clients": 0,
            "video_transport": self.video_transport_name(),
        }

    def webrtc_video_enabled(self) -> bool:
        return (
            bool(getattr(self.args, "hotspot", False))
            and bool(getattr(self.args, "hotspot_udp_video", False))
            and RTCPeerConnection is not None
        )

    def video_transport_name(self) -> str:
        return "webrtc-datachannel" if self.webrtc_video_enabled() else "websocket"

    async def init_renderer(self) -> None:
        self.renderer = await asyncio.to_thread(HubRenderer, self.args.eye_width, self.args.eye_height, not self.args.no_desktop)

    async def index(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(WEB_ROOT / "index.html")

    async def static(self, request: web.Request) -> web.StreamResponse:
        rel = request.match_info.get("path", "")
        target = (WEB_ROOT / rel).resolve()
        if not str(target).startswith(str(WEB_ROOT.resolve())) or not target.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(target)

    async def status(self, request: web.Request) -> web.Response:
        tracking = await self.tracking.snapshot()
        return web.json_response({
            "ok": True,
            "stats": self.stats,
            "trackingPackets": tracking.get("packets", 0),
            "clients": len(self.clients),
            "clientHealth": [
                {k: v for k, v in health.items() if k not in ("ws",)}
                for health in self.client_health.values()
            ],
            "eyeWidth": self.args.eye_width,
            "eyeHeight": self.args.eye_height,
            "fps": self.args.fps,
            "ipcPath": self.args.ipc_path,
            "videoTransport": self.video_transport_name(),
            "webrtcAvailable": RTCPeerConnection is not None,
            "hotspot": {
                "enabled": bool(getattr(self.args, "hotspot", False)),
                "ssid": getattr(self.args, "hotspot_ssid", "foxy"),
                "domain": getattr(self.args, "hotspot_domain", DEFAULT_HOTSPOT_DOMAIN),
                "address": getattr(self.args, "hotspot_address", DEFAULT_HOTSPOT_ADDRESS),
            },
        })

    def summarize_inputs(self, inputs: List[Dict[str, Any]]) -> str:
        parts = []
        for inp in inputs:
            hand = inp.get("handedness") or "unknown"
            b0 = input_button(inp, 0)
            b1 = input_button(inp, 1)
            ax0 = input_axis(inp, 0)
            ax1 = input_axis(inp, 1)
            parts.append(f"{hand}: trigger={b0['value']:.2f} grip={b1['value']:.2f} stick=({ax0:.2f},{ax1:.2f})")
        return " | ".join(parts)

    async def websocket(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024, heartbeat=10)
        await ws.prepare(request)
        self.clients.add(ws)
        self.client_locks[ws] = asyncio.Lock()
        self.client_health[ws] = {
            "peer": request.remote or "local",
            "last_sent_seq": 0,
            "last_sent_ms": 0.0,
            "last_ack_seq": 0,
            "last_ack_ms": 0.0,
            "decode_ms": 0.0,
            "age_ms": 0.0,
            "consecutive_errors": 0,
            "errors": 0,
            "stale_drops": 0,
            "server_drops": 0,
            "server_blocked_ticks": 0,
            "state": "connected",
        }
        self.stats["clients"] = len(self.clients)
        peer = request.remote
        log(f"Quest/client connected: {peer}")

        await ws.send_json({
            "type": "hello",
            "serverTimeMs": now_ms(),
            "eyeWidth": self.args.eye_width,
            "eyeHeight": self.args.eye_height,
            "fps": self.args.fps,
            "message": "Connected to Foxy SDK server",
            "videoTransport": self.video_transport_name(),
            "webrtcAvailable": RTCPeerConnection is not None,
        })

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except Exception:
                        continue
                    typ = payload.get("type")
                    if typ == "tracking":
                        await self.tracking.update(payload)
                        self.stats["last_controller_summary"] = self.summarize_inputs(payload.get("inputs", []))
                    elif typ == "frame-ack":
                        health = self.client_health.get(ws)
                        if health is not None:
                            seq = int(payload.get("streamSeq") or payload.get("frame") or 0)
                            health["last_ack_seq"] = max(int(health.get("last_ack_seq", 0)), seq)
                            health["last_ack_ms"] = now_ms()
                            health["decode_ms"] = float(payload.get("decodeMs") or 0.0)
                            health["age_ms"] = float(payload.get("ageMs") or 0.0)
                            reason = str(payload.get("reason") or "")
                            normal_drop_reasons = {
                                "stale",
                                "local-queue-stale",
                                "superseded",
                                "metadata-superseded",
                                "superseded-by-resync",
                                "superseded-during-decode",
                                "not-newer",
                                "client-backpressure",
                            }
                            if payload.get("ok"):
                                health["consecutive_errors"] = 0
                                health["state"] = "good"
                            elif reason in normal_drop_reasons:
                                # Normal newest-frame streaming drops are not decode errors.
                                # Counting these as errors caused unnecessary resyncs and
                                # made the stream look worse under load.
                                health["stale_drops"] = int(health.get("stale_drops", 0)) + 1
                                health["consecutive_errors"] = 0
                                self.stats["client_stale_drops"] += 1
                                health["state"] = f"client-drop:{reason or 'drop'}"
                            else:
                                health["errors"] = int(health.get("errors", 0)) + 1
                                health["consecutive_errors"] = int(health.get("consecutive_errors", 0)) + 1
                                self.stats["client_decode_errors"] += 1
                                health["state"] = f"client-error:{reason or 'decode'}"
                                if int(health.get("consecutive_errors", 0)) >= self.args.resync_after_errors:
                                    await self.force_resync(ws, reason or "client-decode-error")
                    elif typ == "stream-error":
                        health = self.client_health.get(ws)
                        if health is not None:
                            health["errors"] = int(health.get("errors", 0)) + 1
                            health["consecutive_errors"] = 0
                            health["state"] = "client-requested-resync-keepalive"
                            self.stats["client_decode_errors"] += 1
                        # This is a stream resync, not a connection reset. Keep the
                        # WebSocket open and let the next JPEG become ground truth.
                        await self.force_resync(ws, str(payload.get("reason") or "client-request"))
                    elif typ == "input-event":
                        log(f"Input event: {payload.get('event')} hand={payload.get('handedness')} profile={payload.get('profile')}")
                        if payload.get("event") in ("selectstart", "squeezestart"):
                            await ws.send_json({"type": "haptic", "handedness": payload.get("handedness"), "durationMs": 35, "intensity": 0.35})
                    elif typ == "mic-meta":
                        self.pending_binary = payload
                    elif typ == "audio-control":
                        self.audio.playback_enabled = bool(payload.get("enabled", True))
                        await ws.send_json({"type": "audio-state", "enabled": self.audio.playback_enabled})
                    elif typ == "reset":
                        await self.tracking.reset()
                        if self.renderer:
                            self.renderer.start_time = time.time()
                        await ws.send_json({"type": "reset-ok", "serverTimeMs": now_ms()})
                    elif typ == "ping":
                        await ws.send_json({"type": "pong", "serverTimeMs": now_ms(), "clientTimeMs": payload.get("clientTimeMs")})
                    elif typ == "webrtc-offer":
                        await self.handle_webrtc_offer(ws, payload)
                    elif typ == "webrtc-client-unavailable":
                        self.ws_video_fallback.add(ws)
                        log(f"Client cannot use WebRTC video, falling back to WebSocket: {peer}")
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    if self.pending_binary and self.pending_binary.get("type") == "mic-meta":
                        mic_payload = bytes(msg.data)
                        await self.mic_chunks.push(self.pending_binary, mic_payload)
                        self.stats["mic_chunks_received"] += 1
                        self.stats["mic_bytes_received"] += len(mic_payload)
                        self.pending_binary = None
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    log(f"WebSocket error: {ws.exception()}")
        finally:
            self.clients.discard(ws)
            self.client_health.pop(ws, None)
            self.client_locks.pop(ws, None)
            self.ws_video_fallback.discard(ws)
            await self.close_webrtc(ws)
            self.stats["clients"] = len(self.clients)
            log(f"Quest/client disconnected: {peer}")
        return ws

    async def handle_webrtc_offer(self, ws: web.WebSocketResponse, payload: Dict[str, Any]) -> None:
        if not self.webrtc_video_enabled() or RTCPeerConnection is None or RTCSessionDescription is None:
            await self.send_json_locked(ws, {
                "type": "webrtc-unavailable",
                "reason": "hotspot mode requires aiortc for UDP video transport",
                "videoTransport": "websocket",
            })
            return

        await self.close_webrtc(ws)
        self.ws_video_fallback.discard(ws)
        pc = RTCPeerConnection()
        self.rtc_peers[ws] = pc

        @pc.on("datachannel")
        def on_datachannel(channel):
            log(f"WebRTC data channel opened by client: {channel.label}")
            if channel.label != "foxy-video":
                return
            self.rtc_video_channels[ws] = channel

            @channel.on("close")
            def on_close():
                if self.rtc_video_channels.get(ws) is channel:
                    self.rtc_video_channels.pop(ws, None)
                log("WebRTC video data channel closed")

        desc = RTCSessionDescription(sdp=str(payload.get("sdp") or ""), type=str(payload.get("sdpType") or "offer"))
        await pc.setRemoteDescription(desc)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await self.send_json_locked(ws, {
            "type": "webrtc-answer",
            "sdpType": pc.localDescription.type,
            "sdp": pc.localDescription.sdp,
            "videoTransport": "webrtc-datachannel",
        })

    async def close_webrtc(self, ws: web.WebSocketResponse) -> None:
        self.rtc_video_channels.pop(ws, None)
        pc = self.rtc_peers.pop(ws, None)
        if pc is not None:
            try:
                await pc.close()
            except Exception:
                pass

    async def handle_ipc(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername") or "ipc-client"
        self.stats["ipc_clients"] += 1
        log(f"IPC experience connected: {peer}")
        client_name = "ipc-experience"
        try:
            while True:
                header, payload = await ipc_read_packet(reader)
                typ = header.get("type")
                if typ == "hello":
                    client_name = str(header.get("client") or client_name)
                    await ipc_write_packet(writer, {"type": "hello-ok", "server": "foxy-sdk", "ipcVersion": 1, "time": time.time()})
                elif typ == "get_state":
                    state = await self.tracking.snapshot()
                    await ipc_write_packet(writer, {
                        "type": "state",
                        "time": time.time(),
                        "serverTimeMs": now_ms(),
                        "tracking": state,
                        "views": state.get("views", {}),
                        "inputs": state.get("inputs", []),
                        "events": state.get("events", []),
                        "stats": self.stats,
                        "recommended": {"eyeWidth": self.args.eye_width, "eyeHeight": self.args.eye_height, "fps": self.args.fps},
                    })
                elif typ == "frame":
                    header["client"] = client_name
                    await self.external_frame.set_frame(header, payload)
                elif typ == "raw_frame":
                    header["client"] = client_name
                    try:
                        jpeg, encoded_header = await asyncio.to_thread(encode_raw_sbs_frame, header, payload, self.args.jpeg_quality)
                    except Exception as e:
                        await ipc_write_packet(writer, {"type": "error", "message": f"raw_frame encode failed: {e}"})
                    else:
                        await self.external_frame.set_frame(encoded_header, jpeg)
                elif typ == "audio":
                    await self.broadcast_ipc_audio(header, payload, client_name)
                elif typ == "get_mic_chunk":
                    timeout_ms = float(header.get("timeoutMs", 0.0) or 0.0)
                    item = await self.mic_chunks.pop(timeout_ms)
                    if item is None:
                        await ipc_write_packet(writer, {"type": "mic-timeout", "time": time.time()})
                    else:
                        mic_meta, mic_payload = item
                        mic_meta = dict(mic_meta)
                        mic_meta["type"] = "mic-chunk"
                        self.stats["mic_chunks_served_ipc"] += 1
                        await ipc_write_packet(writer, mic_meta, mic_payload)
                elif typ == "ping":
                    await ipc_write_packet(writer, {"type": "pong", "time": time.time()})
                else:
                    await ipc_write_packet(writer, {"type": "error", "message": f"unknown packet type {typ!r}"})
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception as e:
            log(f"IPC error: {type(e).__name__}: {e}")
        finally:
            self.stats["ipc_clients"] = max(0, self.stats["ipc_clients"] - 1)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            log(f"IPC experience disconnected: {peer}")

    async def broadcast_ipc_audio(self, header: Dict[str, Any], payload: bytes, client_name: str) -> None:
        if not payload:
            return
        sample_rate = int(header.get("sampleRate") or 48000)
        channels = int(header.get("channels") or 2)
        if sample_rate <= 0 or channels <= 0:
            raise RuntimeError("audio packet requires positive sampleRate and channels")
        if len(payload) % (2 * channels) != 0:
            raise RuntimeError("s16le audio payload length must align with channels")

        samples_per_channel = int(header.get("samplesPerChannel") or (len(payload) // (2 * channels)))
        meta = {
            "type": "audio-meta",
            "sampleRate": sample_rate,
            "channels": channels,
            "format": str(header.get("format") or "s16le"),
            "serverTimeMs": now_ms(),
            "samplesPerChannel": samples_per_channel,
            "appName": str(header.get("appName") or client_name),
            "source": "ipc",
        }
        dead = []
        sent = 0
        for ws in list(self.clients):
            try:
                await self.send_binary_pair_locked(ws, meta, payload)
                sent += 1
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)
            self.client_locks.pop(ws, None)
            self.client_health.pop(ws, None)
        self.stats["ipc_audio_chunks_sent"] += sent
        self.stats["ipc_audio_bytes_sent"] += len(payload) * sent

    async def start_ipc(self) -> None:
        path = pathlib.Path(self.args.ipc_path)
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass
        self.ipc_server = await asyncio.start_unix_server(self.handle_ipc, path=str(path))
        os.chmod(path, 0o600)
        log(f"IPC socket listening: {path}")

    async def send_json_locked(self, ws: web.WebSocketResponse, payload: Dict[str, Any]) -> None:
        lock = self.client_locks.get(ws)
        if lock is None:
            await ws.send_json(payload)
            return
        async with lock:
            await ws.send_json(payload)

    async def send_binary_pair_locked(self, ws: web.WebSocketResponse, meta: Dict[str, Any], payload: bytes) -> None:
        """Send JSON metadata and its matching binary payload atomically per client.

        aiohttp preserves message order for a single writer, but this app has
        independent video/audio/control tasks. Without this lock, an audio chunk
        can slip between frame-meta and frame-bytes, making the browser interpret
        the next binary message as the wrong payload type.
        """
        lock = self.client_locks.get(ws)
        if lock is None:
            await ws.send_json(meta)
            await ws.send_bytes(payload)
            return
        async with lock:
            await ws.send_json(meta)
            await ws.send_bytes(payload)

    def build_rtc_video_packets(self, meta: Dict[str, Any], payload: bytes) -> List[bytes]:
        meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        if len(meta_bytes) > 65535:
            raise RuntimeError("WebRTC video metadata is unexpectedly large")
        chunk_size = max(1200, int(self.args.webrtc_chunk_bytes))
        chunks = [payload[i:i + chunk_size] for i in range(0, len(payload), chunk_size)] or [b""]
        if len(chunks) > 65535:
            raise RuntimeError("WebRTC video frame needs too many chunks")
        packets = []
        seq = int(meta.get("streamSeq") or 0)
        count = len(chunks)
        for index, chunk in enumerate(chunks):
            header_meta = meta_bytes if index == 0 else b""
            packets.append(
                RTC_VIDEO_MAGIC
                + struct.pack("!IHHH", seq, index, count, len(header_meta))
                + header_meta
                + chunk
            )
        return packets

    async def send_video_frame(self, ws: web.WebSocketResponse, meta: Dict[str, Any], payload: bytes) -> bool:
        channel = self.rtc_video_channels.get(ws)
        if ws not in self.ws_video_fallback and channel is not None and getattr(channel, "readyState", "") == "open":
            buffered = int(getattr(channel, "bufferedAmount", 0) or 0)
            if buffered > self.args.webrtc_buffered_drop_bytes:
                health = self.client_health.get(ws)
                if health is not None:
                    health["server_drops"] = int(health.get("server_drops", 0)) + 1
                    health["state"] = "webrtc-buffered-drop"
                self.stats["frames_dropped_server"] += 1
                return False
            meta = dict(meta)
            meta["transport"] = "webrtc-datachannel"
            overflowed = False
            for packet in self.build_rtc_video_packets(meta, payload):
                channel.send(packet)
                if int(getattr(channel, "bufferedAmount", 0) or 0) > self.args.webrtc_buffered_drop_bytes:
                    overflowed = True
                    break
            if overflowed:
                health = self.client_health.get(ws)
                if health is not None:
                    health["server_drops"] = int(health.get("server_drops", 0)) + 1
                    health["state"] = "webrtc-mid-frame-drop"
                self.stats["frames_dropped_server"] += 1
                return False
            return True

        if self.webrtc_video_enabled() and ws not in self.ws_video_fallback:
            health = self.client_health.get(ws)
            if health is not None:
                health["state"] = "waiting-for-webrtc-video"
            return False

        meta = dict(meta)
        meta["transport"] = "websocket"
        await self.send_binary_pair_locked(ws, meta, payload)
        return True

    async def force_resync(self, ws: web.WebSocketResponse, reason: str) -> None:
        """Tell one client to throw away all pending stream state and wait for a new full frame.

        JPEG is intra-frame, so the next valid frame is the ground truth.  The
        important part is to kill client/browser queues instead of letting stale
        decoded images keep feeding the XR texture.
        """
        self.stream_epoch += 1
        self.stats["stream_resyncs"] += 1
        health = self.client_health.get(ws)
        if health is not None:
            health["last_sent_seq"] = 0
            health["last_ack_seq"] = 0
            health["last_sent_ms"] = 0.0
            health["consecutive_errors"] = 0
            health["state"] = f"resync:{reason}"
        try:
            await self.send_json_locked(ws, {
                "type": "stream-resync",
                "reason": reason,
                "streamEpoch": self.stream_epoch,
                "serverTimeMs": now_ms(),
            })
        except Exception:
            self.clients.discard(ws)
            self.client_health.pop(ws, None)
            self.client_locks.pop(ws, None)

    def should_send_to_client(self, ws: web.WebSocketResponse, *, count_block: bool = False) -> bool:
        """Continuous latest-frame mode.

        Do not wait for a displayed-frame ACK before sending the next JPEG. Waiting
        for XR texture upload turned normal decode variance into server starvation
        and made the headset look like a slideshow. The browser keeps only a tiny
        newest-frame queue and drops superseded frames locally.
        """
        return ws in self.clients

    async def audio_loop(self) -> None:
        period = self.audio.chunk_ms / 1000.0
        log(f"Audio loop running: {self.audio.sample_rate} Hz, {self.audio.channels} ch, {self.audio.chunk_ms} ms chunks")
        while True:
            started = time.perf_counter()
            if self.clients and self.audio.playback_enabled and self.audio.demo_enabled:
                try:
                    pcm = self.audio.generate_chunk()
                    meta = {"type": "audio-meta", "sampleRate": self.audio.sample_rate, "channels": self.audio.channels, "format": "s16le", "serverTimeMs": now_ms(), "samplesPerChannel": int(self.audio.sample_rate * self.audio.chunk_ms / 1000)}
                    dead = []
                    for ws in list(self.clients):
                        try:
                            await self.send_binary_pair_locked(ws, meta, pcm)
                            self.stats["audio_chunks_sent"] += 1
                        except Exception:
                            dead.append(ws)
                    for ws in dead:
                        self.clients.discard(ws)
                        self.client_locks.pop(ws, None)
                        self.client_health.pop(ws, None)
                except Exception as e:
                    log(f"Audio/send error: {type(e).__name__}: {e}")
            elapsed = time.perf_counter() - started
            await asyncio.sleep(max(0.0, period - elapsed))

    async def frame_loop(self) -> None:
        if not self.renderer:
            raise RuntimeError("Renderer not initialized")
        period = 1.0 / max(1.0, float(self.args.fps))
        log(f"Frame loop running at target {self.args.fps} FPS")
        while True:
            started = time.perf_counter()
            if self.clients:
                try:
                    # Continuous latest-frame send. Render at the target cadence and
                    # let the Quest keep only the newest pending frame. This avoids
                    # ACK-starvation while still allowing the browser to drop stale
                    # work before it becomes visible latency.
                    ready_clients = list(self.clients)
                    if not ready_clients:
                        self.stats["clients"] = 0
                        self.stats["last_render_ms"] = 0.0
                        await asyncio.sleep(period)
                        continue

                    external = await self.external_frame.get_active(self.args.ipc_timeout_ms)
                    if external:
                        jpeg, meta = external
                        meta = dict(meta)
                        meta["type"] = "frame-meta"
                        meta["mode"] = "ipc"
                        meta["appName"] = meta.get("appName") or self.external_frame.client_name
                        meta["serverTimeMs"] = now_ms()
                        meta["eyeWidth"] = int(meta.get("eyeWidth") or self.args.eye_width)
                        meta["eyeHeight"] = int(meta.get("eyeHeight") or self.args.eye_height)
                    else:
                        tracking = await self.tracking.snapshot()
                        jpeg, meta = await asyncio.to_thread(self.renderer.render_stereo_jpeg, tracking, self.args.jpeg_quality)
                        meta["type"] = "frame-meta"

                    self.stream_seq += 1
                    meta["streamSeq"] = self.stream_seq
                    meta["streamEpoch"] = self.stream_epoch
                    meta["payloadBytes"] = len(jpeg)
                    meta["maxClientFrameAgeMs"] = self.args.max_client_frame_age_ms
                    meta["serverQueuePolicy"] = "continuous-latest-frame"

                    dead = []
                    sent = 0
                    for ws in ready_clients:
                        if ws not in self.clients:
                            continue
                        try:
                            per_client_meta = dict(meta)
                            health = self.client_health.get(ws, {})
                            per_client_meta["serverDroppedFrames"] = int(health.get("server_drops", 0) or 0)
                            per_client_meta["serverSkippedNotReady"] = int(health.get("server_blocked_ticks", 0) or 0)
                            frame_sent = await self.send_video_frame(ws, per_client_meta, jpeg)
                            if frame_sent:
                                sent += 1
                            if frame_sent and ws in self.client_health:
                                self.client_health[ws]["last_sent_seq"] = self.stream_seq
                                self.client_health[ws]["last_sent_ms"] = now_ms()
                                self.client_health[ws]["state"] = "sent"
                        except Exception:
                            dead.append(ws)
                    for ws in dead:
                        self.clients.discard(ws)
                        self.client_health.pop(ws, None)
                        self.client_locks.pop(ws, None)

                    self.stats["frames_sent"] += sent
                    self.stats["clients"] = len(self.clients)
                    self.stats["last_render_ms"] = (time.perf_counter() - started) * 1000.0
                    self.stats["last_frame_bytes"] = len(jpeg)
                except Exception as e:
                    log(f"Render/send error: {type(e).__name__}: {e}")
                    await asyncio.sleep(0.5)
            elapsed = time.perf_counter() - started
            await asyncio.sleep(max(0.0, period - elapsed))

    async def on_startup(self, app: web.Application) -> None:
        if getattr(self.args, "hotspot", False):
            self.hotspot = FoxyHotspotProcess(self.args)
            await self.hotspot.start()
        log("Initializing OpenGL hub renderer...")
        await self.init_renderer()
        await self.start_ipc()
        self.frame_task = asyncio.create_task(self.frame_loop())
        self.audio_task = asyncio.create_task(self.audio_loop())

    async def on_cleanup(self, app: web.Application) -> None:
        if self.frame_task:
            self.frame_task.cancel()
        if self.audio_task:
            self.audio_task.cancel()
        if self.ipc_server:
            self.ipc_server.close()
            await self.ipc_server.wait_closed()
        try:
            pathlib.Path(self.args.ipc_path).unlink()
        except Exception:
            pass
        if self.hotspot:
            await self.hotspot.stop()

    @web.middleware
    async def hotspot_domain_middleware(self, request: web.Request, handler):
        allowed_hosts = {
            str(getattr(self.args, "hotspot_domain", DEFAULT_HOTSPOT_DOMAIN)).lower().rstrip("."),
            str(getattr(self.args, "hotspot_address", DEFAULT_HOTSPOT_ADDRESS)),
            "127.0.0.1",
            "localhost",
        }
        host = request.host.rsplit(":", 1)[0].lower().rstrip(".")
        if host and host not in allowed_hosts:
            raise web.HTTPNotFound(text=f"Foxy hotspot only serves {self.args.hotspot_domain}\n")
        return await handler(request)

    def make_app(self) -> web.Application:
        middlewares = []
        if getattr(self.args, "hotspot", False):
            middlewares.append(self.hotspot_domain_middleware)
        app = web.Application(middlewares=middlewares)
        app.router.add_get("/", self.index)
        app.router.add_get("/ws", self.websocket)
        app.router.add_get("/status", self.status)
        app.router.add_get("/{path:.*}", self.static)
        app.on_startup.append(self.on_startup)
        app.on_cleanup.append(self.on_cleanup)
        return app


class FoxyHotspotProcess:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.ready_file = pathlib.Path(tempfile.gettempdir()) / f"foxy-hotspot-ready-{os.getpid()}"

    def build_command(self) -> List[str]:
        helper = ROOT / "scripts" / "foxy-hotspot-helper.py"
        cmd = [
            sys.executable,
            str(helper),
            "--ssid",
            self.args.hotspot_ssid,
            "--password",
            self.args.hotspot_password,
            "--domain",
            self.args.hotspot_domain,
            "--address",
            self.args.hotspot_address,
            "--server-port",
            str(self.args.port),
            "--ready-file",
            str(self.ready_file),
        ]
        if self.args.hotspot_interface:
            cmd.extend(["--interface", self.args.hotspot_interface])
        if self.args.hotspot_country:
            cmd.extend(["--country", self.args.hotspot_country])
        if os.geteuid() != 0 and not self.args.hotspot_no_sudo:
            return ["sudo", "-E", *cmd]
        return cmd

    async def start(self) -> None:
        cmd = self.build_command()
        try:
            self.ready_file.unlink()
        except FileNotFoundError:
            pass
        log(f"Starting Foxy hotspot helper: {' '.join(cmd)}")
        self.proc = await asyncio.create_subprocess_exec(*cmd)
        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            if self.ready_file.exists():
                break
            if self.proc.returncode is not None:
                raise RuntimeError(f"Foxy hotspot helper exited early with status {self.proc.returncode}")
            await asyncio.sleep(0.25)
        else:
            await self.stop()
            raise RuntimeError("Timed out waiting for Foxy hotspot helper to become ready")
        log(f"Hotspot helper running. Open https://{self.args.hotspot_domain}:{self.args.port}")

    async def stop(self) -> None:
        try:
            self.ready_file.unlink()
        except FileNotFoundError:
            pass
        if not self.proc or self.proc.returncode is not None:
            return
        log("Stopping Foxy hotspot helper...")
        self.proc.terminate()
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=8.0)
        except asyncio.TimeoutError:
            self.proc.kill()
            await self.proc.wait()


def build_ssl_context() -> Optional[ssl.SSLContext]:
    cert = ROOT / "certs" / "cert.pem"
    key = ROOT / "certs" / "key.pem"
    if not cert.exists() or not key.exists():
        raise SystemExit("Missing certs. Run ./make-cert.sh first.")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    return ctx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Foxy SDK: PC-rendered stereo frames + Quest WebXR + IPC experiences")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--tls", action="store_true")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--eye-width", type=int, default=960)
    p.add_argument("--eye-height", type=int, default=960)
    p.add_argument("--jpeg-quality", type=int, default=72)
    p.add_argument("--ipc-path", default=DEFAULT_IPC_PATH)
    p.add_argument("--ipc-timeout-ms", type=float, default=650.0)
    p.add_argument("--ack-timeout-ms", type=float, default=300.0, help="Release a stuck per-client frame slot after this many ms")
    p.add_argument("--max-client-frame-age-ms", type=float, default=240.0, help="Display warning threshold only; frame dropping uses Quest-local queue age")
    p.add_argument("--resync-after-errors", type=int, default=8, help="Force a stream resync after this many consecutive client decode errors")
    p.add_argument("--demo-audio", action="store_true", help="Play the built-in PC-to-Quest test tone after the browser enables audio")
    p.add_argument("--no-audio", action="store_true", help="Deprecated compatibility flag; demo audio is off unless --demo-audio is set")
    p.add_argument("--no-desktop", action="store_true", help="Disable desktop capture panel in hub mode")
    p.add_argument("--hotspot", action="store_true", help="Start an isolated Wi-Fi AP helper for Quest access at foxy.local")
    p.add_argument("--hotspot-ssid", default="foxy", help="SSID for --hotspot")
    p.add_argument("--hotspot-password", default="foxy", help="WPA2 password for --hotspot; Wi-Fi standards require 8..63 chars, or empty for open")
    p.add_argument("--hotspot-domain", default=DEFAULT_HOTSPOT_DOMAIN, help="DNS name served by --hotspot")
    p.add_argument("--hotspot-address", default=DEFAULT_HOTSPOT_ADDRESS, help="IPv4 address assigned to the Foxy AP interface")
    p.add_argument("--hotspot-interface", default="", help="Wi-Fi interface to use for --hotspot; autodetected when omitted")
    p.add_argument("--hotspot-country", default=os.environ.get("FOXY_HOTSPOT_COUNTRY", ""), help="Optional two-letter regulatory country for hostapd")
    p.add_argument("--hotspot-no-sudo", action="store_true", help="Do not auto-run the hotspot helper through sudo when not root")
    p.add_argument("--hotspot-udp-video", action="store_true", help="Experimental: send hotspot video over WebRTC DataChannel/UDP instead of WebSocket/TCP")
    p.add_argument("--webrtc-buffered-drop-bytes", type=int, default=2_000_000, help="Drop UDP/WebRTC video frames when the data channel has this many queued bytes")
    p.add_argument("--webrtc-chunk-bytes", type=int, default=16_000, help="WebRTC DataChannel video chunk size; smaller chunks avoid large SCTP message stalls")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.jpeg_quality < 30 or args.jpeg_quality > 95:
        raise SystemExit("--jpeg-quality must be 30..95")
    if args.hotspot:
        if args.host == "127.0.0.1":
            args.host = "0.0.0.0"
        if not args.tls:
            args.tls = True
    server = FoxyServer(args)
    if args.no_audio:
        server.audio.demo_enabled = False
    app = server.make_app()

    scheme = "https" if args.tls else "http"
    log(f"Serving {scheme}://{args.host}:{args.port}")
    log(f"IPC: {args.ipc_path}")
    if args.hotspot:
        log(f"Hotspot mode: SSID={args.hotspot_ssid!r}, URL=https://{args.hotspot_domain}:{args.port}")
        if not args.hotspot_udp_video:
            log("Hotspot video transport: WebSocket/TCP (stable default)")
        elif RTCPeerConnection is None:
            log(f"Warning: aiortc unavailable ({AIORTC_IMPORT_ERROR}); hotspot video will fall back to WebSocket/TCP")
        else:
            log("Hotspot video transport: WebRTC DataChannel over UDP (experimental)")
    elif args.host == "127.0.0.1" and not args.tls:
        log("Quest USB mode: run scripts/adb-localhost.sh and open http://localhost:PORT in Quest Browser")
    elif not args.tls:
        log("Warning: non-localhost HTTP will usually not allow WebXR. Use --tls for Wi-Fi.")
    web.run_app(app, host=args.host, port=args.port, ssl_context=build_ssl_context() if args.tls else None)


if __name__ == "__main__":
    main()
