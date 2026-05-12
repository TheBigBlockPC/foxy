#!/usr/bin/env python3
"""OpenGL Foxy experience example.

Shows:
- left stick movement
- right stick movement
- A/B/X/Y-style primary/secondary button control
- trigger analog control
- grip analog control
- per-eye OpenGL rendering using the real Quest view/projection matrices
- sending stereo frames to Foxy over IPC

Run server first:
    ./run.sh --host 127.0.0.1 --port 8766

Then run this:
    source .venv/bin/activate
    python examples/opengl_experience.py
"""

from __future__ import annotations

import io
import math
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import moderngl
import numpy as np
from PIL import Image, ImageDraw

# Allow running from examples/ without installing package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from foxy_api import FoxyClient, button, head_basis, input_by_hand, thumbstick


EYE_W = 960
EYE_H = 960
FPS = 30


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


def make_cube_vertices() -> np.ndarray:
    p = [
        (-1,-1,1),(1,-1,1),(1,1,1),(-1,-1,1),(1,1,1),(-1,1,1),
        (1,-1,-1),(-1,-1,-1),(-1,1,-1),(1,-1,-1),(-1,1,-1),(1,1,-1),
        (-1,-1,-1),(-1,-1,1),(-1,1,1),(-1,-1,-1),(-1,1,1),(-1,1,-1),
        (1,-1,1),(1,-1,-1),(1,1,-1),(1,-1,1),(1,1,-1),(1,1,1),
        (-1,1,1),(1,1,1),(1,1,-1),(-1,1,1),(1,1,-1),(-1,1,-1),
        (-1,-1,-1),(1,-1,-1),(1,-1,1),(-1,-1,-1),(1,-1,1),(-1,-1,1),
    ]
    colors = [(0.9,0.25,0.2),(0.9,0.25,0.2),(0.9,0.25,0.2),(0.9,0.25,0.2),(0.9,0.25,0.2),(0.9,0.25,0.2),
              (0.2,0.6,1.0),(0.2,0.6,1.0),(0.2,0.6,1.0),(0.2,0.6,1.0),(0.2,0.6,1.0),(0.2,0.6,1.0),
              (0.25,0.9,0.45),(0.25,0.9,0.45),(0.25,0.9,0.45),(0.25,0.9,0.45),(0.25,0.9,0.45),(0.25,0.9,0.45),
              (1.0,0.85,0.2),(1.0,0.85,0.2),(1.0,0.85,0.2),(1.0,0.85,0.2),(1.0,0.85,0.2),(1.0,0.85,0.2),
              (0.7,0.35,1.0),(0.7,0.35,1.0),(0.7,0.35,1.0),(0.7,0.35,1.0),(0.7,0.35,1.0),(0.7,0.35,1.0),
              (0.85,0.85,0.85),(0.85,0.85,0.85),(0.85,0.85,0.85),(0.85,0.85,0.85),(0.85,0.85,0.85),(0.85,0.85,0.85)]
    return np.array([(*pos, *col) for pos, col in zip(p, colors)], dtype=np.float32)


def make_grid_vertices(size=8, step=1.0) -> np.ndarray:
    verts = []
    for i in range(-size, size + 1):
        c = (0.45, 0.55, 0.75) if i == 0 else (0.14, 0.18, 0.28)
        x = i * step
        verts.append((x, 0.0, -size * step, *c))
        verts.append((x, 0.0, size * step, *c))
        z = i * step
        verts.append((-size * step, 0.0, z, *c))
        verts.append((size * step, 0.0, z, *c))
    return np.array(verts, dtype=np.float32)


class ExperienceRenderer:
    def __init__(self, eye_w: int, eye_h: int):
        backends = ["egl", "x11", ""]
        self.ctx = None
        errors = []
        for backend in backends:
            try:
                self.ctx = moderngl.create_standalone_context(backend=backend) if backend else moderngl.create_standalone_context()
                print("ModernGL backend:", backend or "default")
                break
            except Exception as e:
                errors.append(f"{backend or 'default'}: {e}")
        if self.ctx is None:
            raise RuntimeError("Could not create ModernGL context:\n" + "\n".join(errors))

        self.eye_w = eye_w
        self.eye_h = eye_h
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.color_tex = self.ctx.texture((eye_w, eye_h), 4)
        self.depth = self.ctx.depth_renderbuffer((eye_w, eye_h))
        self.fbo = self.ctx.framebuffer(color_attachments=[self.color_tex], depth_attachment=self.depth)

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec3 in_color;

                uniform mat4 u_model;
                uniform mat4 u_mvp;
                uniform float u_time;
                uniform float u_warp;

                out vec3 v_color;
                out vec3 v_world;
                out vec3 v_local;

                void main() {
                    vec3 p = in_pos;

                    // Vertex shader effect: controller trigger/grip can make the
                    // cube pulse organically instead of remaining rigid.
                    float wave = sin(u_time * 4.0 + p.x * 3.0 + p.y * 2.0 + p.z * 2.5);
                    p += normalize(p) * wave * 0.045 * u_warp;

                    vec4 world = u_model * vec4(p, 1.0);
                    v_world = world.xyz;
                    v_local = p;
                    v_color = in_color;
                    gl_Position = u_mvp * vec4(p, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_color;
                in vec3 v_world;
                in vec3 v_local;

                uniform vec3 u_tint;
                uniform float u_brightness;
                uniform float u_time;
                uniform float u_trigger;
                uniform float u_button_mix;

                out vec4 f_color;

                void main() {
                    // Fragment shader effect: procedural rim + scanline + glow,
                    // controlled by buttons and trigger.
                    vec3 base = v_color * u_tint;
                    float bands = 0.5 + 0.5 * sin((v_world.y + v_local.x) * 12.0 + u_time * 5.0);
                    float rim = pow(1.0 - abs(normalize(v_local).z), 2.0);
                    vec3 glow = mix(vec3(0.15, 0.25, 0.55), vec3(1.0, 0.45, 0.18), u_button_mix);

                    vec3 color = base * (0.65 + 0.35 * bands);
                    color += glow * rim * (0.35 + 0.95 * u_trigger);
                    color *= u_brightness;

                    f_color = vec4(color, 1.0);
                }
            """,
        )
        self.cube_vbo = self.ctx.buffer(make_cube_vertices().tobytes())
        self.cube_vao = self.ctx.vertex_array(self.prog, [(self.cube_vbo, "3f 3f", "in_pos", "in_color")])
        self.grid_vbo = self.ctx.buffer(make_grid_vertices().tobytes())
        self.grid_vao = self.ctx.vertex_array(self.prog, [(self.grid_vbo, "3f 3f", "in_pos", "in_color")])

    def view_projection_for_eye(self, state: Dict[str, Any], eye: str) -> Tuple[np.ndarray, np.ndarray]:
        view_data = (state.get("views") or {}).get(eye)
        if view_data:
            view = mat_from_webxr(view_data.get("viewMatrix"))
            proj = mat_from_webxr(view_data.get("projectionMatrix"))
            if view is not None and proj is not None:
                return view, proj
        ipd = 0.064
        x = -ipd / 2 if eye == "left" else ipd / 2
        return look_at((x, 1.6, 2.6), (0, 1.2, -2.4)), perspective(math.radians(88), self.eye_w / self.eye_h, 0.05, 100.0)

    def draw_eye(self, state: Dict[str, Any], eye: str, sim: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        view, proj = self.view_projection_for_eye(state, eye)
        vp = proj @ view

        self.fbo.use()
        self.ctx.viewport = (0, 0, self.eye_w, self.eye_h)
        self.ctx.clear(0.015, 0.018, 0.032, 1.0, depth=1.0)

        grid_model = translate(0, -0.02, -3.0)
        self.prog["u_model"].write(mat_to_gl_bytes(grid_model))
        self.prog["u_mvp"].write(mat_to_gl_bytes(vp @ grid_model))
        self.prog["u_time"].value = sim["time"]
        self.prog["u_warp"].value = 0.0
        self.prog["u_trigger"].value = 0.0
        self.prog["u_button_mix"].value = 0.0
        self.prog["u_tint"].value = (0.55, 0.7, 1.0)
        self.prog["u_brightness"].value = 0.65
        self.grid_vao.render(mode=moderngl.LINES)

        # Main cube controlled by stick/ABXY/triggers.
        model = translate(sim["x"], 1.05 + sim["height"], -3.0 + sim["z"]) @ rotate_y(sim["time"] * 0.8 + sim["spin"]) @ rotate_x(sim["time"] * 0.4) @ scale(sim["scale"], sim["scale"], sim["scale"])
        self.prog["u_model"].write(mat_to_gl_bytes(model))
        self.prog["u_mvp"].write(mat_to_gl_bytes(vp @ model))
        self.prog["u_time"].value = sim["time"]
        self.prog["u_warp"].value = sim["trigger"] + sim["grip"] * 0.6
        self.prog["u_trigger"].value = sim["trigger"]
        self.prog["u_button_mix"].value = sim["button_mix"]
        self.prog["u_tint"].value = sim["tint"]
        self.prog["u_brightness"].value = 0.85 + sim["trigger"] * 1.2
        self.cube_vao.render()

        # Four button indicators: X/Y left and A/B right.
        names = [("X", -0.75, sim["x_pressed"]), ("Y", -0.25, sim["y_pressed"]), ("A", 0.25, sim["a_pressed"]), ("B", 0.75, sim["b_pressed"])]
        for label, x, pressed in names:
            s = 0.10 if not pressed else 0.18
            model = translate(x, 1.72, -2.2) @ rotate_y(sim["time"] * 0.35) @ scale(s, s, s)
            self.prog["u_model"].write(mat_to_gl_bytes(model))
            self.prog["u_mvp"].write(mat_to_gl_bytes(vp @ model))
            self.prog["u_time"].value = sim["time"]
            self.prog["u_warp"].value = 1.0 if pressed else 0.0
            self.prog["u_trigger"].value = 1.0 if pressed else 0.0
            self.prog["u_button_mix"].value = 1.0 if pressed else 0.0
            self.prog["u_tint"].value = (1.5, 1.1, 0.4) if pressed else (0.35, 0.45, 0.75)
            self.prog["u_brightness"].value = 1.0
            self.cube_vao.render()

        data = self.fbo.read(components=4, alignment=1)
        img = Image.frombytes("RGBA", (self.eye_w, self.eye_h), data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        # Overlay small text into the source image so you can verify controls.
        d = ImageDraw.Draw(img)
        d.rectangle((12, 12, 575, 120), fill=(5, 8, 16, 210))
        d.text((24, 24), "OpenGL IPC Experience", fill=(240, 245, 255))
        d.text((24, 54), "Left stick: move cube | Right stick: spin/height", fill=(190, 210, 245))
        d.text((24, 82), f"X={sim['x_pressed']} Y={sim['y_pressed']} A={sim['a_pressed']} B={sim['b_pressed']} Trigger={sim['trigger']:.2f}", fill=(255, 220, 140))
        return img, {"view": mat_to_webgl_list(view), "projection": mat_to_webgl_list(proj), "viewProjection": mat_to_webgl_list(vp)}

    def render(self, state: Dict[str, Any], sim: Dict[str, Any], quality: int = 72) -> Tuple[bytes, Dict[str, Any]]:
        left, lrv = self.draw_eye(state, "left", sim)
        right, rrv = self.draw_eye(state, "right", sim)
        combined = Image.new("RGB", (self.eye_w * 2, self.eye_h))
        combined.paste(left.convert("RGB"), (0, 0))
        combined.paste(right.convert("RGB"), (self.eye_w, 0))
        buf = io.BytesIO()
        combined.save(buf, format="JPEG", quality=quality, optimize=False)
        return buf.getvalue(), {"left": lrv, "right": rrv}


def main():
    client = FoxyClient()
    print("Connecting to Foxy IPC...")
    client.connect()
    print("Connected. Ping ms:", client.ping())

    renderer = ExperienceRenderer(EYE_W, EYE_H)
    x, z = 0.0, 0.0
    spin = 0.0
    height = 0.0
    color_index = 0
    colors = [(1.0, 1.0, 1.0), (1.5, 0.7, 0.45), (0.55, 1.4, 0.8), (0.55, 0.85, 1.6), (1.4, 0.65, 1.45)]
    frame = 0
    last = time.time()

    while True:
        start = time.time()
        state = client.get_state()
        left = input_by_hand(state, "left")
        right = input_by_hand(state, "right")

        lx, ly = thumbstick(left)
        rx, ry = thumbstick(right)

        # Left stick moves the cube in the experience, head-relative so forward
        # follows physical head rotation instead of the fixed world axes.
        dt = min(0.05, start - last)
        last = start
        forward = -ly
        strafe = lx
        right_basis, forward_basis = head_basis(state)
        x += (strafe * right_basis[0] + forward * forward_basis[0]) * dt * 1.6
        z += (strafe * right_basis[1] + forward * forward_basis[1]) * dt * 1.6

        # Right stick controls spin and height.
        spin += rx * dt * 3.0
        height = max(-0.35, min(0.55, height + (-ry) * dt * 1.2))

        # Quest Touch Plus mappings through semantic primary/secondary.
        # Left primary/secondary are usually X/Y. Right primary/secondary are usually A/B.
        x_btn = button(left, 4, "primary")
        y_btn = button(left, 5, "secondary")
        a_btn = button(right, 4, "primary")
        b_btn = button(right, 5, "secondary")

        trigger = max(button(left, 0, "trigger")["value"], button(right, 0, "trigger")["value"])
        grip = max(button(left, 1, "grip")["value"], button(right, 1, "grip")["value"])

        if a_btn["pressed"]:
            color_index = 1
        elif b_btn["pressed"]:
            color_index = 2
        elif x_btn["pressed"]:
            color_index = 3
        elif y_btn["pressed"]:
            color_index = 4
        else:
            color_index = 0

        sim = {
            "time": start,
            "x": x,
            "z": z,
            "spin": spin,
            "height": height,
            "scale": 0.28 + trigger * 0.25 + grip * 0.18,
            "trigger": trigger,
            "grip": grip,
            "button_mix": min(1.0, float(color_index) / max(1.0, len(colors) - 1)),
            "tint": colors[color_index],
            "x_pressed": x_btn["pressed"],
            "y_pressed": y_btn["pressed"],
            "a_pressed": a_btn["pressed"],
            "b_pressed": b_btn["pressed"],
        }

        jpeg, render_views = renderer.render(state, sim)
        client.send_frame(jpeg, eye_width=EYE_W, eye_height=EYE_H, render_views=render_views, app_name="OpenGL Controller Demo", frame_id=frame)
        frame += 1

        sleep = max(0.0, (1.0 / FPS) - (time.time() - start))
        time.sleep(sleep)


if __name__ == "__main__":
    main()
