#!/usr/bin/env python3
"""Raw-frame Foxy IPC example.

This demo sends raw side-by-side RGB frames. Foxy encodes them to JPEG before
streaming them to the Quest browser.

Run server first:
    ./run.sh --host 127.0.0.1 --port 8766

Then run:
    source .venv/bin/activate
    python examples/raw_frame_experience.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

# Allow running from examples/ without installing package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from foxy_api import FoxyClient, input_by_hand, thumbstick


EYE_W = 640
EYE_H = 640
FPS = 30


def draw_disc(img: np.ndarray, cx: float, cy: float, radius: float, color: tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    img[mask] = color


def render_eye(t: float, eye_offset: float, stick_x: float, stick_y: float) -> np.ndarray:
    y = np.linspace(0.0, 1.0, EYE_H, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, EYE_W, dtype=np.float32)[None, :]
    img = np.empty((EYE_H, EYE_W, 3), dtype=np.uint8)
    img[..., 0] = np.clip((0.18 + 0.38 * x + 0.10 * np.sin(t + eye_offset)) * 255, 0, 255).astype(np.uint8)
    img[..., 1] = np.clip((0.12 + 0.42 * y) * 255, 0, 255).astype(np.uint8)
    img[..., 2] = np.clip((0.22 + 0.35 * (1.0 - x) + 0.15 * np.cos(t * 0.7)) * 255, 0, 255).astype(np.uint8)

    cx = EYE_W * (0.5 + 0.30 * np.sin(t * 1.3 + eye_offset) + 0.18 * stick_x)
    cy = EYE_H * (0.5 + 0.24 * np.cos(t * 1.1) + 0.18 * stick_y)
    draw_disc(img, cx, cy, 70, (255, 230, 90))
    draw_disc(img, EYE_W * 0.5, EYE_H * 0.82, 18, (80, 255, 190))
    return img


def main() -> None:
    client = FoxyClient()
    print("Connecting to Foxy IPC...")
    client.connect()
    print("Connected. Sending raw RGB frames; Foxy will JPEG-encode them.")

    frame = 0
    while True:
        start = time.time()
        state = client.get_state()
        left = input_by_hand(state, "left")
        sx, sy = thumbstick(left)

        t = time.time()
        left_eye = render_eye(t, -0.08, sx, sy)
        right_eye = render_eye(t, 0.08, sx, sy)
        raw_sbs = np.concatenate([left_eye, right_eye], axis=1)

        client.send_raw_frame(
            raw_sbs,
            eye_width=EYE_W,
            eye_height=EYE_H,
            app_name="Raw RGB Frame Demo",
            frame_id=frame,
            jpeg_quality=68,
        )
        frame += 1

        time.sleep(max(0.0, (1.0 / FPS) - (time.time() - start)))


if __name__ == "__main__":
    main()
