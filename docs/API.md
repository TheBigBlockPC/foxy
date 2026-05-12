# Foxy IPC Python API

The Foxy IPC API lets a local Python process become the active VR experience.

The server keeps the Quest/WebXR connection and headset transport. Your experience connects locally, reads state, renders frames, and sends those frames back.

## Socket

Default path:

```text
/tmp/foxy_ipc.sock
```

Override:

```bash
FOXY_IPC=/tmp/my_foxy.sock ./run.sh
```

or:

```bash
./run.sh --ipc-path /tmp/my_foxy.sock
```

## Packet protocol

Every packet is:

```text
uint32_be header_length
header JSON bytes
optional binary payload of header["payload_len"] bytes
```

The included `foxy_api.py` handles this for you.

## Minimal Python client

```python
from foxy_api import FoxyClient

client = FoxyClient()
client.connect()

while True:
    state = client.get_state()
    jpeg_sbs, render_views = render(state)

    client.send_frame(
        jpeg_sbs,
        eye_width=960,
        eye_height=960,
        render_views=render_views,
        app_name="My Experience",
    )
```

## State schema

`client.get_state()` returns:

```python
{
    "type": "state",
    "serverTimeMs": 123456.0,

    "views": {
        "left": {
            "eye": "left",
            "projectionMatrix": [16 floats],
            "viewMatrix": [16 floats]
        },
        "right": {
            "eye": "right",
            "projectionMatrix": [16 floats],
            "viewMatrix": [16 floats]
        }
    },

    "inputs": [
        {
            "handedness": "left" | "right",
            "targetRayMode": "tracked-pointer",
            "profiles": ["oculus-touch-v3", ...],

            "grip": {
                "position": {"x": 0, "y": 0, "z": 0},
                "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                "matrix": [16 floats]
            },

            "ray": {
                "position": ...,
                "orientation": ...,
                "matrix": [16 floats]
            },

            "gamepad": {
                "id": "...",
                "mapping": "xr-standard",
                "axes": [floats],
                "buttons": [
                    {"pressed": bool, "touched": bool, "value": float}
                ],
                "semantic": {
                    "trigger": {...},
                    "grip": {...},
                    "thumbstick": {...},
                    "primary": {...},
                    "secondary": {...},
                    "thumbstickX": float,
                    "thumbstickY": float
                }
            }
        }
    ],

    "events": [
        {
            "event": "selectstart" | "selectend" | "squeezestart" | ...,
            "handedness": "left" | "right",
            "profiles": [...]
        }
    ],

    "recommended": {
        "eyeWidth": 960,
        "eyeHeight": 960,
        "fps": 30
    }
}
```

## Controller helpers

`foxy_api.py` includes helpers:

```python
from foxy_api import input_by_hand, button, thumbstick

left = input_by_hand(state, "left")
right = input_by_hand(state, "right")

lx, ly = thumbstick(left)
rx, ry = thumbstick(right)

trigger = button(right, 0, "trigger")["value"]
grip = button(right, 1, "grip")["value"]

# Quest Touch Plus style:
a = button(right, 4, "primary")["pressed"]
b = button(right, 5, "secondary")["pressed"]
x = button(left, 4, "primary")["pressed"]
y = button(left, 5, "secondary")["pressed"]
```

Note: Browser mappings may vary. Foxy sends both raw Gamepad arrays and semantic guesses.

## Sending stereo frames

Use `send_frame()`:

```python
client.send_frame(
    jpeg_sbs,
    eye_width=960,
    eye_height=960,
    render_views={
        "left": {
            "view": [16 floats],
            "projection": [16 floats],
            "viewProjection": [16 floats]
        },
        "right": {
            "view": [16 floats],
            "projection": [16 floats],
            "viewProjection": [16 floats]
        }
    },
    app_name="My OpenGL App",
)
```

The image format is:

```text
single JPEG
width = eye_width * 2
height = eye_height

left eye  = left half
right eye = right half
```

`render_views` is recommended because the Quest client uses it for rotational reprojection.

## OpenGL matrix notes

WebXR matrices arrive as column-major arrays. In Python/Numpy, use:

```python
m = np.array(vals, dtype=np.float32).reshape((4, 4), order="F")
```

For OpenGL uniform upload through ModernGL:

```python
program["u_mvp"].write((proj @ view @ model).T.astype("f4").tobytes())
```

To send a matrix back to the browser:

```python
list_for_webgl = [float(x) for x in matrix.T.reshape(16)]
```

## Experience takeover behavior

Foxy shows the default hub when no IPC experience is sending frames.

When an experience sends frames, the Quest stream switches to IPC mode. If the experience stops sending frames for `--ipc-timeout-ms` milliseconds, Foxy returns to the hub.

Default timeout:

```text
650 ms
```

## Audio and mic

The current API focuses on visual frames and input. The built-in server already supports:

- PC -> Quest demo PCM audio over WebSocket
- Quest mic -> PC `captures/quest_mic_*.webm`

A future IPC extension should add:

```python
client.send_audio_pcm(...)
client.read_mic_chunks(...)
```

## Production transport roadmap

The API is intentionally simple. For serious latency work:

1. Keep this IPC state/control API.
2. Replace JPEG frame payloads with DMA-BUF/GPU texture sharing or encoded frames.
3. Replace WebSocket binary frames with WebRTC data/video channels.
4. Add depth stream for positional reprojection.
5. Add predicted poses or pose timestamps to match render time.


## Hub locomotion

The default hub scene now uses the controller state directly:

- Left stick: head-relative walk/strafe
- Right stick X: smooth turn

This is server-side hub behavior. IPC experiences receive the same raw input and can implement their own locomotion however they want. `foxy_api.head_yaw(state)` is provided for head-relative stick movement.

## Shader example

`examples/opengl_experience.py` now uses explicit GLSL shaders:

- vertex shader trigger/grip-driven deformation
- fragment shader procedural bands/glow/rim lighting
- A/B/X/Y-style buttons change shader tint/glow mix
