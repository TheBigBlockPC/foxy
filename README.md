# Foxy SDK

Foxy SDK is a prototype Linux + Quest VR stack for custom experiences without SteamVR, ALVR, or OpenXR runtime fights.

It gives you:

- A default **Foxy Hub** scene rendered on the PC GPU.
- Quest Browser/WebXR client for headset pose, per-eye matrices, controllers, audio, and mic.
- A local Unix-socket **IPC Python API**.
- External Python experiences that receive all Quest data and send their own left/right eye stream.
- An OpenGL demo experience showing stick movement, A/B/X/Y-style buttons, triggers, grips, and stereo rendering.
- A desktop preview panel in the hub when desktop capture is available.

This is still a prototype transport: JPEG over WebSocket. It is meant to prove the architecture. For production, replace the frame transport with WebRTC + hardware AV1/H.265/H.264.

## Quick start

```bash
unzip foxy_sdk.zip
cd foxy_sdk
./run.sh --host 127.0.0.1 --port 8766 --fps 30 --eye-width 960 --eye-height 960
```

In another terminal, with the Quest connected by USB:

```bash
./scripts/adb-localhost.sh 8766
```

Open in Quest Browser:

```text
http://localhost:8766
```

Click **Enter VR**.

You should see the **Foxy Hub** scene. If desktop capture works on your Linux session, you will also see your desktop in a panel on the right.

Hub locomotion:

- Left stick: head-relative walk/strafe around the hub world
- Right stick X: smooth turn
- Movement is bounded to keep the prototype easy to recover if a stick mapping is weird

## Run the OpenGL IPC experience

Leave the server running. In another terminal:

```bash
cd foxy_sdk
source .venv/bin/activate
python examples/opengl_experience.py
```

The hub should switch to **mode: ipc / OpenGL Controller Demo**.

Controls:

- Left stick: move cube
- Right stick: spin / height
- A/B/X/Y-style primary/secondary buttons: change cube tint and shader glow
- Triggers: scale/brightness plus vertex-warp shader strength
- Grips: scale plus extra shader warp
- The demo uses explicit GLSL vertex/fragment shaders with procedural glow, scanlines, and trigger-driven deformation
- Controller rays/poses: available in the state API

## Run the interactive audio transfer demo

Leave the server running and open the Quest browser page. Press **Enable Audio**
on the page to allow PC -> Quest playback. Press **Start Mic -> PC** if you also
want to test Quest mic transfer back into Python.

```bash
cd foxy_sdk
source .venv/bin/activate
python examples/audio_transfer_demo.py
```

Demo keys:

- `t`: toggle the PCM tone sent from Python to Quest
- `f`: cycle tone frequency
- `+` / `-`: adjust volume
- `q`: quit

Mic chunks are shown in the demo status line and are also appended to
`captures/quest_mic_*.webm`.


## Stream stabilization / ground-truth recovery

This build adds a conservative newest-frame-only stream policy.  The Quest client acknowledges successfully decoded frames, drops only Quest-local backlog/superseded frames, and asks the server for a resync if decoding or texture upload starts failing.  It does **not** drop frames using the PC `serverTimeMs` wall clock, because the Quest and Linux clocks may not be synchronized.  The server will not queue more video behind an unacknowledged frame; it drops instead, because a visible frame drop is safer than letting old frames accumulate and corrupt the live headset view.

New runtime knobs:

```bash
./run.sh --fps 45 --eye-width 1280 --eye-height 1280 --jpeg-quality 65 \
  --ack-timeout-ms 300 --max-client-frame-age-ms 240 --resync-after-errors 8
```

For weaker hardware, prefer stable pacing over resolution:

```bash
./run.sh --fps 36 --eye-width 960 --eye-height 960 --jpeg-quality 60 --no-desktop
```

The browser HUD now shows accepted/received frames, client drops, clock-corrected rough frame age, Quest-local queue age, decode time, and stream state.  You can also inspect `/status` on the PC for `frames_dropped_server`, `client_decode_errors`, `client_stale_drops`, and per-client health.

## Wi-Fi mode

WebXR generally requires a secure context unless you use USB localhost mode.

```bash
./make-cert.sh
./run.sh --host 0.0.0.0 --port 8766 --tls
```

Open on Quest Browser:

```text
https://YOUR_PC_IP:8766
```

Accept the self-signed cert warning.

## Disable desktop capture

```bash
./run.sh --no-desktop
```

Desktop capture depends on the Linux session. It is more likely to work on X11 than locked-down Wayland sessions.

## API docs

See:

- [`docs/API.md`](docs/API.md)
- [`foxy_api.py`](foxy_api.py)
- [`examples/opengl_experience.py`](examples/opengl_experience.py)

## Current architecture

```text
Quest Browser / WebXR
  -> headset pose, per-eye matrices, controllers, mic
  -> WebSocket to Foxy server

Foxy server
  -> default OpenGL hub renderer
  -> desktop panel capture
  -> PCM audio to Quest + Quest mic chunks to IPC/captures
  -> IPC Unix socket for Python experiences
  -> sends side-by-side stereo frames to Quest

Python experience
  -> connects to /tmp/foxy_ipc.sock
  -> receives tracking/controllers/events
  -> renders left/right eye frames with OpenGL
  -> optionally sends PCM audio and reads Quest mic chunks
  -> sends stereo JPEG stream back to Foxy
```

## Limitations

- JPEG/WebSocket is simple but not low-latency enough for production.
- Reprojection is rotational/infinite-depth; positional reprojection would need depth.
- Desktop panel capture may fail under some Wayland security settings.
- WebXR Gamepad button mappings can vary by browser/runtime. Foxy sends both raw arrays and semantic guesses.

## Stream stability behavior

This build treats decode/frame problems as stream-health problems, not connection-health problems.

- Bad or superseded frames are dropped; startup frames are accepted permissively so the first good image can establish ground truth.
- The last good texture stays visible during recovery.
- The WebSocket is kept open during stream resync.
- The client reconnects only when the socket itself is actually closed.
- Server-side ACK timeouts release the send slot so the newest full JPEG becomes ground truth.

Useful weak-hardware preset:

```bash
./run.sh --fps 36 --eye-width 960 --eye-height 960 --jpeg-quality 60 --no-desktop \
  --ack-timeout-ms 400 --max-client-frame-age-ms 240 --resync-after-errors 8
```

## Stabilization update: hold-last-frame drop concealment

This build changes the headset display fallback so dropped frames do not clear the VR view.
When a frame is dropped, rejected, superseded, or the server asks for a stream resync, the client keeps the last successfully uploaded GL texture visible. The display shader clamps invalid reprojected UVs and blends back toward the normal last-frame UVs instead of painting black. This masks transient drops as a short held frame rather than a black flicker.

Practical tuning for weaker hardware:

```bash
./run.sh --fps 36 --eye-width 960 --eye-height 960 --jpeg-quality 60 --no-desktop \
  --ack-timeout-ms 450 --max-client-frame-age-ms 300 --resync-after-errors 12
```

For better hardware:

```bash
./run.sh --fps 45 --eye-width 1280 --eye-height 1280 --jpeg-quality 65 --no-desktop \
  --ack-timeout-ms 450 --max-client-frame-age-ms 300 --resync-after-errors 12
```

If the HUD still reports heavy dropping, lower FPS before lowering quality. Stable frame pacing is more important than peak FPS for headset comfort.


## Stable display rollback build

This build intentionally rolls back the aggressive reconnect/validation logic. Frame drops are handled as display events: the last successfully uploaded XR texture remains visible, pending decoded frames do not become ground truth until the GL upload succeeds, and old WebSocket callbacks are ignored after reconnect/reload. The DOM preview is disabled by default to reduce Quest Browser load; use `?preview=1` to enable it for debugging.

Recommended test command:

```bash
./run.sh --fps 30 --eye-width 960 --eye-height 960 --jpeg-quality 60 --no-desktop --ack-timeout-ms 500 --resync-after-errors 20
```

If it is stable, increase quality before FPS:

```bash
./run.sh --fps 36 --eye-width 1280 --eye-height 1280 --jpeg-quality 62 --no-desktop --ack-timeout-ms 600 --resync-after-errors 20
```


## Demand-paced display ACK build

This build fixes the slideshow failure mode from the previous stabilization builds.

Changes:

- Browser audio is off until you press **Enable Audio**. IPC audio can then stream from Python to Quest. The built-in server tone is opt-in with `--demo-audio`.
- The server no longer renders/encodes frames that the Quest is not ready to display. It waits for a display ACK from the XR texture upload path, then renders the next frame.
- Client ACKs now mean "this frame reached the XR texture", not merely "JPEG decode finished".
- Server `frames_skipped_not_ready` means the server intentionally did not render a useless frame. It is not a visible frame drop.
- Real drops are only counted when a sent frame times out, fails decode, or fails texture upload.

Recommended first test:

```bash
./run.sh --fps 45 --eye-width 960 --eye-height 960 --jpeg-quality 58 --no-desktop --ack-timeout-ms 900 --resync-after-errors 20
```

If it still updates below about 20 FPS at 960x960 per eye, the Quest Browser JPEG decode path is the bottleneck. Lower resolution to prove stability:

```bash
./run.sh --fps 60 --eye-width 640 --eye-height 640 --jpeg-quality 55 --no-desktop --ack-timeout-ms 900 --resync-after-errors 20
```

For high-quality low-latency PCVR, the JPEG/WebSocket browser transport needs to be replaced by hardware H.264/H.265/AV1 decode through a native Quest client.
