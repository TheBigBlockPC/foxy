# Foxy Examples

Run the Foxy server first. USB localhost mode still works as before; for an
isolated local-only Quest AP you can also run:

```bash
./make-cert.sh
./run.sh --hotspot --hotspot-password foxyfoxy
```

Then open `https://foxy.local:8766` on the Quest. The helper tears the hotspot
down when the server exits. WPA/WPA2 cannot use the literal password `foxy`
because Wi-Fi passphrases must be at least 8 characters.

## OpenGL controller demo

```bash
source .venv/bin/activate
python examples/opengl_experience.py
```

Shows:

- OpenGL stereo rendering
- Quest view/projection matrices
- left stick movement
- right stick spin/height
- A/B/X/Y-style primary/secondary buttons
- trigger and grip analog values
- `renderViews` for client-side reprojection

## Audio transfer demo

```bash
source .venv/bin/activate
python examples/audio_transfer_demo.py
```

Before running the demo, open the Foxy browser page on the Quest and press
**Enable Audio**. Press **Start Mic -> PC** to test mic transfer back into
Python.

Keys:

- `t`: toggle Python-generated PCM tone
- `f`: cycle frequency
- `+` / `-`: volume
- `q`: quit
