# Foxy Examples

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
