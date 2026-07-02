#!/usr/bin/env python3
"""Interactive Foxy IPC audio transfer demo.

Run the Foxy server and open the Quest browser page first. Press Enable Audio
on the page to hear PC -> Quest PCM. Press Start Mic -> PC to stream mic blobs
back to this demo and to the captures/ folder.
"""

from __future__ import annotations

import array
import math
import pathlib
import select
import sys
import termios
import time
import tty
from typing import Optional, Tuple

# Allow running from examples/ without installing package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from foxy_api import FoxyClient


SAMPLE_RATE = 48000
CHANNELS = 2
CHUNK_MS = 40
FREQUENCIES = [220.0, 330.0, 440.0, 660.0, 880.0]


def make_tone_chunk(phase: float, freq: float, volume: float) -> Tuple[bytes, float]:
    frames = int(SAMPLE_RATE * CHUNK_MS / 1000)
    out = array.array("h")
    amp = max(0.0, min(1.0, volume)) * 32767.0
    step = (2.0 * math.pi * freq) / SAMPLE_RATE
    for _ in range(frames):
        sample = int(math.sin(phase) * amp)
        phase += step
        if phase > math.tau:
            phase -= math.tau
        for _ch in range(CHANNELS):
            out.append(sample)
    return out.tobytes(), phase


def read_key() -> Optional[str]:
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return None
    return sys.stdin.read(1)


def main() -> None:
    print("Foxy audio transfer demo")
    print("Quest page: press Enable Audio for PC -> Quest sound.")
    print("Quest page: press Start Mic -> PC for Quest -> Python mic chunks.")
    print("Keys: t tone on/off | f frequency | +/- volume | q quit")
    print()

    client = FoxyClient()
    client.connect()

    old_term = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    tone_on = True
    volume = 0.08
    freq_index = 2
    phase = 0.0
    chunks_sent = 0
    mic_chunks = 0
    mic_bytes = 0
    last_mime = "none"
    next_audio_at = time.perf_counter()
    next_status_at = 0.0

    try:
        while True:
            key = read_key()
            if key == "q":
                break
            if key == "t":
                tone_on = not tone_on
            elif key == "f":
                freq_index = (freq_index + 1) % len(FREQUENCIES)
            elif key in ("+", "="):
                volume = min(0.25, volume + 0.02)
            elif key in ("-", "_"):
                volume = max(0.0, volume - 0.02)

            mic = client.get_mic_chunk(timeout_ms=0)
            if mic is not None:
                meta, payload = mic
                mic_chunks += 1
                mic_bytes += len(payload)
                last_mime = str(meta.get("mimeType") or meta.get("format") or "unknown")

            now = time.perf_counter()
            if tone_on and now >= next_audio_at:
                chunk, phase = make_tone_chunk(phase, FREQUENCIES[freq_index], volume)
                client.send_audio_pcm(
                    chunk,
                    sample_rate=SAMPLE_RATE,
                    channels=CHANNELS,
                    samples_per_channel=int(SAMPLE_RATE * CHUNK_MS / 1000),
                    app_name="Audio Transfer Demo",
                )
                chunks_sent += 1
                next_audio_at = now + CHUNK_MS / 1000.0

            if now >= next_status_at:
                status = (
                    f"tone={'on ' if tone_on else 'off'} "
                    f"freq={FREQUENCIES[freq_index]:.0f}Hz "
                    f"vol={volume:.2f} "
                    f"pcm_chunks={chunks_sent} "
                    f"mic_chunks={mic_chunks} "
                    f"mic_bytes={mic_bytes} "
                    f"mic={last_mime}"
                )
                sys.stdout.write("\r" + status + " " * 12)
                sys.stdout.flush()
                next_status_at = now + 0.2

            time.sleep(0.004)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        client.close()
        print("\nDone.")


if __name__ == "__main__":
    main()
