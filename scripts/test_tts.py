"""Phase 8a acceptance test — standalone TTS smoke test.

Run with:
    python scripts/test_tts.py

Writes output.mp3 to the project root if TTS succeeds.
Requires ELEVENLABS_API_KEY (or OPENAI_API_KEY + TTS_PROVIDER=openai) in the environment.
"""

import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from core.tts import synthesize

TEST_TEXT = (
    "The top rated hotel in Dallas is the Magnolia with a 4.8 rating. "
    "Guests especially praise the rooftop pool and the attentive staff."
)

print(f"TTS_PROVIDER={os.environ.get('TTS_PROVIDER', 'elevenlabs')}")
print(f"Synthesizing: {TEST_TEXT!r}")

audio = synthesize(TEST_TEXT)

if audio is None:
    print("FAIL: synthesize() returned None. Check your API key and TTS_PROVIDER env vars.")
    sys.exit(1)

out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output.mp3")
with open(out_path, "wb") as f:
    f.write(audio)

print(f"OK: wrote {len(audio):,} bytes to {out_path}")
print("Play it with: open output.mp3  (macOS) or afplay output.mp3")
