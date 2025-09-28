import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import subprocess

# Load .env for your API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Config
MODEL = "gpt-4o-mini-tts"
VOICES = [
    "alloy", "echo", "fable", "onyx", "nova",
    "shimmer", "coral", "verse", "ballad", "ash",
    "sage", "marin", "cedar"
]
TEXT = "Hello, this is a short audition line for our audiobook."
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

# Generate per-voice samples
parts = []
for idx, voice in enumerate(VOICES):
    out_path = OUT_DIR / f"{idx:02d}_{voice}.wav"
    print(f"🔊 Generating {voice}...")
    with client.audio.speech.with_streaming_response.create(
        model=MODEL,
        voice=voice,
        input=f"{voice.capitalize()} voice speaking. {TEXT}"
    ) as response:
        response.stream_to_file(out_path)
    parts.append(out_path)

# Concatenate them into one file (requires ffmpeg)
final_path = OUT_DIR / "all_voices_demo.wav"
concat_file = OUT_DIR / "concat.txt"
with concat_file.open("w") as f:
    for p in parts:
        f.write(f"file '{p.resolve()}'\n")

subprocess.run([
    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
    "-c", "copy", str(final_path)
])

print(f"✅ Done. All voices stitched into {final_path}")