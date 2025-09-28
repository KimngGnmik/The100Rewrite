import csv, os, sys, time, subprocess, re, shutil
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI

# ---------- CONFIG ----------
VOICE_MAP: Dict[str, str] = {
    "NARRATOR": "alloy",
    "BECCA":    "nova",
    "CHRIS":    "verse",
    "WU":       "cedar",
    "BACKGROUND": "marin",
    "CADOGAN": "ash",
    "A.L.I.E.": "nova",   # will still say "Allie" via alias below
    "MARA": "shimmer",
    "JONAH": "cedar",
}
MODEL = "gpt-4o-mini-tts"     # "tts-1" / "tts-1-hd" also work
OUT_DIR = Path("output_build")
SAMPLE_EXT = "wav"            # "wav" or "mp3" (final per-line files)
LOUDNORM = True               # normalize final concat loudness
TARGET_LUFS = -16             # audiobook-friendly integrated loudness
DEFAULT_INTERLINE_PAUSE = 0.25  # seconds between lines

# Pronunciation aliases (regex -> replacement)
PRONUNCIATION_ALIASES: Dict[str, str] = {
    r"\bA\.L\.I\.E\.\b": "Allie",
}

# ---------- SETUP ----------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OUT_DIR.mkdir(exist_ok=True)

# ---------- UTIL ----------
def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install with:  brew install ffmpeg")

def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    for pattern, repl in PRONUNCIATION_ALIASES.items():
        text = re.sub(pattern, repl, text)
    return text

def _transcode_to_wav(src_mp3: Path, dst_wav: Path):
    """Convert mp3 -> wav (24k mono) with ffmpeg, then remove src mp3."""
    _check_ffmpeg()
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src_mp3), "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(dst_wav)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        src_mp3.unlink(missing_ok=True)
    except Exception:
        pass

def synth_to_file(text: str, voice: str, idx: int) -> Path:
    """
    Call OpenAI TTS once per line.
    We request MP3 (default) and, if SAMPLE_EXT='wav', transcode to WAV so all parts match.
    """
    text = normalize_text(text)
    # always fetch MP3 cleanly from API
    tmp_mp3 = OUT_DIR / f"{idx:05d}_{voice}.mp3"
    with client.audio.speech.with_streaming_response.create(
        model=MODEL,
        voice=voice,
        input=text
    ) as resp:
        resp.stream_to_file(str(tmp_mp3))

    if SAMPLE_EXT.lower() == "mp3":
        return tmp_mp3

    # Convert to WAV
    out_wav = OUT_DIR / f"{idx:05d}_{voice}.wav"
    _transcode_to_wav(tmp_mp3, out_wav)
    return out_wav

def make_silence(seconds: float, idx: int) -> Path:
    """
    Generate silence matching our pipeline (24 kHz mono).
    """
    _check_ffmpeg()
    out = OUT_DIR / f"{idx:05d}_silence.{SAMPLE_EXT.lower()}"
    try:
        if SAMPLE_EXT.lower() == "mp3":
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi",
                 "-i", "anullsrc=channel_layout=mono:sample_rate=24000",
                 "-t", str(seconds), "-q:a", "3", str(out)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:  # wav
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi",
                 "-i", "anullsrc=channel_layout=mono:sample_rate=24000",
                 "-t", str(seconds), "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le",
                 str(out)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    except Exception:
        out.write_bytes(b"")  # keep pipeline from breaking
    return out

def concat_files(parts: List[Path], out_path: Path):
    """Concat with ffmpeg and (optionally) loudness-normalize."""
    _check_ffmpeg()
    list_file = OUT_DIR / "concat.txt"
    with list_file.open("w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{p.resolve()}'\n")

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file)]
    if LOUDNORM:
        cmd += ["-af", f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11"]

    # Force consistent audiobook-friendly output
    out_lower = str(out_path).lower()
    if out_lower.endswith(".wav"):
        cmd += ["-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le"]
    elif out_lower.endswith(".mp3"):
        cmd += ["-ar", "24000", "-ac", "1", "-b:a", "128k"]

    cmd += [str(out_path)]
    subprocess.run(cmd, check=True)

# ---------- MAIN ----------
def main(csv_path: str, final_out: str):
    start = time.time()
    parts: List[Path] = []
    idx = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")  # Expect "SPEAKER|Text" or "[PAUSE=x]"
        for row in reader:
            if not row:
                continue

            # Allow blank rows
            if len(row) == 1 and not row[0].strip():
                continue

            # Pause line like: [PAUSE=2]
            if len(row) == 1:
                cell = row[0].strip()
                # skip headers like "Speaker|Text"
                if cell.lower().startswith("speaker"):
                    continue
                if cell.startswith("[PAUSE=") and cell.endswith("]"):
                    try:
                        seconds = float(cell[7:-1])
                    except Exception:
                        seconds = 0.5
                    parts.append(make_silence(seconds, idx)); idx += 1
                # else single noisy cell -> ignore
                continue

            # Normal line: SPEAKER|text
            speaker, text = row[0].strip(), row[1].strip().strip('"')
            if not speaker and not text:
                continue

            voice = VOICE_MAP.get(speaker.upper(), VOICE_MAP["NARRATOR"])
            print(f"[{idx:05d}] {speaker} → {voice}: {text[:70]}{'...' if len(text)>70 else ''}")

            # Synthesize
            audio_path = synth_to_file(text, voice, idx); idx += 1
            parts.append(audio_path)

            # small natural pause after each line
            parts.append(make_silence(DEFAULT_INTERLINE_PAUSE, idx)); idx += 1

    out_path = Path(final_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat_files(parts, out_path)
    elapsed = time.time() - start
    print(f"✅ Wrote {out_path} in {elapsed:.1f}s")
    print(f"Per-line audio kept in {OUT_DIR}/ for easy retakes.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tts_batch.py script.csv final_output.wav")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])