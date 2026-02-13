#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

API_KEY="${ELEVENLABS_API_KEY:-}"
if [[ -z "$API_KEY" ]]; then
  echo "ELEVENLABS_API_KEY is required."
  echo "Example:"
  echo "  export ELEVENLABS_API_KEY='your_key_here'"
  exit 1
fi

REFERENCE_WAV="${1:-$ROOT/audio/reference/laurent-ref-10s.wav}"
if [[ ! -f "$REFERENCE_WAV" ]]; then
  echo "Reference clip not found: $REFERENCE_WAV"
  exit 1
fi

VOICE_ID_FILE="$ROOT/.elevenlabs_voice_id"
VOICE_ID="${ELEVENLABS_VOICE_ID:-}"
VOICE_NAME="${ELEVENLABS_VOICE_NAME:-Laurent Voice Demo Clone}"

FULL_TEXT_FILE="$ROOT/transcripts/laurent-interstitial-2m45.en.localized.txt"
SHORT_TEXT_FILE="$ROOT/transcripts/xtts-test-short.en.txt"

mkdir -p "$ROOT/audio/cloned" "$ROOT/site/assets"

create_voice_if_needed() {
  if [[ -n "$VOICE_ID" ]]; then
    echo "Using ELEVENLABS_VOICE_ID from environment."
    return 0
  fi

  if [[ -f "$VOICE_ID_FILE" ]]; then
    VOICE_ID="$(cat "$VOICE_ID_FILE")"
    if [[ -n "$VOICE_ID" ]]; then
      echo "Using cached voice id from $VOICE_ID_FILE"
      return 0
    fi
  fi

  echo "Creating a new ElevenLabs cloned voice..."
  local tmp_json
  tmp_json="$(mktemp)"
  curl -sS -X POST "https://api.elevenlabs.io/v1/voices/add" \
    -H "xi-api-key: $API_KEY" \
    -F "name=$VOICE_NAME" \
    -F "description=Voice clone for Deep Search translated demo" \
    -F "files=@$REFERENCE_WAV" > "$tmp_json"

  VOICE_ID="$(python3 - "$tmp_json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
if "voice_id" in data:
    print(data["voice_id"])
    raise SystemExit(0)
detail = data.get("detail") or data
raise SystemExit(f"Failed to create voice: {detail}")
PY
)"

  echo "$VOICE_ID" > "$VOICE_ID_FILE"
  echo "Created voice id: $VOICE_ID"
}

synthesize_mp3() {
  local text_file="$1"
  local out_mp3="$2"
  local payload

  payload="$(python3 - "$text_file" <<'PY'
import json
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
body = {
    "text": text,
    "model_id": "eleven_multilingual_v2",
    "voice_settings": {
        "stability": 0.35,
        "similarity_boost": 0.85,
        "style": 0.25,
        "use_speaker_boost": True
    }
}
print(json.dumps(body))
PY
)"

  curl -sS -X POST "https://api.elevenlabs.io/v1/text-to-speech/$VOICE_ID" \
    -H "xi-api-key: $API_KEY" \
    -H "Accept: audio/mpeg" \
    -H "Content-Type: application/json" \
    --data "$payload" \
    --output "$out_mp3"

  ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "$out_mp3" >/dev/null
}

mix_with_bed() {
  local dry_mp3="$1"
  local mix_mp3="$2"
  local target_audio="$ROOT/audio/segments/laurent-interstitial-2m45.mp3"
  local bed_audio="$ROOT/audio/processed/htdemucs/laurent-interstitial-2m45/no_vocals.mp3"
  local aligned_mp3="$ROOT/audio/cloned/.tmp-aligned.mp3"

  local dry_dur target_dur atempo
  dry_dur="$(ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "$dry_mp3")"
  target_dur="$(ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "$target_audio")"
  atempo="$(python3 - "$dry_dur" "$target_dur" <<'PY'
import sys
d = float(sys.argv[1])
t = float(sys.argv[2])
ratio = d / t
ratio = max(0.5, min(2.0, ratio))
print(f"{ratio:.6f}")
PY
)"

  ffmpeg -y -hide_banner -loglevel error -i "$dry_mp3" \
    -filter:a "atempo=$atempo,volume=1.20" "$aligned_mp3"

  ffmpeg -y -hide_banner -loglevel error \
    -i "$bed_audio" \
    -i "$aligned_mp3" \
    -filter_complex "[0:a]volume=0.72[a0];[1:a]highpass=f=120,lowpass=f=9000,volume=1.25[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[m]" \
    -map "[m]" -c:a libmp3lame -q:a 2 "$mix_mp3"
}

create_voice_if_needed

echo "Generating short sample..."
synthesize_mp3 "$SHORT_TEXT_FILE" "$ROOT/audio/cloned/eleven-en-short.mp3"

echo "Generating full segment..."
synthesize_mp3 "$FULL_TEXT_FILE" "$ROOT/audio/cloned/eleven-en-full.mp3"

echo "Mixing over music bed..."
mix_with_bed "$ROOT/audio/cloned/eleven-en-short.mp3" "$ROOT/audio/cloned/eleven-en-short.mix.mp3"
mix_with_bed "$ROOT/audio/cloned/eleven-en-full.mp3" "$ROOT/audio/cloned/eleven-en-full.mix.mp3"

# Keep website unchanged: overwrite the existing player assets.
cp "$ROOT/audio/cloned/eleven-en-short.mp3" "$ROOT/site/assets/english-clone-xtts-short.mp3"
cp "$ROOT/audio/cloned/eleven-en-short.mix.mp3" "$ROOT/site/assets/english-clone-xtts-short-mix.mp3"
cp "$ROOT/audio/cloned/eleven-en-full.mp3" "$ROOT/site/assets/english-clone-xtts-full.mp3"
cp "$ROOT/audio/cloned/eleven-en-full.mix.mp3" "$ROOT/site/assets/english-clone-xtts-full-mix.mp3"
cp "$ROOT/audio/cloned/eleven-en-full.mix.mp3" "$ROOT/site/assets/english-mix.mp3"

# Keep pipeline output in sync.
cp "$ROOT/audio/cloned/eleven-en-full.mix.mp3" "$ROOT/audio/processed/laurent-interstitial-2m45.en.mix.mp3"

echo "Done. Updated site audio with ElevenLabs renders:"
echo "- $ROOT/site/assets/english-clone-xtts-full.mp3"
echo "- $ROOT/site/assets/english-clone-xtts-full-mix.mp3"
echo "- $ROOT/site/assets/english-mix.mp3"
