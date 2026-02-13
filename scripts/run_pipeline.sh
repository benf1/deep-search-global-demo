#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EPISODE_URL="https://media.radiofrance-podcast.net/podcast09/25253-31.01.2026-ITEMA_24392812-2026Y53750S0031-NET_MFP_D9FFA06C-5661-4E07-93ED-CA368E96B7D2-21-dc2c422803837ad4b7f31e2761369caf.mp3"

mkdir -p "$ROOT/audio/raw" "$ROOT/audio/segments" "$ROOT/audio/processed" "$ROOT/transcripts" "$ROOT/.cache/torch/hub/checkpoints"

if [[ ! -f "$ROOT/audio/raw/deep-search-66-76-full.mp3" ]]; then
  curl -L "$EPISODE_URL" -o "$ROOT/audio/raw/deep-search-66-76-full.mp3"
fi

ffmpeg -y -hide_banner -loglevel error -i "$ROOT/audio/raw/deep-search-66-76-full.mp3" -t 00:10:00 -c copy "$ROOT/audio/segments/deep-search-first-10m.mp3"
bash "$ROOT/scripts/extract_segment.sh" "$ROOT/audio/segments/deep-search-first-10m.mp3" 00:05:08 00:02:45 "$ROOT/audio/segments/laurent-interstitial-2m45.mp3"

source "$ROOT/.venv/bin/activate"
python "$ROOT/scripts/transcribe.py" \
  --input "$ROOT/audio/segments/laurent-interstitial-2m45.mp3" \
  --output-json "$ROOT/transcripts/laurent-interstitial-2m45.fr.json" \
  --output-txt "$ROOT/transcripts/laurent-interstitial-2m45.fr.txt" \
  --model small --language fr

# Preload Demucs model to local cache with curl if needed.
if [[ ! -f "$ROOT/.cache/torch/hub/checkpoints/955717e8-8726e21a.th" ]]; then
  curl -L "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th" \
    -o "$ROOT/.cache/torch/hub/checkpoints/955717e8-8726e21a.th"
fi

export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/.cache"
python -m demucs --two-stems=vocals -n htdemucs --mp3 "$ROOT/audio/segments/laurent-interstitial-2m45.mp3" -o "$ROOT/audio/processed"

# Temporary non-clone English narrator.
edge-tts --voice en-US-BrianNeural --rate=-30% \
  --file "$ROOT/transcripts/laurent-interstitial-2m45.en.localized.txt" \
  --write-media "$ROOT/audio/processed/laurent-interstitial-2m45.en.tts.match.mp3"

ffmpeg -y -hide_banner -loglevel error \
  -i "$ROOT/audio/processed/laurent-interstitial-2m45.en.tts.match.mp3" \
  -filter:a "atempo=0.968,volume=1.25" \
  "$ROOT/audio/processed/laurent-interstitial-2m45.en.tts.aligned.mp3"

ffmpeg -y -hide_banner -loglevel error \
  -i "$ROOT/audio/processed/htdemucs/laurent-interstitial-2m45/no_vocals.mp3" \
  -i "$ROOT/audio/processed/laurent-interstitial-2m45.en.tts.aligned.mp3" \
  -filter_complex "[0:a]volume=0.72[a0];[1:a]highpass=f=120,lowpass=f=9000,volume=1.30[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[m]" \
  -map "[m]" -c:a libmp3lame -q:a 2 \
  "$ROOT/audio/processed/laurent-interstitial-2m45.en.mix.mp3"

echo "Done:"
echo "- $ROOT/audio/segments/laurent-interstitial-2m45.mp3"
echo "- $ROOT/audio/processed/laurent-interstitial-2m45.en.mix.mp3"
