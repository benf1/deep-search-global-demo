#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <input_audio> <start_hh:mm:ss> <duration_hh:mm:ss> <output_audio>"
  exit 1
fi

INPUT="$1"
START="$2"
DURATION="$3"
OUTPUT="$4"

mkdir -p "$(dirname "$OUTPUT")"
ffmpeg -y -hide_banner -loglevel error -ss "$START" -i "$INPUT" -t "$DURATION" -c copy "$OUTPUT"
echo "Wrote: $OUTPUT"
