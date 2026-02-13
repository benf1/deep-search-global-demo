# Deep Search Demo Pipeline

This folder contains a fast pipeline for:

1. Downloading the target episode audio.
2. Cutting the first 10 minutes.
3. Transcribing French speech with timestamped Whisper segments.
4. Finding likely spoken interstitial windows for selection.
5. Separating voice/music with Demucs.
6. Creating a localized English narration mix.
7. Generating XTTS v2 voice-clone outputs (with approved permission).

## Files

- `audio/raw/`: full source file
- `audio/segments/`: extracted windows
- `transcripts/`: transcript text and candidate reports
- `scripts/transcribe.py`: Whisper transcription
- `scripts/find_interstitials.py`: candidate region detection
- `scripts/extract_segment.sh`: extract any chosen window
- `scripts/run_pipeline.sh`: one-command end-to-end run

## Run

Use the virtual environment:

```bash
source /Users/benfrankforter/Desktop/deep-search-global-demo/.venv/bin/activate
```

Transcribe first 10 minutes:

```bash
python /Users/benfrankforter/Desktop/deep-search-global-demo/scripts/transcribe.py \
  --input /Users/benfrankforter/Desktop/deep-search-global-demo/audio/segments/deep-search-first-10m.mp3 \
  --output-json /Users/benfrankforter/Desktop/deep-search-global-demo/transcripts/first-10m.fr.json \
  --output-txt /Users/benfrankforter/Desktop/deep-search-global-demo/transcripts/first-10m.fr.txt \
  --model small \
  --language fr
```

Find candidate interstitial windows:

```bash
python /Users/benfrankforter/Desktop/deep-search-global-demo/scripts/find_interstitials.py \
  --input-json /Users/benfrankforter/Desktop/deep-search-global-demo/transcripts/first-10m.fr.json \
  --output-md /Users/benfrankforter/Desktop/deep-search-global-demo/transcripts/first-10m.candidates.md
```

Extract one chosen region:

```bash
bash /Users/benfrankforter/Desktop/deep-search-global-demo/scripts/extract_segment.sh \
  /Users/benfrankforter/Desktop/deep-search-global-demo/audio/segments/deep-search-first-10m.mp3 \
  00:03:20 \
  00:01:45 \
  /Users/benfrankforter/Desktop/deep-search-global-demo/audio/segments/interstitial-A.mp3
```

Run the full prepared pipeline:

```bash
bash /Users/benfrankforter/Desktop/deep-search-global-demo/scripts/run_pipeline.sh
```

Refresh voice quality with ElevenLabs (keeps the same website layout and player file names):

```bash
export ELEVENLABS_API_KEY="your_api_key"
bash /Users/benfrankforter/Desktop/deep-search-global-demo/scripts/run_elevenlabs_voice_refresh.sh
```

Current generated outputs:

- `/Users/benfrankforter/Desktop/deep-search-global-demo/audio/segments/laurent-interstitial-2m45.mp3`
- `/Users/benfrankforter/Desktop/deep-search-global-demo/audio/processed/htdemucs/laurent-interstitial-2m45/vocals.mp3`
- `/Users/benfrankforter/Desktop/deep-search-global-demo/audio/processed/htdemucs/laurent-interstitial-2m45/no_vocals.mp3`
- `/Users/benfrankforter/Desktop/deep-search-global-demo/audio/processed/laurent-interstitial-2m45.en.mix.mp3`
- `/Users/benfrankforter/Desktop/deep-search-global-demo/audio/cloned/xtts-en-full.wav`
- `/Users/benfrankforter/Desktop/deep-search-global-demo/audio/cloned/xtts-en-full.mix.mp3`
- `/Users/benfrankforter/Desktop/deep-search-global-demo/site/index.html`

Local page preview:

```bash
python3 -m http.server 8080 --directory /Users/benfrankforter/Desktop/deep-search-global-demo/site
```

Then open: `http://localhost:8080`

## High-Fidelity Clone Dataset (New)

Build a cleaner Laurent voice dataset from recent Deep Search episodes (10-15 shows recommended).

What this does:

1. Pulls latest episodes from the Deep Search feed/page.
2. Downloads episode audio into `audio/raw/episodes/`.
3. Optionally isolates vocals with Demucs (`--demucs`) to reduce music bleed.
4. Transcribes + filters high-confidence spoken segments.
5. Exports chunk WAVs, manifests, and merged long reference files.

Script:

- `/Users/benfrankforter/Desktop/deep-search-global-demo/scripts/build_voice_clone_dataset.py`

Quick start:

```bash
cd /Users/benfrankforter/Desktop/deep-search-global-demo

# 1) Stage recent episodes only (no ML dependencies required):
python3 scripts/build_voice_clone_dataset.py \
  --source-url "https://www.radiofrance.fr/fip/podcasts/deep-search-par-laurent-garnier" \
  --episodes 12 \
  --download-only

# 2) Full dataset build (requires faster-whisper + ffmpeg, optionally demucs):
python3 scripts/build_voice_clone_dataset.py \
  --source-url "https://www.radiofrance.fr/fip/podcasts/deep-search-par-laurent-garnier" \
  --episodes 12 \
  --demucs \
  --target-total-seconds 5400 \
  --max-seconds-per-episode 900 \
  --reference-seconds 600,1800
```

Outputs:

- `/Users/benfrankforter/Desktop/deep-search-global-demo/audio/dataset/chunks/*.wav`
- `/Users/benfrankforter/Desktop/deep-search-global-demo/audio/dataset/references/reference_600s.wav`
- `/Users/benfrankforter/Desktop/deep-search-global-demo/audio/dataset/references/reference_1800s.wav`
- `/Users/benfrankforter/Desktop/deep-search-global-demo/audio/dataset/manifests/voice_clone_manifest.jsonl`
- `/Users/benfrankforter/Desktop/deep-search-global-demo/audio/dataset/manifests/voice_clone_manifest.csv`
- `/Users/benfrankforter/Desktop/deep-search-global-demo/audio/dataset/manifests/voice_clone_summary.md`

## Note

Voice cloning / impersonation should only be done with explicit permission from the speaker.
