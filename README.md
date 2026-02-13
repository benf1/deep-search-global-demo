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
source /Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/.venv/bin/activate
```

Transcribe first 10 minutes:

```bash
python /Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/scripts/transcribe.py \
  --input /Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/audio/segments/deep-search-first-10m.mp3 \
  --output-json /Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/transcripts/first-10m.fr.json \
  --output-txt /Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/transcripts/first-10m.fr.txt \
  --model small \
  --language fr
```

Find candidate interstitial windows:

```bash
python /Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/scripts/find_interstitials.py \
  --input-json /Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/transcripts/first-10m.fr.json \
  --output-md /Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/transcripts/first-10m.candidates.md
```

Extract one chosen region:

```bash
bash /Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/scripts/extract_segment.sh \
  /Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/audio/segments/deep-search-first-10m.mp3 \
  00:03:20 \
  00:01:45 \
  /Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/audio/segments/interstitial-A.mp3
```

Run the full prepared pipeline:

```bash
bash /Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/scripts/run_pipeline.sh
```

Refresh voice quality with ElevenLabs (keeps the same website layout and player file names):

```bash
export ELEVENLABS_API_KEY="your_api_key"
bash /Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/scripts/run_elevenlabs_voice_refresh.sh
```

Current generated outputs:

- `/Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/audio/segments/laurent-interstitial-2m45.mp3`
- `/Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/audio/processed/htdemucs/laurent-interstitial-2m45/vocals.mp3`
- `/Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/audio/processed/htdemucs/laurent-interstitial-2m45/no_vocals.mp3`
- `/Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/audio/processed/laurent-interstitial-2m45.en.mix.mp3`
- `/Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/audio/cloned/xtts-en-full.wav`
- `/Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/audio/cloned/xtts-en-full.mix.mp3`
- `/Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/site/index.html`

Local page preview:

```bash
python3 -m http.server 8080 --directory /Users/benfrankforter/Desktop/xvenao/deep-search-voice-demo/site
```

Then open: `http://localhost:8080`

## Note

Voice cloning / impersonation should only be done with explicit permission from the speaker.
