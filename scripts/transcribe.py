#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from faster_whisper import WhisperModel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with faster-whisper and keep timestamped segments."
    )
    parser.add_argument("--input", required=True, help="Input audio path")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    parser.add_argument("--output-txt", required=True, help="Output text path")
    parser.add_argument("--model", default="small", help="Whisper model size")
    parser.add_argument("--language", default="fr", help="Language code")
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Compute type (int8/float16/float32). int8 is fastest on CPU.",
    )
    args = parser.parse_args()

    model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type)
    segments, info = model.transcribe(
        args.input,
        language=args.language,
        vad_filter=True,
        beam_size=5,
        best_of=5,
        temperature=0.0,
    )

    out_json = Path(args.output_json)
    out_txt = Path(args.output_txt)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    segment_rows = []
    txt_lines = []
    for seg in segments:
        row = {
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "avg_logprob": seg.avg_logprob,
            "no_speech_prob": seg.no_speech_prob,
        }
        if row["text"]:
            segment_rows.append(row)
            txt_lines.append(f"[{row['start']:.3f} -> {row['end']:.3f}] {row['text']}")

    payload = {
        "detected_language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": segment_rows,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_txt.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(segment_rows)} segments to {out_json}")
    print(f"Wrote transcript text to {out_txt}")


if __name__ == "__main__":
    main()
