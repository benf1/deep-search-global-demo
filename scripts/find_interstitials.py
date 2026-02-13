#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_segments(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("segments", [])


def merge_speech_regions(
    segments: list[dict], max_gap_s: float = 2.5, min_chars: int = 25
) -> list[dict]:
    regions: list[dict] = []
    current = None

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        if current is None:
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "texts": [text],
                "char_count": len(text),
            }
            continue

        gap = seg["start"] - current["end"]
        if gap <= max_gap_s:
            current["end"] = seg["end"]
            current["texts"].append(text)
            current["char_count"] += len(text)
        else:
            if current["char_count"] >= min_chars:
                regions.append(current)
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "texts": [text],
                "char_count": len(text),
            }

    if current and current["char_count"] >= min_chars:
        regions.append(current)
    return regions


def format_ts(seconds: float) -> str:
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find likely spoken interstitial windows from a Whisper JSON transcript."
    )
    parser.add_argument("--input-json", required=True, help="Transcription JSON path")
    parser.add_argument(
        "--output-md",
        required=True,
        help="Output markdown report path",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=15.0,
        help="Minimum region duration in seconds",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=180.0,
        help="Maximum region duration in seconds",
    )
    args = parser.parse_args()

    segments = load_segments(args.input_json)
    regions = merge_speech_regions(segments)

    ranked = []
    for region in regions:
        duration = region["end"] - region["start"]
        if duration < args.min_duration or duration > args.max_duration:
            continue
        words = sum(len(t.split()) for t in region["texts"])
        density = words / max(duration, 1.0)
        score = density + min(len(region["texts"]) * 0.03, 0.5)
        preview = " ".join(region["texts"])[:220].strip()
        ranked.append(
            {
                "start": region["start"],
                "end": region["end"],
                "duration": duration,
                "words": words,
                "score": score,
                "preview": preview,
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    top = ranked[:10]

    out_path = Path(args.output_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Candidate Interstitials", ""]
    if not top:
        lines.append("No candidate interstitial regions found with current thresholds.")
    else:
        for i, r in enumerate(top, start=1):
            lines.append(
                f"{i}. `{format_ts(r['start'])} -> {format_ts(r['end'])}`"
                f" ({r['duration']:.1f}s, {r['words']} words, score {r['score']:.2f})"
            )
            lines.append(f"   - Preview: {r['preview']}")
            lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {len(top)} candidates to {out_path}")


if __name__ == "__main__":
    main()
