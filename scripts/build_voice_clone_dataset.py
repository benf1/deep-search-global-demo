#!/usr/bin/env python3
"""Build a higher-fidelity voice-clone dataset from recent Deep Search episodes.

Pipeline:
1) Resolve RSS feed from a podcast page (or use RSS URL directly)
2) Download latest episodes
3) Optionally run Demucs to isolate vocals
4) Transcribe episode audio with faster-whisper
5) Select high-confidence speech chunks and export normalized WAV clips
6) Emit manifests + merged reference files for cloning/evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
import unicodedata


DEFAULT_SOURCE_URL = "https://www.radiofrance.fr/fip/podcasts/deep-search-par-laurent-garnier"


@dataclass
class Episode:
    title: str
    audio_url: str
    pub_date_iso: str
    slug: str


def log(message: str) -> None:
    print(message, flush=True)


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def run_capture(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def http_get(url: str, timeout: int = 60) -> str:
    # Use curl for predictable TLS behavior across local Python installs.
    return run_capture(
        [
            "curl",
            "-L",
            "-sS",
            "--fail",
            "--max-time",
            str(timeout),
            url,
        ]
    )


def download_file(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    run(
        [
            "curl",
            "-L",
            "-sS",
            "--fail",
            "--max-time",
            "600",
            "-o",
            str(tmp),
            url,
        ]
    )
    tmp.replace(dest)


def resolve_rss_url(source_url: str) -> str:
    lowered = source_url.lower()
    if lowered.endswith(".xml") or "rss_" in lowered:
        return source_url

    html = http_get(source_url)
    patterns = [
        r'<link[^>]+href="([^"]+)"[^>]+type="application/rss\+xml"',
        r'<link[^>]+type="application/rss\+xml"[^>]+href="([^"]+)"',
        r"(https?://[^\"'\s>]*rss_[0-9]+\.xml)",
    ]

    rss_url = ""
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if match:
            rss_url = match.group(1).strip()
            break

    if not rss_url:
        raise RuntimeError(
            "Could not find RSS feed link in source page. Pass --source-url with an RSS URL directly."
        )
    return rss_url


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_value.lower()).strip("-")
    return slug or "episode"


def parse_pub_date(raw: str) -> str:
    if not raw:
        return "1970-01-01T00:00:00+00:00"
    try:
        return parsedate_to_datetime(raw).isoformat()
    except Exception:
        return "1970-01-01T00:00:00+00:00"


def load_episodes_from_rss(rss_url: str, limit: int) -> list[Episode]:
    xml_text = http_get(rss_url)
    root = ET.fromstring(xml_text)

    items: list[Episode] = []
    seen_slugs: dict[str, int] = {}
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "Untitled").strip()
        pub_iso = parse_pub_date((item.findtext("pubDate") or "").strip())

        enclosure = item.find("enclosure")
        audio_url = ""
        if enclosure is not None:
            audio_url = (enclosure.attrib.get("url") or "").strip()
        if not audio_url:
            continue

        date_tag = pub_iso[:10].replace("-", "") if pub_iso else "unknown"
        base_slug = f"{date_tag}-{slugify(title)}"
        count = seen_slugs.get(base_slug, 0)
        seen_slugs[base_slug] = count + 1
        slug = base_slug if count == 0 else f"{base_slug}-{count + 1}"

        items.append(
            Episode(
                title=title,
                audio_url=audio_url,
                pub_date_iso=pub_iso,
                slug=slug,
            )
        )

    items.sort(key=lambda ep: ep.pub_date_iso, reverse=True)
    return items[:limit]


def module_available(module_name: str) -> bool:
    try:
        code = (
            "import importlib.util,sys;"
            f"sys.exit(0 if importlib.util.find_spec('{module_name}') else 1)"
        )
        subprocess.run([sys.executable, "-c", code], check=True, stdout=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def ffprobe_duration_seconds(audio_path: Path) -> float:
    output = run_capture(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
    )
    try:
        return float(output)
    except ValueError:
        return 0.0


def maybe_build_vocals(
    input_audio: Path,
    demucs_root: Path,
    demucs_model: str,
    use_demucs: bool,
    require_demucs: bool,
) -> Path:
    if not use_demucs:
        return input_audio

    if not module_available("demucs"):
        if require_demucs:
            raise RuntimeError("Demucs module not found in current Python environment")
        log("[warn] Demucs not available; falling back to original audio")
        return input_audio

    target_mp3 = demucs_root / demucs_model / input_audio.stem / "vocals.mp3"
    target_wav = demucs_root / demucs_model / input_audio.stem / "vocals.wav"

    if target_mp3.exists():
        return target_mp3
    if target_wav.exists():
        return target_wav

    demucs_root.mkdir(parents=True, exist_ok=True)
    try:
        run(
            [
                sys.executable,
                "-m",
                "demucs",
                "--two-stems=vocals",
                "-n",
                demucs_model,
                "--mp3",
                str(input_audio),
                "-o",
                str(demucs_root),
            ]
        )
    except subprocess.CalledProcessError:
        if require_demucs:
            raise
        log("[warn] Demucs separation failed; falling back to original audio")
        return input_audio

    if target_mp3.exists():
        return target_mp3
    if target_wav.exists():
        return target_wav

    if require_demucs:
        raise RuntimeError(f"Demucs finished but vocals file was not found for {input_audio}")
    log("[warn] Demucs output missing; falling back to original audio")
    return input_audio


def maybe_transcribe(
    input_audio: Path,
    out_json: Path,
    out_txt: Path,
    model: str,
    language: str,
    compute_type: str,
    force: bool,
    transcribe_script: Path,
) -> None:
    if out_json.exists() and out_txt.exists() and not force:
        return

    if not transcribe_script.exists():
        raise RuntimeError(f"Missing transcribe script: {transcribe_script}")
    if not module_available("faster_whisper"):
        raise RuntimeError(
            "faster-whisper is not available in the current Python environment. "
            "Install dependencies or run with --download-only first."
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            str(transcribe_script),
            "--input",
            str(input_audio),
            "--output-json",
            str(out_json),
            "--output-txt",
            str(out_txt),
            "--model",
            model,
            "--language",
            language,
            "--compute-type",
            compute_type,
        ]
    )


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def select_candidate_chunks(
    transcript_payload: dict[str, Any],
    min_seg_sec: float,
    max_seg_sec: float,
    min_chars: int,
    max_no_speech_prob: float,
    min_avg_logprob: float,
    min_words_per_sec: float,
    max_words_per_sec: float,
    merge_gap_sec: float,
    max_chunk_sec: float,
) -> list[dict[str, Any]]:
    raw_segments = transcript_payload.get("segments") or []

    filtered: list[dict[str, Any]] = []
    for seg in raw_segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue

        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or 0.0)
        duration = max(0.0, end - start)
        if duration < min_seg_sec or duration > max_seg_sec:
            continue

        if len(text) < min_chars:
            continue

        no_speech_prob = float(seg.get("no_speech_prob") or 0.0)
        avg_logprob = float(seg.get("avg_logprob") or -99.0)
        if no_speech_prob > max_no_speech_prob:
            continue
        if avg_logprob < min_avg_logprob:
            continue

        words = max(1, len(text.split()))
        words_per_sec = words / max(duration, 1e-6)
        if words_per_sec < min_words_per_sec or words_per_sec > max_words_per_sec:
            continue

        filtered.append(
            {
                "start": start,
                "end": end,
                "duration": duration,
                "text": text,
                "no_speech_prob": no_speech_prob,
                "avg_logprob": avg_logprob,
            }
        )

    merged: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for seg in filtered:
        if current is None:
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "texts": [seg["text"]],
                "no_speech_values": [seg["no_speech_prob"]],
                "avg_logprob_values": [seg["avg_logprob"]],
            }
            continue

        gap = seg["start"] - current["end"]
        next_duration = seg["end"] - current["start"]
        if gap <= merge_gap_sec and next_duration <= max_chunk_sec:
            current["end"] = seg["end"]
            current["texts"].append(seg["text"])
            current["no_speech_values"].append(seg["no_speech_prob"])
            current["avg_logprob_values"].append(seg["avg_logprob"])
        else:
            merged.append(current)
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "texts": [seg["text"]],
                "no_speech_values": [seg["no_speech_prob"]],
                "avg_logprob_values": [seg["avg_logprob"]],
            }

    if current is not None:
        merged.append(current)

    scored: list[dict[str, Any]] = []
    for row in merged:
        text = " ".join(row["texts"]).strip()
        duration = float(row["end"] - row["start"])
        if duration <= 0.0:
            continue

        mean_no_speech = sum(row["no_speech_values"]) / len(row["no_speech_values"])
        mean_avg_logprob = sum(row["avg_logprob_values"]) / len(row["avg_logprob_values"])

        conf_no_speech = clamp(1.0 - mean_no_speech, 0.0, 1.0)
        conf_logprob = clamp((mean_avg_logprob - min_avg_logprob) / (0.0 - min_avg_logprob), 0.0, 1.0)
        score = 0.6 * conf_no_speech + 0.4 * conf_logprob

        scored.append(
            {
                "start": row["start"],
                "end": row["end"],
                "duration": duration,
                "text": text,
                "mean_no_speech_prob": round(mean_no_speech, 5),
                "mean_avg_logprob": round(mean_avg_logprob, 5),
                "score": round(score, 5),
            }
        )

    scored.sort(key=lambda x: (x["score"], x["duration"]), reverse=True)
    return scored


def cut_chunk_to_wav(
    source_audio: Path,
    output_wav: Path,
    start_sec: float,
    duration_sec: float,
    sample_rate: int,
) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(source_audio),
            "-t",
            f"{duration_sec:.3f}",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ]
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_reference_audio(
    manifest_rows: list[dict[str, Any]],
    output_dir: Path,
    targets_seconds: list[int],
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[dict[str, Any]] = []

    rows_sorted = sorted(manifest_rows, key=lambda r: (float(r["score"]), float(r["duration_sec"])), reverse=True)
    for target in targets_seconds:
        selected_paths: list[Path] = []
        total = 0.0
        for row in rows_sorted:
            clip = Path(row["clip_path"])
            if not clip.exists():
                continue
            selected_paths.append(clip)
            total += float(row["duration_sec"])
            if total >= target:
                break

        if not selected_paths:
            continue

        concat_file = output_dir / f"reference_{target}s.concat.txt"
        out_wav = output_dir / f"reference_{target}s.wav"

        with concat_file.open("w", encoding="utf-8") as f:
            for p in selected_paths:
                escaped = str(p).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(out_wav),
            ]
        )

        created.append(
            {
                "target_seconds": target,
                "actual_seconds": round(total, 3),
                "segments": len(selected_paths),
                "path": str(out_wav),
            }
        )

    return created


def parse_target_seconds(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid reference seconds token: {token}") from exc
        if value > 0:
            values.append(value)

    if not values:
        raise argparse.ArgumentTypeError("At least one positive target seconds value is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a cleaner voice-clone dataset from Deep Search episodes")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL, help="Podcast page URL or RSS URL")
    parser.add_argument("--episodes", type=int, default=12, help="How many latest episodes to process")
    parser.add_argument("--language", default="fr", help="Whisper language code")
    parser.add_argument("--transcribe-model", default="small", help="faster-whisper model")
    parser.add_argument("--compute-type", default="int8", help="faster-whisper compute type")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Export sample rate for chunk WAVs")

    parser.add_argument("--min-seg-sec", type=float, default=2.5)
    parser.add_argument("--max-seg-sec", type=float, default=12.0)
    parser.add_argument("--min-chars", type=int, default=24)
    parser.add_argument("--max-no-speech-prob", type=float, default=0.4)
    parser.add_argument("--min-avg-logprob", type=float, default=-1.0)
    parser.add_argument("--min-words-per-sec", type=float, default=0.8)
    parser.add_argument("--max-words-per-sec", type=float, default=4.8)
    parser.add_argument("--merge-gap-sec", type=float, default=0.45)
    parser.add_argument("--max-chunk-sec", type=float, default=14.0)

    parser.add_argument("--target-total-seconds", type=int, default=5400, help="Stop once this many chunk seconds are collected")
    parser.add_argument("--max-seconds-per-episode", type=int, default=900, help="Cap chunk seconds per episode for variety")
    parser.add_argument(
        "--reference-seconds",
        type=parse_target_seconds,
        default=parse_target_seconds("600,1800"),
        help="Comma-separated merged reference targets, e.g. 600,1800",
    )

    parser.add_argument("--demucs", action="store_true", default=False, help="Use Demucs vocals isolation before transcription")
    parser.add_argument("--require-demucs", action="store_true", default=False, help="Fail if Demucs is unavailable/fails")
    parser.add_argument("--demucs-model", default="htdemucs", help="Demucs model name")

    parser.add_argument("--force-download", action="store_true", help="Re-download episode audio even if it exists")
    parser.add_argument("--force-transcribe", action="store_true", help="Re-run transcription even if JSON/TXT exists")
    parser.add_argument("--download-only", action="store_true", help="Only download episode audio and write metadata")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = repo_root / "audio" / "raw" / "episodes"
    demucs_dir = repo_root / "audio" / "processed" / "demucs"
    transcript_dir = repo_root / "transcripts" / "episodes"
    dataset_dir = repo_root / "audio" / "dataset"
    chunks_dir = dataset_dir / "chunks"
    manifests_dir = dataset_dir / "manifests"
    refs_dir = dataset_dir / "references"

    transcribe_script = repo_root / "scripts" / "transcribe.py"

    rss_url = resolve_rss_url(args.source_url)
    log(f"Resolved RSS feed: {rss_url}")

    episodes = load_episodes_from_rss(rss_url, args.episodes)
    if not episodes:
        raise RuntimeError("No episodes found in feed")

    episodes_meta_path = manifests_dir / "episodes.selected.json"
    write_json(
        episodes_meta_path,
        [
            {
                "title": ep.title,
                "slug": ep.slug,
                "pub_date_iso": ep.pub_date_iso,
                "audio_url": ep.audio_url,
            }
            for ep in episodes
        ],
    )

    manifest_rows: list[dict[str, Any]] = []
    total_seconds = 0.0

    for episode in episodes:
        if total_seconds >= args.target_total_seconds:
            break

        log(f"\\n[episode] {episode.slug}")
        raw_audio_path = raw_dir / f"{episode.slug}.mp3"

        if args.force_download and raw_audio_path.exists():
            raw_audio_path.unlink()

        log(f"- download: {raw_audio_path.name}")
        download_file(episode.audio_url, raw_audio_path)
        duration = ffprobe_duration_seconds(raw_audio_path)
        log(f"- duration: {duration:.1f}s")

        if args.download_only:
            continue

        segment_source = maybe_build_vocals(
            input_audio=raw_audio_path,
            demucs_root=demucs_dir,
            demucs_model=args.demucs_model,
            use_demucs=args.demucs,
            require_demucs=args.require_demucs,
        )
        log(f"- segment source: {segment_source}")

        transcript_json = transcript_dir / f"{episode.slug}.fr.json"
        transcript_txt = transcript_dir / f"{episode.slug}.fr.txt"
        log("- transcribe")
        maybe_transcribe(
            input_audio=segment_source,
            out_json=transcript_json,
            out_txt=transcript_txt,
            model=args.transcribe_model,
            language=args.language,
            compute_type=args.compute_type,
            force=args.force_transcribe,
            transcribe_script=transcribe_script,
        )

        payload = json.loads(transcript_json.read_text(encoding="utf-8"))
        candidates = select_candidate_chunks(
            transcript_payload=payload,
            min_seg_sec=args.min_seg_sec,
            max_seg_sec=args.max_seg_sec,
            min_chars=args.min_chars,
            max_no_speech_prob=args.max_no_speech_prob,
            min_avg_logprob=args.min_avg_logprob,
            min_words_per_sec=args.min_words_per_sec,
            max_words_per_sec=args.max_words_per_sec,
            merge_gap_sec=args.merge_gap_sec,
            max_chunk_sec=args.max_chunk_sec,
        )
        log(f"- candidates: {len(candidates)}")

        episode_seconds = 0.0
        kept = 0
        for idx, cand in enumerate(candidates, start=1):
            if total_seconds >= args.target_total_seconds:
                break
            if episode_seconds >= args.max_seconds_per_episode:
                break

            clip_id = f"{episode.slug}__{idx:04d}"
            clip_path = chunks_dir / f"{clip_id}.wav"
            cut_chunk_to_wav(
                source_audio=segment_source,
                output_wav=clip_path,
                start_sec=float(cand["start"]),
                duration_sec=float(cand["duration"]),
                sample_rate=args.sample_rate,
            )

            row = {
                "clip_id": clip_id,
                "clip_path": str(clip_path),
                "episode_slug": episode.slug,
                "episode_title": episode.title,
                "episode_pub_date_iso": episode.pub_date_iso,
                "source_path": str(segment_source),
                "source_start_sec": round(float(cand["start"]), 3),
                "source_end_sec": round(float(cand["end"]), 3),
                "duration_sec": round(float(cand["duration"]), 3),
                "score": float(cand["score"]),
                "mean_no_speech_prob": float(cand["mean_no_speech_prob"]),
                "mean_avg_logprob": float(cand["mean_avg_logprob"]),
                "text": cand["text"],
            }
            manifest_rows.append(row)
            total_seconds += float(cand["duration"])
            episode_seconds += float(cand["duration"])
            kept += 1

        log(f"- kept clips: {kept}, kept seconds: {episode_seconds:.1f}")

    manifest_rows.sort(key=lambda r: (r["episode_pub_date_iso"], r["clip_id"]), reverse=True)

    jsonl_path = manifests_dir / "voice_clone_manifest.jsonl"
    csv_path = manifests_dir / "voice_clone_manifest.csv"
    summary_path = manifests_dir / "voice_clone_summary.md"

    write_jsonl(jsonl_path, manifest_rows)
    write_csv(csv_path, manifest_rows)

    references = build_reference_audio(
        manifest_rows=manifest_rows,
        output_dir=refs_dir,
        targets_seconds=args.reference_seconds,
    )

    per_episode: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        key = row["episode_slug"]
        entry = per_episode.setdefault(
            key,
            {
                "title": row["episode_title"],
                "seconds": 0.0,
                "clips": 0,
            },
        )
        entry["seconds"] += float(row["duration_sec"])
        entry["clips"] += 1

    lines = [
        "# Voice Clone Dataset Summary",
        "",
        f"- Source URL: {args.source_url}",
        f"- RSS URL: {rss_url}",
        f"- Episodes requested: {args.episodes}",
        f"- Clips exported: {len(manifest_rows)}",
        f"- Total clip seconds: {round(sum(float(r['duration_sec']) for r in manifest_rows), 1)}",
        f"- Segment source mode: {'demucs vocals' if args.demucs else 'original audio'}",
        "",
        "## Per Episode",
        "",
    ]

    for slug, stats in sorted(per_episode.items(), key=lambda x: x[0], reverse=True):
        lines.append(
            f"- `{slug}`: {stats['clips']} clips, {stats['seconds']:.1f}s ({stats['title']})"
        )

    lines.append("")
    lines.append("## Reference Files")
    lines.append("")
    if references:
        for ref in references:
            lines.append(
                f"- `{ref['path']}`: target {ref['target_seconds']}s, actual {ref['actual_seconds']}s, segments {ref['segments']}"
            )
    else:
        lines.append("- No reference files generated (no clips available)")

    lines.append("")
    lines.append("## Manifests")
    lines.append("")
    lines.append(f"- `{jsonl_path}`")
    lines.append(f"- `{csv_path}`")
    lines.append(f"- `{episodes_meta_path}`")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    log("\\nDone.")
    log(f"- Manifest JSONL: {jsonl_path}")
    log(f"- Manifest CSV:   {csv_path}")
    log(f"- Summary:        {summary_path}")
    if references:
        for ref in references:
            log(f"- Reference:      {ref['path']} ({ref['actual_seconds']}s)")


if __name__ == "__main__":
    main()
