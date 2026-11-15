#!/usr/bin/env python3
"""Check a finished recording session before it costs you anything.

    python3 -m voice.check_takes voice/out/

Catches the boring failures that ruin a voice dataset and are invisible to the ear once the
files are in a folder: a take that is silent, one that clipped, one that was read at triple
speed, a numbered file missing, a file whose length has nothing to do with its line.

Only 16-bit PCM WAV can be inspected without extra dependencies. Anything else is reported
as "cannot inspect" rather than passed — a green tick must mean something.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path


def load_manifest(folder: Path) -> list[dict]:
    for name in ("manifest.jsonl", "voice/out/manifest.jsonl"):
        for path in (folder / name, folder.parent / name, Path(name)):
            if path.is_file():
                return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return []


def inspect(path: Path) -> dict:
    """Duration, peak and silence ratio of a 16-bit PCM WAV."""
    with wave.open(str(path), "rb") as fh:
        channels, width, rate, frames = fh.getnchannels(), fh.getsampwidth(), fh.getframerate(), fh.getnframes()
        raw = fh.readframes(frames)
    if width != 2:
        return {"error": f"{width * 8}-bit — convert to 16-bit PCM to inspect"}
    import array
    samples = array.array("h", raw)
    if channels > 1:
        samples = samples[::channels]
    peak = max((abs(s) for s in samples), default=0)
    loud = sum(1 for s in samples if abs(s) > 400)          # crude speech-vs-silence split
    return {
        "duration": round(frames / float(rate), 2),
        "rate": rate,
        "channels": channels,
        "peak": peak,
        "peak_ratio": round(peak / 32768.0, 3),
        "speech_ratio": round(loud / max(1, len(samples)), 3),
        "clipped": peak >= 32700,
    }


def check(folder: Path) -> tuple[list[str], dict]:
    problems: list[str] = []
    manifest = load_manifest(folder)
    by_id = {item.get("id", ""): item for item in manifest}
    wavs = sorted(folder.glob("*.wav"))
    if not wavs:
        return [f"no .wav files in {folder}"], {"files": 0}

    stats = {"files": len(wavs), "seconds": 0.0, "lines": len(manifest)}
    seen_ids: set[str] = set()
    for wav in wavs:
        item = next((v for k, v in by_id.items() if wav.stem == k or wav.stem == f"{k}"), None)
        try:
            info = inspect(wav)
        except (wave.Error, EOFError) as exc:
            problems.append(f"{wav.name}: not a readable WAV ({exc})")
            continue
        if "error" in info:
            problems.append(f"{wav.name}: {info['error']}")
            continue
        seen_ids.add(wav.stem)
        stats["seconds"] += info["duration"]
        if info["speech_ratio"] < 0.15:
            problems.append(f"{wav.name}: mostly silence ({info['speech_ratio']:.0%} speech) — "
                            "wrong input device, or a dead take?")
        if info["clipped"]:
            problems.append(f"{wav.name}: clipped (peak at full scale) — the mic was too hot")
        elif info["peak_ratio"] < 0.02:
            problems.append(f"{wav.name}: very quiet (peak {info['peak_ratio']:.1%}) — move closer")
        if item:
            expected = float(item.get("est_seconds") or 0)
            words = int(item.get("words") or 0)
            if expected and words >= 4:
                ratio = info["duration"] / expected
                if ratio < 0.45:
                    problems.append(f"{wav.name}: {info['duration']:.1f}s for {words} words — "
                                    "far faster than the line, wrong file?")
                elif ratio > 2.2:
                    problems.append(f"{wav.name}: {info['duration']:.1f}s for {words} words — "
                                    "long pause or a run-on take?")

    missing = [i["id"] for i in manifest if i.get("id") and i["id"] not in seen_ids]
    for mid in missing[:10]:
        problems.append(f"missing audio for line {mid} ({by_id[mid]['text'][:48]}…)")
    if len(missing) > 10:
        problems.append(f"… and {len(missing) - 10} more missing lines")

    stats["missing"] = len(missing)
    stats["hours"] = round(stats["seconds"] / 3600, 2)
    return problems, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default="voice/out")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        print(f"No such folder: {folder}")
        return 1
    problems, stats = check(folder)
    print(f"{stats.get('files', 0)} file(s), {stats.get('hours', 0):.2f} h of audio, "
          f"{stats.get('lines', 0)} script line(s), {stats.get('missing', 0)} missing")
    if problems:
        print(f"\n{len(problems)} thing(s) to fix:")
        for p in problems:
            print("  -", p)
        print("\nFix these before training a voice on this set — every one of them teaches the "
              "model a mistake it cannot unlearn.")
        return 1
    print("\nClean. This is enough to fine-tune a voice, if you have the hours "
          "(see VOICE-PLAN.md for how many you actually need).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
