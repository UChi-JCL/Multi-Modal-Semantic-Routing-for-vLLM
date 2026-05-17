#!/usr/bin/env python3
"""Convert repo-local manifest.json into NeMo JSONL format for ASR training.

The synthetic data-gen scripts write per-run manifests with fields
{voice, accent, native_lang?, model?, scenario?, sentence_id, text, file, ...}.
NeMo ASR training expects one JSON per line with at minimum
{audio_filepath, duration, text}. Extra fields are ignored by NeMo but useful
downstream (per-accent WER, voice leakage checks).

Optional resampling: Nemotron-Streaming expects 16 kHz; our TTS emits 24 kHz.
When --resample-to is set, audio is resampled once into a shadow directory
keyed by the original relative path; repeat invocations are idempotent on
already-resampled files.

Splitting: deterministic by MD5-hash of (voice, sentence_id) so reruns keep
the same assignment. Three-way (train/val/test) by default; per-accent mode
applies the same split within each accent subset.

Status: v0. See docs/training-runbook.md for the end-to-end training flow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_PUNCT_RE = re.compile(r"[^a-z0-9 ']")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str, *, keep_punctuation: bool) -> str:
    text = text.lower()
    if not keep_punctuation:
        text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _split_bucket(key: str, train: float, val: float) -> str:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
    bucket = int(digest, 16) % 10000 / 10000.0
    if bucket < train:
        return "train"
    if bucket < train + val:
        return "val"
    return "test"


def _load_manifest(path: Path) -> list[dict]:
    with path.open() as f:
        records = json.load(f)
    skipped = sum(1 for r in records if r.get("file") is None)
    if skipped:
        print(f"  skipped {skipped} failed entries (file=None)", flush=True)
    return [r for r in records if r.get("file") is not None]


def _read_duration_and_sr(wav: Path) -> tuple[float, int]:
    info = sf.info(str(wav))
    return info.frames / info.samplerate, info.samplerate


def _resample_if_needed(src: Path, dst: Path, target_sr: int) -> tuple[float, str]:
    """Returns (duration_seconds, status) where status is 'wrote' | 'skipped'."""
    if dst.exists():
        info = sf.info(str(dst))
        if info.samplerate == target_sr:
            return info.frames / info.samplerate, "skipped"

    data, orig_sr = sf.read(str(src))
    if orig_sr == target_sr:
        resampled = data
    else:
        # resample_poly requires integer up/down factors; GCD handles arbitrary rates cleanly.
        from math import gcd

        g = gcd(orig_sr, target_sr)
        up, down = target_sr // g, orig_sr // g
        resampled = resample_poly(data, up, down)

    # Preserve mono/stereo shape; soundfile expects (frames,) or (frames, channels).
    if resampled.dtype != np.float32 and resampled.dtype != np.float64:
        resampled = resampled.astype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), resampled, target_sr, subtype="PCM_16")
    dur = (resampled.shape[0] if resampled.ndim == 1 else resampled.shape[0]) / target_sr
    return dur, "wrote"


def _build_record(
    entry: dict,
    run_dir: Path,
    resampled_dir: Path | None,
    target_sr: int | None,
    *,
    keep_punctuation: bool,
    minimal: bool,
) -> dict | None:
    rel = entry["file"]
    src_path = (run_dir / rel).resolve()
    if not src_path.exists():
        print(f"  FAILED: source audio missing: {src_path}", flush=True)
        return None

    if resampled_dir is not None and target_sr is not None:
        dst_path = (resampled_dir / rel).resolve()
        dur, _status = _resample_if_needed(src_path, dst_path, target_sr)
        audio_path = dst_path
    else:
        dur, _sr = _read_duration_and_sr(src_path)
        audio_path = src_path

    text = _normalize_text(entry.get("text", ""), keep_punctuation=keep_punctuation)
    rec: dict = {
        "audio_filepath": str(audio_path),
        "duration": round(float(dur), 3),
        "text": text,
    }
    if not minimal:
        if entry.get("accent") is not None:
            rec["accent"] = entry["accent"]
        if entry.get("voice") is not None:
            rec["voice"] = entry["voice"]
    return rec


def _split_records(
    pairs: list[tuple[dict, dict]],
    mode: str,
    train: float,
    val: float,
) -> dict[str, list[dict]]:
    """Split records into buckets.

    pairs: list of (source_entry, nemo_record). Splitting uses the source entry's
    voice + sentence_id so `--minimal` (which drops those fields from the NeMo
    record) doesn't affect partitioning.
    """
    if mode == "none":
        return {"all": [r for _src, r in pairs]}

    def _bucket_for(src: dict) -> str:
        key = f"{src.get('voice', '?')}::{src.get('sentence_id', '?')}"
        return _split_bucket(key, train, val)

    if mode == "three-way":
        splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        for src, rec in pairs:
            splits[_bucket_for(src)].append(rec)
        return splits

    if mode == "per-accent":
        by_accent: dict[str, list[tuple[dict, dict]]] = {}
        for src, rec in pairs:
            by_accent.setdefault(src.get("accent", "unknown"), []).append((src, rec))
        out: dict[str, list[dict]] = {}
        for accent, sub_pairs in by_accent.items():
            sub: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
            for src, rec in sub_pairs:
                sub[_bucket_for(src)].append(rec)
            for split_name, split_recs in sub.items():
                if not split_recs:
                    print(f"  WARN: accent={accent} split={split_name} is empty", flush=True)
                out[f"{accent}.{split_name}"] = split_recs
        return out

    raise ValueError(f"unknown split mode: {mode}")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_summary(
    path: Path,
    splits: dict[str, list[dict]],
    accent_lookup: dict[str, str],
    *,
    source_manifest: Path,
    split_mode: str,
) -> None:
    """Write summary.json. accent_lookup maps audio_filepath -> accent (from source)."""
    summary: dict = {
        "source_manifest": str(source_manifest),
        "split_mode": split_mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "splits": {},
        "overall": {},
    }
    all_recs: list[dict] = []
    for name, recs in splits.items():
        by_accent: dict[str, int] = {}
        for r in recs:
            accent = accent_lookup.get(r["audio_filepath"], "unknown")
            by_accent[accent] = by_accent.get(accent, 0) + 1
        summary["splits"][name] = {
            "n": len(recs),
            "total_duration_s": round(sum(r["duration"] for r in recs), 2),
            "by_accent": by_accent,
        }
        all_recs.extend(recs)
    by_accent_all: dict[str, int] = {}
    for r in all_recs:
        accent = accent_lookup.get(r["audio_filepath"], "unknown")
        by_accent_all[accent] = by_accent_all.get(accent, 0) + 1
    summary["overall"] = {
        "n": len(all_recs),
        "total_duration_s": round(sum(r["duration"] for r in all_recs), 2),
        "by_accent": by_accent_all,
    }
    with path.open("w") as f:
        json.dump(summary, f, indent=2)


def _output_paths(output_stem: Path, split_mode: str, splits: dict[str, list[dict]]) -> dict[str, Path]:
    """Map split name to output file path based on mode."""
    if split_mode == "none":
        return {"all": output_stem}

    # strip trailing .jsonl if the user supplied it to avoid foo.jsonl.train.jsonl
    stem = output_stem.with_suffix("") if output_stem.suffix == ".jsonl" else output_stem
    return {name: Path(f"{stem}.{name}.jsonl") for name in splits}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input-manifest", type=Path, required=True, help="Path to repo manifest.json from a data-gen run."
    )
    parser.add_argument(
        "--output-manifest", type=Path, required=True, help="Output JSONL path (stem for multi-output modes)."
    )
    parser.add_argument(
        "--resample-to", type=int, default=None, help="Target sample rate in Hz (e.g. 16000). Triggers resampling."
    )
    parser.add_argument(
        "--resampled-audio-dir",
        type=Path,
        default=None,
        help="Destination directory for resampled audio (required if --resample-to).",
    )
    parser.add_argument(
        "--split-mode",
        choices=("none", "three-way", "per-accent"),
        default="three-way",
        help="How to partition records.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split fraction.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split fraction.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split fraction.")
    parser.add_argument("--keep-punctuation", action="store_true", help="Preserve punctuation in text.")
    parser.add_argument(
        "--minimal", action="store_true", help="Emit only NeMo core fields (audio_filepath, duration, text)."
    )
    parser.add_argument(
        "--no-write-summary", dest="write_summary", action="store_false", help="Skip writing the .summary.json file."
    )
    parser.set_defaults(write_summary=True)

    args = parser.parse_args()

    if args.resample_to and args.resampled_audio_dir is None:
        parser.error("--resample-to requires --resampled-audio-dir")

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not 0.999 <= ratio_sum <= 1.001:
        parser.error(f"train/val/test ratios must sum to 1.0, got {ratio_sum:.3f}")

    run_dir = args.input_manifest.parent.resolve()
    resampled_dir = args.resampled_audio_dir.resolve() if args.resampled_audio_dir else None

    print(f"Reading {args.input_manifest}", flush=True)
    entries = _load_manifest(args.input_manifest)
    print(f"  loaded {len(entries)} successful entries", flush=True)

    pairs: list[tuple[dict, dict]] = []
    t0 = time.time()
    for i, entry in enumerate(entries, 1):
        voice = entry.get("voice", "?")
        accent = entry.get("accent", "?")
        print(f"[{i}/{len(entries)}] {voice} ({accent})", end=" ", flush=True)
        try:
            rec = _build_record(
                entry,
                run_dir,
                resampled_dir,
                args.resample_to,
                keep_punctuation=args.keep_punctuation,
                minimal=args.minimal,
            )
            if rec is None:
                print("SKIPPED")
                continue
            pairs.append((entry, rec))
            print(f"OK ({rec['duration']:.2f}s)")
        except Exception as e:  # noqa: BLE001 — surface per-file errors without aborting
            print(f"FAILED: {e}")
    elapsed = time.time() - t0
    print(f"\nProcessed {len(pairs)}/{len(entries)} in {elapsed:.1f}s", flush=True)

    splits = _split_records(pairs, args.split_mode, args.train_ratio, args.val_ratio)

    output_paths = _output_paths(args.output_manifest, args.split_mode, splits)
    for name, recs in splits.items():
        out = output_paths[name]
        _write_jsonl(out, recs)
        total_dur = sum(r["duration"] for r in recs)
        print(f"  wrote {len(recs):5d} records ({total_dur:.1f}s) -> {out}", flush=True)

    if args.write_summary:
        if args.split_mode == "none":
            summary_path = args.output_manifest.with_suffix(".summary.json")
        else:
            stem = (
                args.output_manifest.with_suffix("")
                if args.output_manifest.suffix == ".jsonl"
                else args.output_manifest
            )
            summary_path = Path(f"{stem}.summary.json")
        accent_lookup = {rec["audio_filepath"]: src.get("accent", "unknown") for src, rec in pairs}
        _write_summary(
            summary_path,
            splits,
            accent_lookup,
            source_manifest=args.input_manifest,
            split_mode=args.split_mode,
        )
        print(f"  wrote summary -> {summary_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
