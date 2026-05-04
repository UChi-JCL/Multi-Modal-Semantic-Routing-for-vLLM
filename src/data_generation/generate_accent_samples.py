#!/usr/bin/env python3
"""
Generate accented English speech samples using Qwen3-TTS CustomVoice.

Strategy: Use non-English-native voice profiles to speak English text.
The TTS model produces accented English that reflects the speaker's native
language phonology — exactly what we need for accent classification training.

Voices and their native languages:
  - vivian, serena, uncle_fu: Chinese (Mandarin)
  - dylan: Chinese (Beijing dialect)
  - eric: Chinese (Sichuan dialect)
  - ono_anna: Japanese
  - sohee: Korean
  - ryan, aiden: English (native — control group)
"""

import argparse
import json
import time
from pathlib import Path

import httpx

# 10 diverse English sentences covering different phonetic patterns
SENTENCES = [
    "The weather forecast predicts heavy rain throughout the weekend.",
    "Could you please schedule a meeting for three thirty tomorrow afternoon?",
    "I've been studying artificial intelligence and machine learning for years.",
    "The restaurant on Fifth Avenue serves the best chocolate cake in town.",
    "She drove through the countryside enjoying the beautiful autumn leaves.",
    "We need to discuss the quarterly budget report before the deadline.",
    "The children were playing basketball in the park after school.",
    "Please remember to pick up some groceries on your way home tonight.",
    "The professor explained the theory of relativity in simple terms.",
    "Would you rather travel to Japan or visit the beaches in Thailand?",
]

# Voice → accent label mapping
VOICE_ACCENTS = {
    "vivian": "mandarin",
    "serena": "mandarin",
    "uncle_fu": "mandarin",
    "dylan": "mandarin_beijing",
    "eric": "mandarin_sichuan",
    "ono_anna": "japanese",
    "sohee": "korean",
    "ryan": "native_english",
    "aiden": "native_english",
}


def generate_speech(api_base: str, text: str, voice: str, output_path: str) -> float:
    """Generate speech and save to file. Returns generation time in seconds."""
    t0 = time.time()
    with httpx.Client(timeout=120.0, verify=False) as client:
        resp = client.post(
            f"{api_base}/v1/audio/speech",
            json={
                "input": text,
                "voice": voice,
                "response_format": "wav",
            },
        )
        resp.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(resp.content)

    elapsed = time.time() - t0
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Generate accented English speech samples")
    parser.add_argument(
        "--api-base",
        default="https://qwen3-tts-multi-modal-semantic-routing-for-vllm-d266df.apps.shift.nerc.mghpcc.org",
        help="vLLM-Omni API base URL",
    )
    parser.add_argument("--output-dir", default="data/synthetic/pilot", help="Output directory for generated samples")
    parser.add_argument(
        "--voices", nargs="+", default=list(VOICE_ACCENTS.keys()), help="Voices to generate (default: all)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    total = len(args.voices) * len(SENTENCES)
    count = 0

    for voice in args.voices:
        accent = VOICE_ACCENTS[voice]
        voice_dir = output_dir / accent / voice
        voice_dir.mkdir(parents=True, exist_ok=True)

        for i, text in enumerate(SENTENCES):
            count += 1
            filename = f"sentence_{i:02d}.wav"
            output_path = voice_dir / filename

            print(f"[{count}/{total}] {voice} ({accent}) — sentence {i}...", end=" ", flush=True)

            try:
                elapsed = generate_speech(args.api_base, text, voice, str(output_path))
                size_kb = output_path.stat().st_size / 1024
                print(f"OK ({elapsed:.1f}s, {size_kb:.0f}KB)")

                manifest.append(
                    {
                        "voice": voice,
                        "accent": accent,
                        "sentence_id": i,
                        "text": text,
                        "file": str(output_path.relative_to(output_dir)),
                        "generation_time_s": round(elapsed, 2),
                        "file_size_kb": round(size_kb, 1),
                    }
                )
            except Exception as e:
                print(f"FAILED: {e}")
                manifest.append(
                    {
                        "voice": voice,
                        "accent": accent,
                        "sentence_id": i,
                        "text": text,
                        "file": None,
                        "error": str(e),
                    }
                )

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")
    print(f"Generated {sum(1 for m in manifest if m.get('file'))} / {total} samples")


if __name__ == "__main__":
    main()
