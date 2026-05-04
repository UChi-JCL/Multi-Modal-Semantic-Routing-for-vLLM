#!/usr/bin/env python3
"""
Generate accented English speech dataset for accent classification training.

Scenarios: Restaurant, Hospital, Bank, Telecom (T-Mobile style) customer calls.
Models:
  - Qwen3-TTS (localhost:8091): Mandarin + Japanese accents
  - Voxtral TTS (localhost:8092): Hindi + Spanish + Italian accents
  + Native English controls from both models
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

# Customer service scenario sentences
SENTENCES = {
    "restaurant": [
        "Hi, I'd like to make a reservation for four people this Friday evening.",
        "Could you tell me if you have any vegetarian options on the menu?",
        "We have a food allergy in our party. Does the pasta contain any nuts?",
        "I placed an order for delivery about forty five minutes ago and it still hasn't arrived.",
        "Can I change my order? I'd like to add a side of garlic bread please.",
        "What time does the kitchen close tonight? We might be running a little late.",
        "I'm calling to confirm our reservation under the name Johnson for seven thirty.",
        "Do you offer any lunch specials during the week? We're looking for something affordable.",
        "The steak I ordered was supposed to be medium rare but it came out well done.",
        "Could we get a table near the window? It's my wife's birthday and she loves the view.",
    ],
    "hospital": [
        "I need to schedule a follow up appointment with Doctor Chen for next week.",
        "My insurance card number is on file. Can you verify the coverage for this procedure?",
        "I've been experiencing chest pain and shortness of breath since yesterday morning.",
        "Can you transfer me to the pharmacy? I need to refill my prescription for blood pressure medication.",
        "What are the visiting hours for patients in the intensive care unit?",
        "I received a bill that doesn't match what my insurance should have covered.",
        "Is there a specialist available for an urgent referral? My primary care doctor recommended it.",
        "I need to cancel my appointment scheduled for Thursday and reschedule for the following week.",
        "Can you tell me what documents I need to bring for my first visit to the clinic?",
        "The lab results should be ready by now. Can someone from the doctor's office call me back?",
    ],
    "bank": [
        "I'd like to open a new savings account. What are your current interest rates?",
        "There's a charge on my statement that I don't recognize. Can you help me dispute it?",
        "I need to transfer money from my checking account to my savings account.",
        "My debit card was declined at the store even though I have sufficient funds.",
        "Can I set up automatic payments for my mortgage from my checking account?",
        "I'm planning to apply for a home loan. What documents do I need to prepare?",
        "Someone may have accessed my account without authorization. I need to report fraud.",
        "What's the process for ordering new checks? I've almost run out of my current book.",
        "I'd like to increase the daily withdrawal limit on my debit card temporarily.",
        "Can you explain the difference between your premium and standard checking accounts?",
    ],
    "telecom": [
        "I'm calling about my phone bill. It's much higher than usual this month.",
        "I'd like to upgrade my plan to include unlimited data. What options do you have?",
        "My internet connection has been dropping every few hours for the past three days.",
        "Can I add an international calling package? I need to make calls to India regularly.",
        "I want to trade in my old phone and get the new model. How does the promotion work?",
        "There's no signal at my house even though your coverage map says it should be fine.",
        "I need to transfer my phone number from another carrier. How long does that take?",
        "Can you help me set up a family plan? I need to add three additional lines.",
        "I was promised a discount when I renewed my contract but it's not showing on my bill.",
        "My phone screen is cracked. Does my protection plan cover screen replacement?",
    ],
}

# Model → voice → accent mapping
QWEN3_VOICES = {
    "vivian": {"accent": "mandarin", "native_lang": "Chinese (Mandarin)"},
    "serena": {"accent": "mandarin", "native_lang": "Chinese (Mandarin)"},
    "uncle_fu": {"accent": "mandarin", "native_lang": "Chinese (Mandarin)"},
    "dylan": {"accent": "mandarin", "native_lang": "Chinese (Beijing)"},
    "ono_anna": {"accent": "japanese", "native_lang": "Japanese"},
    "aiden": {"accent": "native_english", "native_lang": "English"},
}

VOXTRAL_VOICES = {
    "hi_female": {"accent": "hindi", "native_lang": "Hindi"},
    "hi_male": {"accent": "hindi", "native_lang": "Hindi"},
    "es_female": {"accent": "spanish", "native_lang": "Spanish"},
    "es_male": {"accent": "spanish", "native_lang": "Spanish"},
    "it_female": {"accent": "italian", "native_lang": "Italian"},
    "it_male": {"accent": "italian", "native_lang": "Italian"},
    "neutral_male": {"accent": "native_english", "native_lang": "English"},
    "casual_female": {"accent": "native_english", "native_lang": "English"},
}


def generate_speech(api_base: str, text: str, voice: str, output_path: str, instructions: str = None) -> float:
    """Generate speech and save to file. Returns generation time in seconds."""
    t0 = time.time()
    payload = {
        "input": text,
        "voice": voice,
        "response_format": "wav",
    }
    if instructions:
        payload["instructions"] = instructions

    with httpx.Client(timeout=180.0, verify=False) as client:
        resp = client.post(f"{api_base}/v1/audio/speech", json=payload)
        resp.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(resp.content)

    return time.time() - t0


def run_generation(
    api_base: str, voices: dict, model_name: str, run_id: str, output_base: Path, instructions: str = None
):
    """Generate all sentences for all voices using a given model."""
    manifest = []
    all_sentences = []
    for scenario, sents in SENTENCES.items():
        for i, s in enumerate(sents):
            all_sentences.append((scenario, i, s))

    total = len(voices) * len(all_sentences)
    count = 0

    for voice, info in voices.items():
        accent = info["accent"]
        voice_dir = output_base / accent / voice
        voice_dir.mkdir(parents=True, exist_ok=True)

        for scenario, sent_idx, text in all_sentences:
            count += 1
            filename = f"{scenario}_{sent_idx:02d}.wav"
            output_path = voice_dir / filename

            print(f"[{count}/{total}] {voice} ({accent}) — {scenario}_{sent_idx:02d}...", end=" ", flush=True)
            try:
                elapsed = generate_speech(api_base, text, voice, str(output_path), instructions)
                size_kb = output_path.stat().st_size / 1024
                print(f"OK ({elapsed:.1f}s, {size_kb:.0f}KB)")

                manifest.append(
                    {
                        "voice": voice,
                        "accent": accent,
                        "native_lang": info["native_lang"],
                        "model": model_name,
                        "scenario": scenario,
                        "sentence_id": sent_idx,
                        "text": text,
                        "file": str(output_path.relative_to(output_base)),
                        "instructions": instructions,
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
                        "model": model_name,
                        "scenario": scenario,
                        "sentence_id": sent_idx,
                        "text": text,
                        "file": None,
                        "error": str(e),
                    }
                )

    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-api", default="http://localhost:8091")
    parser.add_argument("--voxtral-api", default="http://localhost:8092")
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--instructions", default=None, help="Optional instruction for Qwen3 voices (e.g. calm tone)")
    args = parser.parse_args()

    timestamp = datetime.utcnow().strftime("%Y%m%d")

    # Find next run number
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in output_base.iterdir() if d.is_dir()]

    # Qwen3 run
    qwen_num = len([e for e in existing if e.startswith("qwen3tts_")]) + 1
    qwen_run_id = f"qwen3tts_{timestamp}_{qwen_num:03d}"
    qwen_dir = output_base / qwen_run_id
    qwen_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"QWEN3-TTS Generation: {qwen_run_id}")
    print(f"{'='*60}\n")

    qwen_manifest = run_generation(
        args.qwen_api, QWEN3_VOICES, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", qwen_run_id, qwen_dir, args.instructions
    )

    # Save Qwen3 run config + manifest
    with open(qwen_dir / "run_config.json", "w") as f:
        json.dump(
            {
                "run_id": qwen_run_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "serving": "vLLM-Omni v0.18.0",
                "hardware": "NVIDIA H100 80GB HBM3",
                "voices": QWEN3_VOICES,
                "instructions": args.instructions,
                "scenarios": list(SENTENCES.keys()),
                "sentences_per_scenario": 10,
                "total_sentences": 40,
                "total_samples": len(qwen_manifest),
                "successful": sum(1 for m in qwen_manifest if m.get("file")),
            },
            f,
            indent=2,
        )

    with open(qwen_dir / "manifest.json", "w") as f:
        json.dump(qwen_manifest, f, indent=2)

    # Voxtral run
    voxtral_num = len([e for e in existing if e.startswith("voxtral_")]) + 1
    voxtral_run_id = f"voxtral_{timestamp}_{voxtral_num:03d}"
    voxtral_dir = output_base / voxtral_run_id
    voxtral_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"VOXTRAL TTS Generation: {voxtral_run_id}")
    print(f"{'='*60}\n")

    voxtral_manifest = run_generation(
        args.voxtral_api, VOXTRAL_VOICES, "mistralai/Voxtral-4B-TTS-2603", voxtral_run_id, voxtral_dir
    )

    # Save Voxtral run config + manifest
    with open(voxtral_dir / "run_config.json", "w") as f:
        json.dump(
            {
                "run_id": voxtral_run_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model": "mistralai/Voxtral-4B-TTS-2603",
                "serving": "vLLM-Omni v0.18.0",
                "hardware": "NVIDIA H100 80GB HBM3",
                "voices": VOXTRAL_VOICES,
                "scenarios": list(SENTENCES.keys()),
                "sentences_per_scenario": 10,
                "total_sentences": 40,
                "total_samples": len(voxtral_manifest),
                "successful": sum(1 for m in voxtral_manifest if m.get("file")),
            },
            f,
            indent=2,
        )

    with open(voxtral_dir / "manifest.json", "w") as f:
        json.dump(voxtral_manifest, f, indent=2)

    # Summary
    qwen_ok = sum(1 for m in qwen_manifest if m.get("file"))
    voxtral_ok = sum(1 for m in voxtral_manifest if m.get("file"))
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Qwen3:   {qwen_ok}/{len(qwen_manifest)} samples → {qwen_run_id}/")
    print(f"Voxtral: {voxtral_ok}/{len(voxtral_manifest)} samples → {voxtral_run_id}/")
    print(f"Total:   {qwen_ok + voxtral_ok} samples across {len(QWEN3_VOICES) + len(VOXTRAL_VOICES)} voices")


if __name__ == "__main__":
    main()
