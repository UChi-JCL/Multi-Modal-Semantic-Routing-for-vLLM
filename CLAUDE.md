# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project generating synthetic accented English speech data using multilingual TTS models, then building accent-aware STT routing for vLLM. University of Chicago, Junchen Jiang Research Group (NSF NAIRR funded).

**Core insight**: TTS models trained on non-English languages naturally produce accented English, providing free ground-truth transcriptions at scale.

**Four research phases**:
1. Data generation pipeline (current phase)
2. Accent classifier training
3. Accent-adaptive STT fine-tuning (Whisper + LoRA)
4. vLLM semantic routing integration

## Session Startup

```bash
cat progress.txt | tail -40            # Last session state
oc whoami 2>/dev/null || echo "Need to login"  # Cluster access
oc get pods -l app=qwen3-tts           # Model serving status
```

## Development Environment

One-time setup:

```bash
uv sync                         # creates .venv from pyproject.toml + uv.lock
uv run pre-commit install       # registers git hook; runs ruff + yamllint on commit
```

Ad-hoc lint:

```bash
uv run pre-commit run --all-files
```

Lint stack: ruff (E/F/I rules, format) for Python, yamllint (relaxed) for k8s manifests, plus trailing-whitespace / EOF / merge-conflict / large-file (<500kb) guards. Config in `pyproject.toml`, `.pre-commit-config.yaml`, `.yamllint.yaml`.

## Cluster & GPU Operations

Infrastructure is NERC MGHPCC OpenShift with H100 GPUs at ~$3/hr. **Always scale down after use.**

```bash
# Login (token expires periodically)
oc login --token=<token> --server=https://k8s.nerc.mghpcc.org:6443

# Scale up a model (takes 3-8 min to load)
oc scale deployment qwen3-tts --replicas=1

# Wait for ready
oc get pods -w   # Watch until STATUS=Running, READY=1/1
oc logs -l app=qwen3-tts --tail=3  # Look for "Application startup complete"

# Port-forward for local API access
oc port-forward svc/qwen3-tts 8091:8000 &
oc port-forward svc/qwen3-tts-base 8094:8000 &
oc port-forward svc/voxtral-tts 8092:8000 &
oc port-forward svc/fish-speech 8093:8000 &

# Verify model is ready
curl http://localhost:8091/v1/audio/voices

# ALWAYS scale down when done
oc scale deployment qwen3-tts --replicas=0
oc scale deployment qwen3-tts-base --replicas=0
oc scale deployment fish-speech --replicas=0
oc scale deployment voxtral-tts --replicas=0

# Verify no pods running
oc get pods   # Should show "No resources found"
```

## Running Data Generation

```bash
# Pilot/single-model generation
python src/data_generation/generate_accent_samples.py --api-base http://localhost:8091

# Production multi-model generation (Qwen3 + Voxtral)
python src/data_generation/generate_accent_dataset.py \
  --qwen-api http://localhost:8091 \
  --voxtral-api http://localhost:8092 \
  --output-dir data/synthetic
```

Python dependency: `httpx` (used for all TTS API calls with 120-180s timeouts).

## API Patterns

**Standard TTS** (Qwen3 CustomVoice — best for accented English):
```python
httpx.post("http://localhost:8091/v1/audio/speech", json={
    "input": "English text to speak",
    "voice": "vivian",
    "response_format": "wav",
    "instructions": "你在跟餐馆打电话，速度快一点，带着chinglish的口音。",  # Chinese instructions work best
})
```

**Voice clone** (Qwen3 Base / Fish Speech — requires base64 reference audio):
```python
httpx.post("http://localhost:8094/v1/audio/speech", json={
    "input": "Text to synthesize",
    "voice": "default",
    "response_format": "wav",
    "ref_audio": "data:audio/wav;base64,{base64_encoded}",
    "ref_text": "Transcript of the reference audio.",
}, timeout=300)
```

## Data & Experiment Conventions

**Run naming**: `{model_short}_{YYYYMMDD}_{NNN}` (e.g., `qwen3tts_20260330_001`)

Each run directory contains:
- `run_config.json` — model metadata, voices, scenarios, hardware config
- `manifest.json` — per-sample metadata (voice, accent, text, file path, generation timing)
- `annotations.json` — optional human feedback on sample quality

Audio layout: `data/synthetic/{run_id}/{accent}/{voice}/{sentence_id}.wav`

Reference benchmark audio: `data/reference_voices/` (Speech Accent Archive samples)

**When adding new experiments**: Always create a new run_id, write manifest.json, and document findings in `progress.txt` with the session date. Never overwrite existing runs.

## Architecture

**API interface**: vLLM-Omni exposes OpenAI-compatible `/v1/audio/speech` (POST with `input`, `voice`, `response_format`, optional `instructions`). Timeout is 120-180s per generation.

**Kubernetes manifests**: `k8s/*.yaml` — separate Deployment + Service + Route per TTS model. All use PVC `vllm-model-cache` for HF model cache, H100 GPU tolerations, `vllm/vllm-omni:v0.18.0` container image.

**Data generation scripts**:
- `generate_accent_samples.py` — single-model pilot generation (9 voices × 10 sentences)
- `generate_accent_dataset.py` — multi-model production generation (Qwen3 + Voxtral, 4 scenarios × 10 sentences per voice)

Scenarios in production dataset: restaurant, hospital, bank, telecom.

## Model Status

| Deployment | Model | Port | Status | Notes |
|---|---|---|---|---|
| qwen3-tts | Qwen3-TTS CustomVoice | 8091 | Best | Chinese instruction for Mandarin/Japanese accent. Good voices: vivian, serena, dylan, ono_anna, aiden |
| qwen3-tts-base | Qwen3-TTS Base | 8094 | Limited | Voice cloning works, but ref_text fragments leak into output |
| voxtral-tts | Voxtral-4B-TTS | 8092 | Blocked | Cross-lingual too slow; speed > 1.0 causes hollow artifacts |
| fish-speech | Fish Speech S2 Pro | 8093 | Not suitable | Clones timbre only, not accent; inline tags ineffective |
| cosyvoice3 | CosyVoice3 | — | Deploy fails | Empty config.json on HuggingFace |

## GitHub Auth

```bash
source .envrc   # Sets GH_TOKEN for bwen-uchicago account
```

Run this before any `gh` command to ensure issues/PRs are created by the correct account.

## Coding Rules

- **NO mentions of Panbot, Enkira, or any company** — this is an academic project
- Git identity: `Bo Wen <t-9bwen@uchicago.edu>` (repo-local config)
- SSH remote: `uchicago-github` alias

## Commit Convention

Follow conventional commits: `type(scope): description`

Types: `feat`, `fix`, `docs`, `exp`, `data`, `k8s`, `refactor`

Examples: `exp: cross-lingual accent test`, `data: commit qwen3tts_20260330_001 samples`
