# Plan: Riva STT Serving + NeMo Training + WER Benchmarking

## Context

The repo currently runs TTS-only on NERC MGHPCC OpenShift H100s using `vllm/vllm-omni` with the shared PVC `vllm-model-cache`. We are adding an STT path for the voice-bot + accent-classifier research phases, with two separate H100 deployments (no co-deployment):

- **Riva** for production-grade streaming STT serving (native streaming, <300ms first-token latency, 560 concurrent streams/H100).
- **NeMo Framework** for LoRA fine-tuning the chosen model (`nvidia/nemotron-speech-streaming-en-0.6b`) on the synthetic accented-English corpus in `data/synthetic/`, and also for running the one-shot `nemo2riva` conversion step.

Baseline model for WER measurement: Nemotron-Streaming-EN-0.6B. Offline ceiling comparison: `nvidia/parakeet-tdt-0.6b-v3` (1.93% / 3.59% WER on LibriSpeech clean/other). Honest-comparison baseline (later): Whisper-v3-turbo + HF PEFT LoRA.

Both Riva and NeMo images live on `nvcr.io/nvidia/...` which requires NGC authentication. This is a new pattern for the repo — existing TTS deployments use public `docker.io/vllm/...` images and have no `imagePullSecrets`.

## Prerequisites — NGC signup tutorial (user runs these manually before `oc apply`)

Once complete, all nvcr.io pulls in this namespace work without per-manifest `imagePullSecrets`.

1. **Sign up.** Go to https://ngc.nvidia.com/signin → "Create an Account" (free; use UChicago email). No payment required.
2. **Generate Personal API Key.** Top-right profile → **Setup** → **Generate Personal Key** → name it (e.g., `openshift-nerc-pull`) → scopes: check **NGC Catalog** and **Private Registry**. Click Generate. **Copy the key once — it is not shown again.**
3. **Local smoke test (optional).**
   ```
   docker login nvcr.io
   # Username: $oauthtoken    (literal string, with the dollar sign)
   # Password: <paste key>
   ```
4. **Create the OpenShift pull secret in this project's namespace.**
   ```
   oc create secret docker-registry ngc-secret \
     --docker-server=nvcr.io \
     --docker-username='$oauthtoken' \
     --docker-password='<KEY>' \
     --docker-email=t-9bwen@uchicago.edu
   ```
5. **Link globally to the default ServiceAccount** so manifests don't need `imagePullSecrets`:
   ```
   oc secrets link default ngc-secret --for=pull
   ```
6. **Verify.**
   ```
   oc get secret ngc-secret -o jsonpath='{.type}{"\n"}'   # kubernetes.io/dockerconfigjson
   oc describe sa default | grep -i ngc                    # "Image pull secrets: ngc-secret"
   ```

If verification passes, we can apply the two new deployments without adding `imagePullSecrets` blocks.

## Files to Create / Modify

**Create:**
- `k8s/riva-stt-deployment.yaml`
- `k8s/nemo-training-deployment.yaml`
- `k8s/nemo-training-data-pvc.yaml`
- `src/benchmarking/__init__.py` (empty)
- `src/benchmarking/measure_wer.py`

**Modify:**
- `k8s/qwen3-tts-deployment.yaml` — add throughput args
- `CLAUDE.md` — NGC section, scale commands, Model Status rows, Fine-tuning Workflow section

Not touched: `progress.txt` (appended post-session), `vllm-api-key-secret.yaml`, other TTS manifests, README.

## Implementation Steps

### 1. `k8s/nemo-training-data-pvc.yaml`
New `PersistentVolumeClaim` named `nemo-training-data`, 50Gi, RWO, storage class matching `vllm-model-cache` (inspect via `oc get pvc vllm-model-cache -o jsonpath='{.spec.storageClassName}'`). Separate from model cache to keep training reads/writes off the serving PVC.

### 2. `k8s/nemo-training-deployment.yaml`
Mirror `k8s/qwen3-tts-deployment.yaml` (lines 1-89) with these deltas:

- `metadata.name: nemo-training`, labels `app: nemo-training`
- `spec.replicas: 0` — scale-up is manual, matching the existing "scale up, use, scale down" workflow (CLAUDE.md lines 34-50)
- Image: `nvcr.io/nvidia/nemo:25.04` (user confirms latest tag before apply via `skopeo list-tags docker://nvcr.io/nvidia/nemo`)
- Command: `["/bin/sh", "-c"]`; Args: `["sleep infinity"]` — container stays alive, user runs training via `oc rsh deployment/nemo-training`
- Toleration block: identical to qwen3-tts lines 18-21
- Env: `HOME=/tmp`, `HF_HOME=/models/hf_home`, `NEMO_CACHE_DIR=/models/nemo_cache`, `TRANSFORMERS_CACHE=/models/hf_home`, `NUMBA_CACHE_DIR=/tmp/numba_cache`, `MPLCONFIGDIR=/tmp/matplotlib`
- Resources: requests cpu 4 / mem 32Gi / `nvidia.com/gpu: 1`; limits cpu 16 / mem 64Gi / `nvidia.com/gpu: 1` (heavier than serving — training needs DataLoader RAM)
- Volumes: `model-cache` (PVC `vllm-model-cache` at `/models`), `training-data` (PVC `nemo-training-data` at `/data`), `shm` (8Gi emptyDir Memory at `/dev/shm`)
- No Service, no Route — internal-only access via `oc rsh`.
- No `imagePullSecrets` block (linked at SA level).

### 3. `k8s/riva-stt-deployment.yaml`
Mirror the TTS template (lines 1-105) with these deltas:

- `metadata.name: riva-stt`, labels `app: riva-stt`
- `replicas: 0`
- Image: `nvcr.io/nvidia/riva/riva-speech:2.19.0` (user verifies latest 2.x stable via skopeo before apply)
- Command runs `riva_start.sh` pointed at `/models/riva/` — expects the `.riva` model file produced by step 4 to be pre-staged on the PVC
- Toleration: identical to qwen3-tts lines 18-21
- Ports:
  - `50051/TCP` (gRPC — cluster-internal only, NO Route: gRPC-over-Route is flaky on OpenShift)
  - `8000/TCP` (HTTP — `/v1/audio/transcriptions` OpenAI-compatible surface; Route with TLS edge)
- Env: `HOME=/tmp`, `HF_HOME=/models/hf_home`, `NUMBA_CACHE_DIR=/tmp/numba_cache`, `MPLCONFIGDIR=/tmp/matplotlib`, `RIVA_MODEL_REPO=/models/riva`. No `VLLM_API_KEY` (Riva uses its own auth model; for research cluster access is Route-level only).
- Throughput args (passed via Riva config or `--ngc-config`): `max-batch-size=32`, `max-concurrent-streams=560` (Jan 2026 Nemotron-Streaming spec for one H100).
- Resources: identical to qwen3-tts lines 52-60 (4-8 CPU, 16-32Gi RAM, 1 H100).
- Volumes: `model-cache` PVC at `/models` + `shm` emptyDir — identical to qwen3-tts lines 61-73.
- Service: two ports (50051 grpc, 8000 http), ClusterIP.
- Route: TLS edge on port 8000 only.

### 4. nemo2riva conversion (one-shot, runs inside nemo-training pod)
Not a file — a documented procedure in CLAUDE.md's new "Fine-tuning Workflow" section:

```
oc scale deployment nemo-training --replicas=1
oc rsh deployment/nemo-training
# inside pod:
pip install nemo2riva
huggingface-cli download nvidia/nemotron-speech-streaming-en-0.6b --local-dir /models/nemo/nemotron-streaming
mkdir -p /models/riva
nemo2riva --out /models/riva/nemotron-streaming-en-0.6b.riva /models/nemo/nemotron-streaming/*.nemo
exit
oc scale deployment nemo-training --replicas=0   # always scale down
```

After this, `riva-stt` deployment can scale up and find the model at `/models/riva/`.

### 5. `src/benchmarking/measure_wer.py`
- Imports: `argparse`, `csv`, `json`, `pathlib`, `time`, `httpx`, `jiwer`
- CLI args: `--manifest` (required), `--audio-root` (default: manifest's parent dir), `--stt-api` (default `http://localhost:8001`), `--model-name` (default `nemotron-streaming-en-0.6b`), `--run-id` (default: inferred from manifest path), `--out` (CSV path)
- Reuse repo convention: `httpx.Client(timeout=30.0)` matching `src/data_generation/generate_accent_dataset.py` line 15
- Manifest schema matches `generate_accent_dataset.py` lines 145-160: fields `voice`, `accent`, `native_lang`, `model`, `scenario`, `sentence_id`, `text`, `file`
- For each sample with non-null `file`: POST multipart `file=@path, model=<name>` to `{stt_api}/v1/audio/transcriptions`; record `total_latency_ms = perf_counter() delta`; `first_token_latency_ms = None` in v1 (HTTP batch; gRPC streaming deferred); capture `hypothesis = response.json()["text"]`
- Scoring: `jiwer.Compose([ToLowerCase(), RemovePunctuation(), Strip(), ReduceToListOfListOfWords()])` applied to both reference and hypothesis before `jiwer.wer`
- CSV columns: `run_id, model, accent, voice, scenario, sentence_id, reference, hypothesis, wer, total_latency_ms, first_token_latency_ms`
- Print aggregate: mean/median WER grouped by `accent` and by `voice`
- Deps installed by user: `pip install httpx jiwer` (consistent with repo's "no requirements.txt" convention)

### 6. `k8s/qwen3-tts-deployment.yaml` edit
In the `vllm serve` args block (around line 31), append:
```
--max-num-seqs 64 --max-num-batched-tokens 16384
```
Keep `--enforce-eager`. No other changes.

### 7. `CLAUDE.md` edits
- Under **Cluster & GPU Operations** (around line 25), insert new subsection **NGC Registry Access** with the 6-step tutorial above.
- Update port-forward block (around line 41): add `oc port-forward svc/riva-stt 8001:8000 &`.
- Update scale-down block (around line 49): add `oc scale deployment riva-stt --replicas=0` and `oc scale deployment nemo-training --replicas=0`.
- Update **Model Status** table (around line 126): add rows for `riva-stt` (Primary STT, 24ms final, 560 concurrent streams/H100) and `nemo-training` (fine-tune sandbox, `oc rsh` for LoRA runs).
- Add new section **Fine-tuning Workflow** after **Running Data Generation**: the scale-up → `nemo2riva` → LoRA train → re-export → scale-down loop.

## Verification

1. Pull secret: `oc describe sa default | grep -i ngc` shows `ngc-secret` as image pull secret.
2. Apply PVC: `oc apply -f k8s/nemo-training-data-pvc.yaml` → `oc get pvc nemo-training-data` → Bound.
3. Apply manifests: `oc apply -f k8s/nemo-training-deployment.yaml -f k8s/riva-stt-deployment.yaml`.
4. **One-shot conversion**: scale up `nemo-training`, run the `nemo2riva` command sequence, verify `/models/riva/nemotron-streaming-en-0.6b.riva` exists (`oc rsh` + `ls -lh`), scale back to 0.
5. **Riva boot**: `oc scale deployment riva-stt --replicas=1`; `oc get pods -l app=riva-stt -w` until Running 1/1 (~2 min). `oc logs -l app=riva-stt --tail=30` should contain `Riva Server started on 0.0.0.0:50051`.
6. **Smoke test Riva HTTP**: `oc port-forward svc/riva-stt 8001:8000 &` then
   ```
   curl -F file=@data/synthetic/qwen3tts_20260330_001/mandarin/vivian/0.wav \
        -F model=nemotron-streaming-en-0.6b \
        http://localhost:8001/v1/audio/transcriptions
   ```
   Should return JSON with a `text` field.
7. **Baseline WER run**:
   ```
   pip install httpx jiwer
   python src/benchmarking/measure_wer.py \
     --manifest data/synthetic/qwen3tts_20260330_001/manifest.json \
     --stt-api http://localhost:8001 \
     --out runs/wer_baseline_nemotron.csv
   ```
   Expected: native_english rows 1-5% WER; accented rows higher (this gap is the research signal).
8. **Parallelism check**: `for i in {1..20}; do curl -F file=@... -F model=... http://localhost:8001/v1/audio/transcriptions & done; wait` — should not error out; Riva handles concurrent clients up to its configured max-batch-size (32).
9. **Scale-down discipline**: confirm `oc get pods -l app=riva-stt` and `-l app=nemo-training` show no pods when done.

## Open Items to Resolve Before or During Implementation

1. **Riva version pin**: `2.19.0` is an assumption. User runs `skopeo list-tags docker://nvcr.io/nvidia/riva/riva-speech | tail -20` before applying and bumps the pin if needed.
2. **NeMo version pin**: same check against `nvcr.io/nvidia/nemo`.
3. **PVC storage class**: inspect `vllm-model-cache` to match; hard-coding to `ocs-storagecluster-ceph-rbd` or similar without checking will cause PVC to hang in Pending.
4. **Riva OpenAI-compat endpoint**: Riva 2.15+ ships `/v1/audio/transcriptions`; confirm on first boot. If missing, fall back to Riva's native gRPC and add a thin FastAPI proxy (deferred — not in this plan).
5. **first_token_latency_ms**: returning `null` in v1 CSV. If the latency metric is needed for the first benchmark round, we add a gRPC streaming path as a follow-up.
