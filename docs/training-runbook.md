# Training Runbook

> **Status: v0 — UNVALIDATED.** Commands here have not been run end-to-end on NERC with NeMo 25.04 × Nemotron-Streaming yet. On first discrepancy, file a PR fix and update the `## Known unknowns` section at the bottom. This document is intended to be the authoritative training guide once validated; until then, treat it as a concrete starting point rather than a guarantee.

End-to-end guide for fine-tuning accent-adaptive ASR models on the synthetic accented-English corpus this repo generates. Covers LoRA (primary recommendation) and full-parameter fine-tuning (alternative), plus the `nemo2riva` handoff to the production STT serving path.

## Prerequisites

1. **Infrastructure merged:**
   - PVC — `k8s/nemo-training-data-pvc.yaml` (commit `e2f7996`).
   - Training pod — `k8s/nemo-training-deployment.yaml` (PR #15 when merged).
2. **NGC registry access** configured per CLAUDE.md § *NGC Registry Access*. In particular:
   ```bash
   oc secrets link default ngc-secret --for=pull
   ```
3. **Pod applied to NERC:**
   ```bash
   oc apply -f k8s/nemo-training-deployment.yaml
   oc get deployment nemo-training   # READY 0/0 by default (scale-to-zero)
   ```
4. **Pre-flight sanity** — quick import check once scaled up:
   ```bash
   oc scale deployment nemo-training --replicas=1
   oc wait --for=condition=available deployment/nemo-training --timeout=600s
   oc exec deployment/nemo-training -- python -c "import nemo; print(nemo.__version__)"
   ```

## Data organization

```
data/
├── synthetic/                        # SOURCE: TTS-generated audio (committed selectively)
│   ├── qwen3tts_20260330_001/
│   │   ├── manifest.json             # repo format — our data-gen schema
│   │   ├── run_config.json
│   │   └── mandarin/vivian/sentence_00.wav
│   └── voxtral_20260330_001/
├── external_accented/                # SOURCE: real accented audio (future; gitignored)
├── reference_voices/                 # reference benchmark samples
├── manifests/                        # DERIVED: NeMo JSONL (gitignored)
│   ├── qwen3tts_20260330_001.train.jsonl
│   ├── qwen3tts_20260330_001.val.jsonl
│   ├── qwen3tts_20260330_001.test.jsonl
│   └── qwen3tts_20260330_001.summary.json
├── synthetic_16k/                    # DERIVED: resampled audio (gitignored)
│   └── qwen3tts_20260330_001/mandarin/vivian/sentence_00.wav
└── experiments/                      # RUNTIME: on-cluster training outputs (PVC only)
```

**Source vs. derived:**
- `data/synthetic/` and `data/external_accented/` are the authoritative pool. Never rewrite in place.
- `data/manifests/`, `data/synthetic_16k/`, and `data/experiments/` are derived artifacts; all three are gitignored and regenerable.

**On-cluster layout** mirrors local `data/` under `/data/` on the `nemo-training` pod (PVC `nemo-training-data`, 120Gi, mounted at `/data`). Serving-side `vllm-model-cache` PVC mounts at `/models` and is where the final `.riva` export lands.

## Workflow overview

```
┌─────────────────────────┐
│ data/synthetic/<run>/   │  source manifest + 24kHz wav
└────────────┬────────────┘
             │  src/training/convert_manifest.py
             │  (lowercase/strip text, resample 24kHz→16kHz,
             │   deterministic train/val/test split by voice+sentence_id,
             │   emit NeMo JSONL + summary)
             ▼
┌─────────────────────────┐
│ data/manifests/*.jsonl  │  + data/synthetic_16k/<run>/
└────────────┬────────────┘
             │  oc rsync ... deployment/nemo-training:/data/...
             ▼
┌─────────────────────────┐
│ /data on nemo-training  │  resampled audio + NeMo manifests
└────────────┬────────────┘
             │  train_asr_adapter.py (LoRA)  or  speech_to_text_rnnt_bpe.py (full-param)
             ▼
┌─────────────────────────┐
│ /data/experiments/<id>/ │  checkpoints (.nemo, .ckpt)
└────────────┬────────────┘
             │  nemo2riva
             ▼
┌─────────────────────────┐
│ /models/riva/<name>.riva│  consumed by riva-stt deployment (issue #9)
└─────────────────────────┘
```

## Step-by-step LoRA recipe (primary)

LoRA fine-tuning adds a small ~1-5M-parameter adapter on top of Nemotron-Streaming's ~600M frozen weights. Best for the current ~400-sample corpus: low catastrophic-forgetting risk, fast iteration, per-accent adapters align with the project's Phase-2/3 routing vision.

### 1. Start the pod

```bash
oc scale deployment nemo-training --replicas=1
oc get pods -l app=nemo-training -w    # wait for READY 1/1
```

### 2. Download the base model (one-time)

```bash
oc rsh deployment/nemo-training
# inside pod:
mkdir -p /models/nemo
huggingface-cli download nvidia/nemotron-speech-streaming-en-0.6b \
    --local-dir /models/nemo/nemotron-streaming
ls /models/nemo/nemotron-streaming/*.nemo
exit
```

### 3. Introspect the model (confirms decoder type)

Still from your laptop:
```bash
oc exec deployment/nemo-training -- python -c "
from nemo.collections.asr.models import ASRModel
import glob
path = glob.glob('/models/nemo/nemotron-streaming/*.nemo')[0]
m = ASRModel.restore_from(path)
print('class:', type(m).__name__)
print('decoder:', m._cfg.decoder._target_)
print('sample_rate:', m._cfg.preprocessor.sample_rate)
"
```

Expected class: some FastConformer variant (`EncDecRNNTBPEModel`, `EncDecHybridRNNTCTCBPEModel`, or TDT variant). **Sample rate** is the key datapoint — our TTS audio is 24 kHz; if the model wants 16 kHz, resampling is required (next step).

### 4. Convert manifests locally

```bash
# from repo root on your laptop
uv run python src/training/convert_manifest.py \
    --input-manifest data/synthetic/qwen3tts_20260330_001/manifest.json \
    --output-manifest data/manifests/qwen3tts_20260330_001 \
    --resample-to 16000 \
    --resampled-audio-dir data/synthetic_16k/qwen3tts_20260330_001 \
    --split-mode three-way
```

Produces:
- `data/manifests/qwen3tts_20260330_001.train.jsonl` (~80% of records)
- `data/manifests/qwen3tts_20260330_001.val.jsonl` (~10%)
- `data/manifests/qwen3tts_20260330_001.test.jsonl` (~10%)
- `data/manifests/qwen3tts_20260330_001.summary.json` (counts + total seconds per split, per accent)
- `data/synthetic_16k/qwen3tts_20260330_001/...` (resampled audio; idempotent — reruns are no-ops for already-resampled files)

See also `src/training/convert_manifest.py --help` for `--minimal` (drop accent/voice extras), `--keep-punctuation`, and `--split-mode per-accent` (separate train/val/test within each accent).

### 5. Upload to the pod

```bash
oc rsync data/synthetic_16k/ deployment/nemo-training:/data/synthetic_16k/ --progress=true
oc rsync data/manifests/     deployment/nemo-training:/data/manifests/     --progress=true
```

**Important:** the NeMo manifests contain absolute `audio_filepath` strings pointing at `/Users/.../synthetic_16k/...` (your laptop paths). Before training, rewrite them to pod paths. Quick sed loop:

```bash
oc rsh deployment/nemo-training
# inside pod:
cd /data/manifests
for f in *.jsonl; do
    sed -i "s|$(echo /Users/chenw615/code/UChi-JCL/Multi-Modal-Semantic-Routing-for-vLLM/data/synthetic_16k | sed 's|/|\\/|g')|/data/synthetic_16k|g" "$f"
done
head -1 qwen3tts_20260330_001.train.jsonl    # verify audio_filepath is now /data/...
```

Alternative: pass `--audio-root` through to `convert_manifest.py` (future enhancement — tracked as TODO).

### 6. Launch training

Inside the pod:

```bash
RUN_ID="lora-smoke-$(date +%Y%m%d-%H%M)"
mkdir -p /data/experiments/$RUN_ID
cd /data/experiments/$RUN_ID

nohup python /opt/NeMo/examples/asr/asr_adapters/train_asr_adapter.py \
    model.restore_from_path=/models/nemo/nemotron-streaming/*.nemo \
    model.train_ds.manifest_filepath=/data/manifests/qwen3tts_20260330_001.train.jsonl \
    model.validation_ds.manifest_filepath=/data/manifests/qwen3tts_20260330_001.val.jsonl \
    model.train_ds.batch_size=8 \
    model.validation_ds.batch_size=8 \
    model.adapter.dim=32 \
    model.optim.lr=3e-4 \
    model.optim.sched.warmup_steps=100 \
    trainer.devices=1 \
    trainer.max_epochs=10 \
    trainer.precision=bf16-mixed \
    exp_manager.exp_dir=/data/experiments/$RUN_ID \
    exp_manager.create_checkpoint_callback=true \
    exp_manager.resume_if_exists=false \
    > train.log 2>&1 &

echo "PID $! — logs at /data/experiments/$RUN_ID/train.log"
```

`nohup ... &` detaches the training from your shell. You can `exit` the `oc rsh` and reconnect later.

### 7. Monitor progress

**From your laptop** (no pod shell needed):
```bash
oc exec deployment/nemo-training -- tail -f /data/experiments/$RUN_ID/train.log
oc exec deployment/nemo-training -- nvidia-smi
```

**Inside the pod:**
```bash
oc rsh deployment/nemo-training
watch -n 5 nvidia-smi
tail -f /data/experiments/$RUN_ID/train.log
```

**Optional TensorBoard:**
```bash
# inside pod
tensorboard --logdir /data/experiments --bind_all --port 6006 &

# from laptop
oc port-forward deployment/nemo-training 6006:6006
# browse http://localhost:6006
```

### 8. Resume after interruption

If the pod was scaled to 0 or crashed, checkpoints survive on `/data` (PVC-backed). Scale back up and resume:

```bash
oc scale deployment nemo-training --replicas=1
oc rsh deployment/nemo-training

# resume from the latest checkpoint
python /opt/NeMo/examples/asr/asr_adapters/train_asr_adapter.py \
    model.restore_from_path=/models/nemo/nemotron-streaming/*.nemo \
    model.train_ds.manifest_filepath=/data/manifests/qwen3tts_20260330_001.train.jsonl \
    model.validation_ds.manifest_filepath=/data/manifests/qwen3tts_20260330_001.val.jsonl \
    ... (same flags as launch) ... \
    exp_manager.resume_if_exists=true \
    +init_from_ptl_ckpt=/data/experiments/$RUN_ID/checkpoints/last.ckpt
```

Lightning's `last.ckpt` is auto-saved every `check_val_every_n_epoch` step. Inspect the filename for the actual epoch/step: `epoch=03-step=2000.ckpt` etc.

## Alternative: full-parameter fine-tuning

Use full fine-tuning when:
- You have **>10k samples per accent** and catastrophic forgetting risk is manageable.
- LoRA adapters have plateaued at non-acceptable WER.
- There's a clear acoustic distribution shift (not just vocabulary/accent).

Exact entry point depends on decoder type (from step 3 above):
- `EncDecRNNTBPEModel` → `/opt/NeMo/examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py`
- `EncDecHybridRNNTCTCBPEModel` → `/opt/NeMo/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py`

### Launch command

```bash
RUN_ID="full-ft-$(date +%Y%m%d-%H%M)"
mkdir -p /data/experiments/$RUN_ID

nohup python /opt/NeMo/examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py \
    --config-path=/opt/NeMo/examples/asr/conf/fastconformer/ \
    --config-name=fast-conformer_transducer_bpe \
    +init_from_nemo_model=/models/nemo/nemotron-streaming/*.nemo \
    model.train_ds.manifest_filepath=/data/manifests/qwen3tts_20260330_001.train.jsonl \
    model.validation_ds.manifest_filepath=/data/manifests/qwen3tts_20260330_001.val.jsonl \
    model.train_ds.batch_size=16 \
    model.optim.lr=1e-4 \
    model.optim.sched.warmup_steps=500 \
    trainer.devices=1 \
    trainer.max_epochs=20 \
    trainer.precision=bf16-mixed \
    exp_manager.exp_dir=/data/experiments/$RUN_ID \
    exp_manager.create_checkpoint_callback=true \
    > train.log 2>&1 &
```

**Flag differences from LoRA:**
- `+init_from_nemo_model=` (not `model.restore_from_path=`) — loads weights but resets optimizer/scheduler. Critical for fine-tuning; `restore_from_path` would inherit the pretraining schedule's end state.
- Lower `model.optim.lr` (1e-4 vs 3e-4) — full-param needs gentler updates to avoid catastrophic forgetting.
- Higher `warmup_steps` — stabilizes the first phase of adaptation.
- `bf16-mixed` — H100-friendly; keeps memory manageable on a single GPU.

### When it plateaus

Watch `val_wer` in `train.log`. If it stops improving for 3+ validation rounds, early-stop (`trainer.max_epochs` or `trainer.val_check_interval` can drive this, or Ctrl+C the `nohup`'d process and pick up from `last.ckpt`).

## `nemo2riva` handoff

Once training produces a final `.nemo` file, convert it for the Riva serving deployment:

```bash
oc rsh deployment/nemo-training
pip install nemo2riva    # not pre-installed; quick

mkdir -p /models/riva
nemo2riva \
    --out /models/riva/nemotron-lora-v1.riva \
    /data/experiments/$RUN_ID/checkpoints/<best-or-final>.nemo

ls -lh /models/riva/
```

After this, the upcoming `riva-stt` deployment (issue #9) will pick up the `.riva` file from the shared `vllm-model-cache` PVC at `/models/riva/`.

## Shut down

H100 is ~$3/hr. When the session is done:

```bash
exit   # leave the oc rsh if you're in it
oc scale deployment nemo-training --replicas=0
oc get pods -l app=nemo-training   # should show "No resources found"
```

Checkpoints on `/data/experiments/` persist across scale cycles.

## Known unknowns

Everything here is provisionally correct but may need adjustment once someone actually runs it. If you hit any of these, please PR a fix + update this section.

1. **NeMo script paths.** `/opt/NeMo/examples/asr/asr_adapters/train_asr_adapter.py` is the canonical location in NeMo 24.x. The 25.04 container may have moved it (possibly under `asr/` root, or renamed). If missing, `ls /opt/NeMo/examples/asr/` to discover the current layout.
2. **Nemotron-Streaming decoder type.** RNN-T vs Hybrid-RNN-T-CTC vs TDT variants each require a different full-param training script. The step-3 introspection is authoritative.
3. **`model.restore_from_path` globbing.** `*.nemo` may not resolve inside NeMo's config loader (Hydra). Use the explicit filename if the glob fails.
4. **Tokenizer.** Nemotron-Streaming ships its tokenizer bundled in the `.nemo` file. `model.restore_from_path` normally handles this; if training complains, try `model.tokenizer.update_tokenizer_model=false`.
5. **Sample rate 16 kHz assumption.** Based on FastConformer's typical preprocessor default. Confirm via the step-3 introspection of `m._cfg.preprocessor.sample_rate`. If it's 16 kHz, our resample is correct; if 24 kHz, skip resampling.
6. **Audio-path rewrite after rsync.** Step 5's `sed` is a workaround; a proper fix is adding `--audio-root` to `convert_manifest.py` so paths are written cluster-ready from the start. Tracked as TODO.
7. **`nemo2riva` output filename convention.** Nemotron-Streaming's Riva-side config may want a specific name; check the Riva model-config docs before assuming arbitrary filenames work.

## See also

- `src/training/convert_manifest.py` — manifest + resample script referenced above.
- `src/training/README.md` — short orientation for the training package.
- CLAUDE.md § *NGC Registry Access* — prerequisite NGC setup.
- GitHub issue #18 — tracking doc for post-first-run corrections.
