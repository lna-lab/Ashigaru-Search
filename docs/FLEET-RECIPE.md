# The Fleet Recipe — serving 大将 + 足軽 locally (NVFP4)

How to stand up the **commander + scout** fleet that Ashigaru-Search drives, using
all-local **NVFP4** models on stock **`vllm/vllm-openai:v0.22.0`** — single-node or
multi-node.

Ashigaru only ever talks to **two OpenAI-compatible endpoints**:

| role | env | who |
|---|---|---|
| 足軽 (scouts) | `ASHIGARU_WORKER_BASE_URL` / `_MODEL` | cheap fast model(s), fanned out |
| 大将 (commander) | `ASHIGARU_ORCH_BASE_URL` / `_MODEL` | smart model: plans, steers, synthesizes |

They can be the **same server**, **two ports on one box**, or **two separate nodes**.
That is the whole multi-node story — no code, just URLs (see §5).

---

## 1. The models

Validated on **7× RTX PRO 2000 Blackwell** (16 GB, ~70 W, SM120, PCIe — no NVLink).

| model | role | params | quant | served-name | key flag |
|---|---|---|---|---|---|
| **LFM2.5-1.2B-JP-202606-NVFP4** | scout (recommended) | 1.2B dense | modelopt | `lfm25-1p2b-jp` | `--quantization modelopt`, TP=1 |
| **LFM2.5-8B-A1B-NVFP4** | scout (higher capacity) | 8B / 1B active | modelopt | `lfm25-8b-a1b` | `--quantization modelopt` |
| **Qwen3.6-27B-MTP-pi-tune-NVFP4** | commander ✅ | 27B dense | modelopt | `qwen36` | `--quantization modelopt` + MTP + tool-call |
| **Huihui-Qwen3.6-35B-A3B-NVFP4** | commander (alt) | 35B / 3B active | compressed-tensors | `qwen36-35b-a3b` | **no `--quantization`** (auto-detect) |

> ⚠️ **modelopt vs compressed-tensors.** The 35B-A3B uses compressed-tensors and must
> omit `--quantization` (vLLM auto-detects it). All other modelopt builds need
> `--quantization modelopt`. Wrong flag → garbage output or load failure.

> ℹ️ **Scout choice: 1.2B dense beats 8B MoE for Ashigaru.** The 1.2B is power-bound
> at ~24k tok/s aggregate (C256, 70 W) — the whole 16 GB card becomes KV+concurrency
> headroom (≈114× concurrent at 16K ctx). Because it's dense TP=1, you scale throughput
> with **data-parallel replicas**, not TP. Two TP=1 replicas → **~46k tok/s** at 140 W
> (~329 tok/J). TP=4 would give ~25k tok/s with 4× the hardware — much less efficient.

---

## 2. Serve each (copy-paste)

All use the official image; mount the model dir read-only at `/model`.

### 足軽 — LFM2.5-1.2B-JP (recommended scout), TP=1, :8012

```bash
# Replica 1
docker run -d --name lfm25-scout \
  --gpus '"device=<GPU_UUID_or_index>"' \
  --shm-size=8g --ipc=host \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -p 8012:8000 \
  -v /path/to/LFM2.5-1.2B-JP-202606-NVFP4:/model:ro \
  vllm/vllm-openai:v0.22.0 \
  /model --served-model-name lfm25-1p2b-jp --quantization modelopt \
  --kv-cache-dtype fp8 --max-model-len 16384 --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 --port 8000

# Replica 2 (optional — doubles throughput; no NVLink required)
docker run -d --name lfm25-scout-2 \
  --gpus '"device=<another_GPU_UUID>"' \
  --shm-size=8g --ipc=host \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -p 8013:8000 \
  -v /path/to/LFM2.5-1.2B-JP-202606-NVFP4:/model:ro \
  vllm/vllm-openai:v0.22.0 \
  /model --served-model-name lfm25-1p2b-jp --quantization modelopt \
  --kv-cache-dtype fp8 --max-model-len 16384 --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 --port 8000

# Round-robin proxy in front of both (ships as rrproxy.py in this repo)
python3 rrproxy.py 8010 http://localhost:8012 http://localhost:8013 &
# Then set ASHIGARU_WORKER_BASE_URL=http://localhost:8010/v1
```

### 足軽 — LFM2.5-8B-A1B (higher-capacity scouts), TP=4, :8000

```bash
docker run -d --name vllm-scouts \
  --gpus '"device=0,1,2,3"' \
  --shm-size=8g --ipc=host \
  -e NCCL_P2P_DISABLE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -p 8000:8000 \
  -v /path/to/LFM2.5-8B-A1B-NVFP4:/model:ro \
  vllm/vllm-openai:v0.22.0 \
  /model --served-model-name lfm25-8b-a1b --quantization modelopt \
  --tensor-parallel-size 4 --disable-custom-all-reduce \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 \
  --reasoning-parser deepseek_r1 \
  --host 0.0.0.0 --port 8000
```

### 大将 — Qwen3.6-27B-MTP-pi-tune (recommended commander), TP=4, :8011

This is the validated configuration: planning, supervision, quality-judging, and synthesis
all confirmed end-to-end with Ashigaru's full feature set.

```bash
docker run -d --name vllm-commander \
  --gpus '"device=0,1,2,3"' \
  --shm-size=8g --ipc=host \
  -e NCCL_P2P_DISABLE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -p 8011:8000 \
  -v /path/to/Qwen3.6-27B-MTP-pi-tune-NVFP4:/model:ro \
  vllm/vllm-openai:v0.22.0 \
  /model --served-model-name qwen36 --trust-remote-code \
  --quantization modelopt \
  --tensor-parallel-size 4 --disable-custom-all-reduce \
  --max-model-len 131072 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
  --host 0.0.0.0 --port 8000
```

### 大将 — Qwen3.6-35B-A3B (alt commander, sparse MoE, faster), TP=4, :8011

```bash
docker run -d --name vllm-commander \
  --gpus '"device=0,1,2,3"' \
  --shm-size=8g --ipc=host \
  -e NCCL_P2P_DISABLE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -p 8011:8000 \
  -v /path/to/Huihui-Qwen3.6-35B-A3B-abliterated-NVFP4:/model:ro \
  vllm/vllm-openai:v0.22.0 \
  /model --served-model-name qwen36-35b-a3b --trust-remote-code \
  --tensor-parallel-size 4 --disable-custom-all-reduce \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --host 0.0.0.0 --port 8000
```

---

## 3. Wire it to Ashigaru

Minimal `.env` for the recommended fleet (1.2B-JP scouts DP×2 + 27B commander):

```bash
# 足軽: 1.2B-JP DP×2 behind rrproxy (or point directly at 8012 for single replica)
ASHIGARU_WORKER_BASE_URL=http://localhost:8010/v1
ASHIGARU_WORKER_MODEL=lfm25-1p2b-jp
ASHIGARU_WORKER_API_KEY=EMPTY

# 大将: Qwen3.6-27B-MTP
ASHIGARU_ORCH_BASE_URL=http://localhost:8011/v1
ASHIGARU_ORCH_MODEL=qwen36
ASHIGARU_ORCH_API_KEY=EMPTY

# Critical: reasoning models burn tokens on <think> before emitting content.
# 16384 covers synthesis of 10 rich scout reports + full reasoning overhead.
# The default (3072) causes empty synthesis with multi-scout runs — do not lower.
ASHIGARU_ORCH_MAX_TOKENS=16384
ASHIGARU_REQUEST_TIMEOUT=300

# Per-role Commander thinking (reasoning-model commanders only; "auto"|"on"|"off").
# MEASURED (8-config sweep, see §6): turning thinking OFF for grounded synthesis is a free win —
# ~4-5x faster synth, equal-or-better accuracy, and it stays HONEST (hedges instead of confidently
# "reasoning" its way to a wrong claim). Planning keeps thinking (sharper decomposition).
ASHIGARU_ORCH_THINK_PLAN=on
ASHIGARU_ORCH_THINK_SYNTH=off
ASHIGARU_ORCH_THINK_JUDGE=off
# Player-coach (send the idle Commander out as one extra premium scout): the sweep did NOT show an
# accuracy gain (more sources != more correct) and it adds wall-time, so it's opt-in, default off.
# ASHIGARU_COMMANDER_SCOUT=1

# Scout count: 5 direct (no thin warm-up round), escalate ×2 to 10 if quality is thin.
# Prefix "M your question" (=5) or "L your question" (=10) to override.
ASHIGARU_MAX_SUBQUESTIONS=5
ASHIGARU_MAX_ESCALATIONS=1

# Full quality features ON
ASHIGARU_ESCALATE=1
ASHIGARU_QUALITY_JUDGE=1
ASHIGARU_SUPERVISE=1
ASHIGARU_SUPERVISE_AFTER=1
ASHIGARU_MAX_CHECKINS=3
```

Then just:
```bash
ashigaru "your question"
```

---

## 4. Numbers (measured, 7× RTX PRO 2000 Blackwell, SM120, PCIe)

### Scout decode throughput (NVFP4, fp8 KV, 70 W power cap)

| model | TP | GPUs | single stream | aggregate peak | tok/joule |
|---|---|---|---|---|---|
| **LFM2.5-1.2B-JP** | 1 | 1 | 212 t/s | **~24k tok/s** @ C256 | ~346 |
| LFM2.5-1.2B-JP DP×2 | 1+1 | 2 | 212 t/s | **~46k tok/s** @ C512 | ~329 |
| LFM2.5-1.2B-JP TP=2 | 2 | 2 | 352 t/s | ~24k tok/s (same ceiling) | ~173 |
| LFM2.5-8B-A1B | 4 | 4 | 297 t/s | ~6.3k tok/s @ C=8 | — |

**Rule of thumb for small dense scouts on a no-NVLink box:** TP buys single-stream latency
and KV headroom; it does NOT increase aggregate throughput (power-bound). For more
throughput, run independent TP=1 replicas (data-parallel).

### End-to-end research run (5→10 scout escalation, same question)

| config | wall time | findings/scout avg | sources | answer |
|---|---|---|---|---|
| qwen36 both roles (A) | 2m 17s | 200 chars | 6 | 167 chars (thin) |
| **qwen36 大将 + lfm25 足軽 (B)** | **1m 57s** | **1,925 chars** | **34** | **3,794 chars** |

**Punchline:** separating roles wins on every metric. The 1.2B scout is fast and
unencumbered by reasoning overhead — it focuses on searching, reading, and citing.
The 27B commander handles the cognitively expensive work (planning, judgment, synthesis).

---

## 5. Two-node scaling

Each model's tensor-parallel group should stay **within one node**. The clean split is
**per-role, across nodes**:

```
┌─ node A (scouts) ─────────────┐     ┌─ node B (commander) ──────────────┐
│ lfm25-1p2b-jp  TP=1 × N       │     │ qwen36  TP=4                       │
│ rrproxy :8010 → :8012,:8013…  │     │ :8011                              │
└───────────────────────────────┘     └────────────────────────────────────┘
            ▲                                        ▲
            │ ASHIGARU_WORKER_BASE_URL               │ ASHIGARU_ORCH_BASE_URL
            └───────────────  ashigaru  ─────────────┘
```

Ashigaru needs **zero code changes** — just point the two base URLs at the two nodes.
Scale scouts by adding more TP=1 replicas behind rrproxy (near-linear throughput scaling).

---

## 6. Gotchas (learned the hard way)

- **`ASHIGARU_ORCH_MAX_TOKENS` must be ≥ 16384 for reasoning-model commanders.**
  Reasoning models (qwen36, LFM2.5) emit `<think>…</think>` before the answer. With
  the default 3072, synthesis of 10 rich scout reports burns the whole budget on thinking
  and returns an **empty `content`** (finish_reason=length). The symptom is a silent empty
  answer with no error. Always set `ASHIGARU_ORCH_MAX_TOKENS=16384`.

- **`ASHIGARU_REQUEST_TIMEOUT=300` for multi-scout runs.**
  9–10 concurrent scouts all hit vLLM + SearXNG simultaneously. The default 120s times out
  under load. 300s is safe; 180s is usually enough if SearXNG is local.

- **compressed-tensors vs modelopt** — see §1. Wrong flag = garbage or load failure.

- **TP>1 on a no-NVLink/no-P2P box:** always add both
  `-e NCCL_P2P_DISABLE=1` **and** `--disable-custom-all-reduce` or NCCL hangs on init.
  TP=1 avoids this entirely (another reason to prefer DP over TP for small scouts).

- **Reasoning parsers differ:** `deepseek_r1` for LFM2.5-8B, `qwen3` for all Qwen3.x.

- **Tool-call parsers differ:** 27B-MTP → `qwen3_xml`, 35B-A3B → `qwen3_coder`.

- **Display GPU:** keep the desktop's GPU out of the TP/device set. On SAZANAMI the
  display GPU moves at every boot — check `nvidia-smi` and pin by UUID, not index.

- **First launch is slow:** NVFP4 kernel warmup is 1–3 min; wait for
  `Application startup complete` before sending requests.

- **Commander is the bottleneck**, not the fleet — with a slow 大将 the cheap scouts sit
  idle waiting for check-in responses. Pick a reasoning-capable commander, raise its
  `--max-num-seqs`, or reduce `ASHIGARU_MAX_CHECKINS`.

- **Start with 5 scouts, not 3.** The default 3 almost always triggers a quality-escalation
  (3→9 in the old config). Starting at 5 with ×2 escalation (5→10) skips the wasted
  thin round and is faster overall.

- **Commander thinking: OFF for synthesis, ON for planning** (measured, 8-config grid sweep,
  blind 3-judge accuracy scoring on a hard technical question). Findings:
  - `synth` thinking OFF was the top config on accuracy AND honesty while being ~4-5x faster on
    the synth step — thinking sometimes "reasons" a reasoning-model into a confident wrong claim;
    no-think stays closer to the grounded scout evidence. **Adopt `ASHIGARU_ORCH_THINK_SYNTH=off`.**
  - `plan` thinking OFF showed no benefit (slightly worse) — keep planning thinking ON.
  - **Player-coach (`COMMANDER_SCOUT`) did not raise accuracy** — the run with the MOST sources (20)
    still got the core fact (GPU generation) wrong. More sources ≠ more correct. Opt-in only.
  - **Caveat:** accuracy is dominated by scout/source quality, NOT the commander knobs — every
    config scored low because the web sources for the niche question were themselves wrong and a
    1.2B scout can't correct them. To actually raise the accuracy ceiling, upgrade the scout
    (e.g. LFM2.5-8B-A1B) or the sources — not the Commander's reasoning toggles.

- **Scout size has a sweet spot; SOURCE LANGUAGE breaks the ceiling** (measured, blind-judged,
  same question):
  - 1.2B-JP scout → overall 3.33 (fell for an H100/Hopper error from JA blogs).
  - 8B-A1B scout → 4.0 (avoided it). **8B is the sweet spot.**
  - 35B-A3B scout (TP=4, 4 GPUs) → 4.0 — *no gain* over 8B despite 4× the hardware. Bigger brains
    re-read the same wrong sources; the ceiling is the EVIDENCE, not the model.
  - **`ASHIGARU_SEARCH_LANGS=en,zh,ja` → 4.67 (best, and faster)** — the Commander writes each
    sub-question in the language whose primary sources are authoritative (English for AI/ML/sci,
    Chinese for much HW/LLM work) and spreads scouts across regions; synthesis translates back to
    the user's language. This is the real accuracy lever once the scout is decent — multilingual
    primary sources beat one language's secondhand blogs, and parallel scouts stop overlapping.
