# The Fleet Recipe — serving 大将 + 足軽 locally (NVFP4)

How to stand up the **commander + scout** fleet that Ashigaru-Search drives, using
all-local **NVFP4** models on stock **`vllm/vllm-openai:v0.22.0`** — single-node or
across two nodes (TP=4 scouts + TP=8 commander).

Ashigaru only ever talks to **two OpenAI-compatible endpoints**:

| role | env | who |
|---|---|---|
| 足軽 (scouts) | `ASHIGARU_WORKER_BASE_URL` / `_MODEL` | many cheap fast models, fanned out |
| 大将 (commander) | `ASHIGARU_ORCH_BASE_URL` / `_MODEL` | one smart model: plans, steers, synthesizes |

They can be the **same server**, **two ports on one box**, or **two separate nodes**.
That is the whole multi-node story — no code, just URLs (see §5).

---

## 1. The models (all NVFP4, all load on stock vLLM 0.22.0)

Measured on **7× RTX PRO 2000 Blackwell** (16 GB, ~70 W, SM120, PCIe — no NVLink).

| model | role | total / active | quant | served-name | vLLM flag that matters |
|---|---|---|---|---|---|
| **LFM2.5-8B-A1B-NVFP4** | scout (or fast 大将) | 8B / **1B** | modelopt | `lfm25-8b-a1b` | `--quantization modelopt` |
| **Huihui-Qwen3.6-27B-NVFP4-MTP** | commander | 27B (hybrid) | modelopt | `huihui-qwen36-27b-local` | `--quantization modelopt` + MTP |
| **Huihui-Qwen3.6-35B-A3B-NVFP4** | commander (best) | 35B / **3B** | **compressed-tensors** | `huihui-qwen36-35b-a3b` | **no `--quantization`** (auto-detect) |

> ⚠️ **modelopt vs compressed-tensors.** The two quant toolchains take different vLLM
> flags. The modelopt builds need `--quantization modelopt`; the compressed-tensors build
> (35B-A3B) must **omit** it and let vLLM auto-detect `quantization=compressed-tensors`.
> Passing `modelopt` on the 35B (or omitting it on the others) → garbage / load failure.

> ℹ️ The 35B-A3B model card says "requires vLLM nightly (cu130)". **Not true for 0.22.0** —
> the image already ships `Qwen3_5MoeForConditionalGeneration` + `Qwen3_5MoeMTP`,
> compressed-tensors 0.15.0.1 (`nvfp4_pack_quantized` + `TENSOR_GROUP`), and the
> Gated-DeltaNet kernels. It loads and serves unmodified.

---

## 2. Serve each (copy-paste)

All use the official image; mount the model dir read-only at `/model`.

**足軽 — LFM2.5-8B-A1B (scouts), TP=4, :8000**
```bash
docker run -d --name vllm-scouts --runtime nvidia --gpus '"device=0,1,2,3"' \
  -p 8000:8000 -v /models/LFM2.5-8B-A1B-NVFP4:/model:ro --shm-size 8g \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  --entrypoint vllm vllm/vllm-openai:v0.22.0 serve /model \
    --served-model-name lfm25-8b-a1b --quantization modelopt \
    --tensor-parallel-size 4 --max-model-len 32768 --max-num-seqs 16 \
    --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 \
    --reasoning-parser deepseek_r1 --chat-template /model/chat_template.jinja \
    --host 0.0.0.0 --port 8000
```

**大将 — Qwen3.6-35B-A3B (recommended), TP=8, :8001**
```bash
docker run -d --name vllm-commander --runtime nvidia --gpus '"device=0,1,2,3,4,5,6,7"' \
  -p 8001:8001 -v /models/Huihui-Qwen3.6-35B-A3B-abliterated-NVFP4:/model:ro --shm-size 32g \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  --entrypoint vllm vllm/vllm-openai:v0.22.0 serve /model \
    --served-model-name huihui-qwen36-35b-a3b --trust-remote-code \
    --tensor-parallel-size 8 --max-model-len 32768 --max-num-seqs 8 \
    --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 \
    --reasoning-parser qwen3 --chat-template /model/chat_template.jinja \
    --host 0.0.0.0 --port 8001
    # agentic tools (optional):  --enable-auto-tool-choice --tool-call-parser qwen3_coder
    # MTP speculative (optional): --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
```

**大将 — Qwen3.6-27B-MTP (alt), TP=4, :8001** — modelopt + MTP speculative decoding:
```bash
  ... serve /model --served-model-name huihui-qwen36-27b-local --trust-remote-code \
    --quantization modelopt --tensor-parallel-size 4 --max-model-len 65536 \
    --max-num-seqs 8 --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
    --chat-template /model/chat_template.jinja \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --host 0.0.0.0 --port 8001
```

---

## 3. Wire it to Ashigaru

`.env`:
```bash
# scouts
ASHIGARU_WORKER_BASE_URL=http://localhost:8000/v1
ASHIGARU_WORKER_MODEL=lfm25-8b-a1b
# commander (point at the smart node)
ASHIGARU_ORCH_BASE_URL=http://localhost:8001/v1
ASHIGARU_ORCH_MODEL=huihui-qwen36-35b-a3b
# fleet size = practical sweet spot
ASHIGARU_MAX_CONCURRENCY=8
ASHIGARU_FLEET_SIZE=8
```
Then `ashigaru "8 your question"` (the leading `8 ` requests 8 scouts).

---

## 4. Numbers (measured, this rig)

**Per-model decode throughput** (NVFP4, `ignore_eos`, fp8 KV):

| model | config | single | C=8 |
|---|---|---|---|
| LFM2.5-8B-A1B | TP=4 | 297 t/s | 1357 t/s |
| LFM2.5-8B-A1B | TP=2 † | 209 t/s | 1016 t/s |
| Qwen3.6-27B-MTP | TP=4, MTP n=3 | 80 t/s | 315 t/s |
| **Qwen3.6-35B-A3B** | TP=4, no MTP | **163 t/s** | 797 t/s |

† TP=2 on a P2P-broken GPU pair (see §6); a healthy pair is faster.

**One full 8-scout research run, same question, by commander brain:**

| commander | wall time | accuracy | sources |
|---|---|---|---|
| LFM2.5-8B-A1B | **71 s** | low (mislabeled H100 as Blackwell) | 5 |
| Qwen3.6-27B-MTP | 173 s | high | 12 |
| **Qwen3.6-35B-A3B** | 126 s | **highest** (concrete TTFT/stack numbers) | 14 |

**Punchline:** the 35B-A3B is *bigger* than the 27B but runs **2× faster** (3B active —
sparse MoE), so it eased the commander bottleneck (173→126 s) *and* gave the best
synthesis. **The commander is the lever: a "big but cheap-to-run" MoE wins.**

---

## 5. Two-node scaling (TP=4 scouts + TP=8 commander)

Each model's tensor-parallel group should stay **within one node** (TP all-reduce wants
the local PCIe/NVLink fabric — spanning nodes needs Ray + a fast network and usually isn't
worth it). The clean split is **per-role, across nodes**:

```
┌─ node A (scouts) ─────────┐     ┌─ node B (commander) ───────┐
│ LFM2.5-8B-A1B  TP=4       │     │ Qwen3.6-35B-A3B  TP=8       │
│ :8000  lfm25-8b-a1b       │     │ :8001  huihui-qwen36-35b-a3b│
└───────────────────────────┘     └─────────────────────────────┘
            ▲                                   ▲
            │ ASHIGARU_WORKER_BASE_URL          │ ASHIGARU_ORCH_BASE_URL
            └──────────────  ashigaru  ─────────┘
```

Ashigaru needs **zero changes** — just point the two base URLs at the two nodes:
```bash
ASHIGARU_WORKER_BASE_URL=http://node-a.lan:8000/v1   # TP=4 scout fleet
ASHIGARU_ORCH_BASE_URL=http://node-b.lan:8001/v1     # TP=8 commander
```
Bind each vLLM to `0.0.0.0` (already in §2) and open the ports between nodes.
Bigger commander node (TP=8) = headroom to run a larger 大将 or raise its `--max-num-seqs`
so the check-in storm (up to `ASHIGARU_MAX_CONCURRENCY` scouts consulting at once) never
queues.

---

## 6. Gotchas (learned the hard way)

- **compressed-tensors vs modelopt** — see §1. Wrong flag = garbage or load failure.
- **A bad GPU P2P pair hangs TP.** On this box GPUs **5↔6** hang NCCL init (topo is all
  `NODE`/PCIe but that pair won't P2P). For TP=2 on such a pair you need **both**
  `-e NCCL_P2P_DISABLE=1` **and** `--disable-custom-all-reduce` (vLLM's CUSTOM all-reduce
  needs P2P → `EngineDeadError` on the first forward otherwise). **TP≥3 on PCIe auto-disables
  custom all-reduce**, so TP=4/8 commanders are unaffected. Prefer known-good pairs for TP=2.
- **Reasoning parsers differ:** `deepseek_r1` for LFM2.5, `qwen3` for the Qwen3.x models.
- **Tool-call parsers differ:** 27B → `qwen3_xml`, 35B-A3B → `qwen3_coder`.
- **Display GPU:** keep the desktop's GPU out of the TP set (here GPU 2 holds gnome-shell).
- **Downloads:** `HF_HUB_DISABLE_XET=1 hf download …` avoids a Xet socket stall on this box.
- **First launch is slow:** torch.compile + NVFP4 warmup is 1–3 min; wait for
  `Application startup complete`.
- **Commander is the bottleneck**, not the fleet — with a slow 大将 the cheap scouts sit
  idle waiting on check-ins. Pick a sparse MoE 大将, raise its `--max-num-seqs`, or lower
  `ASHIGARU_MAX_CHECKINS`.
