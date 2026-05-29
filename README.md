<div align="center">

# 🏯 Ashigaru-Search

**A fleet of small, fast, local LLMs that fan out to search — then report back.**

*One **Commander** plans, many **Ashigaru** scouts search the web + your docs in parallel, the Commander synthesizes a cited answer.*

<sub>足軽 (ashigaru) = a foot-soldier · 大将 (taishō) = the commander</sub>

Apache-2.0 · by [Lna-Lab](https://huggingface.co/sakamakismile) · works with any OpenAI-compatible endpoint

</div>

---

## Why

Big agentic search harnesses spawn sub-agents, gather their results, and synthesize.
Ashigaru-Search gives you that pattern **fully local** with a *swarm of tiny models*:
quantize something like [LFM2.5-8B-A1B-NVFP4](https://huggingface.co/sakamakismile/LFM2.5-8B-A1B-NVFP4)
(fits one 16 GB Blackwell, ~10 full-context sessions, near-linear aggregate throughput)
and suddenly one GPU can run **dozens of search scouts at once**. Cheap tokens →
breadth-first research.

```
                 ┌──────────────────── COMMANDER (orchestrator) ──────────────────────┐
   question ───▶ │  plan: split into K sharp sub-questions                              │
                 └───────────────┬──────────────────────────────────────────┬──────────┘
                                 │ fan-out (async, capped by --concurrency)   │
              ┌──────────────────┼───────────────────┬──────────────────┐    │
          Scout#1            Scout#2             Scout#3   …   Scout#K        │ collect
        (LLM + tools)      (LLM + tools)       (LLM + tools)                  │
            │  web_search → fetch_url → doc_search → read_chunk → <final>     │
            └──────────────────┴───────────────────┴──────────────────┘    ▼
                                                       COMMANDER synthesize → cited answer
```

- **Tools:** `web_search` + `fetch_url` (self-hosted **SearXNG**, no API key) and
  `doc_search` + `read_chunk` (local **BM25** corpus).
- **Model-agnostic:** scouts and commander are any OpenAI-compatible endpoint
  (vLLM, llama.cpp server, SGLang, OpenAI…). LFM2.5's native Pythonic tool calls are
  understood, but the protocol works with *any* instruct model.
- **Two front doors:** a **CLI** and an **MCP server** — so Claude Code / Claude Desktop
  (or any MCP client) can offload research to your local fleet via a `deep_research` tool.

## Quickstart

```bash
# 1) install
pip install -e ".[all]"          # or: pip install ashigaru-search[all]

# 2) start the search engine (self-hosted, no key) — see docker/SEARCH_SETUP.md
#    for English / Chinese regional engine recipes (baidu, sogou, bing, …)
cd docker && docker compose up -d searxng && cd ..

# 3) point at your local LLM fleet (a vLLM server). Example for LFM2.5-8B-A1B-NVFP4:
#    vllm serve sakamakismile/LFM2.5-8B-A1B-NVFP4 --quantization modelopt \
#        --served-model-name lfm25-8b-a1b --max-num-seqs 16 --port 8000
cp .env.example .env             # edit ASHIGARU_WORKER_* if needed

# 4) send the ashigaru
ashigaru "What changed in NVFP4 support across recent vLLM releases?"
```

You'll see the commander plan, the Ashigaru scouts fan out with live tool calls, then a synthesized,
source-cited answer.

### Local documents (RAG)

```bash
ashigaru-index ./my_docs ./index.pkl        # build a BM25 index (.txt/.md/.pdf)
export ASHIGARU_RAG_INDEX=./index.pkl        # now scouts also get doc_search/read_chunk
ashigaru "Summarize our design decisions about the cache layer."
```

### As an MCP server (let Claude Code drive the fleet)

```bash
ashigaru-mcp        # stdio MCP server exposing deep_research(query, max_subquestions)
```

Register it in your MCP client, e.g. Claude Code `settings.json`:

```json
{ "mcpServers": { "ashigaru": { "command": "ashigaru-mcp" } } }
```

Now your cloud agent can say *"research X"* and the **local Ashigaru fleet** does the legwork.

## Configuration

All via env / `.env` (see `.env.example`). Highlights:

| var | default | meaning |
|---|---|---|
| `ASHIGARU_WORKER_BASE_URL` | `http://localhost:8000/v1` | the Ashigaru fleet endpoint (vLLM) |
| `ASHIGARU_WORKER_MODEL` | `lfm25-8b-a1b` | scout model |
| `ASHIGARU_ORCH_BASE_URL` / `_MODEL` | = worker | commander (pluggable; point at a bigger model if you like) |
| `SEARXNG_URL` | `http://localhost:8888` | search backend |
| `ASHIGARU_RAG_INDEX` | — | set to enable local doc tools |
| `ASHIGARU_MAX_SUBQUESTIONS` | `6` | how many scouts to fan out |
| `ASHIGARU_MAX_CONCURRENCY` | `16` | concurrent scouts in flight |
| `ASHIGARU_WORKER_MAX_STEPS` | `6` | tool calls per scout |

## How a scout thinks

Each Ashigaru scout follows a tiny, model-agnostic protocol:

```
<tool>{"name":"web_search","arguments":{"query":"vLLM NVFP4 changelog"}}</tool>
        ← tool result fed back →
<tool>{"name":"fetch_url","arguments":{"url":"https://…"}}</tool>
        ← tool result fed back →
<final>
Findings grounded only in what was read…
Sources:
- https://… — supports claim X
</final>
```

(LFM2.5's native `<|tool_call_start|>[web_search(query="…")]<|tool_call_end|>` is also parsed.)

## Pick your search density

Lead the question with a **half-width tag + space** to tell the Commander how many scouts
to send (overrides `-k`, clamped 1–12):

```bash
# explicit number
ashigaru "3 compare vLLM and SGLang for serving LLMs"     # → exactly 3 scouts
ashigaru "5 大規模言語モデルの量子化手法を比較して"        # → 5 scouts

# S / M / L density — a fraction of the fleet, so agents don't have to pick a number
ashigaru "S quick sanity check on FlashAttention 3"       # → 10% of fleet  (ceil, min 1)
ashigaru "M what's new in NVFP4 tooling?"                 # → 50% of fleet
ashigaru "L deep survey of MoE serving frameworks"        # → 100% of fleet

ashigaru "how does PagedAttention work?"                  # no tag → default fan-out
```

`S`/`M`/`L` map to **10% / 50% / 100%** of `ASHIGARU_FLEET_SIZE` (default 10 → **S=1,
M=5, L=10**), rounded up, minimum 1. So a calling agent can just dial *density* — light,
medium, or deep — and the Commander sizes the raid and splits the question accordingly.

### Quality gate & escalation

After the scouts report, the Commander checks whether the run is **substantive** — each
report must clear a heuristic (long enough, has sources, not "couldn't find…") and,
optionally, an LLM judge (`ASHIGARU_QUALITY_JUDGE`, on by default) that asks *"does this
actually answer the sub-question?"*. If too few reports are solid, it **escalates the
density and redoes the raid**:

```
S (1) ──thin──▶ M (5) ──thin──▶ L (10) ──▶ stop (top tier)
numeric N ──thin──▶ N×3 (capped at 12)
```

Up to `ASHIGARU_MAX_ESCALATIONS` times (default 2). So a quick `S` look that comes back
empty is automatically re-run wider — *"go back and dig harder."* Turn it off with
`ASHIGARU_ESCALATE=0`.

## Live runs — what it actually feels like

Real runs on **one 16 GB RTX PRO 2000 Blackwell**, with **LFM2.5-8B-A1B-NVFP4**
(1.5B active) playing *both* Commander and every scout, searching a local SearXNG.
Wall-clock end to end (plan → parallel scouts → synthesis):

| # | question (angle) | scouts | sources | time | takeaway |
|---|---|---:|---:|---:|---|
| 1 | "compare vLLM and SGLang: what is each best at?" (comparison) | 3 | 2 | **~78 s** | solid, balanced synthesis; source quality varied |
| 2 | "notable small open LLMs <10B in 2026 & strengths" (open survey) | 4 | 3 | **~111 s** | ⚠️ confident **hallucinations** — verify open-ended surveys |
| 3 | "比较 NVFP4 和 INT8 量化的优缺点" (Chinese, technical) | 3 | 5 | **~76 s** | ✅ accurate, fluent Chinese, cited NVIDIA docs |
| — | "what is NVFP4 & why good on Blackwell?" (technical, fewer steps) | 3 | 2 | **~48 s** | ✅ accurate, cited NVIDIA + Red Hat |

What the runs teach:

- **Concrete/technical questions with findable sources are the sweet spot** (#3, #4): the
  scouts fetch primary docs and the synthesis is accurate and cited — even cross-lingual
  (Chinese query → Chinese answer, Chinese + English sources via baidu/google).
- **Open-ended "what's notable in X" surveys are risky** (#2): when web evidence is thin,
  a 1.5B-active model fills gaps *confidently but wrongly* (it invented "Inflection's
  Phi-3", "NVIDIA's Falcon-130"). Treat survey-style answers as leads to verify, not facts.
- **Latency = breadth × depth.** More scouts and more tool steps mean more wall-clock:
  ~48 s for a quick 3-scout run, ~75–110 s when each of 3–4 scouts runs 5–6 tool steps.
  Planning and synthesis also spend a real reasoning budget. Tune with the scout count,
  `ASHIGARU_WORKER_MAX_STEPS`, and `ASHIGARU_MAX_CONCURRENCY`.

All of it is local — no API keys, no data leaving the box.

## Pairs well with

- **[LFM2.5-8B-A1B-NVFP4](https://huggingface.co/sakamakismile/LFM2.5-8B-A1B-NVFP4)** — the
  reference scout: 8B-A1B hybrid MoE in NVFP4, ~10 full-128K sessions and 4.6k tok/s
  aggregate per 16 GB Blackwell card.

## License

Apache-2.0 © Lna-Lab.

---

<div align="center">

**🔬 Lna-Lab** · *send the ashigaru, keep the tokens local*

</div>
