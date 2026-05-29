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
