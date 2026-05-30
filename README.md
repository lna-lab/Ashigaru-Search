<div align="center">

# 🏯 Ashigaru-Search

**A fleet of small, fast, local LLMs that fan out to search — then report back.**

*One **Commander** plans, many **Ashigaru** scouts search the web + your docs in parallel, the Commander synthesizes a cited answer.*

<sub>足軽 (ashigaru) = a foot-soldier · 大将 (taishō) = the commander</sub>

Apache-2.0 · by [Lna-Lab](https://huggingface.co/sakamakismile) · works with any OpenAI-compatible endpoint

</div>

---

## Why it works — cheap scouts, a smart Commander

The bet behind Ashigaru-Search: **don't make one expensive model think hard — make a
swarm of cheap ones search wide, and put the judgment in the orchestration.**

A frontier agent (like Claude Code) researches by planning, fanning out to sub-agents,
reading sources, noticing when a result is thin, and digging more — all *implicitly*,
inside one big, costly context. Ashigaru-Search takes that same loop and **externalizes
it** onto a swarm of tiny local models:

**1. Breadth is almost free.** A small MoE like
[LFM2.5-8B-A1B-NVFP4](https://huggingface.co/sakamakismile/LFM2.5-8B-A1B-NVFP4) (1.5B
active) fits one 16 GB GPU and scales near-linearly — one card runs *dozens* of scouts at
once. Covering a question from many angles at once beats one model going deep on a single
thread.

**2. The Commander makes a weak swarm reliable.** A 1.5B-active scout, alone, wavers — it
hallucinates or chases dead ends. So the *intelligence isn't in the scout, it's in the
scaffold*:
- **Plan** — split the question into sharp, non-overlapping sub-questions.
- **Supervise live** — after each lead, the Commander orders *continue / regroup / return*,
  so a scout never burns its budget on junk.
- **Gate quality** — thin reports (too short, no sources, "couldn't find") are caught.
- **Escalate** — if a run comes back thin, automatically redo wider (S→M→L), so effort
  scales to difficulty instead of being fixed up front.

Cheap foot-soldiers plus a general who plans the formation, reads the field reports, and
redirects forces — *that's the whole design, and the name.* The individual 足軽 is
expendable and not very bright; the 大将's command is what wins the engagement.

**3. You hold the dials.** What's a hidden judgment call inside a big agent is explicit
here — search density (`S/M/L`), step budgets, live supervision, escalation depth — all
knobs, all visible in the trace.

**4. It's yours.** Fully local, no API keys, nothing leaves the box. The cost of
"research 50 things at once" is a warm GPU, not a metered bill.

> **Honest caveat:** it's a swarm of *small* models. On concrete, well-sourced questions
> it's accurate and cited; on open-ended "what's notable" surveys it can still confidently
> invent things (see **Live runs** below). The scaffold raises the floor — it doesn't turn
> a 1.5B model into a domain expert.

In short: one GPU, a swarm of cheap scouts, and a Commander that plans, watches, judges,
and escalates — the agentic-search loop, made local, cheap, and controllable.

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

### ⚡ 30-second taste — no GPU, no Docker, no network

Turn any folder of text into a navigable knowledge **scroll** (here, the repo itself). This is
the offline side of **KURA-Emaki** (below) — no model, no server, ~1 second:

```bash
pip install -e .                                   # core only — no model needed
ashigaru-emaki . ./demo_scroll --no-llm --graph    # cluster + heuristic cards + co-occurrence graph
cat  demo_scroll/SKILL.md                          # a navigable Anthropic Agent-Skill card
head demo_scroll/graph.cypher                      # a knowledge graph for Neo4j / Apache AGE
```

Real, inspectable artifacts, built fully offline. Sanity-check the test suites the same way (no
infra): `PYTHONPATH=. python tests/test_smoke.py && PYTHONPATH=. python tests/test_emaki.py`.

### 🚀 Full fleet — live web search + local LLMs

```bash
# 1) install
pip install -e ".[all]"          # or: pip install ashigaru-search[all]   (--embed also needs [emaki])

# 2) start the search engine (self-hosted, no key) — see docker/SEARCH_SETUP.md
#    for English / Chinese regional engine recipes (baidu, sogou, bing, …)
docker compose -f docker/docker-compose.yml up -d searxng

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

## 🗺️ KURA-Emaki (蔵絵巻) — *don't retrieve, navigate*

The second use-case: compile a bounded local corpus, **offline**, into a navigable **scroll**
the scouts *walk* instead of retrieving top-*k* chunks. 「検索から探索へ」 — from search to
exploration. The **蔵 (storehouse)** of documents becomes an **絵巻 (picture-scroll)** you unroll:
a clustered topic tree of `SKILL.md` cards (a portable Anthropic Agent Skill), **plus** a
**knowledge graph** the scouts pivot across (co-occurrence by default, fully **ontology-typed**
with `--graph-llm`), exported as `graph.cypher` for **Neo4j or PostgreSQL + Apache AGE**.

```bash
# build a scroll — zero infra (no model/network): heuristic cards + co-occurrence graph
ashigaru-emaki ./my_corpus ./my_scroll --no-llm --graph

# better cards: drop --no-llm so your local fleet distils each cluster (needs the worker LLM)
ashigaru-emaki ./my_corpus ./my_scroll --graph

# serve: scouts drill the tree (tree_overview → tree_open → get_document) and pivot the graph
export ASHIGARU_EMAKI=./my_scroll
ashigaru "How does NVFP4 differ from FP8 for MoE inference?"
```

A scout reads the bird's-eye card, **drills coarse→fine**, reads leaves in full, and
**backtracks** from dead ends — the Commander's `continue/regroup/return` becomes
`drill/backtrack/return-grounded`. Web/BM25 stays attached as a hybrid fallback.

It's a **quality lever for hard, single-domain questions**, not a cheap default (navigation reads
far more tokens than a BM25 hit). Full design, ontology attribution (Lna-Lab's
[AIOS](https://github.com/Tonoken3/AIOS) / LNA-ES), trade-offs, and clean-room notes:
**[docs/KURA-Emaki.md](docs/KURA-Emaki.md)**.

## Configuration

All via env / `.env` (see `.env.example`). Highlights:

| var | default | meaning |
|---|---|---|
| `ASHIGARU_WORKER_BASE_URL` | `http://localhost:8000/v1` | the Ashigaru fleet endpoint (vLLM) |
| `ASHIGARU_WORKER_MODEL` | `lfm25-8b-a1b` | scout model |
| `ASHIGARU_ORCH_BASE_URL` / `_MODEL` | = worker | commander (pluggable; point at a bigger model if you like) |
| `SEARXNG_URL` | `http://localhost:8888` | search backend |
| `ASHIGARU_RAG_INDEX` | — | set to enable local BM25 doc tools |
| `ASHIGARU_EMAKI` | — | set to a built KURA-Emaki scroll dir to enable tree+graph navigation |
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

### Mid-search check-ins (the Commander steers live)

A scout is a tiny model and can dig aimlessly. So after each lead it finds, it reports
back — *"Commander, I found '…', continuing?"* — and the Commander (which sees the whole
goal) gives one of three orders:

- **continue** — only if the lead is genuinely valuable and worth more digging;
- **regroup** — a better angle exists: file what's found as an interim note and re-launch
  the scout on a revised focus;
- **return** — enough found, or this dig is low-value: come home and write the report now.

This biases *against wasted digging* — the small scout doesn't burn its step budget chasing
dead ends, because the Commander okays each dig. Controlled by `ASHIGARU_SUPERVISE` (on),
`ASHIGARU_SUPERVISE_AFTER` (1), `ASHIGARU_MAX_CHECKINS` (5). Set `ASHIGARU_SUPERVISE=0` for
fewer LLM calls / lower latency.

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
