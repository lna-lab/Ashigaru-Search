# KURA-Emaki (蔵絵巻) — *don't retrieve, navigate*

> The **蔵 (kura)** is a storehouse of documents. The **絵巻 (emaki)** is a picture-scroll you
> unroll to walk through it. KURA-Emaki compiles a bounded local corpus, **offline**, into a
> navigable scroll of skill-cards that the Ashigaru scouts **walk** at query time — instead
> of doing top-*k* similarity retrieval. 「検索から探索へ」 — from search to exploration.

This is Ashigaru-Search's second use-case. The live fleet searches the open web; KURA-Emaki
points the same Commander+scout machinery *inward* at a corpus you own.

---

## Why navigate instead of retrieve?

Fixed top-*k* retrieval (embed query → nearest neighbours → dump *k* chunks) is a single-shot,
lossy projection: it can't honour exact constraints, combine sparse clues, or refine a
hypothesis over several steps — the agent is a passive consumer of whatever *k* chunks return.

KURA-Emaki gives the agent a **map** instead. A scout reads the root bird's-eye card, **drills
coarse→fine** into the most relevant branch, reads leaf documents in full, and **backtracks**
from dead ends — and can **pivot sideways** along a typed knowledge graph to gather
cross-branch evidence. The same Commander check-ins the live fleet uses (`continue / regroup /
return`) map exactly onto `drill / backtrack / return-grounded`.

This is the publicly-described idea behind **Corpus2Skill** (arXiv:2604.14572, *"Don't Retrieve,
Navigate"*), with the lineage of **RAPTOR** (recursive cluster-summarise trees, 2401.18059) and
**GraphRAG** (entity-relationship graphs, 2404.16130). See [Clean-room note](#clean-room--licensing).

---

## Two artifacts from one corpus

A build produces a scroll directory with **two** navigable structures:

1. **The topic tree (絵巻).** Recursive clustering → an LLM-distilled `SKILL.md` card per node
   (topic label, summary, when-to-navigate, key entities) + an `INDEX.md` of children or leaf
   documents. The cards carry `name` + `description` frontmatter, so the scroll **doubles as a
   portable Anthropic Agent Skill** (github.com/anthropics/skills).

2. **The knowledge graph (蔵の索引).** A **co-occurrence** graph by default (`--graph`: `TERM`
   nodes + a single `CO_OCCURS` relation, zero-GPU), or a **fully ontology-typed**
   entity-relationship graph with `--graph-llm` (typed nodes + the LNA-ES edge vocabulary).
   Emitted as `graph.json` (served in-process) + `graph.cypher` (load into **Neo4j** or
   **PostgreSQL + Apache AGE**). Provenance `MENTIONED_IN` edges tie each entity back to its
   documents (capped at `MAX_DOCLINKS_PER_ENTITY` in the Cypher; `graph.json` keeps the full list).

---

## Build

```bash
# zero infrastructure — no model, no Docker, no network (heuristic cards + co-occurrence graph)
ashigaru-emaki ./my_corpus ./my_scroll --no-llm --graph

# zero-GPU by default: TF-IDF + spherical k-means clustering, cards distilled by your local fleet
ashigaru-emaki ./my_corpus ./my_scroll

# also build a knowledge graph (zero-GPU co-occurrence)
ashigaru-emaki ./my_corpus ./my_scroll --graph

# richer, typed graph via LLM ontology extraction (costs build-time tokens)
ashigaru-emaki ./my_corpus ./my_scroll --graph-llm

# opt-in embedding backend for better topical cohesion (and CJK); pulls sentence-transformers
ashigaru-emaki ./my_corpus ./my_scroll --embed            # default model
ashigaru-emaki ./my_corpus ./my_scroll --embed BAAI/bge-m3
```

Key flags: `--branching N` (tree fan-out, default 8) · `--leaf-max N` (max chunks per leaf,
default 10) · `--max-depth N` · `--chunk / --overlap` (word chunking, reuses `ashigaru-index`) ·
`--no-llm` (heuristic cards, no model needed) · `--quiet`.

The distiller calls your **worker** model (`ASHIGARU_WORKER_*`). It's an embarrassingly-parallel
"summarise every cluster" raid: thin or ungrounded cards fail a quality gate and are re-distilled
once with more context, then fall back to a heuristic card so a build **never hard-fails** on a
flaky small model.

## Serve

Point the existing fleet at a built scroll and ask as usual:

```bash
export ASHIGARU_EMAKI=./my_scroll
ashigaru "How does NVFP4 differ from FP8 for MoE inference?"
```

When `ASHIGARU_EMAKI` is set, the scouts gain the navigation tools and are **steered to start
from the scroll**; web/BM25 tools stay attached as a **hybrid fallback** (see
[trade-offs](#honest-trade-offs)). Via MCP, call the `emaki_navigate` tool (alongside
`deep_research`).

### Navigation tools (spoken in the usual `<tool>/<final>` protocol)

| tool | what it does |
|---|---|
| `tree_overview()` | root bird's-eye card + top branch index |
| `tree_open(node_id)` | drill into a branch — its card + children (or a leaf's documents) |
| `get_document(doc_id)` | read a leaf document in full |
| `graph_neighbors(entity)` | entities related to one, with **typed** edges (graph builds only) |
| `graph_related_docs(entity)` | documents that mention an entity → then `get_document` |

---

## The ontology — `edge_types.yaml`

The graph's relationships come from a fixed, typed vocabulary, **adopted with gratitude from
Lna-Lab's [AIOS](https://github.com/Tonoken3/AIOS) / LNA-ES v4.0** ontology system — 1.5 years of
Ken × Claude Code pair-programming. Ken's design principle, preserved:

> *"Write in a language future AI can read. Edge types carry meaning across millennia; node
> labels are time-bound. A `CONTRASTS` edge tells any future reader 'these are opposites' even
> if the node labels have drifted."*

Each edge type (`CAUSES`, `REQUIRES`, `CONTAINS`/`PART_OF`, `CONTRASTS`, `CONTRADICTS`,
`PRECEDES`/`FOLLOWS`, `TRANSFORMS_INTO`, `CO_OCCURS`, …) is immutable `UPPER_SNAKE_CASE` and
carries an `inverse`, a `weight_default`, a human `description`, and a time-resistant `ai_hint`.
LLM extraction is constrained to these names — which makes typed extraction **easier** for a
small scout than free-form triples, and yields a schema-stable graph. Extend the vocabulary by
editing `ashigaru/emaki/edge_types.yaml`.

### Loading the graph (the "書架" backend)

`graph.cypher` is portable openCypher:

```bash
# Neo4j / Memgraph
cat my_scroll/graph.cypher | cypher-shell

# PostgreSQL + Apache AGE — one server that is BOTH a graph index AND a shelf (書架) for
# documents and future multimodal payloads (JSONB / bytea / large objects; pgvector later)
# wrap each statement:  SELECT * FROM cypher('emaki', $$ <stmt> $$) as (v agtype);
```

The `MENTIONED_IN` provenance edges mean the graph alone can lead you back to source documents
(mirroring AIOS's `RAW_SOURCE` pattern), up to the per-entity cap below.

### Graph bounds (disclosed, never silent)

To keep a large corpus from producing an unusable graph, the builder bounds output and
**reports every drop** (in `graph.json` / `manifest.json` meta and on the build console):

- `MAX_ENTITIES = 4000`, `MAX_EDGES = 12000` — kept by salience / co-occurrence count.
- `MAX_DOCLINKS_PER_ENTITY = 25` provenance links per entity in the Cypher (`graph.json` keeps all).
- co-occurrence mode: drops a term in **>50%** of chunks (stopword-like) when there are ≥8 chunks;
  requires an edge count **≥2** when there are ≥20 chunks; ignores entity text longer than 60 chars.

---

## Honest trade-offs

KURA-Emaki is a **quality lever for hard, single-domain questions**, not a cheap default.

- **Token cost.** Navigation reads far more context than a BM25 hit (Corpus2Skill reports ~53k
  input tokens/query vs ~700 for BM25). For a tok/joule-minded fleet, reach for it when retrieval
  *can't* answer — not for every lookup. The web/BM25 fallback stays attached for that reason.
- **Open-domain / homogeneous corpora.** Flat retrieval often *wins* on open-domain factoid or
  uniform-tabular corpora (the paper's own caveat). Use KURA-Emaki on **bounded, single-domain**
  local corpora; keep the BM25 hybrid for the rest.
- **Zero-GPU by default, by choice.** TF-IDF clustering keeps the default dependency-light. The
  `\w+` tokenizer is coarse on languages without word spaces (CJK) — use `--embed` (a multilingual
  model) there.
- **Stale tree.** A scroll is compiled offline; rebuild when the corpus drifts. (A direct
  index-free `grep` mode over raw text is a natural future option for fast-changing corpora.)
- **Tree quality is upstream of everything.** Bad clusters → unnavigable cards. The quality gate
  filters stopword/boilerplate noise up front (a lesson from the AIOS reading-graph cypher, where
  scraper junk like `"You"/"Access"/"The"` leaked into nodes).

---

## Clean-room & licensing

The **idea** is public (Corpus2Skill 2604.14572; RAPTOR; GraphRAG; Voyager 2305.16291). The
upstream Corpus2Skill **repository is all-rights-reserved (no LICENSE)** — so **none of its code
is used here**. The clustering, distillation, graph, and Cypher emitters are independently
authored for Ashigaru and ship under Ashigaru's own license. The edge-type **ontology** is reused
from Lna-Lab's own AIOS with attribution, by the project owner's direction.

## Scroll layout

```
my_scroll/
  SKILL.md / INDEX.md        # root card (portable Agent Skill) + branch index
  manifest.json              # build params, counts, backend, graph mode
  tree.json                  # the node graph (machine truth)
  documents.json             # {doc_id: {source, text}}  — the 蔵 (full leaf store)
  entity_index.json          # {entity: [doc_id, ...]} cross-index
  nodes/<node_id>/SKILL.md   # per-node cards + INDEX.md
  graph.json / graph.cypher  # knowledge graph (with --graph / --graph-llm)
```

## References

- Corpus2Skill — *Don't Retrieve, Navigate* — arXiv:2604.14572
- RAPTOR — *Recursive Abstractive Processing for Tree-Organized Retrieval* — arXiv:2401.18059
- GraphRAG — *From Local to Global* — arXiv:2404.16130
- Voyager — open-ended embodied agent w/ skill library — arXiv:2305.16291
- Anthropic Agent Skills — github.com/anthropics/skills
- LNA-ES ontology — github.com/Tonoken3/AIOS
