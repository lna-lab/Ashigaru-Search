#!/usr/bin/env bash
# Zero-infrastructure KURA-Emaki demo — no model, no Docker, no network.
# Builds a navigable knowledge scroll from examples/corpus/ and prints the artifacts.
#
#   bash examples/emaki_no_llm.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root
OUT="examples/demo_scroll"

echo "==> building scroll offline (heuristic cards + co-occurrence graph)…"
ashigaru-emaki examples/corpus "$OUT" --no-llm --graph --leaf-max 2 --branching 3

echo
echo "==> root Agent-Skill card  ($OUT/SKILL.md):"
sed -n '1,16p' "$OUT/SKILL.md"

echo
echo "==> knowledge graph  ($OUT/graph.cypher)  — load into Neo4j or PostgreSQL+Apache AGE:"
sed -n '1,8p' "$OUT/graph.cypher"

echo
echo "Done. To navigate it with the fleet (needs your local LLM + SearXNG):"
echo "    export ASHIGARU_EMAKI=$OUT"
echo "    ashigaru \"How does NVFP4 differ from FP8 for MoE inference?\""
