# Quickstart & recipes

## 0. Prereqs
- A running OpenAI-compatible LLM server (the Ashigaru fleet). Reference:
  ```bash
  vllm serve sakamakismile/LFM2.5-8B-A1B-NVFP4 \
      --quantization modelopt --served-model-name lfm25-8b-a1b \
      --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.90 --port 8000
  ```
- Docker (for SearXNG).

## 1. Install + search engine
```bash
pip install -e ".[all]"
cd docker && docker compose up -d searxng
# sanity: JSON API should answer
curl -s 'http://localhost:8888/search?q=hello&format=json' | head -c 300
cd ..
cp .env.example .env
```

## 2. Run
```bash
ashigaru "Compare NVFP4 and MXFP4 for MoE inference on Blackwell."
ashigaru -k 8 "Summarize the current state of open local LLMs"   # 8 scouts
ashigaru "What's new in local LLMs? Answer in Japanese."         # multilingual: EN query, JA answer
ashigaru --json "..." > result.json                             # machine-readable
ashigaru --quiet "..."                                          # no progress on stderr
```

## 3. Add local documents
```bash
ashigaru-index ./papers ./papers.pkl
export ASHIGARU_RAG_INDEX=./papers.pkl
ashigaru "What do my local papers say about speculative decoding?"
```

## 4. Use a bigger Commander, small scouts
```bash
# scouts = LFM2.5 on :8000, commander = a 27B NVFP4 on :8001
export ASHIGARU_ORCH_BASE_URL=http://localhost:8001/v1
export ASHIGARU_ORCH_MODEL=qwen36-27b-nvfp4
ashigaru "Write a literature-style overview of FP4 quantization."
```

## 5. Python API
```python
import asyncio
from ashigaru import research, Config

async def main():
    res = await research("How does SearXNG aggregate engines?", Config())
    print(res.answer)
    for s in res.sources:
        print(s)

asyncio.run(main())
```

## 6. MCP (drive from Claude Code / Desktop)
```bash
ashigaru-mcp        # stdio
```
```json
{ "mcpServers": { "ashigaru": { "command": "ashigaru-mcp" } } }
```
Then ask your agent to research something — it calls `deep_research` and the local
fleet does the work.
