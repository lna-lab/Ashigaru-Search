# SearXNG setup & regional search recipes

The Ashigaru fleet searches through a **self-hosted SearXNG** (no API keys). This guide
covers bringing it up with Docker and tuning it for **English** and **Chinese** regions.

> Verified on Ubuntu 26.04 LTS, Docker 29.1.3, `searxng/searxng:latest`. SearXNG was
> ready ~10 s after `up`; sample result counts below are real.

## 1. Bring it up

```bash
cd docker
docker compose up -d searxng          # pulls searxng/searxng, starts on :8888
# verify the JSON API (this is what the scouts call):
curl -s 'http://localhost:8888/search?q=hello&format=json' | head -c 200
```

Two settings make the API usable by the fleet (already set in `searxng/settings.yml`):

```yaml
server:
  limiter: false        # allow programmatic queries (otherwise SearXNG blocks bots)
search:
  formats: [html, json] # the JSON API is OFF by default — this enables it
```

`use_default_settings: true` keeps **all** of SearXNG's stock engines enabled, so the
engines below work without extra config.

## 2. Query shape

```
GET http://localhost:8888/search?q=<query>&format=json&language=<code>&engines=<a,b,c>
```

- `language` — e.g. `en-US`, `en-GB`, `zh-CN`, `zh-TW`, `ja-JP`. Steer results to a region.
- `engines` — comma list to restrict to specific engines (omit to use all).

In Ashigaru, `web_search` already does `&format=json` and forwards an optional `lang`
arg, so a scout can target a region directly:

```
<tool>{"name":"web_search","arguments":{"query":"大语言模型 最新","lang":"zh-CN"}}</tool>
```

## 3. Engine availability (measured here)

| engine | results | good for |
|---|---:|---|
| google | 18 | EN + general (very strong) |
| duckduckgo | 10 | EN + general, privacy |
| bing | 10 | EN + ZH |
| baidu | 8–9 | **Chinese** |
| sogou | 10 | **Chinese** |
| quark | 9 | **Chinese** |
| brave | 0 | flaky here (often rate-limited) — optional |

## 4. English-speaking regions

```bash
curl -s 'http://localhost:8888/search?q=Blackwell+GPU+NVFP4&format=json&language=en-US' \
  | jq '.results[0:3] | .[] | {title, engine}'
# → 18 results (google/duckduckgo/bing): "Introducing NVFP4 for Efficient…", etc.
```

Recommended engines: `google, duckduckgo, bing, wikipedia`. Use `language=en-US` (or
`en-GB`). That's the default behaviour — nothing extra to configure.

## 5. Chinese-speaking regions

```bash
curl -s 'http://localhost:8888/search?q=大语言模型&format=json&language=zh-CN&engines=baidu,bing,sogou,quark,duckduckgo' \
  | jq '.results[0:3] | .[] | {title, engine}'
# → e.g. "什么是大语言模型?| NVIDIA 词汇表" (baidu), SuperCLUE 中文大模型测评 (google)…
```

For a Chinese-first instance, pin the engines in `searxng/settings.yml` so they're always
on and others stay quiet:

```yaml
use_default_settings: true
search:
  formats: [html, json]
  default_lang: "zh-CN"
engines:
  - { name: baidu,      disabled: false }
  - { name: bing,       disabled: false }
  - { name: sogou,      disabled: false }
  - { name: quark,      disabled: false }
  - { name: duckduckgo, disabled: false }
  - { name: google,     disabled: false }
```

Use `language=zh-CN` (mainland) or `zh-TW` (traditional). Baidu/Sogou/Quark give the best
mainland coverage; Bing and Google add international Chinese-language sources.

## 6. Lifecycle

```bash
docker compose logs -f searxng     # watch
docker compose restart searxng     # after editing settings.yml
docker compose down                # stop
```

Edit `searxng/settings.yml` then `restart` to apply. Change `server.secret_key` before
exposing the instance beyond localhost.
