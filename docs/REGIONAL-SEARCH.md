# Regional SearXNG Fleet — a per-region search backend for the multilingual 足軽

**Goal.** Give the multilingual scout fleet (`ASHIGARU_SEARCH_LANGS`) a search backend that
(a) doesn't collapse when one engine blocks our IP, (b) routes each scout to the engines that are
actually authoritative *and* un-blocked for its language/region, and (c) can spread outbound load
across separate egress IPs so the public engines never see one IP burst-scraping.

This is the scaled form of the 2026-06-22 finding: **a research run's accuracy ceiling is SOURCE
quality, and the search backend's failure mode is upstream engines CAPTCHA-ing our egress IP.
Engine diversity fixes it; more containers behind the *same* IP do not.**

---

## The three levers (in order of cost)

1. **Engine diversity per region** — local, free, biggest immediate win. Each region's instance
   enables the engines that region's primary sources live on, plus robust default-off engines.
2. **Language-aware routing** — the scout's query script picks the matching instance. Pure code.
3. **Separate egress per region** — the real block-dodge: each instance's outbound goes through a
   different proxy / exit IP, so engines see distributed (and locally-originating) traffic. Needs
   external proxy/VPN endpoints — design the hooks now, Ken plugs in the endpoints.

---

## 1. Region → instance → engines

SearXNG's roster on this box (confirmed): `arxiv baidu bing brave duckduckgo github mojeek mwmbl
presearch qwant sogou startpage wikidata wikipedia yandex`. Honest constraint: it ships **native**
engines only for **ZH (baidu, sogou)** and **RU (yandex)**; JA/KO/AR/TR have no dedicated engine,
so those instances use general engines **+ the locale/`language` param** (a `language=ja` query to
bing/presearch returns JA-prioritised results). Native KO (naver) / JA (yahoo-jp) engines would be
custom SearXNG engine modules — a later add.

| instance | port | engines | default locale | notes |
|---|---|---|---|---|
| `searxng-global` | 8888 | bing, presearch, mojeek, mwmbl, qwant, brave, duckduckgo, startpage, arxiv, github, wikidata | en | tech/sci; arxiv+github carry the niche facets |
| `searxng-zh` | 8892 | baidu, sogou, bing, wikipedia | zh-CN | **native** Chinese engines — barely-hammered, robust |
| `searxng-ru` | 8893 | yandex, bing, mwmbl, wikipedia | ru | **native** Yandex; also strong for TR/Gulf adjacency |
| `searxng-ja` | 8894 | bing, presearch, mojeek, duckduckgo, wikipedia | ja | general + ja locale |
| `searxng-ko` | 8895 | bing, presearch, mojeek, wikipedia | ko | general + ko locale (naver = future engine module) |
| `searxng-mena` | 8896 | bing, yandex, presearch, wikipedia | ar / tr | MENA+Türkiye; locale set per query |

Each instance = the same `searxng/searxng` image, its own `settings.yml` (engine list + locale +
optional proxy), ~250 MB RAM. With 1024 GB RDIMM, dozens of instances are free.

## 2. Language-aware routing (Ashigaru `tools/web.py`)

- New config `ASHIGARU_SEARXNG_MAP`, e.g.
  `zh=http://localhost:8892/v1?,ru=...,ja=...,ko=...,ar=...,tr=...` (lang→base URL). `SEARXNG_URL`
  stays the `en`/default fallback so single-instance setups are unchanged.
- `web_search` detects the **script of the query** (no commander change needed):
  hiragana/katakana → `ja`; Hangul → `ko`; Arabic block → `ar`; Cyrillic → `ru`; Han-only → `zh`;
  else → `en`. It routes to the mapped instance **and** sends `language=<lang>` to SearXNG.
- Because the multilingual planner already writes each sub-question in its target language, the
  script heuristic routes each scout to the right regional instance automatically.

## 3. Separate egress — the ECLIPSE fleet IS the egress layer (no proxies to buy)

The hard part (distinct per-region exit IPs) is **already built**: the ECLIPSE remote-agent fleet
(`Lna-Lab/remote-agent-onboarding`) is 7 Ubuntu VMs — `ECLIPSE01-AURORA … 07-HARINA` — each with
**per-VM Surfshark OpenVPN/WireGuard**, i.e. a distinct **external IP + country** (the onboarding
skill verifies tunnel iface + external IP/country per node). So:

- Run a **region-tuned SearXNG on each ECLIPSE node**, VPN'd to that region's country. A JA query
  then leaves a JA IP, a ZH query a ZH/HK IP, etc. — locally-originating, distributed, far less
  CAPTCHA, better regional ranking. SAZANAMI's scouts (on the GPU box) call out to these nodes.
- **Search egress** = SearXNG on the node (the part that was getting blocked — the main win).
- **Fetch egress** (optional, deeper) = `ssh -D <port> <node>` gives SAZANAMI a SOCKS proxy
  exiting through that node's VPN'd IP; point `web_search`/`fetch_url`'s httpx client at the
  region's SOCKS for `fetch_url` too, so reads also originate regionally.
- The VMs are general Ubuntu (Docker-capable), so each just runs the `searxng/searxng` container
  with its region `settings.yml` — same recipe as local, on a VPN'd node.

**Open inputs (Ken holds these — kept out of the repo by the onboarding skill's security rules):**
node ↔ country assignment (which ECLIPSE = which region), and the SAZANAMI→node network path
(SSH alias / Tailscale / LAN). Given those, region routing is just a URL map (§2).

## 4. Rollout

- **Phase 1 (local, free, now):** stand up the region instances *on SAZANAMI* + per-region
  `settings.yml`, add the script-routing to `web_search`. Lights up native ZH/RU engines + locale
  and spreads load by region. (Single egress IP still — engine diversity is the protection here.)
- **Phase 2 (ECLIPSE egress):** move each region's SearXNG onto its country-matched ECLIPSE node;
  point the region URL map at the nodes. This is the real per-IP-block fix — and it's existing
  infra, not a purchase. Add the `ssh -D` SOCKS path for regional `fetch_url` if reads also need
  to originate locally.
- **Phase 3 (native engines):** custom SearXNG engine modules for naver (KO), yahoo-JP (JA), and a
  MENA/TR engine, for true regional-source depth where locale-on-general isn't enough.

The killer application is not tech-doc lookup but **macro / market / geopolitical intelligence for
the trading system** — regional first-language signal (湾岸/Türkiye/中文/русский) reaching the desk
before it surfaces in English media, gathered sovereignly through nodes that look local to each
region. That is the 足軽 fleet's shining stage.

## 5. Honest constraints

- Phase 1 alone (one egress IP) reduces block frequency (load spread across regions/engines) but
  does **not** eliminate it for engines shared across instances (bing). Phase 2 is the real fix.
- True regional-engine specialization is currently only ZH + RU; everyone else is "general +
  locale" until Phase 3 adds native modules.
- Keep concurrency modest per instance; diversity buys headroom, it isn't a license to burst.
