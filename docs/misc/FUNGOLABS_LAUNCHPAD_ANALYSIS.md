# FungoLabs Launchpad — external reference analysis

**Date:** 2026-11-08 · **Target:** <https://fungolabs.org/launchpad/> · **Method:** live walkthrough in
headless Chrome (a11y snapshots, screenshots, network/console capture, Lighthouse), plus static
analysis of the shipped bundle, route chunks, API responses and response headers.

Scope: what the product is, how it is built, what is worth porting into sloughGPT, and what to
avoid. Three kanban cards were raised from this (see _Cards raised_). This is a reference note, not
a spec — nothing here changes scope on its own; `docs/PRODUCT_ENGINEERING.md` stays the source of
what we build.

---

## 1. What the product is

An NFT launchpad for "encrypted NFTs on Ethereum": collections whose **traits and pictures are
sealed with FHE (Zama's protocol)** until the holder chooses to open them. Tagline: _Seen by all.
Known by one._

| Surface                                             | What it does                                                                                                                                                            |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Home                                                | WebGL voxel hero; `01 · SEALING` drag-a-look demo; `02 · BEFORE THE REVEAL` fake TV comparing "Standard Ethereum" vs "With FHE"; `03 · GAMES` playable sealed-card demo |
| `/drops`                                            | Filter rail (status, sale type, 11 categories, view modes), search; one live drop: _Fungo Genesis_ (2038/3232 minted)                                                   |
| `/d/:slug`                                          | Gallery, phase tabs, uniform-price sealed-auction explainer, whitelist panel, 11/11 "Sealed" traits, Story/Details/Activity, 12-question FAQ, refunds                   |
| `/how`                                              | Long-form explainer: sealed draw, holder-controlled viewing, prove-without-showing, auction rules, allowlists, studio terms                                             |
| `/apply`                                            | Studio onboarding: `Create your studio → Launch it on-chain → Review by the platform → You hear back here`                                                              |
| `/campaigns/*`                                      | Promotion layer over drops: `rules.maxLive` concurrency cap, `can.og/can.wl/can.token` permissions, explicit takedown and preview/publish states                        |
| `/vault`, `/funds`, `/account`, `/studio`, `/admin` | Holder and operator surfaces (not walked in depth)                                                                                                                      |

Economics stated in-product: contracts belong to the studio, revenue goes to studio wallets, the
platform fee is **on-chain and capped** (`feeBps: 500`, `defaultTerms` in the config document).

## 2. Stack (evidence)

- **Build:** Vite/rolldown, React SPA (`<div id="root">`), react-router (route table read out of
  the bundle), **158 JS chunks** — aggressive route/component splitting.
- **Data:** TanStack Query (27 `queryKey`, `staleTime` ×6, `gcTime`, `QueryClient` retry defaults).
- **Chain:** **no wagmi/viem/web3.js** — a hand-rolled 200 KB JSON-RPC provider
  (`provider-jsonrpc-*.js`), a 259 KB FHE contract layer (`encrypt` ×32, `inputProof` ×15), a 6.8 KB
  SDK wrapper.
- **3D:** `three.min` 608 KB (161 KB gz) + ~15 own scene modules (`vox`, `world`, `stations`,
  `rocket`, `crowd`, `coins`, `cycle`, `peek`, `base`, `kit`, `props`).
- **API:** own REST — `/api/config`, `/api/drops`, `/api/campaigns/*`, `/api/auth/me`,
  `/api/auth/x/start`, `/api/holds`, `/api/time`, `/api/site/slides`, `/api/reports`,
  `/api/studio/*`, `/api/rpc/1`, `/api/relayer/1`.
- **Infra:** Cloudflare, strict CSP (`script-src 'self' 'wasm-unsafe-eval' challenges.cloudflare.com`),
  HSTS, `X-Frame-Options: DENY`, nosniff, Referrer/Permissions-Policy, COOP/CORP.
- **Perf (home, measured):** FCP 2.0 s · DCL 1.29 s · load 1.54 s · 63 requests · **1.65 MB**
  transferred · 0 longtasks after load.

## 3. Quality findings (ranked)

| Sev | Finding                                                                                                                                                             | Evidence                                                                                               |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| P1  | **Their own CSP blocks Cloudflare Insights** — analytics can never load                                                                                             | Console: `Loading the script '…beacon.min.js…' violates … script-src`                                  |
| P1  | **Whole app `noindex`**, `sitemap.xml` → 404                                                                                                                        | `X-Robots-Tag: noindex, nofollow, noarchive` on every response; Lighthouse SEO 69, `is-crawlable` fail |
| P1  | **CLS 0.832** on mobile `/drops` (threshold 0.1)                                                                                                                    | Lighthouse mobile                                                                                      |
| P2  | **Mobile home has no navigation** ≤760 px: `Explore drops` / `Apply to launch` are `display:none` and no hamburger exists (inner pages do have a 42×42 menu button) | getComputedStyle audit at 390/760/1100 px                                                              |
| P2  | **Lighthouse returns `PAGE_HUNG` on home** — the continuous three.js rAF scene saturates a throttled main thread; succeeded on `/drops`                             | Lighthouse navigation audit                                                                            |
| P3  | `label-content-name-mismatch` on footer brand link (`aria-label="FungoLabs Launchpad, home"` vs visible text)                                                       | Lighthouse a11y (otherwise **100**)                                                                    |
| P3  | Duplicate polling: `/api/time` ×3, `/api/holds` ×4 on one page load                                                                                                 | Network capture                                                                                        |
| P3  | Wallet reach: injected extension only + X login — no WalletConnect/Coinbase                                                                                         | Sign-in modal: "No wallet found in this browser"                                                       |

**Done well:** a11y 100 (skip link, landmark roles, named icon buttons, gallery `roledescription`,
focus-trapped modal), global `prefers-reduced-motion` **plus** a separate JS gate for
`scroll-behavior`, hashed assets `public, max-age=31536000, immutable`, pre-paint theme script
(zero FOUC), self-hosted fonts, honest contractual copy.

## 4. Patterns worth porting

1. **Runtime config descriptor** — `GET /api/config?v=<hash>` returns all mutable truth as data
   (chains, RPC/relayer URLs, factory address, fee bps, `defaultTerms`, payment tokens,
   `blockSeconds`); the client only ever _projects_ it. Same idea as our descriptor rule, applied to
   the data plane.
2. **Server-authoritative clock** — `GET /api/time` → `{now}` is fetched into the query cache;
   drops carry `clock:{lag, held}` and the UI renders _"sales on hold"_ from it. Client `Date.now()`
   skew kills auctions; ours affects lease/TTL/ETA logic.
3. **Per-endpoint cache ladder** — `private, no-store` (auth) · `private, max-age=10/30` (data) ·
   `private, no-cache` (HTML) · `public, … immutable` (hashed assets). Four tiers, each right for
   its volatility.
4. **Same-origin proxy for third-party I/O** — `rpc: …/api/rpc/1`, `relayerUrl: …/api/relayer/1`,
   resolved against `window.location.origin`. No keys in the browser, CORS never becomes a topic,
   rate limits stay server-side; they own the wrapper's error taxonomy
   (`401|403|415|429 on /api/relayer/`).
5. **OAuth done properly** — PKCE S256 + `state`, scopes `users.read tweet.read`, cookie
   `__Secure-lp_xbind; Max-Age=600; Path=/launchpad; HttpOnly; Secure; SameSite=Lax`, and
   `GET /api/auth/me → 200 null` (anonymous is a state, not an error).
6. **Heavy compute off the main thread** — FHE encryption in a versioned Web Worker
   (`/launchpad/zama/encrypt.worker.3.7.0.js`) with `batchRpcCalls: true`, an explicit
   `timeout: 300_000` budget, a short `permitTTL`, and `terminate?.()` teardown; per-chain graceful
   degradation ("Encrypted features are not available on ${chain}").
7. **Canvas alt text as art direction** — the hero canvas carries a full narrative `alt`/`role="img"`
   description; screen readers get the value proposition, not "canvas".
8. **Explicit product states in plain copy** — takedown (_"Nothing runs from a drop the platform took
   down"_), preview vs publish, "sales on hold", whitelist closed, anonymous-is-a-state.
9. **Refuse to transact on config drift** (from the vault/funds surface, §7) — _"This page and the
   platform name different Genesis pools, **so nothing is sent from here**. Reload the page; if it
   stays, tell the team."_ Client-side contract ≠ server-side contract ⇒ halt **before** the
   signature rather than sending funds to the wrong pool. The safety half of pattern #1: a versioned
   descriptor needs a fail-closed check, not just a cache-buster.
10. **Pre-sign simulation in our own words** — _"Shown in your wallet before you confirm"_, _"About
    0.0x ETH, paid from this wallet"_. The amount, token and wallet are stated in the page's copy so
    the user compares two independent sources of truth instead of trusting the wallet popup alone.
11. **State machines written as sentences** — queue → signing → published → marketplaces-refresh,
    each with its own honest line (_"An opening waits to be published"_ · _"Zama's network is still
    signing the answer. This keeps trying."_ · _"{n} openings published. Marketplaces show them after
    they refresh."_), plus who pays what (_"Their fee is paid. Publishing … no fee, only the
    network's gas"_), mid-flight honesty (_"a transaction already sent still goes through; what is
    not sent yet stays in the pool for you"_) and real edge cases (_"This wallet does not take ETH
    sent by a contract … Leave ETH out to take the rest"_). Directly applicable to our banner/toast
    tiers for long-running jobs.
12. **State what does _not_ exist** (tokenomics) — _"The platform's own share, held in the open as
    ETH and ZAMA. **There is no FUNGO treasury.**"_ A disclosure page that names its own absences
    preempts the rumor-and-FAQ loop instead of answering it later. Same page states the human step
    plainly (_"Accounts approved in a manual review"_) and the lockup mechanically (_"Locked in its
    contract for a full year, then out evenly"_).
13. **Unavailable, but explained** (funds / studio) — features _"open here once this site's network
    is live"_, and _"Applications are closed for now."_ Not a dead button, not a hidden section: the
    global state is announced with its reason, in the same voice as the rest of the UI.
14. **Wrong-context states are first-class** — _"Your wallet is on another network"_ is its own
    screen with its own copy, not an error toast under a broken action.

## 5. What NOT to copy

- `noindex` on a marketing launchpad + missing sitemap.
- A CSP that blocks your own analytics beacon.
- Eager 608 KB WebGL for a decorative hero (`PAGE_HUNG`, CLS 0.832).
- Hand-built REST with no descriptor/parity gate — we are structurally ahead here; don't regress.
- Query keys that don't dedupe (4× `/api/holds` per load): owning the transport does not buy
  discipline; dedupe is an asserted invariant.
- A mobile header with no menu at all.
- Their 3D lesson inverted: **the cryptography is code-split behind the journey that needs it and
  the toy is not.** Apply that ordering to ourselves.

## 6. Cards raised

| Card                                                                           | Priority | Gist                                                                                                                                                                                 |
| ------------------------------------------------------------------------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `20261108_134211_runtime-config-descriptor-for-web-client-flagslimitsendpoint` | medium   | Audit `routers/config.py` (deprecated → `settings.py`) first; if gaps remain, one public versioned descriptor for deploy-surviving client values, projected through `http-client.ts` |
| `20261108_134212_server-authoritative-clock-for-leasettleta-computation`       | medium   | One server time source + offset helper; convert gating `Date.now()` usages (leave pure display alone)                                                                                |
| `20261108_134212_off-critical-path-policy-for-heavy-visuals-and-heavy-compute` | low      | Inventory above-the-fold heavy work; intersection/idle gating + workers with explicit budgets; record the rule in `PRODUCT_ENGINEERING.md`                                           |

## 7. Follow-up probes (resolved 2026-11-08, read-only — no transactions sent)

| Question                                  | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Headers on `encrypt.worker.3.7.0.js`      | `cache-control: private, max-age=3600`, `nosniff`, **no `age`/`etag`**. Path is versioned, so it qualifies for `public, …, immutable` like every other hashed asset; instead a browser re-fetches after an hour — in the one flow where the UI promises a 300 s budget. Correct security posture, suboptimal caching (same class as the eager hero).                                                                                                                                                                |
| What `/api/holds`'s `{from, until}` gates | **On-chain sale-hold windows.** `{"1":{"from":0,"until":0}}` = no hold active. `Hold-*.js` shows `clock` is a **contract address** written via `encodeFunctionData` + `run({chain,label})`, with `heldBefore` accumulating prior holds — copy: _"Hold every sale for N…"_, _"Every sale is on hold"_. The drop document mirrors it as `clock:{lag, held}` → one on-chain fact → one read API (10 s TTL) → one derived field → one sentence of copy. Holding sales is an on-chain admin action, not a platform flag. |
| Relayer rate-limit posture                | **Not observable** — no `ratelimit`/`retry-after` on any probe. Their `tx` chunk parses `401\|403\|415\|429 on …/api/relayer/…`, so a limiter exists upstream of their parser (Cloudflare WAF).                                                                                                                                                                                                                                                                                                                     |
| — bonus —                                 | `POST /api/rpc/1` → **`403 {"error":"csrf","message":"Missing request header"}`**: the RPC proxy requires a custom header, which a cross-origin page cannot send without a passing preflight. That makes the proxy **unusable from third-party pages** — a same-origin capability check. Directly applicable to any proxy we put in front of an LLM provider.                                                                                                                                                       |
| — bonus —                                 | **`x-settings-version: 36172f87c8268367`** is stamped on every response, and `/api/config?v=<same hash>` is the cache-buster. Any response tells a client its cached config went stale — no `/version` endpoint, no extra round-trip. This is the missing half of pattern #1: a versioned descriptor needs an observability channel.                                                                                                                                                                                |

| `/vault` holder journey (the last unopened surface) | **It is a money-and-openings ledger, not an "my NFTs" gallery**: pool fees, token shares, claims, confidential-token conversion, and the pipeline that publishes decrypted openings on-chain. Only **one** server call in the whole 74 KB chunk — `/api/wl/drop/{id}/token/proof/{wallet}`; everything else is chain reads + local state. Source of patterns **#9–#11** above. |

**Method note:** Cloudflare began timing out the automated browser after ~30 navigations (the bot
protection works), so the studio/apply, campaign, FHE, **vault, tokenomics, funds** and studio-home
sections were read from shipped route chunks rather than clicked through; the probes above were plain
`curl` requests. The site **redeployed repeatedly during this review** — the entry chunk moved
`index-CL7oyVN3.js` → `index-DTh5XEGT.js` → `index-N6EsAsFY.js` (375 427 → 376 137 bytes) and old
content-hashed chunk URLs began returning 404, so every hash quoted here is point-in-time and had to
be re-derived from the live `index.html` each pass. Findings are behavioural (headers, routes, copy,
architecture) and survive a rebuild; exact hashes will not.
