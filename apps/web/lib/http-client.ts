// ═══════════════════════════════════════════════════════════════════════════
// http-client — Full-featured HTTP request library for sloughGPT frontend
// ═══════════════════════════════════════════════════════════════════════════
//
// The implementation lives in ../lib/ (apps/lib — shared across apps), one
// module per concern:
//   errors · types · corr-id · url · retry · interceptors · resilience ·
//   state · request · verbs · api-client · client · auth · sse
//
// This file is a PURE FACADE for the public surface, plus the one thing only
// this file can do: wire the host into the transport. No implementation here —
// put it in the matching apps/lib/ module and (if public) expose it through
// apps/lib/index.ts. The wiring below is not implementation, it is the seam:
// apps/lib must never import apps/web (its ports are read at call time), so
// the host hands over base URL, bearer token and logger exactly here — on
// import of the facade, which by construction precedes any request.
//
// Features:
//   1. Interceptors — request/response middleware chain
//   2. Cache layer — TTL-based with stale-while-revalidate
//   3. Circuit breaker — closed → open → half-open state machine
//   4. Lifecycle hooks — beforeRequest, afterResponse, onError
//   5. Progress tracking — upload/download percentage
//   6. Dedup with TTL — recent-dedup beyond in-flight only
//   7. Request throttling — max concurrent with queue
//   8. Response metadata — timing, retry, cache hit, circuit state
//
// All existing exports (apiGet, apiPost, apiClient, etc.) are backward-
// compatible. New features are opt-in via RequestOptions or httpClient.
// ═══════════════════════════════════════════════════════════════════════════

import { configureTransport } from '../../lib/ports'
import { PUBLIC_API_URL } from './config'
import { useAuthStore } from './auth'

// Host → transport wiring. Every read in apps/lib is at call time, and this
// module body runs before anything can make a request, so the ports are
// populated before first use. An unwired host (SDK, a test importing apps/lib
// directly) simply gets the inert defaults: same-origin paths, no auth header,
// silent logging — never a crash.
//
// The logger port is deliberately NOT wired here: dev-log owns it and
// registers itself, so a host that mocks/supplies its own logger keeps full
// control (and a test mocking dev-log silently falls back to the no-op port).
configureTransport({
  getBaseUrl: () => PUBLIC_API_URL,
  getToken: () => useAuthStore.getState().token,
})

export * from '../../lib/index'
