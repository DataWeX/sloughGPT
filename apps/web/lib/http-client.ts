// ═══════════════════════════════════════════════════════════════════════════
// http-client — Full-featured HTTP request library for sloughGPT frontend
// ═══════════════════════════════════════════════════════════════════════════
//
// The implementation lives in ./http/ — one module per concern:
//   errors · types · corr-id · url · retry · interceptors · resilience ·
//   state · request · verbs · api-client · client · auth · sse
//
// This file is a PURE FACADE: it only re-exports the exact public surface.
// Never add implementation here — put it in the matching ./http/ module and
// (if public) expose it through ./http/index.ts.
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

export * from './http/index'
