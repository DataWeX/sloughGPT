// ═══════════════════════════════════════════════════════════════════════════
// http — public surface of the split http-client
// ═══════════════════════════════════════════════════════════════════════════
//
// This barrel is the ONLY thing re-exported by apps/web/lib/http-client.ts
// (the facade).
// The export set below is kept byte-equivalent to the pre-split http-client:
// internal helpers (_resolveUrl, _corrId, state singletons, …) stay private
// to this directory on purpose.
//
// Module map (one-way dependency flow, no cycles):
//   types ← errors ← url/retry/corr-id ← interceptors/resilience ← state
//        ← request ← verbs ← api-client/client
//   auth, sse — leaf paths built on corr-id/url
// ═══════════════════════════════════════════════════════════════════════════

// Error model
export { ApiError } from './errors'

// Types
export type {
  RequestInterceptor,
  ResponseInterceptor,
  RequestConfig,
  ResponseEnvelope,
  CacheOptions,
  CircuitBreakerOptions,
  CircuitState,
  ThrottleOptions,
  ResponseMetadata,
  ProgressCallbacks,
  RequestOptions,
  HttpClientResponse,
  HttpClientOptions,
} from './types'

// Correlation IDs
export { getRecentCorrelationIds } from './corr-id'

// Middleware + resilience primitives
export { InterceptorManager } from './interceptors'
export { HttpCache, CircuitBreaker, Throttler } from './resilience'

// Verb wrappers
export { apiGet, apiPost, apiPostForm, apiPut, apiDelete, apiPatch } from './verbs'

// Convenience client
export { apiClient, createApiClient } from './api-client'

// Full-featured client
export type { HttpClient } from './client'
export { httpClient, createHttpClient } from './client'

// SSE streaming + raw auth fetch
export type { SSEEvent } from './sse'
export { streamSSE } from './sse'
export { authFetch } from './auth'
