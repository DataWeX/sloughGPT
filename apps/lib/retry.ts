// Retry policies + the mutable per-process request defaults.

export const RETRYABLE_STATUSES = new Set([408, 429, 502, 503, 504])
const MAX_RETRIES = 2
const BASE_DELAY = 500
const DEFAULT_TIMEOUT_MS = 30_000

// 400 is intentionally NOT retryable: docstore 400s are client errors (e.g.
// "error parsing the body" from an aborted request) — retrying amplifies
// bursts without ever succeeding. 500 stays (docstore upsert races resolve).
export const DOCSTORE_RETRYABLE_STATUSES = new Set([408, 429, 500, 502, 503, 504])
export const DOCSTORE_MAX_RETRIES = 4
export const DOCSTORE_BASE_DELAY = 300

/** Mutable request defaults — single source of truth (httpClient.configure() writes here). */

export const httpDefaults = {

  timeout: DEFAULT_TIMEOUT_MS,

  maxRetries: MAX_RETRIES,

  baseDelay: BASE_DELAY,

}
