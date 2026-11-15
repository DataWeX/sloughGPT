// runtime-config — typed consumer for GET /runtime-config (descriptor
// runtime_config.get). One mutable-truth document for values that must outlive
// a deploy: the server's feature flags with their *resolved* state (an
// SLO_FF_* env override beats the registered status), versioned by a sha256
// config_version that doubles as the response ETag.
//
// Contract (apps/api/server/routers/runtime_config.py):
//   {config_version: sha256hex, flags: [{name, status, enabled, version,
//    since, description}]}  — raw wire (no {status, data} envelope), sorted by
//    name, Cache-Control: private, max-age=30.
//
// All API traffic goes through http-client (the single source of truth) — no
// raw fetch here. The in-memory TTL mirrors the response header: one document
// per 30s window no matter how many components ask.
//
// UI wiring (surface ladder): no journey page consumes flags yet — but the
// module DOES carry the config-drift guards (card 20261108_222308): every
// action presents this document's config_version, and a server refusal
// raises the global "Reload the page" banner (see bottom of file).

import { _globalInterceptors } from '../../lib/state'
import type { RequestConfig, ResponseEnvelope } from '../../lib/types'
import { apiGet } from './http-client'
import { notify } from './notify'

export type FlagStatus = 'enabled' | 'disabled' | 'experimental'

export interface RuntimeConfigFlag {
  /** Registry name, e.g. "slonet_provider". */
  name: string
  /** Registered intent (what the ledger says). */
  status: FlagStatus
  /** Effective state — env override wins. This is what code should act on. */
  enabled: boolean
  /** Semver of the feature behind the flag (bumps are ledgered server-side). */
  version: string
  /** ISO date the flag entered its current version. */
  since: string
  description: string
}

export interface RuntimeConfig {
  /** sha256 over sorted (name, status, version, enabled) — also the ETag. */
  config_version: string
  flags: RuntimeConfigFlag[]
}

/** In-memory TTL — deliberately equal to Cache-Control's max-age=30. */
const TTL_MS = 30_000

let cache: { config: RuntimeConfig; fetchedAt: number } | null = null

/**
 * Fetch the runtime-config document, at most once per TTL window.
 *
 * Rejects when the transport fails — no invented fallback document. Callers
 * that degrade gracefully should use {@link isFlagEnabled}, which fails safe.
 */
export async function fetchRuntimeConfig(): Promise<RuntimeConfig> {
  const now = Date.now()
  if (cache && now - cache.fetchedAt < TTL_MS) return cache.config
  const config = await apiGet<RuntimeConfig>('/runtime-config')
  cache = { config, fetchedAt: Date.now() }
  return config
}

/**
 * Resolve one flag from the server's document (30s TTL shared across calls).
 *
 * Fails safe: a flag the server never declared, or an unreachable server,
 * yields `false` — matching the backend's own unknown-flag default. Callers
 * that need to distinguish "off" from "unknown" should catch the rejection
 * from {@link fetchRuntimeConfig} directly.
 */
export async function isFlagEnabled(name: string): Promise<boolean> {
  try {
    const cfg = await fetchRuntimeConfig()
    return cfg.flags.find((f) => f.name === name)?.enabled ?? false
  } catch {
    return false
  }
}

/** Test seam — drops the in-memory TTL cache. */
export function _resetRuntimeConfigCache(): void {
  cache = null
}

// ═══════════════════════════════════════════════════════════════════════════
// Drift guards (card 20261108_222308) — fail closed before an irreversible
// action when this client's held document and the server's current one
// disagree. Server half: infrastructure/config_drift_middleware.py refuses a
// stale/unknown presented version with 412 E_CONFIG_DRIFT. Client half here:
// PRESENT the version on every action, and on refusal surface the one global
// banner sentence instead of letting a stale-calibration call proceed.
// ═══════════════════════════════════════════════════════════════════════════

/** Header the server's drift precondition reads (mirror of the middleware). */
export const CONFIG_VERSION_HEADER = 'X-Slo-Config-Version'

/** Methods that can execute something irreversible — reads never carry it. */
const UNSAFE_METHODS = new Set(['POST', 'PUT', 'PATCH', 'DELETE'])

/** Registry code the server refuses with (ERROR_REGISTRY → 412). */
const DRIFT_CODE = 'E_CONFIG_DRIFT'

/** Fallback sentence if the response body carries none (registry mirrors it). */
const DRIFT_FALLBACK =
  'This page and the server disagree about the active configuration, ' +
  'so nothing was sent. Reload the page to continue.'

/**
 * Request guard — present the held `config_version` on every action call.
 *
 * Fail-closed on the CLAIM: an explicit header is never overwritten or
 * invented. The document is fetched through the 30s TTL (at most one wire
 * read per window); when it cannot be fetched no claim is made — the server
 * then treats the request as headerless, and an unreachable server fails the
 * request at the transport anyway.
 */
export async function attachConfigVersion(config: RequestConfig): Promise<RequestConfig> {
  if (!UNSAFE_METHODS.has(config.method)) return config // reads are the recovery path
  if (CONFIG_VERSION_HEADER in config.headers) return config // explicit claim wins
  try {
    const cfg = await fetchRuntimeConfig()
    config.headers[CONFIG_VERSION_HEADER] = cfg.config_version
  } catch {
    /* no document held → make no claim (pass-through, never a fake one) */
  }
  return config
}

/**
 * Response guard — a 412 E_CONFIG_DRIFT means the server already refused the
 * action. Drop the stale held document (next read is fresh truth) and raise
 * the one global banner with the refusal sentence + reload remedy.
 *
 * The envelope is returned UNCHANGED: the ApiError still propagates to the
 * caller (the action did not happen — fail closed, no auto-replay).
 * `envelope.json` only exists for `ok` responses, so the error body is
 * parsed from `text` here.
 */
export function handleConfigDrift(envelope: ResponseEnvelope): ResponseEnvelope {
  if (envelope.status !== 412) return envelope
  let body: { code?: string; error?: string } | undefined
  try {
    body = JSON.parse(envelope.text) as { code?: string; error?: string }
  } catch {
    return envelope
  }
  if (body?.code !== DRIFT_CODE) return envelope

  _resetRuntimeConfigCache() // stale calibration must not survive the refusal
  notify.banner({
    tone: 'destructive',
    title: 'Configuration out of date',
    message: body.error || DRIFT_FALLBACK,
    action: { label: 'Reload the page', onAction: () => window.location.reload() },
    key: 'config-drift', // one banner slot — dedupes repeat refusals
  })
  return envelope
}

let installed = false

/**
 * Wire both guards onto the global interceptor chain (idempotent).
 * Called exactly once, from the http-client facade's module body — its import
 * precedes any request by construction.
 */
export function installRuntimeConfigGuards(): void {
  if (installed) return
  installed = true
  _globalInterceptors.request.use((c) => attachConfigVersion(c))
  _globalInterceptors.response.use((e) => handleConfigDrift(e))
}

/** Test seam — allows re-installation after interceptor clears. */
export function _resetRuntimeConfigGuards(): void {
  installed = false
}
