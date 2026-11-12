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
// No UI wiring yet (surface ladder): a journey page consumes this when it
// needs to gate on a flag; until then the module exists so that wiring is a
// one-liner, not another endpoint.

import { apiGet } from './http-client'

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
