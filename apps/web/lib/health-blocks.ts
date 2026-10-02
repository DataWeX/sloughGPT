/**
 * Health blocks — the typed contract between the two health transports and the UI.
 *
 * Why this module exists
 * ----------------------
 * Two transports deliver the same datum in different shapes: HTTP `/health/detailed`
 * nests it (`d.system.cpu_percent`, `d.inference.is_inferencing`,
 * `d.health_score.score`) while SSE `/health/stream` flattens it (`d.cpu_percent`,
 * `d.is_inferencing`, `d.health_score` as an int). Each transport used to carry its
 * own ~85-line hand-written snapshot literal, and each silently dropped a *different*
 * field — HTTP hardcoded `diagnoses: []`, SSE never set `bandwidth`. One shared
 * normalizer turns "dropped a field" into a TypeScript error instead of a silent
 * omission, because both transports must produce a complete `HealthBlocks` or fail
 * to compile.
 *
 * Shape
 * -----
 *     HTTP DetailedHealth ─▶ blocksFromDetailed ─┐
 *                                                ├─▶ HealthBlocks ─▶ snapshotFromBlocks ─▶ LiveHealthSnapshot ─▶ cards
 *     SSE envelope data   ─▶ blocksFromSse     ─┘
 *
 * Source-specific reads (nested vs flat, `status_message` vs `health_summary`) live
 * in the two adapters — that is where source differences belong. The normalizer only
 * flattens; it knows nothing about either payload.
 *
 * Absent vs null vs zero
 * ----------------------
 * Three states are NOT interchangeable, and cards that conflate them lie to the user:
 *
 *   absent  the key is `undefined` — the subsystem is not in the payload at all
 *           (no gateway reachable ⇒ `bandwidth` absent). Cards hide: no data exists.
 *   null    the key is present with no value yet (model not loaded ⇒ `device: null`).
 *           Cards show an em-dash / "n/a": the subsystem exists, it just has nothing.
 *   zero    a measured zero (`error_count: 0`). Cards show "0": nothing happened.
 *
 * Reporting absent as zero claims "fine" when we actually do not know. Blocks
 * document which states they allow.
 *
 * @module health-blocks
 */

import type { BandwidthStats, DetailedHealth } from '@/lib/system-controller'

// ---------------------------------------------------------------------------
// Startup vocabulary
// ---------------------------------------------------------------------------

export type StartupStage = 'init' | 'critical' | 'ready' | 'background' | 'unknown'

export interface HookStatus {
  name: string
  stage: StartupStage
  status: 'pending' | 'running' | 'ok' | 'timeout' | 'error'
  duration_seconds: number
  error: string | null
}

export interface StagedLoaderStatus {
  stage: StartupStage
  stage_value: number
  elapsed_seconds: number
  model_progress: number
  model_progress_message: string
  errors: Record<string, string>
  hooks: Record<string, HookStatus>
  stages: Record<string, { hooks: string[]; time: number | null }>
}

// ---------------------------------------------------------------------------
// Collection element types (shared by snapshots and cards)
// ---------------------------------------------------------------------------

export interface Diagnosis {
  check: string
  severity: string
  score: number
  message: string
}

export interface ModelMetric {
  model: string
  count: number
  total_tokens: number
  tokens_per_sec: number
  avg_tokens: number
}

export interface ModelEvent {
  type: string
  model: string
  detail: string
  ts: number
}

export interface RateViolation {
  path: string
  count: number
  limit: number
  ts: number
}

export interface HealthPoint {
  score: number
  status: string
  ts: number
}

export interface MemoryPoint {
  rss_mb: number
  virtual_mb: number
  system_percent: number
  ts: number
}

export interface PathLatency {
  path: string
  avg_ms: number
  count: number
  p95_ms: number
}

export interface RecentError {
  path: string
  method: string
  status: number
  message: string
  error_type: string
  ts: number
}

export interface TrainingPool {
  active_jobs: number
  max_workers: number
  total_tracked: number
}

// ---------------------------------------------------------------------------
// Blocks — one coherent group of health data with an explicit state contract
// ---------------------------------------------------------------------------

/** Identity and liveness. Fields are always present in both transports. */
export interface CoreBlock {
  model_loaded: boolean
  model_loading: boolean
  uptime_seconds: number
  /** `null` until a model resolves — not absent. */
  model_type: string | null
  /** `null` until the device is known — not absent. */
  device: string | null
  /** `null` before a soul is selected — not absent. */
  soul: string | null
}

/** Staged-loader progress driving the startup overlay. */
export interface StartupBlock {
  startup_stage: StartupStage
  startup_stage_value: number
  startup_elapsed: number
  startup_model_progress: number
  startup_model_progress_message: string
  startup_hooks: Record<string, HookStatus>
}

/** Generation counters and throughput. */
export interface InferenceBlock {
  /** False when no inference is running (measured), not when the payload lacked it. */
  is_inferencing: boolean
  inference_count: number
  total_tokens: number
  tokens_per_sec: number
  avg_tokens_per_request: number
  /** Absent ⇒ `null` — the basic payload never produced it (known backend gap). */
  num_parameters: number | null
}

/** Request volume, errors and latency. Zero is measured and means "clean". */
export interface TrafficBlock {
  request_count: number
  error_count: number
  avg_latency_ms: number
  p95_latency_ms: number
  requests_per_minute: number
}

/** Host-level gauges. `null` means "unreadable", never 0% — 0% is a reading. */
export interface SystemBlock {
  cpu_percent: number | null
  memory_percent: number | null
}

/** Composite score, one-line summary and per-check diagnoses. */
export interface ScoringBlock {
  health_score: number
  health_status: string
  health_summary: string
  diagnoses: Diagnosis[]
}

/** Bounded history series. Empty array means "no samples yet", not absent. */
export interface CollectionsBlock {
  model_metrics: ModelMetric[]
  model_events: ModelEvent[]
  rate_violations: RateViolation[]
  health_history: HealthPoint[]
  memory_history: MemoryPoint[]
  path_latencies: PathLatency[]
  recent_errors: RecentError[]
}

/** Quantization and executor pool — both genuinely optional subsystems. */
export interface ExecutionBlock {
  quantization: unknown | null
  training_pool: TrainingPool | null
}

/** Edge gateway byte counters.
 *
 *  - absent (`undefined`) — payload predates the gateway counters, or the
 *    projection dropped them. Card hides: there is nothing to report.
 *  - `null` — gateway known but not reachable. Card hides.
 *  - object with `identity_bytes: 0` — gateway reachable, measured zero. Card
 *    hides only because a 0% saving is not worth a card, *not* for lack of data.
 */
export interface BandwidthBlock {
  bandwidth?: BandwidthStats | null
}

/** The canonical, transport-agnostic form every source adapter must produce. */
export interface HealthBlocks {
  core: CoreBlock
  startup: StartupBlock
  inference: InferenceBlock
  traffic: TrafficBlock
  system: SystemBlock
  scoring: ScoringBlock
  collections: CollectionsBlock
  execution: ExecutionBlock
  bandwidth: BandwidthBlock
}

/** Every block name, for exhaustive checks and tests. */
export type BlockName = keyof HealthBlocks

export const BLOCK_NAMES: readonly BlockName[] = [
  'core',
  'startup',
  'inference',
  'traffic',
  'system',
  'scoring',
  'collections',
  'execution',
  'bandwidth',
] as const

/**
 * The store/card-facing view: a flat snapshot assembled from every block.
 * Produced by {@link snapshotFromBlocks} — never hand-written, or the transports
 * start drifting again.
 */
export type LiveHealthSnapshot = CoreBlock &
  StartupBlock &
  InferenceBlock &
  TrafficBlock &
  SystemBlock &
  ScoringBlock &
  CollectionsBlock &
  ExecutionBlock &
  BandwidthBlock

/**
 * The exact slice of the flat snapshot one block owns. Cards take this instead of
 * the whole snapshot, so a card physically cannot reach into another block's data
 * and a renamed field becomes a compile error at every consumer.
 *
 * Cards spanning two blocks intersect the slices:
 * `BlockSlice<TrafficBlock> & BlockSlice<InferenceBlock>`.
 *
 * `B` is intentionally the block *type* rather than a block-name key: `keyof`
 * over a generic indexed access collapses to `string` in TypeScript, which would
 * silently widen the Pick and quietly undo the narrowing.
 */
export type BlockSlice<B> = Pick<LiveHealthSnapshot, Extract<keyof B, keyof LiveHealthSnapshot>>

// ---------------------------------------------------------------------------
// Source adapters — the only place that knows a payload's shape
// ---------------------------------------------------------------------------

const toArray = <T>(value: unknown): T[] => (Array.isArray(value) ? (value as T[]) : [])

/** Finite number, or `null` when the input is missing/unreadable. */
const toNumber = (value: unknown): number | null => {
  if (value == null || value === '') return null
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}

/** Finite number with a default — for counters where `null` has no meaning. */
const toCount = (value: unknown, fallback = 0): number => toNumber(value) ?? fallback

/** Gateway counters, preserving absent vs null. */
const toBandwidthBlock = (source: unknown): BandwidthBlock => {
  if (!source || typeof source !== 'object' || !('bandwidth' in source)) return {}
  return { bandwidth: (source as { bandwidth?: BandwidthStats | null }).bandwidth ?? null }
}

/**
 * Lift a `/health/detailed` payload into {@link HealthBlocks}.
 *
 * Keeps every read the HTTP path performed before, including the startup-overlay
 * hardening (an `unknown` stage on a loaded model means startup finished) and the
 * `status_message` summary. Adds the one field HTTP dropped: `health_score.diagnoses`.
 */
export function blocksFromDetailed(d: DetailedHealth): HealthBlocks {
  const src = d as DetailedHealth & { startup_progress?: StagedLoaderStatus }
  const score = src.health_score
  const staged = src.startup_progress
  const rawStage = staged?.stage ?? 'unknown'

  return {
    core: {
      model_loaded: Boolean(src.model_loaded),
      model_loading: Boolean(src.model_loading),
      uptime_seconds: toCount(src.uptime_seconds),
      model_type: src.model_type ?? null,
      device: src.device ?? null,
      soul: src.soul ?? null,
    },
    startup: {
      // Same hardening as the SSE path: unknown + loaded model ⇒ background.
      startup_stage: rawStage === 'unknown' && Boolean(src.model_loaded) ? 'background' : rawStage,
      startup_stage_value: toCount(staged?.stage_value),
      startup_elapsed: toCount(staged?.elapsed_seconds),
      startup_model_progress: toCount(staged?.model_progress),
      startup_model_progress_message: staged?.model_progress_message ?? '',
      startup_hooks: staged?.hooks ?? {},
    },
    inference: {
      is_inferencing: Boolean(src.inference?.is_inferencing),
      inference_count: toCount(src.inference_count),
      total_tokens: toCount(src.total_tokens),
      tokens_per_sec: toCount(src.tokens_per_sec),
      avg_tokens_per_request: toCount(src.avg_tokens_per_request),
      num_parameters: toNumber(src.num_parameters),
    },
    traffic: {
      request_count: toCount(src.request_count),
      error_count: toCount(src.error_count),
      avg_latency_ms: toCount(src.avg_latency_ms),
      p95_latency_ms: toCount(src.p95_latency_ms),
      requests_per_minute: toCount(src.requests_per_minute),
    },
    system: {
      cpu_percent: toNumber(src.system?.cpu_percent),
      memory_percent: toNumber(src.system?.memory_percent),
    },
    scoring: {
      health_score: toCount(score?.score),
      health_status: String(score?.status || src.status || 'unknown'),
      // HTTP's summary source is the human-readable status message, not the
      // health flow's one-liner — this is a source difference, not a drift.
      health_summary: String(src.status_message || ''),
      diagnoses: toArray<Diagnosis>(score?.diagnoses),
    },
    collections: {
      model_metrics: toArray<ModelMetric>(src.model_metrics),
      model_events: toArray<ModelEvent>(src.model_events),
      rate_violations: toArray<RateViolation>(src.rate_violations),
      health_history: toArray<HealthPoint>(src.health_history),
      memory_history: toArray<MemoryPoint>(src.memory_history),
      path_latencies: toArray<PathLatency>(src.path_latencies),
      recent_errors: toArray<RecentError>(src.recent_errors),
    },
    execution: {
      quantization: src.quantization ?? null,
      training_pool: src.training_pool ?? null,
    },
    bandwidth: toBandwidthBlock(src),
  }
}

/**
 * Lift a `/health/stream` envelope into {@link HealthBlocks}.
 *
 * SSE flattens what HTTP nests, so this adapter does the lifting: `health_score`
 * arrives as an int, `is_inferencing` sits at the top level, and `system.*` is
 * unwrapped. It also reads the nested `startup_progress` object when present,
 * falling back to the flat `startup_*` keys older emitters still send.
 *
 * Adds the one field SSE dropped: `bandwidth`.
 */
export function blocksFromSse(raw: Record<string, unknown>): HealthBlocks {
  const src = raw as Record<string, unknown> & Partial<LiveHealthSnapshot>
  const staged = src.startup_progress as StagedLoaderStatus | undefined
  // Hardening: very old / minimal snapshots omit every startup field, which
  // used to pin the StartupOverlay on "Connecting" forever. A loaded model
  // means startup finished — resolve to "background" (overlay-clearing).
  const rawStage = staged?.stage ?? src.startup_stage ?? 'unknown'
  const resolvedStage =
    rawStage === 'unknown' && Boolean(src.model_loaded) ? 'background' : rawStage

  return {
    core: {
      model_loaded: Boolean(src.model_loaded),
      model_loading: Boolean(src.model_loading),
      uptime_seconds: toCount(src.uptime_seconds),
      model_type: src.model_type ?? null,
      device: src.device ?? null,
      soul: src.soul ?? null,
    },
    startup: {
      startup_stage: resolvedStage,
      startup_stage_value: toCount(staged?.stage_value ?? src.startup_stage_value),
      startup_elapsed: toCount(staged?.elapsed_seconds ?? src.startup_elapsed),
      startup_model_progress: toCount(staged?.model_progress ?? src.startup_model_progress),
      startup_model_progress_message:
        staged?.model_progress_message ?? src.startup_model_progress_message ?? '',
      startup_hooks: staged?.hooks ?? src.startup_hooks ?? {},
    },
    inference: {
      is_inferencing: Boolean(src.is_inferencing),
      inference_count: toCount(src.inference_count),
      total_tokens: toCount(src.total_tokens),
      tokens_per_sec: toCount(src.tokens_per_sec),
      avg_tokens_per_request: toCount(src.avg_tokens_per_request),
      num_parameters: toNumber(src.num_parameters),
    },
    traffic: {
      request_count: toCount(src.request_count),
      error_count: toCount(src.error_count),
      avg_latency_ms: toCount(src.avg_latency_ms),
      p95_latency_ms: toCount(src.p95_latency_ms),
      requests_per_minute: toCount(src.requests_per_minute),
    },
    system: {
      cpu_percent: toNumber(src.cpu_percent),
      memory_percent: toNumber(src.memory_percent),
    },
    scoring: {
      health_score: toCount(src.health_score),
      health_status: String(src.health_status || 'unknown'),
      health_summary: String(src.health_summary || ''),
      diagnoses: toArray<Diagnosis>(src.diagnoses),
    },
    collections: {
      model_metrics: toArray<ModelMetric>(src.model_metrics),
      model_events: toArray<ModelEvent>(src.model_events),
      rate_violations: toArray<RateViolation>(src.rate_violations),
      health_history: toArray<HealthPoint>(src.health_history),
      memory_history: toArray<MemoryPoint>(src.memory_history),
      path_latencies: toArray<PathLatency>(src.path_latencies),
      recent_errors: toArray<RecentError>(src.recent_errors),
    },
    execution: {
      quantization: src.quantization ?? null,
      training_pool: src.training_pool ?? null,
    },
    bandwidth: toBandwidthBlock(src),
  }
}

// ---------------------------------------------------------------------------
// Normalizer — the single place a snapshot is assembled
// ---------------------------------------------------------------------------

/**
 * Flatten {@link HealthBlocks} into the flat {@link LiveHealthSnapshot} the store
 * and cards consume. The only writer of that shape: if a field must appear in the
 * snapshot it must belong to a block, and if it belongs to a block both adapters
 * must already supply it or this file does not compile.
 *
 * A block's keys are spread verbatim, so `bandwidth` stays absent when the source
 * never carried it — the absent/null/zero contract survives into the store.
 */
export function snapshotFromBlocks(blocks: HealthBlocks): LiveHealthSnapshot {
  return {
    ...blocks.core,
    ...blocks.startup,
    ...blocks.inference,
    ...blocks.traffic,
    ...blocks.system,
    ...blocks.scoring,
    ...blocks.collections,
    ...blocks.execution,
    ...blocks.bandwidth,
  }
}
