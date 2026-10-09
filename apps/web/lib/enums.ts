/**
 * Shared enums — single source of truth for all status/phase/type strings.
 *
 * Pattern: `as const` object + derived type. IDE autocomplete + compile-time
 * checks. Add new values here, then import in consumers.
 *
 * Convention:
 *   UPPER_CASE key  →  lowercase value (matches API wire format)
 *   PascalCase type →  union of values
 */

// ── Training ────────────────────────────────────────────────────────

/** Training job status — matches API response `status` field. */
export const TrainingJobStatus = {
  PENDING: 'pending',
  RUNNING: 'running',
  COMPLETED: 'completed',
  FAILED: 'failed',
  CANCELLED: 'cancelled',
} as const
export type TrainingJobStatus = (typeof TrainingJobStatus)[keyof typeof TrainingJobStatus]

/** Training phase — UI lifecycle (idle → training → complete/error). */
export const TrainingPhase = {
  IDLE: 'idle',
  TRAINING: 'training',
  COMPLETE: 'complete',
  ERROR: 'error',
} as const
export type TrainingPhase = (typeof TrainingPhase)[keyof typeof TrainingPhase]

/** Training method — user-facing selection. */
export const TrainingMethod = {
  DISTILL: 'distill',
  FINETUNE: 'finetune',
  VLM: 'vlm',
  NATIVE: 'native',
} as const
export type TrainingMethod = (typeof TrainingMethod)[keyof typeof TrainingMethod]

/** Training method — internal/API aliases (maps to same concepts). */
export const TrainingMethodInternal = {
  SLONET: 'slonet',
  HF: 'hf',
  TURBO: 'turbo',
} as const
export type TrainingMethodInternal =
  (typeof TrainingMethodInternal)[keyof typeof TrainingMethodInternal]

// ── Health / Status ─────────────────────────────────────────────────

/** Health level — service health check result. */
export const HealthLevel = {
  HEALTHY: 'healthy',
  DEGRADED: 'degraded',
  OFFLINE: 'offline',
  ERROR: 'error',
} as const
export type HealthLevel = (typeof HealthLevel)[keyof typeof HealthLevel]

const HEALTH_LEVEL_VALUES: readonly string[] = Object.values(HealthLevel)

/** Narrow a wire string to HealthLevel, falling back when unknown. */
export function asHealthLevel(value: string | null | undefined): HealthLevel {
  return (HEALTH_LEVEL_VALUES as readonly string[]).includes(value ?? '')
    ? (value as HealthLevel)
    : HealthLevel.OFFLINE
}

/** Severity — alerts, errors, banners. */
export const Severity = {
  INFO: 'info',
  WARNING: 'warning',
  ERROR: 'error',
  CRITICAL: 'critical',
} as const
export type Severity = (typeof Severity)[keyof typeof Severity]

/** Message tone — toast, banner, notification. */
export const MessageTone = {
  INFO: 'info',
  SUCCESS: 'success',
  WARNING: 'warning',
  ERROR: 'error',
  DESTRUCTIVE: 'destructive',
} as const
export type MessageTone = (typeof MessageTone)[keyof typeof MessageTone]

/** Derived types for backwards compat. */
export type ToastType = Extract<MessageTone, 'info' | 'success' | 'error'>
export type BannerTone = MessageTone
export type BannerVariant = Extract<MessageTone, 'info' | 'success' | 'warning' | 'error'>

// ── SSE / Streaming ─────────────────────────────────────────────────

/** SSE stream phase — matches Python `StreamPhase` enum. */
export const StreamPhase = {
  IDLE: 'IDLE',
  STREAMING: 'STREAMING',
  ERROR: 'ERROR',
  CLIENT_ERROR: 'CLIENT_ERROR',
} as const
export type StreamPhase = (typeof StreamPhase)[keyof typeof StreamPhase]

/** SSE stream status — matches Python `StreamStatus` enum. */
export const StreamStatus = {
  WORKING: 'working',
  ERROR: 'error',
  COMPLETE: 'complete',
  IDLE: 'idle',
} as const
export type StreamStatus = (typeof StreamStatus)[keyof typeof StreamStatus]

// ── Consciousness ───────────────────────────────────────────────────

/** Qualia dimensions — 7D emotional/cognitive state. */
export const QualiaDimension = {
  VALENCE: 'valence',
  AROUSAL: 'arousal',
  NOVELTY: 'novelty',
  COHERENCE: 'coherence',
  SALIENCE: 'salience',
  CERTAINTY: 'certainty',
  COMPLEXITY: 'complexity',
} as const
export type QualiaDimension = (typeof QualiaDimension)[keyof typeof QualiaDimension]
export type Qualia = Record<QualiaDimension, number>

/** Consciousness level — 0=off, 1=basic, 2=standard, 3=deep. */
export const ConsciousnessLevel = {
  OFF: 0,
  BASIC: 1,
  STANDARD: 2,
  DEEP: 3,
} as const
export type ConsciousnessLevel = (typeof ConsciousnessLevel)[keyof typeof ConsciousnessLevel]
