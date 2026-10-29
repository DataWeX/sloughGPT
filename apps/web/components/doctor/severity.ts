/**
 * Severity vocabulary shared by DoctorSummary and FindingsList.
 *
 * Colors are Noir Violet semantic tokens only (success / warning / primary /
 * destructive) — never raw hex or rgb literals.
 */

import type { DoctorSeverity } from './types'

/** Worst-first display order. */
export const SEVERITY_ORDER: DoctorSeverity[] = ['critical', 'warn', 'info', 'ok']

export const SEVERITY_LABEL: Record<DoctorSeverity, string> = {
  critical: 'Critical',
  warn: 'Warning',
  info: 'Info',
  ok: 'OK',
}

/** `StatusDot` / `StatusBadge` tone per severity. */
export const SEVERITY_TONE: Record<
  DoctorSeverity,
  'destructive' | 'warning' | 'primary' | 'success'
> = {
  critical: 'destructive',
  warn: 'warning',
  info: 'primary',
  ok: 'success',
}

/** Text token class per severity (contrast-safe on both themes). */
export const SEVERITY_TEXT_CLASS: Record<DoctorSeverity, string> = {
  critical: 'text-destructive',
  warn: 'text-warning',
  info: 'text-primary',
  ok: 'text-success',
}

/** Subtle chip background per severity. */
export const SEVERITY_CHIP_CLASS: Record<DoctorSeverity, string> = {
  critical: 'bg-destructive/10 text-destructive border-destructive/30',
  warn: 'bg-warning/10 text-warning border-warning/30',
  info: 'bg-primary/10 text-primary border-primary/30',
  ok: 'bg-success/10 text-success border-success/30',
}

/**
 * `StatusBadge` tone per severity — its vocabulary spells the informational
 * band `info` where `StatusDot` spells it `primary`.
 */
export const SEVERITY_BADGE_TONE: Record<
  DoctorSeverity,
  'destructive' | 'warning' | 'info' | 'success'
> = {
  critical: 'destructive',
  warn: 'warning',
  info: 'info',
  ok: 'success',
}
