/**
 * `/mole` API contract types — mirrored from
 * `apps/api/server/routers/mole.py` (the frontend is built to this shape).
 *
 *   GET  /mole/report  -> { report: MoleReport | null, path, age_s }
 *   POST /mole/run     -> { report: MoleReport }
 */

export type MoleSeverity = 'ok' | 'info' | 'warn' | 'critical'

export interface MoleFinding {
  /** Probe that produced it: http | sse | journey | preflight */
  source: string
  /** Stable check id, e.g. "api.health_score" */
  check: string
  severity: MoleSeverity
  score: number
  message: string
  detail?: string
  /** Core component: api | inference | training | system | journeys */
  component?: string
}

export interface MoleProbe {
  name: string
  /** null = skipped */
  ok: boolean | null
  error?: string
}

export interface MoleReport {
  schema_version: number
  /** Report timestamp (unix seconds) */
  ts: number
  targets: Record<string, string>
  probes: MoleProbe[]
  findings: MoleFinding[]
  summary: { total: number; by_severity: Record<MoleSeverity, number> }
  overall: MoleSeverity
}

export interface MoleReportResponse {
  report: MoleReport | null
  path: string
  /** Seconds since `report.ts`; null when there is no report */
  age_s: number | null
}
