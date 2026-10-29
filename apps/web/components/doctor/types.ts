/**
 * `/doctor` API contract types — mirrored from
 * `apps/api/server/routers/doctor.py` (the frontend is built to this shape).
 *
 *   GET  /doctor/report  -> { report: DoctorReport | null, path, age_s }
 *   POST /doctor/run     -> { report: DoctorReport }
 */

export type DoctorSeverity = 'ok' | 'info' | 'warn' | 'critical'

export interface DoctorFinding {
  /** Probe that produced it: http | sse | journey | preflight */
  source: string
  /** Stable check id, e.g. "api.health_score" */
  check: string
  severity: DoctorSeverity
  score: number
  message: string
  detail?: string
  /** Core component: api | inference | training | system | journeys */
  component?: string
}

export interface DoctorProbe {
  name: string
  /** null = skipped */
  ok: boolean | null
  error?: string
}

export interface DoctorReport {
  schema_version: number
  /** Report timestamp (unix seconds) */
  ts: number
  targets: Record<string, string>
  probes: DoctorProbe[]
  findings: DoctorFinding[]
  summary: { total: number; by_severity: Record<DoctorSeverity, number> }
  overall: DoctorSeverity
}

export interface DoctorReportResponse {
  report: DoctorReport | null
  path: string
  /** Seconds since `report.ts`; null when there is no report */
  age_s: number | null
}
