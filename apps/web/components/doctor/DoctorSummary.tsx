'use client'

import { cn, StatusBadge, StatusDot } from '@sloughgpt/strui'
import { timeAgo } from '@/lib/time-ago'
import {
  SEVERITY_BADGE_TONE,
  SEVERITY_CHIP_CLASS,
  SEVERITY_LABEL,
  SEVERITY_ORDER,
  SEVERITY_TONE,
} from './severity'
import type { DoctorReport } from './types'

interface DoctorSummaryProps {
  report: DoctorReport
  /** Seconds since the report was written — fallback when `report.ts` is absent. */
  ageS?: number | null
}

function reportAge(report: DoctorReport, ageS?: number | null): string {
  if (report.ts) return timeAgo(report.ts)
  if (ageS != null) return `${Math.max(0, Math.round(ageS))}s ago`
  return 'unknown'
}

/**
 * Overall status pill + severity counts + report age.
 *
 * Summary, not a dump: counts and the worst severity only — the per-finding
 * detail lives in `FindingsList`.
 */
export function DoctorSummary({ report, ageS }: DoctorSummaryProps) {
  const counts = report.summary?.by_severity ?? ({} as Record<string, number>)
  const overall = report.overall ?? 'ok'
  const findings = report.findings ?? []
  const probes = report.probes ?? []
  const probesOk = probes.filter((p) => p.ok === true).length

  return (
    <section
      data-testid="doctor-summary"
      aria-label="Doctor summary"
      className="rounded-lg border border-border/60 bg-card/50 px-4 py-3"
    >
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <div className="flex items-center gap-2">
          <StatusDot
            tone={SEVERITY_TONE[overall]}
            pulse={overall === 'critical'}
            aria-hidden
          />
          <span className="text-sm font-medium" data-testid="doctor-overall">
            Overall {SEVERITY_LABEL[overall]}
          </span>
          <StatusBadge tone={SEVERITY_BADGE_TONE[overall]} size="md">
            {overall}
          </StatusBadge>
        </div>
        <span className="text-xs text-muted-foreground">
          Checked {reportAge(report, ageS)}
        </span>
      </div>

      <div
        className="mt-3 flex flex-wrap gap-1.5"
        role="list"
        aria-label="Findings by severity"
      >
        {SEVERITY_ORDER.map((sev) => (
          <span
            key={sev}
            role="listitem"
            data-testid={`severity-count-${sev}`}
            className={cn(
              'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs',
              SEVERITY_CHIP_CLASS[sev],
            )}
          >
            <span className="font-semibold">{counts[sev] ?? 0}</span>
            <span>{SEVERITY_LABEL[sev]}</span>
          </span>
        ))}
      </div>

      <p className="mt-2 text-xs text-muted-foreground">
        {report.summary?.total ?? findings.length} findings · {probesOk}/{probes.length}{' '}
        probes ok
      </p>
    </section>
  )
}
