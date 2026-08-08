'use client'

import { Card, CardContent } from '@sloughgpt/strui'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'

interface DiagnosticsCardProps {
  liveHealth: LiveHealthSnapshot | null
}

function severityClass(severity: string): string {
  switch (severity) {
    case 'critical': return 'bg-destructive'
    case 'warn': return 'bg-warning'
    case 'info': return 'bg-primary'
    default: return 'bg-success'
  }
}

export function DiagnosticsCard({ liveHealth }: DiagnosticsCardProps) {
  if (!liveHealth) return null
  const { diagnoses, health_summary, health_score, health_status } = liveHealth
  if (diagnoses.length === 0 && !health_summary) return null

  return (
    <Card className="p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Diagnostics</span>
        <span className={`inline-flex items-center gap-1.5 px-1.5 py-0.5 rounded-full text-[10px] font-medium ${
          health_status === 'healthy' ? 'bg-success/10 text-success' :
          health_status === 'degraded' ? 'bg-warning/10 text-warning' :
          'bg-destructive/10 text-destructive'
        }`}>
          {health_score > 0 ? `${health_score}/100` : health_status}
        </span>
      </div>
      <CardContent className="p-0 space-y-2">
        {health_summary && <p className="text-xs text-muted-foreground">{health_summary}</p>}
        <div className="space-y-1">
          {diagnoses.map((d, i) => (
            <div key={`${d.check}-${i}`} className="flex items-start gap-2 text-xs">
              <span className={`mt-1 inline-block w-2 h-2 rounded-full shrink-0 ${severityClass(d.severity)}`} />
              <span className="capitalize font-medium w-20 shrink-0">{d.check}</span>
              <span className="text-muted-foreground flex-1">{d.message}</span>
              <span className="font-mono text-[10px] text-muted-foreground/70">{Math.round(d.score)}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
