'use client'

import { useState, useMemo } from 'react'
import { cn, Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'
import { timeAgo } from '@/lib/time-ago'

interface AuditLog {
  event_type: string
  timestamp: string
  user?: string
  resource?: string
  detail?: string
  extra?: Record<string, unknown>
  ip?: string
}

interface ThreatLogCardProps {
  logs: AuditLog[]
}

type Severity = 'critical' | 'high' | 'medium' | 'low' | 'info'

interface ThreatEntry {
  log: AuditLog
  severity: Severity
  reason: string
}

const SEVERITY_STYLES: Record<Severity, string> = {
  critical: 'bg-destructive/15 text-destructive border-destructive/30',
  high: 'bg-orange-500/15 text-orange-500 border-orange-500/30',
  medium: 'bg-warning/15 text-warning border-warning/30',
  low: 'bg-primary/15 text-primary border-primary/30',
  info: 'bg-muted text-muted-foreground border-border/40',
}

const SEVERITY_DOT: Record<Severity, string> = {
  critical: 'bg-destructive',
  high: 'bg-orange-500',
  medium: 'bg-warning',
  low: 'bg-primary',
  info: 'bg-muted-foreground',
}

function classifySeverity(log: AuditLog): { severity: Severity; reason: string } {
  const t = log.event_type.toLowerCase()
  const detail = (log.detail ?? '').toLowerCase()

  if (t.includes('delete') || t.includes('revoke') || t.includes('destroy')) {
    return { severity: 'high', reason: 'Destructive action' }
  }
  if (t.includes('auth') && (t.includes('fail') || t.includes('denied') || detail.includes('fail'))) {
    return { severity: 'high', reason: 'Authentication failure' }
  }
  if (t.includes('rotate') || t.includes('create') && t.includes('key')) {
    return { severity: 'medium', reason: 'Key management' }
  }
  if (t.includes('train') || t.includes('checkpoint')) {
    return { severity: 'medium', reason: 'Training activity' }
  }
  if (t.includes('upload') || t.includes('import')) {
    return { severity: 'low', reason: 'Data import' }
  }
  if (t.includes('model') || t.includes('load')) {
    return { severity: 'low', reason: 'Model operation' }
  }
  return { severity: 'info', reason: 'General event' }
}

function detectAnomalies(logs: AuditLog[]): ThreatEntry[] {
  const threats: ThreatEntry[] = []

  const authFails = logs.filter(l => l.event_type.toLowerCase().includes('auth') && (l.detail?.toLowerCase().includes('fail') || l.event_type.toLowerCase().includes('fail')))
  if (authFails.length >= 3) {
    threats.push({
      log: authFails[authFails.length - 1],
      severity: 'critical',
      reason: `Possible brute force: ${authFails.length} auth failures detected`,
    })
  }

  const deletes = logs.filter(l => l.event_type.toLowerCase().includes('delete'))
  if (deletes.length >= 5) {
    threats.push({
      log: deletes[deletes.length - 1],
      severity: 'critical',
      reason: `Bulk deletions: ${deletes.length} delete events`,
    })
  }

  const ipCounts: Record<string, number> = {}
  for (const l of logs) {
    if (l.ip) ipCounts[l.ip] = (ipCounts[l.ip] ?? 0) + 1
  }
  const suspiciousIps = Object.entries(ipCounts).filter(([, c]) => c >= 10)
  for (const [ip, count] of suspiciousIps) {
    const log = logs.find(l => l.ip === ip)
    if (log) {
      threats.push({
        log,
        severity: 'high',
        reason: `High activity from IP ${ip}: ${count} events`,
      })
    }
  }

  return threats
}

export function ThreatLogCard({ logs }: ThreatLogCardProps) {
  const [expanded, setExpanded] = useState(false)
  const [showAnomalies, setShowAnomalies] = useState(false)

  const classified = useMemo(() => {
    return logs.map(log => ({
      log,
      ...classifySeverity(log),
    }))
  }, [logs])

  const anomalies = useMemo(() => detectAnomalies(logs), [logs])

  const severityCounts = useMemo(() => {
    const counts: Record<Severity, number> = { critical: 0, high: 0, medium: 0, low: 0, info: 0 }
    for (const c of classified) counts[c.severity]++
    return counts
  }, [classified])

  const displayLogs = showAnomalies
    ? anomalies
    : expanded
      ? classified
      : classified.slice(0, 10)

  const totalAnomalies = anomalies.length

  return (
    <Card data-testid="threat-log">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Threat Log</CardTitle>
          {totalAnomalies > 0 && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-destructive/15 text-destructive font-medium">
              {totalAnomalies} anomal{totalAnomalies === 1 ? 'y' : 'ies'}
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap gap-1.5">
          {(Object.entries(severityCounts) as [Severity, number][]).map(([sev, count]) => (
            count > 0 && (
              <span key={sev} className={cn('text-[9px] px-1.5 py-0.5 rounded font-medium border', SEVERITY_STYLES[sev])}>
                {sev} ({count})
              </span>
            )
          ))}
        </div>

        <div className="flex gap-1">
          <Button
            size="sm"
            variant={!showAnomalies ? 'default' : 'ghost'}
            className="h-6 text-[10px]"
            onClick={() => setShowAnomalies(false)}
          >
            All Events
          </Button>
          <Button
            size="sm"
            variant={showAnomalies ? 'default' : 'ghost'}
            className="h-6 text-[10px]"
            onClick={() => setShowAnomalies(true)}
            disabled={totalAnomalies === 0}
          >
            Anomalies ({totalAnomalies})
          </Button>
        </div>

        {displayLogs.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-3">
            {logs.length === 0 ? 'No events recorded yet.' : 'No events match filter.'}
          </p>
        ) : (
          <div className="space-y-1 max-h-64 overflow-y-auto">
            {displayLogs.map((entry, i) => (
              <div
                key={i}
                className={cn('flex items-start gap-2 rounded-md border px-2 py-1.5 text-[10px]', SEVERITY_STYLES[entry.severity])}
              >
                <span className={cn('h-2 w-2 rounded-full shrink-0 mt-0.5', SEVERITY_DOT[entry.severity])} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium truncate">{entry.log.event_type}</span>
                    <span className="text-[9px] text-muted-foreground shrink-0">{timeAgo(entry.log.timestamp)}</span>
                  </div>
                  <p className="text-[9px] text-muted-foreground/70 mt-0.5">{entry.reason}</p>
                  {entry.log.user && <span className="text-[9px] text-muted-foreground/60">@{entry.log.user} </span>}
                  {entry.log.ip && <span className="text-[9px] text-muted-foreground/60">from {entry.log.ip}</span>}
                </div>
              </div>
            ))}
          </div>
        )}

        {!showAnomalies && classified.length > 10 && (
          <Button size="sm" variant="ghost" className="h-6 text-[10px] w-full" onClick={() => setExpanded(!expanded)}>
            {expanded ? 'Show less' : `Show all ${classified.length} events`}
          </Button>
        )}
      </CardContent>
    </Card>
  )
}
