'use client'

import { useMemo } from 'react'
import { cn, Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

interface AuditLog {
  event_type: string
  timestamp: string
  user?: string
  ip?: string
}

interface ApiKey {
  id: string
  name: string
  scopes: string[]
  created_at: number
  revoked: boolean
  expires_at?: number
}

interface SecurityAuditCardProps {
  logs: AuditLog[]
  keys: ApiKey[]
}

type Severity = 'critical' | 'warning' | 'info'

interface Finding {
  severity: Severity
  title: string
  detail: string
}

const SEVERITY_STYLE: Record<Severity, string> = {
  critical: 'bg-destructive/15 text-destructive border-destructive/30',
  warning: 'bg-warning/15 text-warning border-warning/30',
  info: 'bg-primary/15 text-primary border-primary/30',
}

const SEVERITY_ICON: Record<Severity, string> = {
  critical: '!',
  warning: '⚠',
  info: 'i',
}

function generateFindings(logs: AuditLog[], keys: ApiKey[]): Finding[] {
  const findings: Finding[] = []
  const activeKeys = keys.filter(k => !k.revoked)

  if (activeKeys.length === 0) {
    findings.push({
      severity: 'info',
      title: 'No active API keys',
      detail: 'Create an API key to enable programmatic access.',
    })
  }

  const wildcardKeys = activeKeys.filter(k => k.scopes.includes('*'))
  if (wildcardKeys.length > 0) {
    findings.push({
      severity: 'critical',
      title: `${wildcardKeys.length} key(s) have wildcard scope`,
      detail: `Keys [${wildcardKeys.map(k => k.name).join(', ')}] have full access. Restrict scopes to least privilege.`,
    })
  }

  if (activeKeys.length > 5) {
    findings.push({
      severity: 'warning',
      title: 'Many active API keys',
      detail: `${activeKeys.length} active keys. Revoke unused keys to reduce attack surface.`,
    })
  }

  const now = Date.now() / 1000
  const oldKeys = activeKeys.filter(k => (now - k.created_at) > 90 * 24 * 60 * 60)
  if (oldKeys.length > 0) {
    findings.push({
      severity: 'warning',
      title: 'Old API keys detected',
      detail: `${oldKeys.length} key(s) older than 90 days. Consider rotating them.`,
    })
  }

  const expiredKeys = activeKeys.filter(k => k.expires_at && k.expires_at < now)
  if (expiredKeys.length > 0) {
    findings.push({
      severity: 'critical',
      title: 'Expired API keys still active',
      detail: `${expiredKeys.length} key(s) have expired but are not revoked.`,
    })
  }

  const authFails = logs.filter(l => l.event_type.toLowerCase().includes('auth') && l.event_type.toLowerCase().includes('fail'))
  if (authFails.length >= 5) {
    findings.push({
      severity: 'critical',
      title: 'Multiple authentication failures',
      detail: `${authFails.length} auth failures detected. Possible brute force attempt.`,
    })
  } else if (authFails.length > 0) {
    findings.push({
      severity: 'warning',
      title: 'Authentication failures detected',
      detail: `${authFails.length} failed auth event(s) in logs.`,
    })
  }

  const deletes = logs.filter(l => l.event_type.toLowerCase().includes('delete'))
  if (deletes.length >= 10) {
    findings.push({
      severity: 'critical',
      title: 'High volume of deletions',
      detail: `${deletes.length} delete events. Verify these are intentional.`,
    })
  }

  const uniqueIps = new Set(logs.filter(l => l.ip).map(l => l.ip))
  if (uniqueIps.size > 10) {
    findings.push({
      severity: 'warning',
      title: 'Many unique IP addresses',
      detail: `${uniqueIps.size} unique IPs in audit logs. Review for unauthorized access.`,
    })
  }

  const uniqueUsers = new Set(logs.filter(l => l.user).map(l => l.user))
  if (uniqueUsers.size > 5) {
    findings.push({
      severity: 'warning',
      title: 'Many unique users',
      detail: `${uniqueUsers.size} distinct users in logs. Verify all are authorized.`,
    })
  }

  const now24h = Date.now() / 1000 - 24 * 60 * 60
  const recentLogs = logs.filter(l => new Date(l.timestamp).getTime() / 1000 > now24h)
  if (recentLogs.length > 100) {
    findings.push({
      severity: 'info',
      title: 'High activity in last 24h',
      detail: `${recentLogs.length} events in the last 24 hours.`,
    })
  }

  if (findings.length === 0) {
    findings.push({
      severity: 'info',
      title: 'Security posture looks good',
      detail: 'No critical issues found. Keep monitoring.',
    })
  }

  return findings
}

function calculateScore(findings: Finding[]): number {
  let score = 100
  for (const f of findings) {
    if (f.severity === 'critical') score -= 20
    else if (f.severity === 'warning') score -= 10
  }
  return Math.max(0, score)
}

export function SecurityAuditCard({ logs, keys }: SecurityAuditCardProps) {
  const findings = useMemo(() => generateFindings(logs, keys), [logs, keys])
  const score = useMemo(() => calculateScore(findings), [findings])

  const scoreColor = score >= 80 ? 'text-success' : score >= 50 ? 'text-warning' : 'text-destructive'
  const scoreLabel = score >= 80 ? 'Good' : score >= 50 ? 'Fair' : 'Poor'

  const criticals = findings.filter(f => f.severity === 'critical').length
  const warnings = findings.filter(f => f.severity === 'warning').length

  return (
    <Card data-testid="security-audit">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Security Audit</CardTitle>
          <div className="flex items-center gap-2">
            {criticals > 0 && <span className="text-[9px] px-1.5 py-0.5 rounded bg-destructive/15 text-destructive font-medium">{criticals} critical</span>}
            {warnings > 0 && <span className="text-[9px] px-1.5 py-0.5 rounded bg-warning/15 text-warning font-medium">{warnings} warning{warnings !== 1 ? 's' : ''}</span>}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/30">
          <div className={cn('text-2xl font-bold font-mono', scoreColor)}>
            {score}
          </div>
          <div>
            <p className={cn('text-sm font-medium', scoreColor)}>{scoreLabel}</p>
            <p className="text-[9px] text-muted-foreground">Security score</p>
          </div>
          <div className="flex-1">
            <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
              <div
                className={cn('h-full rounded-full transition-all duration-500', score >= 80 ? 'bg-success' : score >= 50 ? 'bg-warning' : 'bg-destructive')}
                style={{ width: `${score}%` }}
              />
            </div>
          </div>
        </div>

        <div className="space-y-1.5 max-h-64 overflow-y-auto">
          {findings.map((f, i) => (
            <div key={i} className={cn('flex items-start gap-2 rounded-md border px-2 py-1.5 text-[10px]', SEVERITY_STYLE[f.severity])}>
              <span className="font-bold text-xs shrink-0 mt-px">{SEVERITY_ICON[f.severity]}</span>
              <div className="min-w-0">
                <p className="font-medium">{f.title}</p>
                <p className="text-[9px] text-muted-foreground/70 mt-0.5">{f.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
