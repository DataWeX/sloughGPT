'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

interface AuditLog {
  event_type: string
  timestamp: string
  details?: Record<string, unknown>
  ip?: string
  user?: string
}

interface SecurityOverviewCardProps {
  logs: AuditLog[]
  apiKeyConfigured: boolean
  apiKeyCount: number
}

function categorize(eventType: string): string {
  const t = eventType.toLowerCase()
  if (t.includes('auth') || t.includes('login') || t.includes('token') || t.includes('key')) return 'Auth'
  if (t.includes('delet') || t.includes('remov')) return 'Destructive'
  if (t.includes('model') || t.includes('load') || t.includes('inference')) return 'Model'
  if (t.includes('train') || t.includes('checkpoint')) return 'Training'
  if (t.includes('upload') || t.includes('file') || t.includes('dataset')) return 'Data'
  return 'System'
}

const CATEGORY_COLORS: Record<string, string> = {
  Auth: 'bg-primary/15 text-primary',
  Model: 'bg-success/15 text-success',
  Training: 'bg-warning/15 text-warning',
  Data: 'bg-accent/15 text-accent',
  Destructive: 'bg-destructive/15 text-destructive',
  System: 'bg-muted text-muted-foreground',
}

function timeAgo(ts: string): string {
  try {
    const d = new Date(ts)
    if (isNaN(d.getTime())) return ''
    const diffMs = Date.now() - d.getTime()
    const diffM = Math.floor(diffMs / 60000)
    const diffH = Math.floor(diffM / 60)
    const diffD = Math.floor(diffH / 24)
    if (diffD > 0) return `${diffD}d ago`
    if (diffH > 0) return `${diffH}h ago`
    if (diffM > 0) return `${diffM}m ago`
    return 'just now'
  } catch { return '' }
}

export function SecurityOverviewCard({ logs, apiKeyConfigured, apiKeyCount }: SecurityOverviewCardProps) {
  const categories: Record<string, number> = {}
  for (const log of logs) {
    const cat = categorize(log.event_type)
    categories[cat] = (categories[cat] ?? 0) + 1
  }

  const ips = new Set(logs.filter(l => l.ip).map(l => l.ip))
  const users = new Set(logs.filter(l => l.user).map(l => l.user))

  const recent = logs.slice(0, 5)

  return (
    <Card data-testid="security-overview">
      <CardHeader>
        <CardTitle className="text-base">Security Overview</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-4 gap-2 mb-3">
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">API Keys</div>
            <div className={`text-sm font-mono font-medium ${apiKeyConfigured ? 'text-success' : 'text-warning'}`}>
              {apiKeyConfigured ? apiKeyCount : 'None'}
            </div>
          </div>
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">Events</div>
            <div className="text-sm font-mono font-medium">{logs.length}</div>
          </div>
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">IPs</div>
            <div className="text-sm font-mono font-medium">{ips.size}</div>
          </div>
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">Users</div>
            <div className="text-sm font-mono font-medium">{users.size}</div>
          </div>
        </div>

        {Object.keys(categories).length > 0 && (
          <div className="mb-3">
            <div className="text-[10px] text-muted-foreground mb-1.5">Event Categories</div>
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(categories).sort((a, b) => b[1] - a[1]).map(([cat, count]) => (
                <span key={cat} className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${CATEGORY_COLORS[cat] ?? CATEGORY_COLORS.System}`}>
                  {cat} ({count})
                </span>
              ))}
            </div>
          </div>
        )}

        {recent.length > 0 && (
          <div>
            <div className="text-[10px] text-muted-foreground mb-1.5">Recent Activity</div>
            <div className="space-y-1">
              {recent.map((log, i) => (
                <div key={i} className="flex items-center justify-between text-[11px] py-0.5 border-b border-border/30 last:border-0">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <span className={`h-1.5 w-1.5 rounded-full shrink-0 ${
                      log.event_type.toLowerCase().includes('delete') ? 'bg-destructive' :
                      log.event_type.toLowerCase().includes('auth') || log.event_type.toLowerCase().includes('login') ? 'bg-primary' :
                      'bg-success'
                    }`} />
                    <span className="font-medium truncate">{log.event_type}</span>
                    {log.user && <span className="text-muted-foreground truncate">@{log.user}</span>}
                  </div>
                  <span className="text-muted-foreground shrink-0 ml-2">{timeAgo(log.timestamp)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {logs.length === 0 && (
          <p className="text-sm text-muted-foreground text-center py-2">No audit events recorded.</p>
        )}
      </CardContent>
    </Card>
  )
}
