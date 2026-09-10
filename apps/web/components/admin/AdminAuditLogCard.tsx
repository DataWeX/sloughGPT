'use client'

import { cn, Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'
import { RefreshCw, ChevronDown } from 'lucide-react'
import { timeAgo } from '@/lib/time-ago'

interface AuditLogEntry {
  event_type: string
  timestamp: string
  user?: string
  resource?: string
  detail?: string
  extra?: Record<string, unknown>
}

interface AdminAuditLogCardProps {
  logs: AuditLogEntry[]
  filter: string
  historyMode: 'session' | 'persisted'
  loadingMore: boolean
  onFilterChange: (filter: string) => void
  onToggleHistory: () => void
  onLoadOlder: () => void
  onRefresh: () => void
}

export function AdminAuditLogCard({
  logs,
  filter,
  historyMode,
  loadingMore,
  onFilterChange,
  onToggleHistory,
  onLoadOlder,
  onRefresh,
}: AdminAuditLogCardProps) {
  const filtered = filter
    ? logs.filter(l =>
        l.event_type.toLowerCase().includes(filter.toLowerCase()) ||
        l.user?.toLowerCase().includes(filter.toLowerCase()) ||
        l.resource?.toLowerCase().includes(filter.toLowerCase()) ||
        l.detail?.toLowerCase().includes(filter.toLowerCase())
      )
    : logs

  return (
    <Card data-testid="admin-audit-log">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Audit Logs</CardTitle>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={onRefresh}
              aria-label="Refresh logs"
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <Input
          placeholder="Filter logs..."
          value={filter}
          onChange={(e) => onFilterChange(e.target.value)}
          aria-label="Filter audit logs"
        />

        <div className="flex items-center gap-2">
          <Button
            variant={historyMode === 'session' ? 'default' : 'outline'}
            size="sm"
            onClick={onToggleHistory}
          >
            {historyMode === 'session' ? 'Session' : 'Persisted'}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={onLoadOlder}
            disabled={loadingMore}
          >
            <ChevronDown className="h-3.5 w-3.5 mr-1" />
            {loadingMore ? 'Loading...' : 'Load Older'}
          </Button>
        </div>

        <div className="space-y-1 max-h-80 overflow-y-auto" data-testid="log-list">
          {filtered.length === 0 && (
            <p className="text-sm text-muted-foreground text-center py-4">No logs found.</p>
          )}
          {filtered.map((log, i) => (
            <div
              key={i}
              className="flex items-start gap-2 rounded-md border border-border/30 px-2 py-1.5 text-[11px]"
            >
              <span className={cn(
                'h-1.5 w-1.5 rounded-full shrink-0 mt-1.5',
                log.event_type.includes('error') || log.event_type.includes('fail')
                  ? 'bg-destructive'
                  : log.event_type.includes('warn')
                    ? 'bg-warning'
                    : 'bg-success'
              )} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1.5">
                  <span className="font-medium truncate">{log.event_type}</span>
                  {log.user && <span className="text-muted-foreground truncate">@{log.user}</span>}
                  {log.resource && <span className="text-muted-foreground truncate">on {log.resource}</span>}
                </div>
                {log.detail && (
                  <p className="text-[9px] text-muted-foreground/70 mt-0.5 truncate">{log.detail}</p>
                )}
              </div>
              <span className="text-muted-foreground shrink-0 ml-2 text-[9px]">{timeAgo(log.timestamp)}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
