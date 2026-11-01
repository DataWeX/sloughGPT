'use client'

import { Card, CardContent, CardHeader, CardTitle, Button, Input, Skeleton, Checkbox } from '@sloughgpt/strui'
import { ConfirmDialog } from '@/components/ConfirmDialog'
import {
  useWebhooks, AVAILABLE_EVENTS, eventLabel, formatTimestamp,
} from '@/hooks/useWebhooks'

interface Props {
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
}

export function WebhooksCard({ addToast }: Props) {
  const {
    webhooks, loading, newUrl, newEvents, adding, testingUrl,
    expandedId, deliveries, deliveriesLoading, pendingDelete,
    retryQueue, deadLetters, showRetries, stats,
    setNewUrl, setNewEvents, setExpandedId, setShowRetries, setPendingDelete,
    fetchWebhooks, addWebhook, deleteWebhook, testWebhook, loadDeliveries,
  } = useWebhooks()

  const toggleEvent = (event: string) => {
    setNewEvents(newEvents.includes(event) ? newEvents.filter(e => e !== event) : [...newEvents, event])
  }

  const handleToggleExpand = (id: string) => {
    if (expandedId === id) {
      setExpandedId(null)
      return
    }
    setExpandedId(id)
    void loadDeliveries(id)
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Webhooks ({webhooks.length})</CardTitle>
        {stats && stats.total_deliveries > 0 && (
          <p className="text-xs text-muted-foreground">
            {stats.successful_deliveries}/{stats.total_deliveries} delivered ({stats.success_rate})
            {stats.failed_deliveries > 0 && <span className="text-destructive"> · {stats.failed_deliveries} failed</span>}
          </p>
        )}
      </CardHeader>
      <CardContent className="space-y-3">
        {loading ? (
          <div className="space-y-1.5">
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        ) : webhooks.length === 0 && !newUrl ? (
          <p className="text-[10px] text-muted-foreground/60">No webhooks yet. Add one to get notified when training starts or finishes.</p>
        ) : null}

        {webhooks.map(w => (
          <div key={w.id} className="rounded-lg border border-border/40 text-xs">
            <div
              className="flex cursor-pointer items-center justify-between p-2.5 hover:bg-muted/20 transition-colors"
              onClick={() => handleToggleExpand(w.id)}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleToggleExpand(w.id); } }}
              role="button"
              tabIndex={0}
            >
              <div className="min-w-0 flex-1">
                <p className="truncate font-mono text-[10px]">{w.url}</p>
                <div className="flex gap-1 text-[10px] text-muted-foreground/60">
                  {w.events.map(e => (
                    <span key={e} className="rounded-full bg-muted/50 px-1.5 py-0.5">{eventLabel(e)}</span>
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-0.5">
                <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={e => { e.stopPropagation(); void testWebhook(w.url, addToast) }} disabled={testingUrl === w.url}>
                  {testingUrl === w.url ? 'Testing...' : 'Test'}
                </Button>
                <Button size="sm" variant="ghost" className="h-6 text-[10px] text-destructive" onClick={e => { e.stopPropagation(); setPendingDelete(w.id) }}>
                  Delete
                </Button>
              </div>
            </div>
            {expandedId === w.id && (
              <div className="border-t border-border/30 px-2.5 py-1.5">
                {deliveriesLoading ? (
                  <p className="text-[10px] text-muted-foreground/60">Loading deliveries...</p>
                ) : deliveries.length === 0 ? (
                  <p className="text-[10px] text-muted-foreground/60">No deliveries yet</p>
                ) : (
                  <div className="space-y-0.5">
                    {deliveries.map(d => (
                      <div key={d.id} className="flex items-center justify-between text-[10px] text-muted-foreground/60">
                        <span>{eventLabel(d.event)}</span>
                        <span className={d.success ? 'text-success' : 'text-destructive'}>
                          {d.success ? 'Delivered' : 'Failed'} ({d.status})
                        </span>
                        <span className="tabular-nums">{formatTimestamp(d.delivered_at)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        <div className="space-y-1.5 border-t border-border/30 pt-2.5">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Add webhook</p>
          <Input
            id="webhook-url"
            aria-label="Webhook URL"
            placeholder="https://example.com/webhook"
            value={newUrl}
            onChange={e => setNewUrl(e.target.value)}
            className="h-7 font-mono text-[10px]"
          />
          <div className="flex flex-wrap gap-1.5">
            {AVAILABLE_EVENTS.map(({ key, label }) => (
              <label key={key} className="flex items-center gap-1 text-[10px] text-muted-foreground/60">
                <Checkbox
                  checked={newEvents.includes(key)}
                  onCheckedChange={() => toggleEvent(key)}
                  className="h-3 w-3"
                />
                {label}
              </label>
            ))}
          </div>
          <Button size="sm" className="h-7 text-[10px]" onClick={() => void addWebhook(addToast)} disabled={adding || !newUrl.trim()}>
            {adding ? 'Adding...' : 'Add webhook'}
          </Button>
        </div>

        {(retryQueue.length > 0 || deadLetters.length > 0) && (
          <div className="border-t border-border/30 pt-2.5 space-y-1.5">
            <button
              type="button"
              className="flex items-center gap-1.5 text-[10px] font-medium text-muted-foreground/60 hover:text-foreground"
              onClick={() => setShowRetries(!showRetries)}
              aria-expanded={showRetries}
            >
              <span>{showRetries ? '▼' : '▶'}</span>
              <span>Retry queue ({retryQueue.length})</span>
              {deadLetters.length > 0 && <span className="text-destructive">· {deadLetters.length} dead</span>}
            </button>

            {showRetries && (
              <div className="space-y-1.5 text-[10px]">
                {retryQueue.length > 0 && (
                  <div className="space-y-0.5">
                    <p className="text-muted-foreground/60 font-medium">Pending retries</p>
                    {retryQueue.map(r => (
                      <div key={r.delivery_id} className="flex items-center justify-between rounded bg-muted/30 px-2 py-1">
                        <span className="truncate font-mono">{r.webhook_id}</span>
                        <span>{eventLabel(r.event)}</span>
                        <span>attempt {r.attempt_count}/5</span>
                        <span className="text-muted-foreground/60 tabular-nums">next: {new Date(r.next_retry_at * 1000).toLocaleTimeString()}</span>
                      </div>
                    ))}
                  </div>
                )}

                {deadLetters.length > 0 && (
                  <div className="space-y-0.5">
                    <p className="text-destructive font-medium">Dead letters (permanently failed)</p>
                    {deadLetters.map(dl => (
                      <div key={dl.delivery_id} className="flex items-center justify-between rounded bg-destructive/5 px-2 py-1">
                        <span className="truncate font-mono">{dl.webhook_id}</span>
                        <span>{eventLabel(dl.event)}</span>
                        <span className="text-destructive">{dl.error || `HTTP ${dl.status_code}`}</span>
                        <span>{dl.attempt_count} attempts</span>
                        <span className="text-muted-foreground/60 tabular-nums">{formatTimestamp(dl.dead_lettered_at)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </CardContent>

      <ConfirmDialog
        open={pendingDelete !== null}
        onOpenChange={(open) => { if (!open) setPendingDelete(null) }}
        title="Delete webhook"
        description="This webhook will stop receiving training notifications. This cannot be undone."
        confirmLabel="Delete Webhook"
        onConfirm={() => void deleteWebhook(pendingDelete!, addToast)}
      />
    </Card>
  )
}
