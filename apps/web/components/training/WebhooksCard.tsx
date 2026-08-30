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
      <CardContent className="space-y-4">
        {loading ? (
          <div className="space-y-2">
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        ) : webhooks.length === 0 && !newUrl ? (
          <p className="text-xs text-muted-foreground">No webhooks yet. Add one to get notified when training starts or finishes.</p>
        ) : null}

        {webhooks.map(w => (
          <div key={w.id} className="rounded border text-sm">
            <div
              className="flex cursor-pointer items-center justify-between p-3 hover:bg-muted/30"
              onClick={() => handleToggleExpand(w.id)}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleToggleExpand(w.id); } }}
              role="button"
              tabIndex={0}
            >
              <div className="min-w-0 flex-1">
                <p className="truncate font-mono text-xs">{w.url}</p>
                <div className="flex gap-2 text-xs text-muted-foreground">
                  {w.events.map(e => (
                    <span key={e} className="rounded bg-muted px-1.5 py-0.5">{eventLabel(e)}</span>
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-1">
                <Button size="sm" variant="ghost" onClick={e => { e.stopPropagation(); void testWebhook(w.url, addToast) }} disabled={testingUrl === w.url}>
                  {testingUrl === w.url ? 'Testing...' : 'Test'}
                </Button>
                <Button size="sm" variant="ghost" className="text-destructive" onClick={e => { e.stopPropagation(); setPendingDelete(w.id) }}>
                  Delete
                </Button>
              </div>
            </div>
            {expandedId === w.id && (
              <div className="border-t px-3 py-2">
                {deliveriesLoading ? (
                  <p className="text-xs text-muted-foreground">Loading deliveries...</p>
                ) : deliveries.length === 0 ? (
                  <p className="text-xs text-muted-foreground">No deliveries yet</p>
                ) : (
                  <div className="space-y-1">
                    {deliveries.map(d => (
                      <div key={d.id} className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>{eventLabel(d.event)}</span>
                        <span className={d.success ? 'text-success' : 'text-destructive'}>
                          {d.success ? 'Delivered' : 'Failed'} ({d.status})
                        </span>
                        <span>{formatTimestamp(d.delivered_at)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        <div className="space-y-2 border-t pt-3">
          <p className="text-xs font-medium">Add webhook</p>
          <Input
            id="webhook-url"
            aria-label="Webhook URL"
            placeholder="https://example.com/webhook"
            value={newUrl}
            onChange={e => setNewUrl(e.target.value)}
            className="h-8 font-mono text-xs"
          />
          <div className="flex flex-wrap gap-2">
            {AVAILABLE_EVENTS.map(({ key, label }) => (
              <label key={key} className="flex items-center gap-1 text-xs text-muted-foreground">
                <Checkbox
                  checked={newEvents.includes(key)}
                  onCheckedChange={() => toggleEvent(key)}
                  className="h-3 w-3"
                />
                {label}
              </label>
            ))}
          </div>
          <Button size="sm" onClick={() => void addWebhook(addToast)} disabled={adding || !newUrl.trim()}>
            {adding ? 'Adding...' : 'Add webhook'}
          </Button>
        </div>

        {(retryQueue.length > 0 || deadLetters.length > 0) && (
          <div className="border-t pt-3 space-y-2">
            <button
              type="button"
              className="flex items-center gap-2 text-xs font-medium text-muted-foreground hover:text-foreground"
              onClick={() => setShowRetries(!showRetries)}
              aria-expanded={showRetries}
            >
              <span>{showRetries ? '▼' : '▶'}</span>
              <span>Retry queue ({retryQueue.length})</span>
              {deadLetters.length > 0 && <span className="text-destructive">· {deadLetters.length} dead</span>}
            </button>

            {showRetries && (
              <div className="space-y-2 text-xs">
                {retryQueue.length > 0 && (
                  <div className="space-y-1">
                    <p className="text-muted-foreground font-medium">Pending retries</p>
                    {retryQueue.map(r => (
                      <div key={r.delivery_id} className="flex items-center justify-between rounded bg-muted/30 px-2 py-1">
                        <span className="truncate font-mono">{r.webhook_id}</span>
                        <span>{eventLabel(r.event)}</span>
                        <span>attempt {r.attempt_count}/5</span>
                        <span className="text-muted-foreground">next: {new Date(r.next_retry_at * 1000).toLocaleTimeString()}</span>
                      </div>
                    ))}
                  </div>
                )}

                {deadLetters.length > 0 && (
                  <div className="space-y-1">
                    <p className="text-destructive font-medium">Dead letters (permanently failed)</p>
                    {deadLetters.map(dl => (
                      <div key={dl.delivery_id} className="flex items-center justify-between rounded bg-destructive/5 px-2 py-1">
                        <span className="truncate font-mono">{dl.webhook_id}</span>
                        <span>{eventLabel(dl.event)}</span>
                        <span className="text-destructive">{dl.error || `HTTP ${dl.status_code}`}</span>
                        <span>{dl.attempt_count} attempts</span>
                        <span className="text-muted-foreground">{formatTimestamp(dl.dead_lettered_at)}</span>
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
