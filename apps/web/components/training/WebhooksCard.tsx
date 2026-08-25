'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label, Skeleton } from '@sloughgpt/strui'
import { ConfirmDialog } from '@/components/ConfirmDialog'
import { trainingJobsController, type Webhook, type WebhookStats } from '@/lib/training-controller'

interface Props {
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
}

const AVAILABLE_EVENTS = [
  { key: 'training.complete', label: 'Training Completed' },
  { key: 'training.failed', label: 'Training Failed' },
  { key: 'training.started', label: 'Training Started' },
]

function eventLabel(key: string): string {
  return AVAILABLE_EVENTS.find(e => e.key === key)?.label ?? key
}

function formatTimestamp(ts: string): string {
  try { return new Date(ts).toLocaleString() } catch { return ts }
}

interface RetryEntry {
  delivery_id: string
  webhook_id: string
  event: string
  attempt_count: number
  next_retry_at: number
}

interface DeadLetter {
  delivery_id: string
  webhook_id: string
  event: string
  error: string | null
  status_code: number | null
  attempt_count: number
  dead_lettered_at: string
}

export function WebhooksCard({ addToast }: Props) {
  const [webhooks, setWebhooks] = useState<Webhook[]>([])
  const [loading, setLoading] = useState(true)
  const [newUrl, setNewUrl] = useState('')
  const [newEvents, setNewEvents] = useState<string[]>(['training.complete'])
  const [adding, setAdding] = useState(false)
  const [testingUrl, setTestingUrl] = useState<string | null>(null)
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [deliveries, setDeliveries] = useState<Array<{ id: string; event: string; status: number; success: boolean; delivered_at: string }>>([])
  const [deliveriesLoading, setDeliveriesLoading] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)
  const [retryQueue, setRetryQueue] = useState<RetryEntry[]>([])
  const [deadLetters, setDeadLetters] = useState<DeadLetter[]>([])
  const [showRetries, setShowRetries] = useState(false)
  const [stats, setStats] = useState<WebhookStats | null>(null)

  const fetchWebhooks = useCallback(async () => {
    setLoading(true)
    try {
      const result = await trainingJobsController.listWebhooks()
      setWebhooks(result ?? [])
    } catch {
      setWebhooks([])
    } finally {
      setLoading(false)
    }
  }, [])

  const fetchRetryData = useCallback(async () => {
    try {
      const [retries, deads, statsResult] = await Promise.all([
        trainingJobsController.getWebhookRetryQueue(),
        trainingJobsController.getWebhookDeadLetters(),
        trainingJobsController.webhookStats(),
      ])
      setRetryQueue(retries.retries ?? [])
      setDeadLetters(deads.dead_letters ?? [])
      setStats(statsResult)
    } catch {
      // Silently fail
    }
  }, [])

  useEffect(() => { void fetchWebhooks() }, [fetchWebhooks])
  useEffect(() => { void fetchRetryData() }, [fetchRetryData])

  const handleAdd = useCallback(async () => {
    if (!newUrl.trim()) return
    setAdding(true)
    try {
      await trainingJobsController.createWebhook(newUrl, newEvents)
      addToast('Webhook added', 'success')
      setNewUrl('')
      setNewEvents(['training.complete'])
      void fetchWebhooks()
    } catch {
      addToast('Could not add webhook', 'error')
    } finally {
      setAdding(false)
    }
  }, [newUrl, newEvents, addToast, fetchWebhooks])

  const handleDelete = useCallback(async () => {
    if (!pendingDelete) return
    const id = pendingDelete
    setPendingDelete(null)
    try {
      await trainingJobsController.deleteWebhook(id)
      addToast('Webhook deleted', 'success')
      void fetchWebhooks()
    } catch {
      addToast('Could not delete webhook', 'error')
    }
  }, [pendingDelete, addToast, fetchWebhooks])

  const handleTest = useCallback(async (url: string) => {
    setTestingUrl(url)
    try {
      await trainingJobsController.testWebhook(url)
      addToast('Webhook test sent', 'success')
    } catch {
      addToast('Could not webhook test', 'error')
    } finally {
      setTestingUrl(null)
    }
  }, [addToast])

  const toggleDeliveries = useCallback(async (id: string) => {
    if (expandedId === id) {
      setExpandedId(null)
      setDeliveries([])
      return
    }
    setExpandedId(id)
    setDeliveriesLoading(true)
    try {
      const result = await trainingJobsController.getWebhookDeliveries(id, 10)
      setDeliveries(result ?? [])
    } catch {
      setDeliveries([])
    } finally {
      setDeliveriesLoading(false)
    }
  }, [expandedId])

  const toggleEvent = useCallback((event: string) => {
    setNewEvents(prev => prev.includes(event) ? prev.filter(e => e !== event) : [...prev, event])
  }, [])

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
              onClick={() => void toggleDeliveries(w.id)}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); void toggleDeliveries(w.id); } }}
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
                <Button size="sm" variant="ghost" onClick={e => { e.stopPropagation(); void handleTest(w.url) }} disabled={testingUrl === w.url}>
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
                <input
                  type="checkbox"
                  checked={newEvents.includes(key)}
                  onChange={() => toggleEvent(key)}
                  className="h-3 w-3"
                />
                {label}
              </label>
            ))}
          </div>
          <Button size="sm" onClick={handleAdd} disabled={adding || !newUrl.trim()}>
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
        onConfirm={() => void handleDelete()}
      />
    </Card>
  )
}
