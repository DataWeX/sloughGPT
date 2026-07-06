'use client'

import { useCallback, useEffect, useState } from 'react'
import { Button } from '@sloughgpt/strui'
import { trainingJobsController } from '@/lib/training-controller'
import type { Webhook } from '@/lib/training-controller'
import { devDebug } from '@/lib/dev-log'
import { useToastStore } from '@/lib/toast-store'

export function WebhookManager() {
  const addToast = useToastStore(s => s.addToast)
  const [webhooks, setWebhooks] = useState<Webhook[]>([])
  const [showAdd, setShowAdd] = useState(false)
  const [newWebhook, setNewWebhook] = useState({
    url: '',
    description: '',
    events: ['training.completed'],
  })
  const [loading, setLoading] = useState(false)
  const [availableEvents, setAvailableEvents] = useState<string[]>(['training.completed'])

  const fetchWebhooks = useCallback(async () => {
    try {
      const wh = await trainingJobsController.listWebhooks()
      setWebhooks(wh)
    } catch (error) {
      devDebug('Failed to fetch webhooks:', error)
    }
  }, [])

  const fetchStats = useCallback(async () => {
    try {
      await trainingJobsController.webhookStats()
    } catch (error) {
      devDebug('Failed to fetch webhook stats:', error)
    }
  }, [])

  useEffect(() => {
    void fetchWebhooks()
    void fetchStats()
  }, [fetchWebhooks, fetchStats])

  const handleAddWebhook = async () => {
    if (!newWebhook.url.trim()) return
    setLoading(true)
    try {
      await trainingJobsController.createWebhook(newWebhook.url, newWebhook.events)
      addToast('Webhook registered', 'success')
      setShowAdd(false)
      setNewWebhook({ url: '', description: '', events: ['training.completed'] })
      void fetchWebhooks()
    } catch (error) {
      addToast(error instanceof Error ? error.message : 'Failed to register webhook', 'error')
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteWebhook = async (id: string) => {
    if (!confirm('Delete this webhook?')) return
    try {
      await trainingJobsController.deleteWebhook(id)
      addToast('Webhook deleted', 'success')
      void fetchWebhooks()
    } catch (error) {
      addToast(error instanceof Error ? error.message : 'Delete failed', 'error')
    }
  }

  const handleTestWebhook = async (url: string) => {
    try {
      await trainingJobsController.testWebhook(url)
      addToast('Test sent', 'success')
    } catch (error) {
      addToast(error instanceof Error ? error.message : 'Test failed', 'error')
    }
  }

  const toggleEvent = (event: string) => {
    setNewWebhook(prev => ({
      ...prev,
      events: prev.events.includes(event)
        ? prev.events.filter(e => e !== event)
        : [...prev.events, event],
    }))
  }

  return (
    <div className="space-y-3">
      {webhooks.length > 0 ? (
        <div className="space-y-2">
          {webhooks.map(webhook => (
            <div key={webhook.id} className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${webhook.is_active ? 'bg-green-500' : 'bg-gray-400'}`} />
                  <span className="font-medium text-sm truncate">{webhook.url}</span>
                </div>
                <div className="flex gap-1 mt-1 flex-wrap">
                  {webhook.events.map(event => (
                    <span key={event} className="text-xs bg-muted px-2 py-0.5 rounded">
                      {event.split('.')[1]}
                    </span>
                  ))}
                </div>
              </div>
              <div className="flex gap-1 ml-2">
                <Button size="sm" variant="ghost" onClick={() => handleTestWebhook(webhook.url)}>
                  Test
                </Button>
                <Button size="sm" variant="ghost" onClick={() => handleDeleteWebhook(webhook.id)} className="text-destructive/60 hover:text-destructive">
                  ✕
                </Button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground text-center py-4">No webhooks configured</p>
      )}

      {showAdd ? (
        <div className="p-4 border border-border rounded-lg space-y-3">
          <input
            type="url"
            value={newWebhook.url}
            onChange={e => setNewWebhook(prev => ({ ...prev, url: e.target.value }))}
            placeholder="https://example.com/webhook"
            className="sl-input w-full"
          />
          <input
            type="text"
            value={newWebhook.description}
            onChange={e => setNewWebhook(prev => ({ ...prev, description: e.target.value }))}
            placeholder="Description (optional)"
            className="sl-input w-full"
          />
          <div className="flex gap-2 flex-wrap">
            {availableEvents.map(event => (
              <label key={event} className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={newWebhook.events.includes(event)}
                  onChange={() => toggleEvent(event)}
                />
                {event.split('.')[1]}
              </label>
            ))}
          </div>
          <div className="flex gap-2">
            <Button size="sm" onClick={handleAddWebhook} disabled={loading || !newWebhook.url.trim()}>
              {loading ? 'Adding...' : 'Add Webhook'}
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setShowAdd(false)}>
              Cancel
            </Button>
          </div>
        </div>
      ) : (
        <Button size="sm" onClick={() => setShowAdd(true)}>
          + Add Webhook
        </Button>
      )}
    </div>
  )
}
