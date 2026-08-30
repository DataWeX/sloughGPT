'use client'

import { useState, useEffect, useCallback } from 'react'
import { trainingJobsController, type Webhook, type WebhookStats } from '@/lib/training-controller'

export const AVAILABLE_EVENTS = [
  { key: 'training.completed', label: 'Training Completed' },
  { key: 'training.failed', label: 'Training Failed' },
  { key: 'training.started', label: 'Training Started' },
]

export function eventLabel(key: string): string {
  return AVAILABLE_EVENTS.find(e => e.key === key)?.label ?? key
}

export function formatTimestamp(ts: string): string {
  try { return new Date(ts).toLocaleString() } catch { return ts }
}

export interface RetryEntry {
  delivery_id: string
  webhook_id: string
  event: string
  attempt_count: number
  next_retry_at: number
}

export interface DeadLetter {
  delivery_id: string
  webhook_id: string
  event: string
  error: string | null
  status_code: number | null
  attempt_count: number
  dead_lettered_at: string
}

export interface DeliveryEntry {
  id: string
  event: string
  status: number
  success: boolean
  delivered_at: string
}

export interface UseWebhooksReturn {
  webhooks: Webhook[]
  loading: boolean
  newUrl: string
  newEvents: string[]
  adding: boolean
  testingUrl: string | null
  expandedId: string | null
  deliveries: DeliveryEntry[]
  deliveriesLoading: boolean
  pendingDelete: string | null
  retryQueue: RetryEntry[]
  deadLetters: DeadLetter[]
  showRetries: boolean
  stats: WebhookStats | null
  setNewUrl: (url: string) => void
  setNewEvents: (events: string[]) => void
  setExpandedId: (id: string | null) => void
  setShowRetries: (show: boolean) => void
  setPendingDelete: (id: string | null) => void
  fetchWebhooks: () => Promise<void>
  fetchRetryData: () => Promise<void>
  addWebhook: (addToast: (msg: string, type?: 'success' | 'error' | 'info') => void) => Promise<void>
  deleteWebhook: (id: string, addToast: (msg: string, type?: 'success' | 'error' | 'info') => void) => Promise<void>
  testWebhook: (url: string, addToast: (msg: string, type?: 'success' | 'error' | 'info') => void) => Promise<void>
  loadDeliveries: (webhookId: string) => Promise<void>
}

export function useWebhooks(): UseWebhooksReturn {
  const [webhooks, setWebhooks] = useState<Webhook[]>([])
  const [loading, setLoading] = useState(true)
  const [newUrl, setNewUrl] = useState('')
  const [newEvents, setNewEvents] = useState<string[]>(['training.completed'])
  const [adding, setAdding] = useState(false)
  const [testingUrl, setTestingUrl] = useState<string | null>(null)
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [deliveries, setDeliveries] = useState<DeliveryEntry[]>([])
  const [deliveriesLoading, setDeliveriesLoading] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)
  const [retryQueue, setRetryQueue] = useState<RetryEntry[]>([])
  const [deadLetters, setDeadLetters] = useState<DeadLetter[]>([])
  const [showRetries, setShowRetries] = useState(false)
  const [stats, setStats] = useState<WebhookStats | null>(null)

  const fetchWebhooks = useCallback(async () => {
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
      // ignore
    }
  }, [])

  useEffect(() => {
    let active = true
    const load = async () => {
      try {
        const result = await trainingJobsController.listWebhooks()
        if (active) setWebhooks(result ?? [])
      } catch {
        if (active) setWebhooks([])
      } finally {
        if (active) setLoading(false)
      }
    }
    load()
    return () => { active = false }
  }, [])

  useEffect(() => {
    let active = true
    const load = async () => {
      try {
        const [retries, deads, statsResult] = await Promise.all([
          trainingJobsController.getWebhookRetryQueue(),
          trainingJobsController.getWebhookDeadLetters(),
          trainingJobsController.webhookStats(),
        ])
        if (active) {
          setRetryQueue(retries.retries ?? [])
          setDeadLetters(deads.dead_letters ?? [])
          setStats(statsResult)
        }
      } catch {
        // ignore
      }
    }
    load()
    return () => { active = false }
  }, [])

  const addWebhook = useCallback(async (addToast: (msg: string, type?: 'success' | 'error' | 'info') => void) => {
    if (!newUrl.trim()) return
    setAdding(true)
    try {
      await trainingJobsController.createWebhook(newUrl.trim(), newEvents)
      setNewUrl('')
      setNewEvents(['training.completed'])
      addToast('Webhook added', 'success')
      await fetchWebhooks()
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Failed to add webhook', 'error')
    } finally {
      setAdding(false)
    }
  }, [newUrl, newEvents, fetchWebhooks])

  const deleteWebhook = useCallback(async (id: string, addToast: (msg: string, type?: 'success' | 'error' | 'info') => void) => {
    try {
      await trainingJobsController.deleteWebhook(id)
      addToast('Webhook deleted', 'success')
      setPendingDelete(null)
      await fetchWebhooks()
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Failed to delete webhook', 'error')
    }
  }, [fetchWebhooks])

  const testWebhook = useCallback(async (url: string, addToast: (msg: string, type?: 'success' | 'error' | 'info') => void) => {
    setTestingUrl(url)
    try {
      await trainingJobsController.testWebhook(url)
      addToast('Test sent', 'success')
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Test failed', 'error')
    } finally {
      setTestingUrl(null)
    }
  }, [])

  const loadDeliveries = useCallback(async (webhookId: string) => {
    setDeliveriesLoading(true)
    try {
      const result = await trainingJobsController.getWebhookDeliveries(webhookId)
      setDeliveries(result ?? [])
    } catch {
      setDeliveries([])
    } finally {
      setDeliveriesLoading(false)
    }
  }, [])

  return {
    webhooks, loading, newUrl, newEvents, adding, testingUrl,
    expandedId, deliveries, deliveriesLoading, pendingDelete,
    retryQueue, deadLetters, showRetries, stats,
    setNewUrl, setNewEvents, setExpandedId, setShowRetries, setPendingDelete,
    fetchWebhooks, fetchRetryData, addWebhook, deleteWebhook, testWebhook, loadDeliveries,
  }
}
