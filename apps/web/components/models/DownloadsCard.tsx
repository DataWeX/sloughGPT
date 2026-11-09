'use client'
import { notify } from '@/lib/notify'

import { useState, useEffect, useCallback } from 'react'
import { cn, ActionCard, Progress, Button } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { modelController } from '@/lib/model-controller'
import { extractErrorMessage } from '@/lib/error-utils'
import { formatBytes } from '@/lib/format-bytes'
// Module scope: `addToast` must keep a stable identity — it sits in
// useCallback/useEffect dependency arrays, and a per-render arrow would
// re-fire those effects on every render.
const addToast = (message: string, type?: string) =>
  notify.toast(message, { type: type as 'success' | 'error' | 'info' })

interface Download {
  model_id: string
  status: string
  progress: number
  bytes_downloaded: number
  total_bytes: number
  speed_bps: number
}

function formatSpeed(bps: number): string {
  if (bps === 0) return '--'
  return `${formatBytes(bps)}/s`
}

export default function DownloadsCard() {
  const [downloads, setDownloads] = useState<Download[]>([])
  const [loading, setLoading] = useState(true)

  const fetchDownloads = useCallback(async () => {
    try {
      const result = await modelController.listDownloads()
      setDownloads(result.downloads ?? [])
    } catch (err) {
      notify.toast(extractErrorMessage(err, 'Could not load downloads'), { type: 'error' })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    let active = true
    const load = async () => {
      setLoading(true)
      try {
        const result = await modelController.listDownloads()
        if (active) setDownloads(result.downloads ?? [])
      } catch {
        if (active) {
          notify.toast('Could not load downloads', { type: 'error' })
          setDownloads([])
        }
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    const hasActive = downloads.some((d) => d.status === 'downloading' || d.status === 'queued')
    if (!hasActive) return
    const timer = setInterval(fetchDownloads, 2000)
    return () => clearInterval(timer)
  }, [downloads, fetchDownloads])

  const handleCancel = async (modelId: string) => {
    try {
      await modelController.cancelDownload(modelId)
      notify.toast(`Cancelled download: ${modelId}`, { type: 'success' })
      await fetchDownloads()
    } catch (err) {
      notify.toast(extractErrorMessage(err, 'Could not cancel'), { type: 'error' })
    }
  }

  const handleRetry = async (modelId: string) => {
    try {
      await modelController.retryDownload(modelId)
      notify.toast(`Retrying download: ${modelId}`, { type: 'success' })
      await fetchDownloads()
    } catch (err) {
      notify.toast(extractErrorMessage(err, 'Could not retry'), { type: 'error' })
    }
  }

  const handleVerify = async (modelId: string) => {
    try {
      const result = await modelController.verifyDownload(modelId)
      if (result.verified) {
        notify.toast(`${modelId} verified`, { type: 'success' })
      } else {
        notify.toast(`${modelId} verification failed: ${result.error || 'unknown'}`, {
          type: 'error',
        })
      }
    } catch (err) {
      notify.toast(extractErrorMessage(err, 'Could not verify'), { type: 'error' })
    }
  }

  if (loading) return null

  if (downloads.length === 0) return null

  return (
    <ActionCard
      title="Downloads"
      actions={
        <Button size="sm" variant="ghost" onClick={fetchDownloads} aria-label="Refresh downloads">
          <IconRefresh className="h-3.5 w-3.5" />
        </Button>
      }
    >
      <div className="space-y-1.5">
        {downloads.map((d) => {
          const pct = Math.round(d.progress * 100)
          const isActive = d.status === 'downloading' || d.status === 'queued'
          const isFailed = d.status === 'failed' || d.status === 'error'
          return (
            <div
              key={d.model_id}
              className="rounded-lg border border-border/40 px-2.5 py-2 text-xs"
            >
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="font-medium truncate text-[11px]">{d.model_id}</span>
                  <span
                    className={cn(
                      'text-[9px] px-1.5 py-0.5 rounded-full font-medium',
                      isActive
                        ? 'bg-primary/10 text-primary'
                        : isFailed
                          ? 'bg-destructive/10 text-destructive'
                          : d.status === 'completed'
                            ? 'bg-success/10 text-success'
                            : 'bg-muted text-muted-foreground',
                    )}
                  >
                    {d.status}
                  </span>
                </div>
                <div className="flex items-center gap-1 text-[9px] text-muted-foreground/60 shrink-0 tabular-nums">
                  <span>
                    {formatBytes(d.bytes_downloaded)} / {formatBytes(d.total_bytes)}
                  </span>
                  <span className="ml-0.5">{formatSpeed(d.speed_bps)}</span>
                </div>
              </div>
              {isActive && (
                <Progress value={pct} size="sm" variant={pct > 0 ? 'default' : 'warning'} />
              )}
              {isFailed && (
                <div className="flex gap-0.5 mt-1">
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 text-[10px]"
                    onClick={() => handleRetry(d.model_id)}
                  >
                    Retry
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 text-[10px]"
                    onClick={() => handleVerify(d.model_id)}
                  >
                    Verify
                  </Button>
                </div>
              )}
              {isActive && (
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-6 text-[10px] mt-1"
                  onClick={() => handleCancel(d.model_id)}
                >
                  Cancel
                </Button>
              )}
            </div>
          )
        })}
      </div>
    </ActionCard>
  )
}
