'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Progress, Button } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { modelController } from '@/lib/model-controller'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'

interface Download {
  model_id: string
  status: string
  progress: number
  bytes_downloaded: number
  total_bytes: number
  speed_bps: number
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`
}

function formatSpeed(bps: number): string {
  if (bps === 0) return '--'
  return `${formatBytes(bps)}/s`
}

export default function DownloadsCard() {
  const [downloads, setDownloads] = useState<Download[]>([])
  const [loading, setLoading] = useState(true)
  const addToast = useToastStore(s => s.addToast)

  const fetchDownloads = useCallback(async () => {
    try {
      const result = await modelController.listDownloads()
      setDownloads(result.downloads)
    } catch (err) {
      addToast(extractErrorMessage(err, 'Could not load downloads'), 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => {
    let active = true
    const load = async () => {
      setLoading(true)
      try {
        const result = await modelController.listDownloads()
        if (active) setDownloads(result.downloads)
      } catch {
        if (active) {
          addToast('Could not load downloads', 'error')
          setDownloads([])
        }
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => { active = false }
  }, [addToast])

  useEffect(() => {
    const hasActive = downloads.some(d => d.status === 'downloading' || d.status === 'queued')
    if (!hasActive) return
    const timer = setInterval(fetchDownloads, 2000)
    return () => clearInterval(timer)
  }, [downloads, fetchDownloads])

  const handleCancel = async (modelId: string) => {
    try {
      await modelController.cancelDownload(modelId)
      addToast(`Cancelled download: ${modelId}`, 'success')
      await fetchDownloads()
    } catch (err) {
      addToast(extractErrorMessage(err, 'Could not cancel'), 'error')
    }
  }

  const handleRetry = async (modelId: string) => {
    try {
      await modelController.retryDownload(modelId)
      addToast(`Retrying download: ${modelId}`, 'success')
      await fetchDownloads()
    } catch (err) {
      addToast(extractErrorMessage(err, 'Could not retry'), 'error')
    }
  }

  const handleVerify = async (modelId: string) => {
    try {
      const result = await modelController.verifyDownload(modelId)
      if (result.verified) {
        addToast(`${modelId} verified`, 'success')
      } else {
        addToast(`${modelId} verification failed: ${result.error || 'unknown'}`, 'error')
      }
    } catch (err) {
      addToast(extractErrorMessage(err, 'Could not verify'), 'error')
    }
  }

  if (loading) return null

  if (downloads.length === 0) return null

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Downloads</CardTitle>
        <Button size="sm" variant="ghost" onClick={fetchDownloads} aria-label="Refresh downloads">
          <IconRefresh className="h-3.5 w-3.5" />
        </Button>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {downloads.map(d => {
            const pct = Math.round(d.progress * 100)
            const isActive = d.status === 'downloading' || d.status === 'queued'
            const isFailed = d.status === 'failed' || d.status === 'error'
            return (
              <div key={d.model_id} className="rounded-md border border-border/60 px-3 py-2 text-sm">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="font-medium truncate text-xs">{d.model_id}</span>
                    <span className={`text-[10px] px-1 rounded ${
                      isActive ? 'bg-primary/10 text-primary' :
                      isFailed ? 'bg-destructive/10 text-destructive' :
                      d.status === 'completed' ? 'bg-success/10 text-success' :
                      'bg-muted text-muted-foreground'
                    }`}>{d.status}</span>
                  </div>
                  <div className="flex items-center gap-1 text-[10px] text-muted-foreground shrink-0">
                    <span>{formatBytes(d.bytes_downloaded)} / {formatBytes(d.total_bytes)}</span>
                    <span className="ml-1">{formatSpeed(d.speed_bps)}</span>
                  </div>
                </div>
                {isActive && (
                  <Progress value={pct} size="sm" variant={pct > 0 ? 'default' : 'warning'} />
                )}
                {isFailed && (
                  <div className="flex gap-1 mt-1">
                    <Button size="sm" variant="ghost" onClick={() => handleRetry(d.model_id)}>Retry</Button>
                    <Button size="sm" variant="ghost" onClick={() => handleVerify(d.model_id)}>Verify</Button>
                  </div>
                )}
                {isActive && (
                  <Button size="sm" variant="ghost" className="mt-1" onClick={() => handleCancel(d.model_id)}>Cancel</Button>
                )}
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
