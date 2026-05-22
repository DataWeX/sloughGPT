'use client'

import { useEffect, useState, useCallback } from 'react'
import { getDownloadStatus, cancelDownload, type DownloadProgress } from '@/lib/download-controller'
import { Button, IconX, IconRefresh } from '@/components/ui'

interface DownloadManagerProps {
  modelId: string
  sizeGb?: number
  onProgress?: (progress: DownloadProgress) => void
  onComplete?: () => void
  onError?: (error: string) => void
}

export function DownloadManager({ modelId, sizeGb, onProgress, onComplete, onError }: DownloadManagerProps) {
  const [progress, setProgress] = useState<DownloadProgress | null>(null)
  const [polling, setPolling] = useState(false)

  const fetchStatus = useCallback(async () => {
    try {
      const status = await getDownloadStatus(modelId)
      setProgress(status)
      onProgress?.(status)

      if (status.status === 'complete' || status.status === 'already_cached') {
        setPolling(false)
        onComplete?.()
      } else if (status.status === 'failed') {
        setPolling(false)
        onError?.(status.error || 'Download failed')
      } else if (status.status === 'cancelled') {
        setPolling(false)
      }
    } catch {
      // Ignore polling errors
    }
  }, [modelId, onProgress, onComplete, onError])

  const startDownload = useCallback(async () => {
    setPolling(true)
    await fetchStatus()
  }, [fetchStatus])

  const handleCancel = useCallback(async () => {
    await cancelDownload(modelId)
    setPolling(false)
    setProgress(prev => prev ? { ...prev, status: 'cancelled' } : null)
  }, [modelId])

  useEffect(() => {
    if (!polling) return
    fetchStatus()
    const interval = setInterval(fetchStatus, 1000)
    return () => clearInterval(interval)
  }, [polling, fetchStatus])

  if (!progress) {
    return (
      <div className="flex items-center gap-2 p-3 rounded-lg border border-border/50 bg-muted/30">
        <Button size="sm" variant="ghost" onClick={startDownload} className="text-xs gap-1.5">
          <IconRefresh className="h-3 w-3" />
          Download model
        </Button>
        {sizeGb && <span className="text-xs text-muted-foreground">{sizeGb.toFixed(2)} GB</span>}
      </div>
    )
  }

  const isActive = progress.status === 'downloading' || progress.status === 'queued' || progress.status === 'started'
  const isComplete = progress.status === 'complete' || progress.status === 'already_cached'
  const isFailed = progress.status === 'failed'
  const isCancelled = progress.status === 'cancelled'

  return (
    <div className="p-3 rounded-lg border border-border/50 bg-muted/30 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isActive && <IconRefresh className="h-3 w-3 animate-spin text-primary" />}
          {isComplete && <span className="h-2 w-2 rounded-full bg-success" />}
          {isFailed && <span className="h-2 w-2 rounded-full bg-error" />}
          {isCancelled && <span className="h-2 w-2 rounded-full bg-muted-foreground" />}
          <span className="text-xs font-medium">
            {isActive ? 'Downloading' : isComplete ? 'Ready' : isFailed ? 'Failed' : 'Cancelled'}
          </span>
          {progress.current_file && (
            <span className="text-[10px] text-muted-foreground truncate max-w-[120px]">
              {progress.current_file}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-muted-foreground">
            {progress.percentage.toFixed(1)}%
          </span>
          {isActive && (
            <Button size="sm" variant="ghost" onClick={handleCancel} className="h-5 px-1.5 text-[10px]">
              <IconX className="h-2.5 w-2.5" />
            </Button>
          )}
        </div>
      </div>

      {isActive && (
        <div className="space-y-1">
          <div className="h-1.5 rounded-full bg-muted overflow-hidden">
            <div
              className="h-full bg-primary transition-all duration-300 ease-out"
              style={{ width: `${Math.min(progress.percentage, 100)}%` }}
            />
          </div>
          <div className="flex items-center justify-between text-[10px] text-muted-foreground">
            <span>
              {(progress.bytes_downloaded / (1024 * 1024 * 1024)).toFixed(2)} / {(progress.total_bytes / (1024 * 1024 * 1024)).toFixed(2)} GB
            </span>
            <span>
              {progress.speed_mb_per_sec.toFixed(1)} MB/s
              {progress.eta_seconds > 0 && ` · ${Math.ceil(progress.eta_seconds)}s left`}
            </span>
          </div>
        </div>
      )}

      {isFailed && progress.error && (
        <p className="text-[10px] text-error">{progress.error}</p>
      )}

      {isComplete && (
        <p className="text-[10px] text-success">Model downloaded and loaded successfully.</p>
      )}
    </div>
  )
}
