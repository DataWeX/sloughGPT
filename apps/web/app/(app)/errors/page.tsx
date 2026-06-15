'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Chip, KpiGrid, StatCard, Skeleton, Toggle } from '@/components/ui'
import { IconAlert, IconRefresh, IconX, IconTrash } from '@/components/ui'
import { errorController, type ErrorEntry } from '@/lib/error-controller'

const PAGE_SIZE = 50

function ErrorDetail({ error, onClose }: { error: ErrorEntry; onClose: () => void }) {
  const dialogRef = useRef<HTMLDivElement>(null)
  const previousFocusRef = useRef<HTMLElement | null>(null)

  useEffect(() => {
    previousFocusRef.current = document.activeElement as HTMLElement
    dialogRef.current?.focus()
    return () => { previousFocusRef.current?.focus() }
  }, [])

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', handleEscape)
    return () => document.removeEventListener('keydown', handleEscape)
  }, [onClose])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" role="dialog" aria-modal="true" aria-labelledby="error-detail-title" ref={dialogRef} tabIndex={-1}>
      <Card className="max-w-2xl w-full max-h-[80vh] flex flex-col">
        <CardHeader className="flex flex-row items-center justify-between shrink-0">
          <CardTitle id="error-detail-title" className="text-base truncate">{error.message.slice(0, 80)}</CardTitle>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground" aria-label="Close error details">
            <IconX className="h-4 w-4" />
          </button>
        </CardHeader>
        <CardContent className="space-y-3 overflow-y-auto">
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div><span className="text-muted-foreground">Source:</span> {error.source}</div>
            <div><span className="text-muted-foreground">Time:</span> {new Date(error.timestamp).toLocaleString()}</div>
            <div><span className="text-muted-foreground">URL:</span> <span className="break-all">{error.url || '-'}</span></div>
            <div><span className="text-muted-foreground">Line:</span> {error.line ?? '-'}:{error.col ?? '-'}</div>
            <div><span className="text-muted-foreground">Client:</span> {error.client_host}</div>
            <div><span className="text-muted-foreground">ID:</span> {error.id}</div>
          </div>
          {error.stack && (
            <div>
              <div className="text-xs text-muted-foreground mb-1">Stack trace</div>
              <pre className="rounded bg-muted p-3 text-xs font-mono whitespace-pre-wrap max-h-48 overflow-y-auto">{error.stack}</pre>
            </div>
          )}
          {error.metadata && Object.keys(error.metadata).length > 0 && (
            <div>
              <div className="text-xs text-muted-foreground mb-1">Metadata</div>
              <pre className="rounded bg-muted p-3 text-xs font-mono whitespace-pre-wrap max-h-32 overflow-y-auto">{JSON.stringify(error.metadata, null, 2)}</pre>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function SourceDot({ source }: { source: string }) {
  const color =
    source === 'hydration' ? 'bg-yellow-500' :
    source === 'window.onerror' ? 'bg-red-500' :
    source === 'unhandledrejection' ? 'bg-orange-500' :
    source === 'error-boundary' ? 'bg-purple-500' :
    'bg-blue-500'
  return <span className={`inline-block w-2 h-2 rounded-full ${color} shrink-0`} />
}

export default function ErrorMonitorPage() {
  const [errors, setErrors] = useState<ErrorEntry[]>([])
  const [unreadCount, setUnreadCount] = useState(0)
  const [totalCount, setTotalCount] = useState(0)
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [selected, setSelected] = useState<ErrorEntry | null>(null)
  const [sourceFilter, setSourceFilter] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const sentinelRef = useRef<HTMLDivElement | null>(null)

  const fetchErrors = useCallback(async () => {
    try {
      setLoading(true)
      const res = await errorController.getRecent(PAGE_SIZE, 0)
      setErrors(res.errors)
      setUnreadCount(res.unread_count)
      setTotalCount(res.total)
    } catch {
      // silent
    } finally {
      setLoading(false)
    }
  }, [])

  const loadMore = useCallback(async () => {
    if (loadingMore || errors.length >= totalCount) return
    setLoadingMore(true)
    try {
      const res = await errorController.getRecent(PAGE_SIZE, errors.length)
      setErrors(prev => [...prev, ...res.errors])
    } catch {
      // silent
    } finally {
      setLoadingMore(false)
    }
  }, [loadingMore, errors.length, totalCount])

  useEffect(() => { fetchErrors() }, [fetchErrors])

  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(fetchErrors, 5000)
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
  }, [autoRefresh, fetchErrors])

  const handleClear = useCallback(async () => {
    await errorController.clear()
    setErrors([])
    setUnreadCount(0)
    setTotalCount(0)
  }, [])

  const sources = [...new Set(errors.map(e => e.source))]
  const filtered = sourceFilter ? errors.filter(e => e.source === sourceFilter) : errors

  const now = Date.now()
  const err1h = errors.filter(e => now - new Date(e.timestamp).getTime() < 3600000).length

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Error Monitor" subtitle="Client-side errors reported to the server" />}
        right={
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-muted-foreground hidden sm:inline">
              {totalCount} total
            </span>
            <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer select-none">
              <Toggle checked={autoRefresh} onChange={setAutoRefresh} />
              Auto
            </label>
            <Button variant="outline" size="sm" onClick={handleClear} disabled={errors.length === 0} aria-label="Clear all errors">
              <IconTrash className="h-3.5 w-3.5" />
            </Button>
            <Button variant="outline" size="sm" onClick={fetchErrors} disabled={loading} aria-label="Refresh errors">
              <IconRefresh className="h-3.5 w-3.5" />
            </Button>
          </div>
        }
      />

      {selected && <ErrorDetail error={selected} onClose={() => setSelected(null)} />}

      <div className="space-y-4">
        <KpiGrid columns={4}>
          <StatCard label="Total" value={loading ? '...' : totalCount} icon={<IconAlert />} />
          <StatCard label="Last hour" value={loading ? '...' : err1h} />
          <StatCard label="Sources" value={loading ? '...' : sources.length} />
          <StatCard label="Unread" value={loading ? '...' : unreadCount} />
        </KpiGrid>

        {sources.length > 0 && (
          <div className="flex flex-wrap gap-2">
            <Chip label="All" selected={sourceFilter === null} onClick={() => setSourceFilter(null)} />
            {sources.map(s => (
              <Chip key={s} label={s} selected={sourceFilter === s} onClick={() => setSourceFilter(sourceFilter === s ? null : s)} />
            ))}
          </div>
        )}

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Recent Errors</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            {loading ? (
              Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))
            ) : filtered.length === 0 ? (
              <div className="py-8 text-center text-xs text-muted-foreground">
                {errors.length === 0
                  ? 'No client errors reported yet.'
                  : 'No errors match the selected source filter.'}
              </div>
            ) : (
              <>
                {filtered.map((err) => (
                  <button
                    key={err.id}
                    onClick={() => setSelected(err)}
                    className="w-full flex items-start gap-3 rounded-md p-2.5 text-left hover:bg-muted/50 transition-colors group"
                    aria-label={`View error: ${err.message.slice(0, 60)}`}
                  >
                    <SourceDot source={err.source} />
                    <div className="flex-1 min-w-0">
                      <div className="text-sm truncate">{err.message}</div>
                      <div className="flex items-center gap-2 mt-0.5 text-xs text-muted-foreground">
                        <span>{err.source}</span>
                        <span>·</span>
                        <span>{new Date(err.timestamp).toLocaleTimeString()}</span>
                        {err.url && (
                          <>
                            <span>·</span>
                            <span className="truncate max-w-[200px]">{err.url}</span>
                          </>
                        )}
                      </div>
                    </div>
                    <span className="text-xs text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity">
                      {err.line}:{err.col}
                    </span>
                  </button>
                ))}
                {loadingMore && (
                  <div className="py-3 flex items-center justify-center gap-2 text-xs text-muted-foreground">
                    <div className="h-3 w-3 animate-spin rounded-full border border-muted-foreground border-t-transparent" />
                    Loading more...
                  </div>
                )}
                <div ref={sentinelRef} className="h-1" />
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {/* IntersectionObserver for infinite scroll */}
      {!loading && !loadingMore && filtered.length > 0 && filtered.length < totalCount && (
        <InfiniteScrollTrigger sentinelRef={sentinelRef} onTrigger={loadMore} />
      )}
    </div>
  )
}

function InfiniteScrollTrigger({ sentinelRef, onTrigger }: { sentinelRef: React.RefObject<HTMLDivElement | null>; onTrigger: () => void }) {
  useEffect(() => {
    const el = sentinelRef.current
    if (!el) return
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) onTrigger()
      },
      { rootMargin: '200px' },
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [sentinelRef, onTrigger])
  return null
}
