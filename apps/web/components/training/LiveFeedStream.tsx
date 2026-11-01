'use client'

import { memo, useCallback, useEffect, useRef, useState } from 'react'
import { Badge, StatusDot } from '@sloughgpt/strui'
import { createSSEStream, type SSEEnvelope, type SSEStream } from '@/lib/sse-client'

interface FeedSourceView {
  name: string
  spec: string
  total: number
  delta: number
  last_updated: string | null
}

interface ActiveJob {
  id: string
  name: string
  status: string
  dataset: string
  progress: number
}

interface LiveSnapshot {
  sources: FeedSourceView[]
  totals: Record<string, number>
  jobs: { active: ActiveJob[]; count: number }
  at: string
  changed: boolean
}

interface LogEntry {
  key: number
  kind: 'arrival' | 'job'
  text: string
  at: string
}

type ConnectionState = 'connecting' | 'live' | 'reconnecting' | 'idle'

const MAX_LOG_ENTRIES = 50

function relTime(iso: string): string {
  const then = Date.parse(iso)
  if (Number.isNaN(then)) return ''
  const secs = Math.max(0, Math.round((Date.now() - then) / 1000))
  if (secs < 5) return 'now'
  if (secs < 60) return `${secs}s ago`
  const mins = Math.round(secs / 60)
  if (mins < 60) return `${mins}m ago`
  return `${Math.round(mins / 60)}h ago`
}

function isSnapshot(data: unknown): data is LiveSnapshot {
  const d = data as LiveSnapshot | null
  return !!d && Array.isArray(d.sources) && !!d.jobs && typeof d.totals === 'object'
}

function deriveEntries(prevJobs: ActiveJob[], snap: LiveSnapshot): LogEntry[] {
  const entries: LogEntry[] = []
  let key = Date.now() * 1000

  for (const src of snap.sources) {
    if (src.delta > 0) {
      entries.push({
        key: key++,
        kind: 'arrival',
        text: `${src.spec} received ${src.delta} new record${src.delta === 1 ? '' : 's'}`,
        at: snap.at,
      })
    }
  }

  const prevById = new Map(prevJobs.map((j) => [j.id, j]))
  for (const job of snap.jobs.active) {
    const prev = prevById.get(job.id)
    if (!prev) {
      entries.push({
        key: key++,
        kind: 'job',
        text: `${job.name || 'Training job'} started on ${job.dataset || 'a dataset'}`,
        at: snap.at,
      })
    } else if (prev.status !== job.status) {
      entries.push({
        key: key++,
        kind: 'job',
        text: `${job.name || 'Training job'} → ${job.status}`,
        at: snap.at,
      })
    }
  }

  return entries
}

export const LiveFeedStream = memo(function LiveFeedStream() {
  const [snapshot, setSnapshot] = useState<LiveSnapshot | null>(null)
  const [log, setLog] = useState<LogEntry[]>([])
  const [conn, setConn] = useState<ConnectionState>('connecting')
  const prevJobsRef = useRef<ActiveJob[] | null>(null)
  const logKeyRef = useRef(0)

  const handleEnvelope = useCallback((env: SSEEnvelope) => {
    if (env.stream !== 'training-live') return
    if (!isSnapshot(env.data)) return
    const snap = env.data

    setSnapshot(snap)
    if (snap.changed) {
      const entries = deriveEntries(prevJobsRef.current ?? [], snap)
      if (entries.length > 0) {
        const keyed = entries.map((e) => ({ ...e, key: ++logKeyRef.current }))
        setLog((prev) => [...keyed.reverse(), ...prev].slice(0, MAX_LOG_ENTRIES))
      }
    }
    prevJobsRef.current = snap.jobs.active
  }, [])

  useEffect(() => {
    const stream: SSEStream = createSSEStream({
      url: '/training/live-stream',
      onEvent: handleEnvelope,
      onOpen: () => setConn('live'),
      onClose: () => setConn((c) => (c === 'live' ? 'reconnecting' : 'idle')),
      onError: () => setConn('reconnecting'),
    })
    stream.start()
    return () => stream.stop()
  }, [handleEnvelope])

  const connTone =
    conn === 'live'
      ? 'primary'
      : conn === 'reconnecting'
        ? 'warning'
        : conn === 'connecting'
          ? 'muted'
          : 'muted'
  const connLabel =
    conn === 'live'
      ? 'Live — following feeds'
      : conn === 'reconnecting'
        ? 'Reconnecting…'
        : conn === 'connecting'
          ? 'Connecting…'
          : 'Disconnected'

  return (
    <section
      aria-label="Live feed stream"
      className="rounded-lg border border-border/40 bg-card/50 p-4"
    >
      <header className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <StatusDot tone={connTone} pulse={conn === 'live'} />
          <span className="text-sm font-medium">{connLabel}</span>
        </div>
        {snapshot ? (
          <span className="text-xs text-muted-foreground">
            {snapshot.jobs.count > 0
              ? `${snapshot.jobs.count} job${snapshot.jobs.count === 1 ? '' : 's'} running`
              : 'no jobs running'}
          </span>
        ) : null}
      </header>

      {/* Feed totals — the trainable specs and what has landed so far */}
      <ul className="mb-3 flex flex-wrap gap-2" aria-label="Feed totals">
        {(snapshot?.sources ?? []).map((src) => (
          <li
            key={src.name}
            className="flex items-center gap-1.5 rounded-md border border-border/40 px-2 py-1 text-xs"
          >
            <span className="font-mono text-muted-foreground">{src.spec}</span>
            <span className="tabular-nums">{src.total}</span>
            {src.delta > 0 ? <span className="text-success">+{src.delta}</span> : null}
          </li>
        ))}
        {!snapshot ? (
          <li className="px-1 py-1 text-xs text-muted-foreground">Loading feed totals…</li>
        ) : null}
      </ul>

      {/* Active jobs — status at a glance */}
      {(snapshot?.jobs.active ?? []).length > 0 ? (
        <ul className="mb-3 flex flex-wrap gap-2" aria-label="Active jobs">
          {snapshot?.jobs.active.map((job) => (
            <li key={job.id}>
              <Badge variant="outline" className="gap-1.5 text-xs font-normal">
                <span className="max-w-40 truncate">{job.name || job.id}</span>
                <span className="text-muted-foreground">
                  {job.status}
                  {typeof job.progress === 'number' ? ` ${Math.round(job.progress)}%` : ''}
                </span>
              </Badge>
            </li>
          ))}
        </ul>
      ) : null}

      {/* Event log — arrivals and job transitions, newest first */}
      <ol
        role="log"
        aria-live="polite"
        aria-label="Feed activity"
        className="max-h-56 space-y-1 overflow-y-auto"
      >
        {log.length === 0 ? (
          <li className="py-2 text-xs text-muted-foreground">
            No activity yet — new records from the feed sources above appear here as they land,
            alongside training job changes.
          </li>
        ) : (
          log.map((entry) => (
            <li key={entry.key} className="flex items-baseline gap-2 text-xs">
              <span
                aria-hidden="true"
                className={
                  entry.kind === 'arrival'
                    ? 'size-1.5 shrink-0 translate-y-[-1px] rounded-full bg-[rgb(var(--primary))]'
                    : 'size-1.5 shrink-0 translate-y-[-1px] rounded-full bg-[rgb(var(--accent))]'
                }
              />
              <time className="w-14 shrink-0 tabular-nums text-muted-foreground" dateTime={entry.at}>
                {relTime(entry.at)}
              </time>
              <span className="min-w-0 break-words">{entry.text}</span>
            </li>
          ))
        )}
      </ol>
    </section>
  )
})
