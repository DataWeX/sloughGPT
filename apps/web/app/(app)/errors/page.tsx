'use client'

export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback, useRef } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Button } from '@sloughgpt/strui'
import { Chip } from '@sloughgpt/strui'
import { IconRefresh, IconDownload } from '@sloughgpt/strui'
import { apiGet } from '@/lib/http-client'

interface LogEntry {
  timestamp: string
  command: string
  category?: string
  pattern: string
  snippet: string
  cwd?: string
}

function timeSince(ts: string): string {
  const diff = Date.now() - new Date(ts).getTime()
  if (diff < 60_000) return 'just now'
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`
  return `${Math.floor(diff / 86_400_000)}d ago`
}

function categoryColor(cat: string): string {
  const map: Record<string, string> = {
    hydration: '#60a5fa',
    'null-access': '#fb923c',
    'type-error': '#c084fc',
    'chunk-load': '#facc15',
    network: '#f87171',
    auth: '#fbbf24',
    'not-found': '#9ca3af',
    'server-error': '#ef4444',
    cors: '#f472b6',
    build: '#818cf8',
    'infinite-loop': '#f87171',
    'hook-warning': '#34d399',
    'memory-leak': '#fb7185',
    router: '#38bdf8',
    ssr: '#a78bfa',
    unknown: '#6b7280',
  }
  return map[cat] || '#6b7280'
}

export default function ErrorsPage() {
  const [entries, setEntries] = useState<LogEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [filterCat, setFilterCat] = useState('')
  const [filterText, setFilterText] = useState('')
  const [expanded, setExpanded] = useState<number | null>(null)
  const logRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const fetchLogs = useCallback(async () => {
    try {
      const data = await apiGet<{ entries: LogEntry[] }>('/errors/log', undefined, { silent: true })
      setEntries(data.entries || [])
    } catch {
      // server may not be running
    }
    setLoading(false)
  }, [])

  useEffect(() => {
    fetchLogs()
  }, [fetchLogs])

  useEffect(() => {
    if (!autoRefresh) return
    const id = setInterval(fetchLogs, 5000)
    return () => clearInterval(id)
  }, [autoRefresh, fetchLogs])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        inputRef.current?.focus()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const categories = Array.from(new Set(entries.map(e => e.category || 'unknown'))).sort()

  const filtered = entries.filter(e => {
    if (filterCat && (e.category || 'unknown') !== filterCat) return false
    if (filterText) {
      const q = filterText.toLowerCase()
      return (
        e.command.toLowerCase().includes(q) ||
        e.pattern.toLowerCase().includes(q) ||
        e.snippet.toLowerCase().includes(q)
      )
    }
    return true
  })

  const exportLog = () => {
    const text = filtered
      .map(e => `[${e.timestamp}] [${e.category || 'unknown'}] ${e.pattern}\n  cmd: ${e.command}\n  ${e.snippet.slice(0, 200)}`)
      .join('\n\n')
    const blob = new Blob([text], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `error-log-${new Date().toISOString().slice(0, 10)}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="sl-page mx-auto max-w-5xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Console" />}
        right={
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-muted-foreground font-mono">
              {entries.length} errors
            </span>
            <Button
              variant={autoRefresh ? 'default' : 'outline'}
              size="sm"
              onClick={() => setAutoRefresh(a => !a)}
            >
              {autoRefresh ? '● Live' : '○ Paused'}
            </Button>
            <Button variant="outline" size="sm" onClick={fetchLogs}>
              <IconRefresh className="w-3.5 h-3.5" />
            </Button>
            <Button variant="outline" size="sm" onClick={exportLog} disabled={filtered.length === 0}>
              <IconDownload className="w-3.5 h-3.5" />
            </Button>
          </div>
        }
      />

      {/* Toolbar */}
      <div className="mb-3 flex items-center gap-2">
        <div className="relative flex-1">
          <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground text-xs font-mono">$</span>
          <input
            ref={inputRef}
            value={filterText}
            onChange={e => setFilterText(e.target.value)}
            placeholder="grep..."
            className="w-full rounded-md border border-border/60 bg-background pl-6 pr-3 py-1.5 text-xs font-mono text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/40"
          />
          {filterText && (
            <button
              onClick={() => setFilterText('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground text-xs"
            >
              ×
            </button>
          )}
        </div>
      </div>

      {/* Category chips */}
      <div className="mb-3 flex flex-wrap gap-1.5">
        <button
          onClick={() => setFilterCat('')}
          className={`rounded px-2 py-0.5 text-[11px] font-mono transition-colors ${
            !filterCat ? 'bg-primary/15 text-primary' : 'bg-muted/50 text-muted-foreground hover:text-foreground'
          }`}
        >
          all ({entries.length})
        </button>
        {categories.map(cat => {
          const count = entries.filter(e => (e.category || 'unknown') === cat).length
          return (
            <button
              key={cat}
              onClick={() => setFilterCat(filterCat === cat ? '' : cat)}
              className={`rounded px-2 py-0.5 text-[11px] font-mono transition-colors ${
                filterCat === cat ? 'bg-primary/15 text-primary' : 'bg-muted/50 text-muted-foreground hover:text-foreground'
              }`}
            >
              <span
                className="inline-block w-1.5 h-1.5 rounded-full mr-1"
                style={{ backgroundColor: categoryColor(cat) }}
              />
              {cat} ({count})
            </button>
          )
        })}
      </div>

      {/* Log output */}
      <div
        ref={logRef}
        className="rounded-lg border border-border/40 bg-[#0d1117] text-[#c9d1d9] font-mono text-[12px] leading-[1.6] overflow-auto"
        style={{ maxHeight: 'calc(100vh - 260px)' }}
      >
        {loading ? (
          <div className="p-4 text-muted-foreground">Loading...</div>
        ) : filtered.length === 0 ? (
          <div className="p-4 text-muted-foreground">
            {entries.length === 0 ? '$ no errors logged' : '$ no matching entries'}
          </div>
        ) : (
          filtered.map((entry, i) => {
            const cat = entry.category || 'unknown'
            const color = categoryColor(cat)
            const isExpanded = expanded === i
            const ts = new Date(entry.timestamp).toLocaleTimeString()
            return (
              <div key={i}>
                <button
                  className="w-full text-left px-3 py-[3px] hover:bg-white/5 transition-colors border-b border-white/5"
                  onClick={() => setExpanded(isExpanded ? null : i)}
                >
                  <span className="text-[#8b949e]">{ts}</span>
                  {' '}
                  <span style={{ color }} className="font-bold">[{cat}]</span>
                  {' '}
                  <span className="text-[#f0883e]">{entry.pattern}</span>
                  {' '}
                  <span className="text-[#8b949e] text-[11px]">({timeSince(entry.timestamp)})</span>
                  {!isExpanded && (
                    <span className="text-[#484f58] ml-2 hidden sm:inline">
                      {entry.command.slice(0, 60)}
                    </span>
                  )}
                </button>
                {isExpanded && (
                  <div className="px-3 py-2 bg-white/[0.02] border-b border-white/5">
                    <div className="text-[#8b949e] text-[11px] mb-1">
                      $ {entry.command}
                    </div>
                    {entry.cwd && (
                      <div className="text-[#484f58] text-[10px] mb-1">
                        cwd: {entry.cwd}
                      </div>
                    )}
                    <pre className="text-[11px] text-[#c9d1d9] whitespace-pre-wrap break-all max-h-32 overflow-y-auto bg-black/30 rounded p-2 mt-1">
                      {entry.snippet}
                    </pre>
                  </div>
                )}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
