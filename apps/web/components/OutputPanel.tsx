/**
 * OutputPanel — floating collapsible panel for service output.
 * Toggled via status bar button or Ctrl+` keyboard shortcut.
 * Supports level/source/search filtering, pause, and export.
 */

'use client'

import { useState, useEffect, useMemo } from 'react'
import { Button, Input } from '@sloughgpt/strui'
import { useServerOutput } from '@/hooks/useServerOutput'

interface OutputPanelProps {
  open: boolean
  onClose: () => void
}

const LEVELS = ['info', 'warning', 'error'] as const
type LevelFilter = typeof LEVELS[number]

export function OutputPanel({ open, onClose }: OutputPanelProps) {
  const { lines, streaming, clear, scrollRef, paused, togglePause, exportLines } = useServerOutput({ tail: 100, maxLines: 500 })
  const [filterLevel, setFilterLevel] = useState<Set<LevelFilter>>(new Set())
  const [filterSource, setFilterSource] = useState('')
  const [searchText, setSearchText] = useState('')

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === '`' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        if (open) onClose()
        else window.dispatchEvent(new CustomEvent('toggle-output-panel'))
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, onClose])

  const sources = useMemo(() => {
    const set = new Set(lines.map(l => l.source))
    return Array.from(set).sort()
  }, [lines])

  const filtered = useMemo(() => {
    return lines.filter(line => {
      if (filterLevel.size > 0 && !filterLevel.has(line.level as LevelFilter)) return false
      if (filterSource && line.source !== filterSource) return false
      if (searchText && !line.text.toLowerCase().includes(searchText.toLowerCase())) return false
      return true
    })
  }, [lines, filterLevel, filterSource, searchText])

  const toggleLevel = (level: LevelFilter) => {
    setFilterLevel(prev => {
      const next = new Set(prev)
      if (next.has(level)) next.delete(level)
      else next.add(level)
      return next
    })
  }

  if (!open) return null

  return (
    <div className="fixed bottom-12 right-4 w-[calc(100vw-2rem)] max-w-[520px] max-h-[400px] z-50 bg-background border rounded-lg shadow-lg flex flex-col">
      <div className="flex items-center justify-between px-3 py-2 border-b">
        <div className="flex items-center gap-2">
          <span className={`inline-block w-2 h-2 rounded-full ${streaming && !paused ? 'bg-success animate-pulse' : paused ? 'bg-warning' : 'bg-muted-foreground/50'}`} />
          <span className="text-sm font-medium">Service Output</span>
          <span className="text-xs text-muted-foreground">({filtered.length}/{lines.length})</span>
        </div>
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="sm" onClick={togglePause} className="h-7 px-2 text-xs" aria-label={paused ? 'Resume output' : 'Pause output'}>
            {paused ? '▶' : '⏸'}
          </Button>
          <Button variant="ghost" size="sm" onClick={() => exportLines('text')} className="h-7 px-2 text-xs" aria-label="Export as log file">
            ↓
          </Button>
          <Button variant="ghost" size="sm" onClick={clear} className="h-7 px-2 text-xs">
            Clear
          </Button>
          <Button variant="ghost" size="sm" onClick={onClose} className="h-7 px-2 text-xs" aria-label="Close">
            &times;
          </Button>
        </div>
      </div>

      <div className="flex items-center gap-2 px-3 py-1.5 border-b text-[11px]">
        <div className="flex items-center gap-1">
          {LEVELS.map(level => (
            <button
              key={level}
              onClick={() => toggleLevel(level)}
              className={`px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors ${
                filterLevel.size === 0 || filterLevel.has(level)
                  ? level === 'error' ? 'bg-destructive/15 text-destructive' :
                    level === 'warning' ? 'bg-warning/15 text-warning' :
                    'bg-muted text-muted-foreground'
                  : 'bg-muted/30 text-muted-foreground/50'
              }`}
            >
              {level}
            </button>
          ))}
        </div>
        <select
          value={filterSource}
          onChange={e => setFilterSource(e.target.value)}
          className="h-5 px-1 rounded border bg-background text-[10px] text-muted-foreground"
          aria-label="Filter by source"
        >
          <option value="">all sources</option>
          {sources.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <Input
          value={searchText}
          onChange={e => setSearchText(e.target.value)}
          placeholder="Search..."
          className="h-5 w-24 px-1.5 text-[10px]"
          aria-label="Search output"
        />
      </div>

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto font-mono text-[11px] p-2 space-y-0.5"
        role="log"
        aria-label="Service output panel"
      >
        {filtered.length === 0 ? (
          <div className="text-muted-foreground py-8 text-center text-xs">
            {lines.length === 0
              ? (streaming ? 'Waiting for output...' : 'No output yet')
              : 'No matching lines'}
          </div>
        ) : (
          filtered.map((line, i) => (
            <div key={`${line.ts}-${i}`} className="flex gap-2 leading-tight">
              <span className="text-muted-foreground shrink-0 w-14">
                {new Date(line.ts * 1000).toLocaleTimeString()}
              </span>
              <span className={`shrink-0 w-10 ${
                line.level === 'error' ? 'text-destructive' :
                line.level === 'warning' ? 'text-warning' :
                'text-muted-foreground'
              }`}>
                {line.level}
              </span>
              <span className="text-muted-foreground/60 shrink-0 w-16 truncate" title={line.source}>
                {line.source}
              </span>
              <span className="flex-1 min-w-0 break-all">{line.text}</span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
