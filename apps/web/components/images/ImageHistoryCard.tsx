'use client'

import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, cn } from '@sloughgpt/strui'
import { timeAgo } from '@/lib/time-ago'

interface HistoryEntry {
  id: string
  prompt: string
  style: string
  timestamp: number
  thumbnail?: string
}

const STORAGE_KEY = 'sloughgpt-image-history'

function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch { return [] }
}

function saveHistory(history: HistoryEntry[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(history))
}

interface ImageHistoryCardProps {
  onReUse?: (prompt: string, style: string) => void
}

export function ImageHistoryCard({ onReUse }: ImageHistoryCardProps) {
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory())
  const [filter, setFilter] = useState('')
  const [expanded, setExpanded] = useState<string | null>(null)

  const filtered = history.filter(h =>
    !filter || h.prompt.toLowerCase().includes(filter.toLowerCase()) || h.style.includes(filter)
  )

  const styles = [...new Set(history.map(h => h.style))]

  const handleDelete = (id: string) => {
    const updated = history.filter(h => h.id !== id)
    setHistory(updated)
    saveHistory(updated)
  }

  const handleClear = () => {
    setHistory([])
    saveHistory([])
  }

  if (history.length === 0) return null

  return (
    <Card data-testid="image-history">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">
            Generation History
            <span className="text-muted-foreground font-normal ml-2">({history.length})</span>
          </CardTitle>
          <div className="flex gap-1">
            <input
              type="text"
              placeholder="Filter..."
              value={filter}
              onChange={e => setFilter(e.target.value)}
              className="text-[10px] border border-border rounded px-2 py-0.5 bg-background w-24"
            />
            <Button size="sm" variant="ghost" className="text-destructive text-[10px]" onClick={handleClear}>
              Clear
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex gap-1 mb-3 flex-wrap">
          <Button
            size="sm"
            variant={filter === '' ? 'default' : 'ghost'}
            className="text-[9px]"
            onClick={() => setFilter('')}
          >
            All ({history.length})
          </Button>
          {styles.map(s => {
            const count = history.filter(h => h.style === s).length
            return (
              <Button
                key={s}
                size="sm"
                variant={filter === s ? 'default' : 'ghost'}
                className="text-[9px]"
                onClick={() => setFilter(s)}
              >
                {s} ({count})
              </Button>
            )
          })}
        </div>
        <div className="space-y-1.5 max-h-80 overflow-y-auto">
          {filtered.map(h => (
            <div
              key={h.id}
              className={cn(
                'group p-2 rounded border border-border hover:border-primary/50 transition-colors',
                expanded === h.id && 'border-primary/50'
              )}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-medium truncate">{h.prompt}</div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className="text-[9px] text-muted-foreground">{h.style}</span>
                    <span className="text-[9px] text-muted-foreground">{timeAgo(h.timestamp)}</span>
                  </div>
                </div>
                <div className="flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                  {onReUse && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-5 text-[9px]"
                      onClick={() => onReUse(h.prompt, h.style)}
                    >
                      Re-use
                    </Button>
                  )}
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-5 text-[9px] text-destructive"
                    onClick={() => handleDelete(h.id)}
                  >
                    Del
                  </Button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export function recordImageGeneration(prompt: string, style: string, thumbnail?: string) {
  const history = loadHistory()
  const entry: HistoryEntry = {
    id: `img-hist-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    prompt,
    style,
    timestamp: Date.now(),
    thumbnail,
  }
  const updated = [entry, ...history].slice(0, 200)
  saveHistory(updated)
  return entry.id
}
