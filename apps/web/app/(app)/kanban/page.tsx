'use client'

import { useState, useEffect, useCallback, useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge, Button, cn } from '@sloughgpt/strui'
import { IconRefresh, IconPlus } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'

interface KanbanCard {
  id: string
  title: string
  description: string
  column: string
  priority: 'low' | 'medium' | 'high' | 'critical'
  tags: string[]
  created_at: string
  updated_at: string
  due_date: string
  assignee: string
  notes: string[]
}

interface KanbanColumn {
  name: string
  wip_limit: number
  order: number
}

interface KanbanBoard {
  name: string
  columns: KanbanColumn[]
  cards: KanbanCard[]
}

const COLUMN_LABELS: Record<string, string> = {
  todo: 'To Do',
  in_progress: 'In Progress',
  review: 'Review',
  done: 'Done',
}

const COLUMN_COLORS: Record<string, string> = {
  todo: 'bg-secondary text-secondary-foreground',
  in_progress: 'bg-primary/15 text-primary',
  review: 'bg-accent/20 text-accent-foreground',
  done: 'bg-success/15 text-success',
}

const PRIORITY_COLORS: Record<string, string> = {
  low: 'bg-secondary text-secondary-foreground',
  medium: 'bg-warning/15 text-warning',
  high: 'bg-accent/20 text-accent',
  critical: 'bg-destructive/15 text-destructive',
}

export default function KanbanPage() {
  const [board, setBoard] = useState<KanbanBoard | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filterTag, setFilterTag] = useState<string | null>(null)

  const fetchBoard = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/kanban')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setBoard(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load board')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchBoard()
  }, [fetchBoard])

  const allTags = useMemo(() => {
    if (!board) return []
    const tags = new Set<string>()
    for (const card of board.cards) {
      for (const t of card.tags) tags.add(t)
    }
    return Array.from(tags).sort()
  }, [board])

  const filteredCards = useMemo(() => {
    if (!board) return []
    if (!filterTag) return board.cards
    return board.cards.filter(c => c.tags.includes(filterTag))
  }, [board, filterTag])

  const cardsByColumn = useMemo(() => {
    const map: Record<string, KanbanCard[]> = {}
    for (const card of filteredCards) {
      if (!map[card.column]) map[card.column] = []
      map[card.column].push(card)
    }
    return map
  }, [filteredCards])

  const sortedColumns = useMemo(() => {
    if (!board) return []
    return [...board.columns].sort((a, b) => a.order - b.order)
  }, [board])

  const stats = useMemo(() => {
    if (!board) return { total: 0, byCol: {} }
    const byCol: Record<string, number> = {}
    for (const card of board.cards) {
      byCol[card.column] = (byCol[card.column] || 0) + 1
    }
    return { total: board.cards.length, byCol }
  }, [board])

  return (
    <PageContainer title="Kanban Board">
      <div className="space-y-4">
        {/* Header bar */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="sm" onClick={fetchBoard} disabled={loading}>
              <IconRefresh className={cn('h-4 w-4', loading && 'animate-spin')} />
              Refresh
            </Button>
            <span className="text-sm text-muted-foreground">
              {stats.total} cards
            </span>
          </div>
          {allTags.length > 0 && (
            <div className="flex items-center gap-1 flex-wrap">
              <Button
                variant={filterTag === null ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setFilterTag(null)}
              >
                All
              </Button>
              {allTags.map(tag => (
                <Button
                  key={tag}
                  variant={filterTag === tag ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setFilterTag(filterTag === tag ? null : tag)}
                >
                  {tag}
                </Button>
              ))}
            </div>
          )}
        </div>

        {/* Error state */}
        {error && (
          <Card>
            <CardContent className="p-4 text-destructive text-sm">
              {error}
            </CardContent>
          </Card>
        )}

        {/* Loading state */}
        {loading && !board && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {[1, 2, 3, 4].map(i => (
              <Card key={i} className="animate-pulse">
                <CardContent className="p-4 space-y-3">
                  <div className="h-5 bg-muted rounded w-24" />
                  <div className="h-4 bg-muted rounded w-full" />
                  <div className="h-4 bg-muted rounded w-3/4" />
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {/* Board */}
        {board && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {sortedColumns.map(col => {
              const cards = cardsByColumn[col.name] || []
              const atWip = col.wip_limit > 0 && cards.length >= col.wip_limit
              return (
                <div key={col.name} className="space-y-3">
                  {/* Column header */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <h3 className="text-sm font-medium text-foreground">
                        {COLUMN_LABELS[col.name] || col.name}
                      </h3>
                      <span className={cn(
                        'inline-flex items-center justify-center h-5 min-w-5 px-1.5 rounded-full text-xs font-medium',
                        COLUMN_COLORS[col.name] || 'bg-muted text-muted-foreground'
                      )}>
                        {cards.length}
                      </span>
                      {col.wip_limit > 0 && (
                        <span className="text-xs text-muted-foreground">
                          / {col.wip_limit}
                        </span>
                      )}
                    </div>
                    {atWip && (
                      <span className="text-xs text-warning font-medium">WIP limit</span>
                    )}
                  </div>

                  {/* Cards */}
                  <div className="space-y-2 min-h-[4rem]">
                    {cards.length === 0 && (
                      <div className="text-xs text-muted-foreground italic p-3 text-center border border-dashed border-border rounded-lg">
                        No cards
                      </div>
                    )}
                    {cards.map(card => (
                      <Card
                        key={card.id}
                        className="transition-shadow hover:shadow-md"
                      >
                        <CardContent className="p-3 space-y-2">
                          <p className="text-sm font-medium text-foreground leading-snug">
                            {card.title}
                          </p>
                          {card.description && (
                            <p className="text-xs text-muted-foreground line-clamp-2">
                              {card.description}
                            </p>
                          )}
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className={cn(
                              'inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium uppercase tracking-wider',
                              PRIORITY_COLORS[card.priority] || 'bg-muted text-muted-foreground'
                            )}>
                              {card.priority}
                            </span>
                            {card.tags.slice(0, 3).map(tag => (
                              <span
                                key={tag}
                                className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] bg-muted text-muted-foreground"
                              >
                                {tag}
                              </span>
                            ))}
                            {card.tags.length > 3 && (
                              <span className="text-[10px] text-muted-foreground">
                                +{card.tags.length - 3}
                              </span>
                            )}
                          </div>
                          {card.due_date && (
                            <p className="text-[10px] text-muted-foreground">
                              Due: {new Date(card.due_date).toLocaleDateString()}
                            </p>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </PageContainer>
  )
}
