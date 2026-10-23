'use client'

/* Kanban page — client component. Fetches board from planner API,
   renders interactive board with engine-validated transitions. */

import { useState, useEffect, useCallback, lazy, Suspense } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Button, IconRefresh } from '@sloughgpt/strui'
import { fetchBoard, transitionCard, type TransitionResult } from '@/lib/planner-client'
import type { Board, Card, Column } from '@/components/planner/types'

const KanbanScene = lazy(() => import('@/components/kanban3d/KanbanScene'))

function hasWebGL(): boolean {
  try {
    const canvas = document.createElement('canvas')
    return Boolean(canvas.getContext('webgl2') || canvas.getContext('webgl'))
  } catch {
    return false
  }
}

const STATUS_COLORS: Record<string, string> = {
  card_moved: 'rgb(var(--success))',
  card_blocked: 'rgb(var(--destructive))',
  card_unblocked: 'rgb(var(--primary))',
  wip_exceeded: 'rgb(var(--destructive))',
  transition_rejected: 'rgb(var(--destructive))',
}

function orderedNames(columns: Column[]): string[] {
  return columns
    .slice()
    .sort((a, b) => a.order - b.order)
    .map((c) => c.name)
}

export default function KanbanPage() {
  const [board, setBoard] = useState<Board | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedCard, setSelectedCard] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [viewMode, setViewMode] = useState<'dom' | 'immersive'>('dom')
  const [feedback, setFeedback] = useState<{
    type: string
    message: string
    target?: string
  } | null>(null)

  const switchMode = useCallback((mode: 'dom' | 'immersive') => {
    if (mode === 'immersive' && !hasWebGL()) {
      setFeedback({
        type: 'transition_rejected',
        message: 'WebGL is not available — staying in DOM view',
      })
      return
    }
    setViewMode(mode)
  }, [])

  const loadBoard = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetchBoard()
      setBoard(res.board)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load board')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadBoard()
  }, [loadBoard])

  const selected = board?.cards.find((c) => c.id === selectedCard) ?? null
  const colNames = board ? orderedNames(board.columns) : []
  let legalTargets: string[] = []
  if (selected && board) {
    const fromIdx = colNames.indexOf(selected.column)
    if (fromIdx === -1) {
      legalTargets = colNames.filter((n) => n !== selected.column)
    } else if (fromIdx === colNames.length - 1) {
      legalTargets = [] // last column (done) is terminal
    } else {
      legalTargets = [...(fromIdx > 0 ? [colNames[fromIdx - 1]] : []), colNames[fromIdx + 1]]
    }
  }

  const handleMove = useCallback(
    async (cardId: string, toState: string, force = false) => {
      setBusy(true)
      setFeedback(null)
      try {
        const result: TransitionResult = await transitionCard({
          card_id: cardId,
          to_state: toState,
          force,
        })
        if (result.ok) {
          setFeedback({
            type: result.event?.type ?? 'card_moved',
            message: `${result.from_state} → ${result.to_state}${result.event?.metadata?.forced ? ' (forced)' : ''}`,
          })
          setSelectedCard(null)
          await loadBoard()
        } else {
          setFeedback({
            type: result.event?.type ?? 'transition_rejected',
            message: result.error,
            target: toState,
          })
        }
      } catch {
        setFeedback({
          type: 'transition_rejected',
          message: 'Failed to reach planner API',
          target: toState,
        })
      } finally {
        setBusy(false)
      }
    },
    [loadBoard],
  )

  return (
    <PageContainer
      title="Kanban"
      subtitle="Planner board with engine-validated transitions"
      loading={loading}
      loadingContent={
        <div className="py-8 text-sm" style={{ color: 'rgb(var(--muted-foreground))' }}>
          Loading board…
        </div>
      }
      error={error}
      onRetry={loadBoard}
      headerRight={
        <div className="flex items-center gap-2">
          <div
            className="flex items-center rounded-md p-0.5"
            style={{ backgroundColor: 'rgb(var(--muted) / 0.5)' }}
            role="group"
            aria-label="Board view mode"
          >
            {(['dom', 'immersive'] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => switchMode(mode)}
                aria-pressed={viewMode === mode}
                className="rounded px-2.5 py-1 text-xs font-medium transition-colors"
                style={
                  viewMode === mode
                    ? {
                        backgroundColor: 'rgb(var(--card))',
                        color: 'rgb(var(--foreground))',
                        boxShadow: '0 1px 2px rgb(0 0 0 / 0.08)',
                      }
                    : { color: 'rgb(var(--muted-foreground))' }
                }
              >
                {mode === 'dom' ? 'List' : '3D'}
              </button>
            ))}
          </div>
          <Button variant="ghost" size="sm" onClick={loadBoard} aria-label="Refresh board">
            <IconRefresh className="h-4 w-4" />
          </Button>
        </div>
      }
      maxWidth="max-w-7xl"
    >
      {feedback && (
        <div
          className="mb-4 flex items-center justify-between gap-3 rounded-lg px-3 py-2 text-xs"
          style={{
            backgroundColor: `rgb(${STATUS_COLORS[feedback.type] ?? 'var(--muted)'} / 0.1)`,
            color: STATUS_COLORS[feedback.type] ?? 'rgb(var(--muted-foreground))',
          }}
        >
          <span>{feedback.message}</span>
          {feedback.type === 'wip_exceeded' && feedback.target && selectedCard && (
            <button
              disabled={busy}
              onClick={() => handleMove(selectedCard, feedback.target!, true)}
              className="shrink-0 rounded px-2 py-1 text-xs font-medium disabled:opacity-50"
              style={{
                backgroundColor: 'rgb(var(--destructive) / 0.15)',
                color: 'rgb(var(--destructive))',
              }}
            >
              Force move
            </button>
          )}
        </div>
      )}

      {selectedCard && (
        <div
          className="mb-3 flex items-center justify-between rounded-lg px-3 py-2"
          style={{ backgroundColor: 'rgb(var(--card))' }}
        >
          <span className="text-xs font-medium" style={{ color: 'rgb(var(--foreground))' }}>
            {selected?.title ?? selectedCard}
          </span>
          <div className="flex items-center gap-2">
            {legalTargets.map((target) => (
              <button
                key={target}
                disabled={busy}
                onClick={() => handleMove(selectedCard, target)}
                className="rounded px-2 py-1 text-xs font-medium transition-colors disabled:opacity-50"
                style={{
                  backgroundColor: 'rgb(var(--primary) / 0.15)',
                  color: 'rgb(var(--primary))',
                }}
              >
                → {target.replace(/_/g, ' ')}
              </button>
            ))}
            <button
              onClick={() => setSelectedCard(null)}
              className="rounded px-2 py-1 text-xs"
              style={{ color: 'rgb(var(--muted-foreground))' }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {board && board.columns.length > 0 ? (
        viewMode === 'immersive' ? (
          <Suspense
            fallback={
              <div
                className="flex h-96 items-center justify-center rounded-lg text-sm"
                style={{ color: 'rgb(var(--muted-foreground))' }}
              >
                Loading 3D scene…
              </div>
            }
          >
            <KanbanScene board={board} selectedCardId={selectedCard} onSelect={setSelectedCard} />
          </Suspense>
        ) : (
          <BoardColumns board={board} selectedCard={selectedCard} onSelect={setSelectedCard} />
        )
      ) : (
        <div
          className="rounded-lg border border-dashed p-8 text-center"
          style={{ borderColor: 'rgb(var(--border))', color: 'rgb(var(--muted-foreground))' }}
        >
          <p className="text-sm">No board found.</p>
          <p className="mt-1 text-xs">Expected .kanban/board.jsonl at the repo root.</p>
        </div>
      )}
    </PageContainer>
  )
}

function BoardColumns({
  board,
  selectedCard,
  onSelect,
}: {
  board: Board
  selectedCard: string | null
  onSelect: (id: string) => void
}) {
  const cardsByColumn = new Map<string, Card[]>()
  for (const col of board.columns) {
    cardsByColumn.set(col.name, [])
  }
  for (const card of board.cards) {
    cardsByColumn.get(card.column)?.push(card)
  }

  return (
    <div className="flex gap-4 overflow-x-auto pb-4" style={{ minHeight: 'calc(100vh - 240px)' }}>
      {board.columns
        .slice()
        .sort((a: Column, b: Column) => a.order - b.order)
        .map((col) => {
          const cards = cardsByColumn.get(col.name) ?? []
          const overWip = col.wip_limit > 0 && cards.length > col.wip_limit
          return (
            <div key={col.name} className="min-w-[280px]">
              <div
                className="mb-3 flex items-center justify-between rounded-lg px-3 py-2"
                style={{ backgroundColor: 'rgb(var(--card))' }}
              >
                <span
                  className="text-sm font-medium capitalize"
                  style={{ color: 'rgb(var(--foreground))' }}
                >
                  {col.label || col.name.replace(/_/g, ' ')}
                </span>
                <span
                  className="rounded-full px-1.5 py-0.5 text-[10px] font-medium"
                  style={{
                    backgroundColor: overWip
                      ? 'rgb(var(--destructive) / 0.15)'
                      : 'rgb(var(--muted) / 0.15)',
                    color: overWip ? 'rgb(var(--destructive))' : 'rgb(var(--muted-foreground))',
                  }}
                >
                  {cards.length}
                  {col.wip_limit > 0 ? ` / ${col.wip_limit}` : ''}
                </span>
              </div>

              <div className="flex flex-col gap-2">
                {cards.length === 0 ? (
                  <div
                    className="rounded-lg border border-dashed p-4 text-center text-xs"
                    style={{
                      borderColor: 'rgb(var(--border))',
                      color: 'rgb(var(--muted-foreground))',
                    }}
                  >
                    No cards
                  </div>
                ) : (
                  cards.map((card) => (
                    <button
                      key={card.id}
                      onClick={() => onSelect(card.id)}
                      className="rounded-lg border p-3 text-left transition-colors"
                      style={{
                        borderColor:
                          selectedCard === card.id ? 'rgb(var(--primary))' : 'rgb(var(--border))',
                        backgroundColor: 'rgb(var(--card))',
                      }}
                    >
                      <span
                        className="text-sm font-medium"
                        style={{ color: 'rgb(var(--foreground))' }}
                      >
                        {card.title}
                      </span>
                      {card.tags.length > 0 && (
                        <div className="mt-1 flex flex-wrap gap-1">
                          {card.tags.map((tag) => (
                            <span
                              key={tag}
                              className="rounded px-1.5 py-0.5 text-[10px]"
                              style={{
                                backgroundColor: 'rgb(var(--primary) / 0.15)',
                                color: 'rgb(var(--primary))',
                              }}
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                      )}
                    </button>
                  ))
                )}
              </div>
            </div>
          )
        })}
    </div>
  )
}
