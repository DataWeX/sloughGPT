'use client'

/* Kanban page — client component. Fetches board from planner API,
   renders interactive board with engine-validated transitions. */

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Button, IconRefresh } from '@sloughgpt/strui'
import { fetchBoard, transitionCard, type TransitionResult } from '@/lib/planner-client'
import type { Board, Card, Column } from '@/components/planner/types'

const STATUS_COLORS: Record<string, string> = {
  card_moved: 'rgb(var(--color-success))',
  card_blocked: 'rgb(var(--color-error))',
  card_unblocked: 'rgb(var(--color-accent))',
  wip_exceeded: 'rgb(var(--color-error))',
  transition_rejected: 'rgb(var(--color-error))',
}

const LEGAL_TARGETS: Record<string, string[]> = {
  inbox: ['ready'],
  ready: ['doing', 'blocked'],
  doing: ['review', 'ready', 'blocked'],
  review: ['verify', 'doing', 'blocked'],
  verify: ['done', 'review', 'blocked'],
  done: [],
  blocked: ['inbox', 'ready', 'doing'],
}

export default function KanbanPage() {
  const [board, setBoard] = useState<Board | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedCard, setSelectedCard] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [feedback, setFeedback] = useState<{ type: string; message: string } | null>(null)

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
  const legalTargets = selected ? (LEGAL_TARGETS[selected.column] ?? []) : []

  const handleMove = useCallback(
    async (cardId: string, toState: string) => {
      setBusy(true)
      setFeedback(null)
      try {
        const result: TransitionResult = await transitionCard({
          card_id: cardId,
          to_state: toState,
        })
        if (result.ok) {
          setFeedback({
            type: result.event?.type ?? 'card_moved',
            message: `${result.from_state} → ${result.to_state}`,
          })
          setSelectedCard(null)
          await loadBoard()
        } else {
          setFeedback({ type: 'transition_rejected', message: result.error })
        }
      } catch {
        setFeedback({ type: 'transition_rejected', message: 'Failed to reach planner backend' })
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
        <div className="py-8 text-sm" style={{ color: 'rgb(var(--color-muted))' }}>
          Loading board…
        </div>
      }
      error={error}
      onRetry={loadBoard}
      headerRight={
        <Button variant="ghost" size="sm" onClick={loadBoard} aria-label="Refresh board">
          <IconRefresh className="h-4 w-4" />
        </Button>
      }
      maxWidth="max-w-7xl"
    >
      {feedback && (
        <div
          className="mb-4 rounded-lg px-3 py-2 text-xs"
          style={{
            backgroundColor: `rgb(${STATUS_COLORS[feedback.type] ?? 'var(--color-muted)'} / 0.1)`,
            color: STATUS_COLORS[feedback.type] ?? 'rgb(var(--color-muted))',
          }}
        >
          {feedback.message}
        </div>
      )}

      {selectedCard && (
        <div
          className="mb-3 flex items-center justify-between rounded-lg px-3 py-2"
          style={{ backgroundColor: 'rgb(var(--color-surface))' }}
        >
          <span className="text-xs font-medium" style={{ color: 'rgb(var(--color-foreground))' }}>
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
                  backgroundColor: 'rgb(var(--color-accent) / 0.15)',
                  color: 'rgb(var(--color-accent))',
                }}
              >
                → {target.replace(/_/g, ' ')}
              </button>
            ))}
            <button
              onClick={() => setSelectedCard(null)}
              className="rounded px-2 py-1 text-xs"
              style={{ color: 'rgb(var(--color-muted))' }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {board && board.columns.length > 0 ? (
        <BoardColumns board={board} selectedCard={selectedCard} onSelect={setSelectedCard} />
      ) : (
        <div
          className="rounded-lg border border-dashed p-8 text-center"
          style={{ borderColor: 'rgb(var(--color-border))', color: 'rgb(var(--color-muted))' }}
        >
          <p className="text-sm">No board found. Start the planner backend:</p>
          <code className="mt-2 block text-xs" style={{ color: 'rgb(var(--color-foreground))' }}>
            planner gui --port 8787
          </code>
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
                style={{ backgroundColor: 'rgb(var(--color-surface))' }}
              >
                <span
                  className="text-sm font-medium capitalize"
                  style={{ color: 'rgb(var(--color-foreground))' }}
                >
                  {col.label || col.name.replace(/_/g, ' ')}
                </span>
                <span
                  className="rounded-full px-1.5 py-0.5 text-[10px] font-medium"
                  style={{
                    backgroundColor: overWip
                      ? 'rgb(var(--color-error) / 0.15)'
                      : 'rgb(var(--color-muted) / 0.15)',
                    color: overWip ? 'rgb(var(--color-error))' : 'rgb(var(--color-muted))',
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
                      borderColor: 'rgb(var(--color-border))',
                      color: 'rgb(var(--color-muted))',
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
                          selectedCard === card.id
                            ? 'rgb(var(--color-accent))'
                            : 'rgb(var(--color-border))',
                        backgroundColor: 'rgb(var(--color-surface))',
                      }}
                    >
                      <span
                        className="text-sm font-medium"
                        style={{ color: 'rgb(var(--color-foreground))' }}
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
                                backgroundColor: 'rgb(var(--color-accent) / 0.15)',
                                color: 'rgb(var(--color-accent))',
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
