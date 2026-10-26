'use client'

import { cn, StatusDot } from '@sloughgpt/strui'
import type { Card, Column } from './types'
import { CardItem } from './CardItem'
import { COLUMN_LABELS } from './types'

interface TrayProps {
  column: Column
  cards: Card[]
  onDragStart: (e: React.DragEvent, cardId: string, column: string) => void
  onDragEnd: (e: React.DragEvent) => void
  onDragOver: (e: React.DragEvent) => void
  onDrop: (e: React.DragEvent, column: string) => void
  onCardClick: (card: Card) => void
  draggingId: string | null
}

/**
 * Ledger column — the log-entry board's section header + entry list.
 * WIP health is a StatusDot instrument, not a sticker: it pulses
 * destructive when the column is over its limit.
 */
export function Tray({
  column,
  cards,
  onDragStart,
  onDragEnd,
  onDragOver,
  onDrop,
  onCardClick,
  draggingId,
}: TrayProps) {
  const isOverWip = column.wip_limit > 0 && cards.length > column.wip_limit

  return (
    <div
      className={cn(
        'flex flex-col min-h-[200px] rounded-xl p-3',
        'bg-muted/40 border border-border shadow-sm',
        'relative overflow-hidden',
        isOverWip && 'border-destructive/40',
      )}
      onDragOver={onDragOver}
      onDrop={(e) => onDrop(e, column.name)}
    >
      {/* Column header — running head of the ledger section */}
      <div className="relative flex items-center justify-between mb-3">
        <h3 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          {COLUMN_LABELS[column.name] || column.name}
        </h3>
        <span
          className={cn(
            'inline-flex items-center gap-1.5 text-[10px] font-medium px-1.5 py-0.5 rounded',
            'bg-card text-muted-foreground border border-border',
            isOverWip && 'bg-destructive/15 text-destructive border-destructive/30',
          )}
        >
          {isOverWip && <StatusDot tone="destructive" pulse aria-hidden />}
          {cards.length}
          {column.wip_limit > 0 && ` / ${column.wip_limit}`}
        </span>
      </div>

      {/* Entries area */}
      <div className="relative space-y-1.5 flex-1">
        {cards.length === 0 ? (
          <div className="rounded-lg border-2 border-dashed border-border p-4 text-center">
            <p className="text-xs text-muted-foreground/70">Drop cards here</p>
          </div>
        ) : (
          cards.map((card) => (
            <CardItem
              key={card.id}
              card={card}
              onDragStart={onDragStart}
              onDragEnd={onDragEnd}
              onClick={onCardClick}
            />
          ))
        )}
      </div>
    </div>
  )
}
