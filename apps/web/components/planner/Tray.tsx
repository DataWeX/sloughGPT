'use client'

import { cn } from '@sloughgpt/strui'
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
        'bg-[hsl(40,30%,95%)]',
        'border border-[hsl(40,20%,85%)]',
        'shadow-[4px_4px_0_0_hsl(40,20%,80%)]',
        'relative overflow-hidden',
        isOverWip && 'border-destructive/40 shadow-[4px_4px_0_0_hsl(0,60%,70%)]',
      )}
      onDragOver={onDragOver}
      onDrop={(e) => onDrop(e, column.name)}
    >
      {/* Halftone dot pattern overlay */}
      <div
        className={cn(
          'absolute inset-0 pointer-events-none opacity-30',
          '[background-image:',
            'radial-gradient(circle,hsl(40,20%,75%)_1.5px,transparent_1.5px)',
          ']',
          '[background-size:12px_12px]',
          '[background-position:0_0]',
        )}
        aria-hidden="true"
      />

      {/* Column header */}
      <div className="relative flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-[hsl(40,20%,40%)]">
          {COLUMN_LABELS[column.name] || column.name}
        </h3>
        <span
          className={cn(
            'text-[10px] font-medium px-1.5 py-0.5 rounded',
            'bg-[hsl(40,20%,88%)] text-[hsl(40,20%,40%)]',
            'border border-[hsl(40,20%,80%)]',
            isOverWip && 'bg-destructive/15 text-destructive border-destructive/30',
          )}
        >
          {cards.length}
          {column.wip_limit > 0 && ` / ${column.wip_limit}`}
        </span>
      </div>

      {/* Cards area */}
      <div className="relative space-y-2 flex-1">
        {cards.length === 0 ? (
          <div className="rounded-lg border-2 border-dashed border-[hsl(40,20%,78%)] p-4 text-center bg-[hsl(40,30%,92%)]">
            <p className="text-xs text-[hsl(40,20%,55%)]">Drop cards here</p>
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
