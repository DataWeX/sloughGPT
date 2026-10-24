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
  onDrop: (e: React.DragEvent) => void
  onCardClick: (card: Card) => void
  draggingId: string | null
  isFocused: boolean
  focusCard: number
  dropAllowed: boolean
  overWip: boolean
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
  isFocused,
  focusCard,
  dropAllowed,
  overWip,
}: TrayProps) {
  const count = cards.length
  const wipFull = column.wip_limit > 0 && count >= column.wip_limit
  const showOverWip = overWip && !wipFull

  return (
    <div
      className={cn(
        'flex flex-col rounded-lg relative overflow-hidden',
        'transition-all duration-200',
        isFocused && 'ring-1 ring-zinc-600/50',
      )}
      style={{
        background: dropAllowed
          ? 'linear-gradient(180deg, rgba(0,0,0,0.18) 0%, rgba(0,0,0,0.08) 100%)'
          : 'linear-gradient(180deg, rgba(0,0,0,0.28) 0%, rgba(0,0,0,0.18) 100%)',
        boxShadow: dropAllowed
          ? 'inset 0 1px 6px rgba(0,0,0,0.3), inset 0 -1px 0 rgba(255,255,255,0.03), 0 1px 0 rgba(255,255,255,0.04)'
          : 'inset 0 1px 6px rgba(0,0,0,0.4)',
        border: showOverWip
          ? '1px solid rgba(200,50,50,0.25)'
          : wipFull
            ? '1px solid rgba(200,160,40,0.2)'
            : '1px solid rgba(255,255,255,0.05)',
        opacity: dropAllowed ? 1 : 0.45,
        minHeight: 320,
      }}
      onDragOver={onDragOver}
      onDrop={onDrop}
      role="listitem"
      aria-label={`${COLUMN_LABELS[column.name] || column.name}, ${count} cards`}
    >
      {/* Sticky header */}
      <div
        className="flex items-center justify-between px-3 py-2.5 shrink-0"
        style={{
          background: 'linear-gradient(180deg, rgba(0,0,0,0.25) 0%, rgba(0,0,0,0.15) 100%)',
          borderBottom: '1px solid rgba(255,255,255,0.05)',
        }}
      >
        <h3 className="text-[10px] font-semibold uppercase tracking-[0.12em] text-zinc-500">
          {COLUMN_LABELS[column.name] || column.name}
        </h3>
        <span
          className={cn(
            'inline-flex items-center gap-1 text-[10px] font-mono px-1.5 py-0.5 rounded',
            'transition-colors duration-200',
            showOverWip
              ? 'bg-red-950/50 text-red-400 border border-red-900/50'
              : wipFull
                ? 'bg-amber-950/50 text-amber-400 border border-amber-900/50'
                : 'bg-zinc-900 text-zinc-600 border border-zinc-800',
          )}
        >
          {(wipFull || showOverWip) && (
            <StatusDot tone={showOverWip ? 'destructive' : 'warning'} pulse aria-hidden />
          )}
          {count}
          {column.wip_limit > 0 && ` / ${column.wip_limit}`}
        </span>
      </div>

      {/* Scrollable cards area */}
      <div
        className="flex-1 overflow-y-auto overflow-x-hidden px-2 py-2 space-y-1.5 min-h-0"
        style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(255,255,255,0.08) transparent' }}
      >
        {cards.length === 0 ? (
          <div
            className={cn(
              'rounded-md border border-dashed p-5 text-center transition-colors duration-200 h-20 flex items-center justify-center',
              dropAllowed ? 'border-zinc-800' : 'border-red-900/40',
            )}
          >
            <p
              className={cn(
                'text-[10px] font-mono',
                dropAllowed ? 'text-zinc-700' : 'text-red-800',
              )}
            >
              {dropAllowed ? 'drop here' : 'WIP full'}
            </p>
          </div>
        ) : (
          cards.map((card, idx) => (
            <div
              key={card.id}
              className={cn(
                'transition-[transform,opacity] duration-200',
                '[transition-timing-function:cubic-bezier(0.34,1.56,0.64,1)]',
                idx === focusCard && 'ring-1 ring-zinc-500/50 rounded-md',
                draggingId === card.id && 'opacity-20',
              )}
            >
              <CardItem
                card={card}
                onDragStart={onDragStart}
                onDragEnd={onDragEnd}
                onClick={onCardClick}
              />
            </div>
          ))
        )}
      </div>
    </div>
  )
}
