'use client'

import { cn } from '@sloughgpt/strui'
import type { Card } from './types'
import { PRIORITY_CLASSES } from './types'

interface CardItemProps {
  card: Card
  onDragStart: (e: React.DragEvent, cardId: string, column: string) => void
  onDragEnd: (e: React.DragEvent) => void
  onClick: (card: Card) => void
}

/**
 * Board entry — the log-entry primitive (see DESIGN_SYSTEM.md "The Log Entry").
 *
 * Anatomy: gutter (short hash prefix, monospace) | content (title + summary +
 * tags) | margin (priority, assignee, due). No floating-card stack: the entry
 * reads as a ledger line you drag between sections.
 */
export function CardItem({ card, onDragStart, onDragEnd, onClick }: CardItemProps) {
  return (
    <div
      draggable
      onDragStart={(e) => onDragStart(e, card.id, card.column)}
      onDragEnd={onDragEnd}
      onClick={() => onClick(card)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onClick(card)
        }
      }}
      role="button"
      tabIndex={0}
      aria-label={`Card: ${card.title}`}
      className={cn(
        'group grid grid-cols-[2rem_minmax(0,1fr)] gap-x-2 items-start',
        'rounded-md border border-border bg-card px-3 py-2 shadow-sm cursor-grab',
        'transition-all duration-150',
        'hover:border-primary/50 hover:shadow-md',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
        'active:cursor-grabbing',
      )}
    >
      {/* Gutter — stable short id, the entry's line number */}
      <span
        className="pt-0.5 font-mono text-[10px] leading-5 text-muted-foreground/50 select-none truncate"
        aria-hidden="true"
        title={card.id}
      >
        {(card.root_hash || card.id).slice(0, 4)}
      </span>

      <div className="min-w-0">
        <div className="flex items-start justify-between gap-2">
          <p className="text-sm font-medium text-foreground leading-snug flex-1">{card.title}</p>
          {card.root_hash && (
            <span
              className="mt-0.5 h-2 w-2 rounded-full bg-primary shrink-0"
              title="Hash tree active"
              aria-label="Hash tree active"
            />
          )}
        </div>
        {card.description && (
          <p className="mt-0.5 text-xs text-muted-foreground line-clamp-2">{card.description}</p>
        )}
        <div className="mt-1 flex items-center gap-1.5 flex-wrap font-mono text-[10px] leading-4 text-muted-foreground/70">
          <span
            className={cn(
              'inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium uppercase tracking-wider',
              PRIORITY_CLASSES[card.priority] || 'bg-muted text-muted-foreground',
            )}
          >
            {card.priority}
          </span>
          {card.tags.slice(0, 3).map((tag) => (
            <span key={tag} className="px-1 py-0.5 rounded bg-muted text-muted-foreground">
              {tag}
            </span>
          ))}
          {card.tags.length > 3 && (
            <span className="text-muted-foreground/60">+{card.tags.length - 3}</span>
          )}
          {card.assignee && <span>{card.assignee}</span>}
          {card.due_date && <span>{card.due_date}</span>}
        </div>
      </div>
    </div>
  )
}
