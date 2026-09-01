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
        'rounded-lg border border-border bg-card p-3 shadow-sm cursor-grab',
        'transition-all duration-150',
        'hover:-translate-y-0.5 hover:shadow-md',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
        'active:cursor-grabbing',
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <p className="text-sm font-medium text-foreground leading-snug flex-1">
          {card.title}
        </p>
        {card.root_hash && (
          <span
            className="mt-0.5 h-2 w-2 rounded-full bg-purple-500 shrink-0"
            title="Hash tree active"
            aria-label="Hash tree active"
          />
        )}
      </div>
      {card.description && (
        <p className="mt-1 text-xs text-muted-foreground line-clamp-2">
          {card.description}
        </p>
      )}
      <div className="mt-2 flex items-center gap-1.5 flex-wrap">
        <span
          className={cn(
            'inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium uppercase tracking-wider',
            PRIORITY_CLASSES[card.priority] || 'bg-muted text-muted-foreground',
          )}
        >
          {card.priority}
        </span>
        {card.tags.slice(0, 3).map((tag) => (
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
      {(card.assignee || card.due_date) && (
        <div className="mt-2 flex items-center gap-2 text-[10px] text-muted-foreground">
          {card.assignee && <span>{card.assignee}</span>}
          {card.due_date && <span>{card.due_date}</span>}
        </div>
      )}
    </div>
  )
}
