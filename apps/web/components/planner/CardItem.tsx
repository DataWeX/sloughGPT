'use client'

import { useState } from 'react'
import { cn } from '@sloughgpt/strui'
import type { Card } from './types'

interface CardItemProps {
  card: Card
  onDragStart: (e: React.DragEvent, cardId: string, column: string) => void
  onDragEnd: (e: React.DragEvent) => void
  onClick: (card: Card) => void
}

const PRIORITY_COLOR: Record<string, string> = {
  critical: '#ef4444',
  high: '#f59e0b',
  medium: '#a1a1aa',
  low: '#22c55e',
}

export function CardItem({ card, onDragStart, onDragEnd, onClick }: CardItemProps) {
  const [hovered, setHovered] = useState(false)
  const priorityColor = PRIORITY_COLOR[card.priority] || PRIORITY_COLOR.medium

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
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      role="button"
      tabIndex={0}
      aria-label={`Card: ${card.title}`}
      className={cn(
        'group relative cursor-grab active:cursor-grabbing',
        // Spring curves for lift and landing
        'transition-[box-shadow,transform] duration-200',
        '[transition-timing-function:cubic-bezier(0.34,1.56,0.64,1)]',
        // Drag tilt
        'active:rotate-[1.5deg] active:scale-[1.02]',
      )}
      style={{
        // Card material — off-white paper
        background: hovered
          ? 'linear-gradient(178deg, #fafaf9 0%, #f5f5f4 100%)'
          : 'linear-gradient(178deg, #f0f0ef 0%, #e8e8e7 100%)',
        borderRadius: '4px',
        border: '1px solid rgba(0,0,0,0.07)',
        // Shadow: lifts on hover
        boxShadow: hovered
          ? '0 4px 16px rgba(0,0,0,0.25), 0 1px 3px rgba(0,0,0,0.15), 0 0 0 1px rgba(0,0,0,0.03)'
          : '0 1px 3px rgba(0,0,0,0.1), 0 0 0 1px rgba(0,0,0,0.04)',
        transform: hovered ? 'translateY(-2px)' : 'translateY(0)',
      }}
    >
      {/* Priority edge stripe */}
      <div
        className="absolute left-0 top-0 bottom-0 w-[3px] rounded-l-[4px]"
        style={{ background: priorityColor }}
        aria-hidden
      />

      <div className="relative pl-3.5 pr-3 py-2.5">
        <div className="flex items-start justify-between gap-2">
          <p className="text-[13px] font-medium leading-snug flex-1 text-zinc-800">{card.title}</p>
          {card.root_hash && (
            <span
              className="mt-1 h-1.5 w-1.5 rounded-full shrink-0 bg-emerald-500"
              style={{ boxShadow: '0 0 4px rgba(16,185,129,0.4)' }}
              title="Hash chain active"
              aria-label="Hash chain active"
            />
          )}
        </div>
        {card.description && (
          <p className="mt-1 text-[11px] leading-snug line-clamp-2 text-zinc-500">
            {card.description}
          </p>
        )}
        <div className="mt-1.5 flex items-center gap-1.5 flex-wrap text-[9px] font-mono text-zinc-400">
          {card.tags.slice(0, 3).map((tag) => (
            <span key={tag} className="px-1 py-px rounded bg-zinc-100 text-zinc-500">
              {tag}
            </span>
          ))}
          {card.assignee && <span>{card.assignee}</span>}
          {card.due_date && <span>{card.due_date}</span>}
        </div>
      </div>

      {/* Bottom paper edge */}
      <div
        className="absolute bottom-0 left-0 right-0 h-px rounded-b-[4px]"
        style={{
          background:
            'linear-gradient(90deg, transparent 10%, rgba(255,255,255,0.5) 50%, transparent 90%)',
        }}
        aria-hidden
      />
    </div>
  )
}
