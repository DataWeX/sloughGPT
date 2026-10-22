'use client'

import type { Card, Column } from './types'
import { Tray } from './Tray'

interface TableProps {
  columns: Column[]
  cardsByColumn: Record<string, Card[]>
  onDragStart: (e: React.DragEvent, cardId: string, column: string) => void
  onDragEnd: (e: React.DragEvent) => void
  onDragOver: (e: React.DragEvent, column: string) => void
  onDrop: (e: React.DragEvent, column: string) => void
  onCardClick: (card: Card) => void
  draggingId: string | null
  focusCol: number
  focusCard: number
  canDrop: (column: string) => boolean
  isOverWip: (column: string) => boolean
}

export function Table({
  columns,
  cardsByColumn,
  onDragStart,
  onDragEnd,
  onDragOver,
  onDrop,
  onCardClick,
  draggingId,
  focusCol,
  focusCard,
  canDrop,
  isOverWip,
}: TableProps) {
  const sorted = [...columns].sort((a, b) => a.order - b.order)
  const cols = sorted.length

  return (
    <div
      className="grid gap-3 h-full"
      style={{
        gridTemplateColumns: `repeat(auto-fill, minmax(${cols <= 4 ? 240 : 200}px, 1fr))`,
        gridAutoRows: 'minmax(0, 1fr)',
      }}
      role="list"
    >
      {sorted.map((col, i) => (
        <Tray
          key={col.name}
          column={col}
          cards={cardsByColumn[col.name] || []}
          onDragStart={onDragStart}
          onDragEnd={onDragEnd}
          onDragOver={(e) => onDragOver(e, col.name)}
          onDrop={(e) => onDrop(e, col.name)}
          onCardClick={onCardClick}
          draggingId={draggingId}
          isFocused={i === focusCol}
          focusCard={i === focusCol ? focusCard : -1}
          dropAllowed={canDrop(col.name)}
          overWip={isOverWip(col.name)}
        />
      ))}
    </div>
  )
}
