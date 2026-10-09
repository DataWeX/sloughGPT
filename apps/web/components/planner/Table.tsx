'use client'

import type { Card, Column } from './types'
import { Tray } from './Tray'

interface TableProps {
  columns: Column[]
  cards: Card[]
  onDragStart: (e: React.DragEvent, cardId: string, column: string) => void
  onDragEnd: (e: React.DragEvent) => void
  onDragOver: (e: React.DragEvent) => void
  onDrop: (e: React.DragEvent, column: string) => void
  onCardClick: (card: Card) => void
  draggingId: string | null
}

export function Table({
  columns,
  cards,
  onDragStart,
  onDragEnd,
  onDragOver,
  onDrop,
  onCardClick,
  draggingId,
}: TableProps) {
  const sortedColumns = [...columns].sort((a, b) => a.order - b.order)

  const cardsByColumn = cards.reduce<Record<string, Card[]>>((acc, card) => {
    if (!acc[card.column]) acc[card.column] = []
    acc[card.column].push(card)
    return acc
  }, {})

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {sortedColumns.map((col) => (
        <Tray
          key={col.name}
          column={col}
          cards={cardsByColumn[col.name] || []}
          onDragStart={onDragStart}
          onDragEnd={onDragEnd}
          onDragOver={onDragOver}
          onDrop={onDrop}
          onCardClick={onCardClick}
          draggingId={draggingId}
        />
      ))}
    </div>
  )
}
