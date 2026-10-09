'use client'

import type { Card, Board } from './types'
import { Table } from './Table'

interface SceneProps {
  board: Board
  filteredCards: Card[]
  onDragStart: (e: React.DragEvent, cardId: string, column: string) => void
  onDragEnd: (e: React.DragEvent) => void
  onDragOver: (e: React.DragEvent) => void
  onDrop: (e: React.DragEvent, column: string) => void
  onCardClick: (card: Card) => void
  draggingId: string | null
}

export function Scene({
  board,
  filteredCards,
  onDragStart,
  onDragEnd,
  onDragOver,
  onDrop,
  onCardClick,
  draggingId,
}: SceneProps) {
  return (
    <Table
      columns={board.columns}
      cards={filteredCards}
      onDragStart={onDragStart}
      onDragEnd={onDragEnd}
      onDragOver={onDragOver}
      onDrop={onDrop}
      onCardClick={onCardClick}
      draggingId={draggingId}
    />
  )
}
