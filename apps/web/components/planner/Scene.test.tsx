import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { Scene } from './Scene'
import type { Card, Column } from './types'

afterEach(() => cleanup())

const board = {
  columns: [{ name: 'todo', label: 'To Do', wip_limit: 5, order: 0 }] as Column[],
  cards: [],
}
const cards: Card[] = [
  { id: 'c1', title: 'Scene Card', description: '', column: 'todo', priority: 'low', tags: [], due_date: '', assignee: '', sprint: '', gh: '', notes: [], created_at: '', updated_at: '', root_hash: '' },
]

const defaultProps = {
  onDragStart: vi.fn(), onDragEnd: vi.fn(), onDragOver: vi.fn(),
  onDrop: vi.fn(), onCardClick: vi.fn(), draggingId: null,
}

describe('Scene', () => {
  it('renders the Table with columns and cards', () => {
    render(<Scene board={board} filteredCards={cards} {...defaultProps} />)
    expect(screen.getAllByText('Scene Card').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('To Do').length).toBeGreaterThanOrEqual(1)
  })
  it('renders empty when no cards', () => {
    render(<Scene board={board} filteredCards={[]} {...defaultProps} />)
    expect(screen.getAllByText('Drop cards here').length).toBeGreaterThanOrEqual(1)
  })
})
