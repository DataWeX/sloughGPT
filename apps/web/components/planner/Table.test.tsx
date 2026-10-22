import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { Table } from './Table'
import type { Card, Column } from './types'

afterEach(() => cleanup())

const cols: Column[] = [
  { name: 'todo', label: 'To Do', wip_limit: 5, order: 0 },
  { name: 'done', label: 'Done', wip_limit: 0, order: 1 },
]
const cards: Card[] = [
  {
    id: 'c1',
    title: 'Card A',
    description: '',
    column: 'todo',
    priority: 'low',
    tags: [],
    due_date: '',
    assignee: '',
    sprint: '',
    gh: '',
    notes: [],
    created_at: '',
    updated_at: '',
    root_hash: '',
  },
  {
    id: 'c2',
    title: 'Card B',
    description: '',
    column: 'done',
    priority: 'high',
    tags: [],
    due_date: '',
    assignee: '',
    sprint: '',
    gh: '',
    notes: [],
    created_at: '',
    updated_at: '',
    root_hash: '',
  },
]

const defaultProps = {
  onDragStart: vi.fn(),
  onDragEnd: vi.fn(),
  onDragOver: vi.fn(),
  onDrop: vi.fn(),
  onCardClick: vi.fn(),
  draggingId: null,
  focusCol: 0,
  focusCard: -1,
  canDrop: () => true,
  isOverWip: () => false,
}

describe('Table', () => {
  it('renders all columns', () => {
    render(
      <Table
        columns={cols}
        cardsByColumn={{ todo: [cards[0]], done: [cards[1]] }}
        {...defaultProps}
      />,
    )
    expect(screen.getAllByText('To Do').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Done').length).toBeGreaterThanOrEqual(1)
  })
  it('distributes cards to correct columns', () => {
    render(
      <Table
        columns={cols}
        cardsByColumn={{ todo: [cards[0]], done: [cards[1]] }}
        {...defaultProps}
      />,
    )
    expect(screen.getAllByText('Card A').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Card B').length).toBeGreaterThanOrEqual(1)
  })
  it('renders empty tray for column with no cards', () => {
    render(
      <Table columns={cols} cardsByColumn={{ todo: [cards[0]], done: [] }} {...defaultProps} />,
    )
    expect(screen.getAllByText('drop here').length).toBeGreaterThanOrEqual(1)
  })
  it('sorts columns by order', () => {
    const reversed = [{ ...cols[1] }, { ...cols[0] }]
    render(
      <Table
        columns={reversed}
        cardsByColumn={{ todo: [cards[0]], done: [cards[1]] }}
        {...defaultProps}
      />,
    )
    const headings = screen.getAllByText(/To Do|Done/)
    expect(headings[0].textContent).toBe('To Do')
  })
})
