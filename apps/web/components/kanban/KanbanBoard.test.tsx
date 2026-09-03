import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { KanbanBoard } from './KanbanBoard'
import type { KanbanBoard as BoardData } from './types'

afterEach(() => cleanup())

const makeBoard = (overrides: Partial<BoardData> = {}): BoardData => ({
  name: 'test',
  columns: [
    { name: 'todo', wip_limit: 5, order: 0 },
    { name: 'done', wip_limit: 0, order: 1 },
  ],
  cards: [],
  ...overrides,
})

describe('KanbanBoard', () => {
  it('renders all columns', () => {
    render(<KanbanBoard board={makeBoard()} />)
    expect(screen.getAllByText(/todo/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/done/).length).toBeGreaterThanOrEqual(1)
  })
  it('distributes cards to columns', () => {
    const board = makeBoard({
      cards: [
        { id: 'c1', title: 'Card A', description: '', column: 'todo', priority: 'low', tags: [], due_date: '', assignee: '', notes: [] },
        { id: 'c2', title: 'Card B', description: '', column: 'done', priority: 'high', tags: [], due_date: '', assignee: '', notes: [] },
      ],
    })
    render(<KanbanBoard board={board} />)
    expect(screen.getAllByText('Card A').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Card B').length).toBeGreaterThanOrEqual(1)
  })
  it('shows No cards for empty columns', () => {
    render(<KanbanBoard board={makeBoard()} />)
    expect(screen.getAllByText('No cards').length).toBeGreaterThanOrEqual(1)
  })
  it('sorts columns by order', () => {
    const board = makeBoard({
      columns: [
        { name: 'done', wip_limit: 0, order: 1 },
        { name: 'todo', wip_limit: 5, order: 0 },
      ],
    })
    render(<KanbanBoard board={board} />)
    const headings = screen.getAllByText(/todo|done/)
    expect(headings[0].textContent).toMatch(/todo/)
  })
})
