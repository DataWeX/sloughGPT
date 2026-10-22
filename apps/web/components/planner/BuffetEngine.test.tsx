import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { BuffetEngine } from './BuffetEngine'

const mockBoard = { columns: [{ name: 'todo', label: 'To Do', wip_limit: 5, order: 0 }], cards: [] }
const mockRefresh = vi.fn()

vi.mock('@/lib/usePlannerBoard', () => ({
  usePlannerBoard: () => ({
    board: mockBoard,
    columns: mockBoard.columns,
    tags: [],
    loading: false,
    error: null,
    refresh: mockRefresh,
    moveCard: vi.fn(),
    createCard: vi.fn(),
    updateCard: vi.fn(),
    deleteCard: vi.fn(),
    undo: vi.fn(),
    canUndo: false,
    isOverWip: () => false,
    canDrop: () => true,
    drag: null,
    setDrag: vi.fn(),
  }),
}))

vi.mock('@/lib/planner-client', () => ({
  moveCard: vi.fn(),
  createCard: vi.fn(),
  updateCard: vi.fn(),
  deleteCard: vi.fn(),
  syncNotes: vi.fn(),
  fetchBoard: vi.fn(),
  fetchTags: vi.fn(),
}))

afterEach(() => cleanup())

describe('BuffetEngine', () => {
  it('renders Planner heading', () => {
    render(<BuffetEngine />)
    expect(screen.getAllByText('Planner').length).toBeGreaterThanOrEqual(1)
  })
  it('renders card count', () => {
    render(<BuffetEngine />)
    expect(screen.getAllByText(/0 cards/).length).toBeGreaterThanOrEqual(1)
  })
  it('has search input', () => {
    render(<BuffetEngine />)
    expect(screen.getAllByPlaceholderText(/Search/).length).toBeGreaterThanOrEqual(1)
  })
  it('has New button', () => {
    render(<BuffetEngine />)
    expect(screen.getAllByText(/New/).length).toBeGreaterThanOrEqual(1)
  })
})
