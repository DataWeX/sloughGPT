import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { BuffetEngine } from './BuffetEngine'

const mockBoard = { columns: [], cards: [] }
const mockRefresh = vi.fn()

vi.mock('@/lib/useOonBoard', () => ({
  useOonBoard: () => ({
    board: mockBoard,
    tags: [],
    loading: false,
    error: null,
    refresh: mockRefresh,
    optimisticMove: vi.fn(),
    optimisticAdd: vi.fn(),
    optimisticDelete: vi.fn(),
  }),
}))

vi.mock('@/lib/oon', () => ({
  oon: { move: vi.fn(), create: vi.fn(), update: vi.fn(), delete: vi.fn(), sync: vi.fn() },
}))

afterEach(() => cleanup())

describe('BuffetEngine', () => {
  it('renders Planner heading', () => {
    render(<BuffetEngine />)
    expect(screen.getAllByText('Planner').length).toBeGreaterThanOrEqual(1)
  })
  it('renders Board subtitle', () => {
    render(<BuffetEngine />)
    expect(screen.getAllByText('Board').length).toBeGreaterThanOrEqual(1)
  })
  it('has search input', () => {
    render(<BuffetEngine />)
    expect(screen.getAllByPlaceholderText(/Search/).length).toBeGreaterThanOrEqual(1)
  })
  it('has New card button', () => {
    render(<BuffetEngine />)
    expect(screen.getAllByText(/New/).length).toBeGreaterThanOrEqual(1)
  })
})
