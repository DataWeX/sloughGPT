import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { Tray } from './Tray'
import type { Card, Column } from './types'

afterEach(() => cleanup())

const makeColumn = (overrides: Partial<Column> = {}): Column => ({
  name: 'todo', label: 'To Do', wip_limit: 5, order: 0, ...overrides,
})
const makeCard = (overrides: Partial<Card> = {}): Card => ({
  id: 'c1', title: 'Card 1', description: '', column: 'todo',
  priority: 'medium', tags: [], due_date: '', assignee: '', sprint: '', gh: '',
  notes: [], created_at: '', updated_at: '', root_hash: '', ...overrides,
})

const defaultProps = {
  onDragStart: vi.fn(), onDragEnd: vi.fn(), onDragOver: vi.fn(),
  onDrop: vi.fn(), onCardClick: vi.fn(), draggingId: null,
}

describe('Tray', () => {
  it('renders column label', () => {
    render(<Tray column={makeColumn()} cards={[]} {...defaultProps} />)
    expect(screen.getAllByText('To Do').length).toBeGreaterThanOrEqual(1)
  })
  it('shows card count 0', () => {
    render(<Tray column={makeColumn()} cards={[]} {...defaultProps} />)
    expect(screen.getAllByText('0 / 5').length).toBeGreaterThanOrEqual(1)
  })
  it('shows card count with cards', () => {
    render(<Tray column={makeColumn()} cards={[makeCard({ id: 'c1' }), makeCard({ id: 'c2' })]} {...defaultProps} />)
    expect(screen.getAllByText('2 / 5').length).toBeGreaterThanOrEqual(1)
  })
  it('shows Drop cards when empty', () => {
    render(<Tray column={makeColumn()} cards={[]} {...defaultProps} />)
    expect(screen.getAllByText('Drop cards here').length).toBeGreaterThanOrEqual(1)
  })
  it('renders cards', () => {
    render(<Tray column={makeColumn()} cards={[makeCard({ title: 'My Card' })]} {...defaultProps} />)
    expect(screen.getAllByText('My Card').length).toBeGreaterThanOrEqual(1)
  })
  it('uses COLUMN_LABELS lookup', () => {
    render(<Tray column={makeColumn({ name: 'review' })} cards={[]} {...defaultProps} />)
    expect(screen.getAllByText('Review').length).toBeGreaterThanOrEqual(1)
  })
  it('shows WIP warning style when over limit', () => {
    const { container } = render(<Tray column={makeColumn({ wip_limit: 1 })} cards={[makeCard({ id: 'c1' }), makeCard({ id: 'c2' })]} {...defaultProps} />)
    expect(container.querySelectorAll('[class*="border-destructive"]').length).toBeGreaterThanOrEqual(1)
  })
  it('hides wip limit when 0', () => {
    render(<Tray column={makeColumn({ wip_limit: 0 })} cards={[]} {...defaultProps} />)
    expect(screen.getAllByText('0').length).toBeGreaterThanOrEqual(1)
  })
})
