import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { KanbanColumn } from './KanbanColumn'

afterEach(() => cleanup())

const makeColumn = (overrides = {}) => ({ name: 'in_progress', wip_limit: 5, order: 0, ...overrides })
const makeCard = (overrides = {}) => ({
  id: 'c1', title: 'Card A', description: '', column: 'in_progress',
  priority: 'low', tags: [], due_date: '', assignee: '', notes: [], ...overrides,
})

describe('KanbanColumn', () => {
  it('renders column name', () => {
    render(<KanbanColumn column={makeColumn()} cards={[]} />)
    expect(screen.getAllByText(/in progress/).length).toBeGreaterThanOrEqual(1)
  })
  it('shows card count', () => {
    render(<KanbanColumn column={makeColumn()} cards={[makeCard()]} />)
    expect(screen.getAllByText('1 / 5').length).toBeGreaterThanOrEqual(1)
  })
  it('shows No cards when empty', () => {
    render(<KanbanColumn column={makeColumn()} cards={[]} />)
    expect(screen.getAllByText('No cards').length).toBeGreaterThanOrEqual(1)
  })
  it('renders cards', () => {
    render(<KanbanColumn column={makeColumn()} cards={[makeCard()]} />)
    expect(screen.getAllByText('Card A').length).toBeGreaterThanOrEqual(1)
  })
  it('hides wip limit when 0', () => {
    render(<KanbanColumn column={makeColumn({ wip_limit: 0 })} cards={[makeCard()]} />)
    expect(screen.getAllByText('1').length).toBeGreaterThanOrEqual(1)
  })
  it('replaces underscores in name with spaces', () => {
    render(<KanbanColumn column={makeColumn({ name: 'code_review' })} cards={[]} />)
    expect(screen.getAllByText(/code review/).length).toBeGreaterThanOrEqual(1)
  })
})
