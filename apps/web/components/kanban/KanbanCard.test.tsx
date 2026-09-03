import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { KanbanCard } from './KanbanCard'

afterEach(() => cleanup())

const makeCard = (overrides = {}) => ({
  id: 'c1', title: 'Test Card', description: 'A description', column: 'todo',
  priority: 'medium', tags: [], due_date: '', assignee: '', notes: [], ...overrides,
})

describe('KanbanCard', () => {
  it('renders title', () => {
    render(<KanbanCard card={makeCard()} />)
    expect(screen.getAllByText('Test Card').length).toBeGreaterThanOrEqual(1)
  })
  it('renders description', () => {
    render(<KanbanCard card={makeCard()} />)
    expect(screen.getAllByText('A description').length).toBeGreaterThanOrEqual(1)
  })
  it('renders priority dot', () => {
    const { container } = render(<KanbanCard card={makeCard({ priority: 'high' })} />)
    const dots = container.querySelectorAll('[class*="rounded-full"]')
    expect(dots.length).toBeGreaterThanOrEqual(1)
  })
  it('renders tags', () => {
    render(<KanbanCard card={makeCard({ tags: ['bug', 'urgent'] })} />)
    expect(screen.getAllByText('bug').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('urgent').length).toBeGreaterThanOrEqual(1)
  })
  it('hides description when empty', () => {
    render(<KanbanCard card={makeCard({ description: '' })} />)
    expect(screen.queryByText('A description')).toBeNull()
  })
})
