import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { CardItem } from './CardItem'
import type { Card } from './types'

afterEach(() => cleanup())

const makeCard = (overrides: Partial<Card> = {}): Card => ({
  id: 'c1', title: 'Test Card', description: 'A description', column: 'todo',
  priority: 'medium', tags: ['frontend', 'urgent'], due_date: '2026-09-10',
  assignee: 'alice', sprint: 's1', gh: '', notes: [],
  created_at: '2026-09-01', updated_at: '2026-09-01', root_hash: '',
  ...overrides,
})

const defaultProps = {
  onDragStart: vi.fn(),
  onDragEnd: vi.fn(),
  onClick: vi.fn(),
}

describe('CardItem', () => {
  it('renders title', () => {
    render(<CardItem card={makeCard()} {...defaultProps} />)
    expect(screen.getAllByText('Test Card').length).toBeGreaterThanOrEqual(1)
  })
  it('renders description', () => {
    render(<CardItem card={makeCard()} {...defaultProps} />)
    expect(screen.getAllByText('A description').length).toBeGreaterThanOrEqual(1)
  })
  it('renders priority badge', () => {
    render(<CardItem card={makeCard({ priority: 'high' })} {...defaultProps} />)
    expect(screen.getAllByText('high').length).toBeGreaterThanOrEqual(1)
  })
  it('renders tags', () => {
    render(<CardItem card={makeCard({ tags: ['a', 'b'] })} {...defaultProps} />)
    expect(screen.getAllByText('a').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('b').length).toBeGreaterThanOrEqual(1)
  })
  it('shows +N when more than 3 tags', () => {
    render(<CardItem card={makeCard({ tags: ['a', 'b', 'c', 'd', 'e'] })} {...defaultProps} />)
    expect(screen.getAllByText('+2').length).toBeGreaterThanOrEqual(1)
  })
  it('renders assignee and due_date', () => {
    render(<CardItem card={makeCard()} {...defaultProps} />)
    expect(screen.getAllByText('alice').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('2026-09-10').length).toBeGreaterThanOrEqual(1)
  })
  it('calls onClick on click', () => {
    const onClick = vi.fn()
    render(<CardItem card={makeCard()} {...defaultProps} onClick={onClick} />)
    fireEvent.click(screen.getAllByRole('button')[0])
    expect(onClick).toHaveBeenCalled()
  })
  it('calls onClick on Enter key', () => {
    const onClick = vi.fn()
    render(<CardItem card={makeCard()} {...defaultProps} onClick={onClick} />)
    fireEvent.keyDown(screen.getAllByRole('button')[0], { key: 'Enter' })
    expect(onClick).toHaveBeenCalled()
  })
  it('has aria-label', () => {
    render(<CardItem card={makeCard()} {...defaultProps} />)
    expect(screen.getAllByLabelText('Card: Test Card').length).toBeGreaterThanOrEqual(1)
  })
  it('is draggable', () => {
    render(<CardItem card={makeCard()} {...defaultProps} />)
    expect(screen.getAllByRole('button')[0]).toHaveAttribute('draggable', 'true')
  })
  it('shows hash tree dot when root_hash present', () => {
    const { container } = render(<CardItem card={makeCard({ root_hash: 'abc123' })} {...defaultProps} />)
    expect(container.querySelectorAll('[title="Hash tree active"]').length).toBeGreaterThanOrEqual(1)
  })
  it('hides hash tree dot when no root_hash', () => {
    const { container } = render(<CardItem card={makeCard({ root_hash: '' })} {...defaultProps} />)
    expect(container.querySelectorAll('[title="Hash tree active"]').length).toBe(0)
  })
})
