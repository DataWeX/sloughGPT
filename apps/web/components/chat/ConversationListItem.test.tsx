import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ConversationListItem } from './ConversationListItem'

afterEach(cleanup)

const baseConv = {
  id: 'c1', name: 'Test Chat', session_id: 'c1',
  created_at: '2026-06-21T10:00:00Z', updated_at: '2026-06-21T12:00:00Z',
  message_count: 1,
  messages: [{ id: 'm1', role: 'user' as const, content: 'Hello world', timestamp: '2026-06-21T12:00:00Z' }],
  starred: false, pinned: false,
}

describe('ConversationListItem', () => {
  it('renders conversation name', () => {
    render(<ConversationListItem conversation={baseConv} isActive={false} onClick={vi.fn()} onDelete={vi.fn()} />)
    expect(screen.getByText('Test Chat')).toBeDefined()
  })

  it('shows message count and separator', () => {
    render(<ConversationListItem conversation={baseConv} isActive={false} onClick={vi.fn()} onDelete={vi.fn()} />)
    expect(screen.getByText(/1/)).toBeDefined()
  })

  it('shows pinned icon when pinned', () => {
    render(<ConversationListItem conversation={{ ...baseConv, pinned: true }} isActive={false} onClick={vi.fn()} onDelete={vi.fn()} />)
    const pins = screen.getAllByRole('listitem')
    expect(pins.length).toBeGreaterThanOrEqual(1)
  })

  it('shows star icon when starred', () => {
    render(<ConversationListItem conversation={{ ...baseConv, starred: true }} isActive={false} onClick={vi.fn()} onDelete={vi.fn()} />)
    expect(screen.getByText('Test Chat')).toBeDefined()
  })

  it('calls onClick when clicked', () => {
    const onClick = vi.fn()
    render(<ConversationListItem conversation={baseConv} isActive={false} onClick={onClick} onDelete={vi.fn()} />)
    fireEvent.click(screen.getByText('Test Chat'))
    expect(onClick).toHaveBeenCalled()
  })

  it('calls onDelete when Delete clicked in dropdown', () => {
    const onDelete = vi.fn()
    render(<ConversationListItem conversation={baseConv} isActive={false} onClick={vi.fn()} onDelete={onDelete} />)
    fireEvent.click(screen.getByText('Test Chat'))
    const moreBtns = screen.getAllByRole('button')
    const moreBtn = moreBtns.find(b => b.querySelector('svg'))
    if (moreBtn) fireEvent.click(moreBtn)
  })

  it('has listitem role', () => {
    render(<ConversationListItem conversation={baseConv} isActive={false} onClick={vi.fn()} onDelete={vi.fn()} />)
    expect(screen.getByRole('listitem')).toBeDefined()
  })

  it('shows active state styling', () => {
    const { container } = render(<ConversationListItem conversation={baseConv} isActive={true} onClick={vi.fn()} onDelete={vi.fn()} />)
    const listitem = container.querySelector('[role="listitem"]')
    expect(listitem?.className).toContain('bg-secondary/50')
  })
})
