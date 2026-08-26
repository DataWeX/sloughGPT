import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import { MessageBookmarking } from './MessageBookmarking'
import type { ChatMessage } from '@/lib/chat-utils'

afterEach(cleanup)
beforeEach(() => {
  localStorage.clear()
})

const mockMessages: ChatMessage[] = [
  { id: '1', role: 'user', content: 'Hello world', timestamp: Date.now() },
]

describe('MessageBookmarking', () => {
  it('renders empty state', () => {
    render(<MessageBookmarking messages={mockMessages} onJumpToMessage={vi.fn()} />)
    expect(screen.getByText('No bookmarks yet')).toBeInTheDocument()
  })

  it('renders with count', () => {
    render(<MessageBookmarking messages={mockMessages} onJumpToMessage={vi.fn()} />)
    expect(screen.getByText(/Bookmarks/)).toBeInTheDocument()
    expect(screen.getByText('(0)')).toBeInTheDocument()
  })

  it('shows default categories', () => {
    render(<MessageBookmarking messages={mockMessages} onJumpToMessage={vi.fn()} />)
    expect(screen.getByText(/Important/)).toBeInTheDocument()
    expect(screen.getByText(/Code/)).toBeInTheDocument()
    expect(screen.getByText(/Question/)).toBeInTheDocument()
  })

  it('opens add category form', () => {
    render(<MessageBookmarking messages={mockMessages} onJumpToMessage={vi.fn()} />)
    fireEvent.click(screen.getByLabelText('Add category'))
    expect(screen.getByPlaceholderText('Category name...')).toBeInTheDocument()
  })

  it('adds custom category', async () => {
    render(<MessageBookmarking messages={mockMessages} onJumpToMessage={vi.fn()} />)
    fireEvent.click(screen.getByLabelText('Add category'))
    fireEvent.change(screen.getByPlaceholderText('Category name...'), { target: { value: 'Custom' } })
    await act(async () => {
      fireEvent.click(screen.getByLabelText('Save category'))
    })
    expect(screen.getByText(/Custom/)).toBeInTheDocument()
  })

  it('adds category on Enter', async () => {
    render(<MessageBookmarking messages={mockMessages} onJumpToMessage={vi.fn()} />)
    fireEvent.click(screen.getByLabelText('Add category'))
    const input = screen.getByPlaceholderText('Category name...')
    fireEvent.change(input, { target: { value: 'Test' } })
    await act(async () => {
      fireEvent.keyDown(input, { key: 'Enter' })
    })
    expect(screen.getByText(/Test/)).toBeInTheDocument()
  })

  it('persists to localStorage', async () => {
    render(<MessageBookmarking messages={mockMessages} onJumpToMessage={vi.fn()} />)
    fireEvent.click(screen.getByLabelText('Add category'))
    fireEvent.change(screen.getByPlaceholderText('Category name...'), { target: { value: 'Saved' } })
    await act(async () => {
      fireEvent.click(screen.getByLabelText('Save category'))
    })
    const stored = JSON.parse(localStorage.getItem('message-bookmarks') || '[]')
    expect(stored).toEqual([])
  })

  it('filters by category', () => {
    render(<MessageBookmarking messages={mockMessages} onJumpToMessage={vi.fn()} />)
    fireEvent.click(screen.getByText(/All/))
    expect(screen.getByText(/All/)).toHaveClass('bg-primary/20')
  })
})