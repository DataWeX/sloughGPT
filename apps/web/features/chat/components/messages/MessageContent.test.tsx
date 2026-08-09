import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('./Markdown', () => ({
  Markdown: ({ content }: { content: string }) => <span data-testid="markdown">{content}</span>,
}))

import { MessageContent } from './MessageContent'

describe('MessageContent', () => {
  afterEach(cleanup)

  it('renders assistant message content', () => {
    render(<MessageContent content="Hello world" role="assistant" />)
    expect(screen.getByTestId('markdown')).toBeDefined()
    expect(screen.getByText('Hello world')).toBeDefined()
  })

  it('renders user message content', () => {
    render(<MessageContent content="Hello world" role="user" />)
    expect(screen.getByText('Hello world')).toBeDefined()
  })

  it('returns null for empty user message', () => {
    const { container } = render(<MessageContent content="" role="user" />)
    expect(container.innerHTML).toBe('')
  })

  it('shows bounce dots for empty assistant message', () => {
    const { container } = render(<MessageContent content="" role="assistant" />)
    const dots = container.querySelectorAll('.animate-bounce')
    expect(dots.length).toBe(3)
  })

  it('highlights search query in assistant message', () => {
    render(<MessageContent content="The quick brown fox" role="assistant" searchQuery="quick" />)
    const marks = screen.getAllByText('quick')
    expect(marks.length).toBeGreaterThanOrEqual(1)
    expect(marks[0].tagName).toBe('MARK')
  })

  it('highlights search query in user message', () => {
    render(<MessageContent content="The quick brown fox" role="user" searchQuery="quick" />)
    const marks = screen.getAllByText('quick')
    expect(marks.length).toBeGreaterThanOrEqual(1)
    expect(marks[0].tagName).toBe('MARK')
  })

  it('shows streaming cursor when isStreaming', () => {
    const { container } = render(<MessageContent content="Hello" role="assistant" isStreaming />)
    const cursor = container.querySelector('.animate-pulse')
    expect(cursor).toBeTruthy()
  })

  it('truncates content when collapsible', () => {
    const long = 'a'.repeat(200)
    render(<MessageContent content={long} role="assistant" collapsibleLength={50} />)
    expect(screen.getByText(/Show more/)).toBeDefined()
    expect(screen.getByText(/150 more/)).toBeDefined()
  })

  it('expands collapsed content on click', () => {
    const long = 'a'.repeat(200)
    render(<MessageContent content={long} role="assistant" collapsibleLength={50} />)
    fireEvent.click(screen.getByText(/Show more/))
    expect(screen.getByText('Show less')).toBeDefined()
  })

  it('collapses expanded content on "Show less"', () => {
    const long = 'a'.repeat(200)
    render(<MessageContent content={long} role="assistant" collapsibleLength={50} />)
    fireEvent.click(screen.getByText(/Show more/))
    fireEvent.click(screen.getByText('Show less'))
    expect(screen.getByText(/Show more/)).toBeDefined()
  })

  it('shows edit form when isEditing', () => {
    render(<MessageContent content="Original" role="user" isEditing messageId="m1" onEdit={vi.fn()} onEditCancel={vi.fn()} />)
    expect(screen.getByLabelText('Edit message form')).toBeDefined()
    expect(screen.getByText('Cancel')).toBeDefined()
    expect(screen.getByText('Resend')).toBeDefined()
  })

  it('calls onEdit on form submit', () => {
    const onEdit = vi.fn()
    render(<MessageContent content="Original" role="user" isEditing messageId="m1" onEdit={onEdit} onEditCancel={vi.fn()} />)
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Updated' } })
    fireEvent.submit(screen.getByLabelText('Edit message form'))
    expect(onEdit).toHaveBeenCalledWith('m1', 'Updated')
  })

  it('calls onEditCancel when Cancel clicked', () => {
    const onEditCancel = vi.fn()
    render(<MessageContent content="Original" role="user" isEditing messageId="m1" onEdit={vi.fn()} onEditCancel={onEditCancel} />)
    fireEvent.click(screen.getByText('Cancel'))
    expect(onEditCancel).toHaveBeenCalled()
  })

  it('resets edit content to original on cancel', () => {
    const onEditCancel = vi.fn()
    render(<MessageContent content="Original" role="user" isEditing messageId="m1" onEdit={vi.fn()} onEditCancel={onEditCancel} />)
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Changed' } })
    fireEvent.click(screen.getByText('Cancel'))
    expect(onEditCancel).toHaveBeenCalled()
  })

  it('sets aria-expanded on collapsible buttons', () => {
    const long = 'a'.repeat(200)
    render(<MessageContent content={long} role="assistant" collapsibleLength={50} />)
    const btn = screen.getByText(/Show more/)
    expect(btn.getAttribute('aria-expanded')).toBe('false')
    fireEvent.click(btn)
    const btnAfter = screen.getByText('Show less')
    expect(btnAfter.getAttribute('aria-expanded')).toBe('true')
  })

  it('collapsible for user role', () => {
    const long = 'a'.repeat(200)
    render(<MessageContent content={long} role="user" collapsibleLength={50} />)
    expect(screen.getByText(/Show more/)).toBeDefined()
    fireEvent.click(screen.getByText(/Show more/))
    expect(screen.getByText('Show less')).toBeDefined()
  })
})
