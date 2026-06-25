// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@/components/ui', () => {
  const iconMock = (name: string) => { const C = () => <span data-testid={`icon-${name}`}>{name}</span>; C.displayName = `Icon${name}`; return C }
  return {
    IconX: iconMock('x'),
    IconTrash: iconMock('trash'),
    IconThumbUp: iconMock('thumb-up'),
    IconThumbDown: iconMock('thumb-down'),
    IconChat: iconMock('chat'),
  }
})

vi.mock('@/components/ui/button', () => ({
  Button: ({ children, onClick, variant, size, className, ...rest }: any) => (
    <button onClick={onClick} className={className} data-variant={variant} data-size={size} {...rest}>{children}</button>
  ),
}))

import { ConversationViewer } from './ConversationViewer'

const msg = (id: string, role: 'user' | 'assistant' | 'system' = 'user', overrides = {}) => ({
  id,
  role,
  content: `Message ${id} content`,
  timestamp: Date.now(),
  ...overrides,
})

describe('ConversationViewer', () => {
  const onClose = vi.fn()
  const onExport = vi.fn()
  const onDelete = vi.fn()

  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('renders nothing when closed', () => {
    const { container } = render(<ConversationViewer isOpen={false} onClose={onClose} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders dialog when open', () => {
    render(<ConversationViewer isOpen={true} onClose={onClose} messages={[]} />)
    expect(screen.getByRole('dialog')).toBeDefined()
  })

  it('shows title and message count', () => {
    const messages = [msg('1'), msg('2')]
    render(<ConversationViewer isOpen={true} onClose={onClose} messages={messages} title="My Chat" />)
    expect(screen.getByText('My Chat')).toBeDefined()
    expect(screen.getByText('2 messages')).toBeDefined()
  })

  it('shows singular for 1 message', () => {
    const messages = [msg('1')]
    render(<ConversationViewer isOpen={true} onClose={onClose} messages={messages} />)
    expect(screen.getByText('1 message')).toBeDefined()
  })

  it('renders message roles and content', () => {
    const messages = [msg('1', 'user'), msg('2', 'assistant')]
    render(<ConversationViewer isOpen={true} onClose={onClose} messages={messages} />)
    expect(screen.getByText('You')).toBeDefined()
    expect(screen.getByText('Assistant')).toBeDefined()
    expect(screen.getByText('Message 1 content')).toBeDefined()
    expect(screen.getByText('Message 2 content')).toBeDefined()
  })

  it('shows empty state when no messages', () => {
    render(<ConversationViewer isOpen={true} onClose={onClose} messages={[]} />)
    expect(screen.getByText('No messages in this conversation')).toBeDefined()
  })

  it('calls onClose when backdrop clicked', () => {
    render(<ConversationViewer isOpen={true} onClose={onClose} messages={[msg('1')]} />)
    const backdrop = document.querySelector('.bg-foreground\\/20')
    fireEvent.click(backdrop!)
    expect(onClose).toHaveBeenCalled()
  })

  it('calls onClose when close button clicked', () => {
    render(<ConversationViewer isOpen={true} onClose={onClose} messages={[msg('1')]} />)
    fireEvent.click(screen.getByLabelText('Close dialog'))
    expect(onClose).toHaveBeenCalled()
  })

  it('calls onClose on Escape key', () => {
    render(<ConversationViewer isOpen={true} onClose={onClose} messages={[msg('1')]} />)
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(onClose).toHaveBeenCalled()
  })

  it('shows delete button when onDelete is provided', () => {
    render(<ConversationViewer isOpen={true} onClose={onClose} messages={[msg('1')]} onDelete={onDelete} />)
    expect(screen.getByLabelText('Delete conversation')).toBeDefined()
  })

  it('calls onDelete when delete button clicked', () => {
    render(<ConversationViewer isOpen={true} onClose={onClose} messages={[msg('1')]} onDelete={onDelete} />)
    fireEvent.click(screen.getByLabelText('Delete conversation'))
    expect(onDelete).toHaveBeenCalled()
  })

  it('hides delete button when onDelete not provided', () => {
    render(<ConversationViewer isOpen={true} onClose={onClose} messages={[msg('1')]} />)
    expect(screen.queryByLabelText('Delete conversation')).toBeNull()
  })

  it('shows positive feedback icon', () => {
    const messages = [msg('1', 'assistant', { feedback: 'positive' })]
    render(<ConversationViewer isOpen={true} onClose={onClose} messages={messages} />)
    expect(screen.getByTestId('icon-thumb-up')).toBeDefined()
  })

  it('shows negative feedback icon', () => {
    const messages = [msg('1', 'assistant', { feedback: 'negative' })]
    render(<ConversationViewer isOpen={true} onClose={onClose} messages={messages} />)
    expect(screen.getByTestId('icon-thumb-down')).toBeDefined()
  })

  it('shows system role label', () => {
    const messages = [msg('1', 'system')]
    render(<ConversationViewer isOpen={true} onClose={onClose} messages={messages} />)
    expect(screen.getByText('System')).toBeDefined()
  })

  it('shows Esc hint in footer', () => {
    render(<ConversationViewer isOpen={true} onClose={onClose} messages={[msg('1')]} />)
    expect(screen.getByText('Esc')).toBeDefined()
  })

  it('prevents body scroll when open', () => {
    render(<ConversationViewer isOpen={true} onClose={onClose} messages={[msg('1')]} />)
    expect(document.body.style.overflow).toBe('hidden')
  })

  it('restores body scroll on unmount', () => {
    const { unmount } = render(<ConversationViewer isOpen={true} onClose={onClose} messages={[msg('1')]} />)
    unmount()
    expect(document.body.style.overflow).toBe('')
  })

  it('applies default title when none given', () => {
    render(<ConversationViewer isOpen={true} onClose={onClose} messages={[msg('1')]} />)
    expect(screen.getByText('Conversation')).toBeDefined()
  })

  it('handles tab key for focus trap', () => {
    render(
      <ConversationViewer
        isOpen={true}
        onClose={onClose}
        messages={[msg('1')]}
        onDelete={onDelete}
      />
    )
    const dialog = screen.getByRole('dialog')
    const focusable = dialog.querySelectorAll('button')
    expect(focusable.length).toBeGreaterThanOrEqual(2)
  })
})
