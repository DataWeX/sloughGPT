// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

import ConversationRow from './ConversationRow'
import type { Conversation } from '@/lib/session-controller'

describe('ConversationRow', () => {
  afterEach(cleanup)

  const msg = (role: string, content: string) => ({ id: 'm1', role, content, timestamp: '2026-01-01T00:00:00Z' })

  const base = {
    conversation: {
      id: '1', name: 'Test Chat', session_id: 's1', created_at: '2026-01-01T00:00:00Z',
      messages: [msg('user', 'Hello world'), msg('assistant', 'Hi there')],
      updated_at: '2026-06-01T12:00:00Z', pinned: false, starred: false, message_count: 2,
    } as Conversation,
    selected: false,
    onToggleSelect: vi.fn(),
    onSelect: vi.fn(),
    onPin: vi.fn(),
    onStar: vi.fn(),
    onArchive: vi.fn(),
    onDelete: vi.fn(),
    onRename: vi.fn(),
    onExport: vi.fn(),
  }

  it('renders conversation name and message count', () => {
    render(<ConversationRow {...base} />)
    expect(screen.getByText('Test Chat')).toBeDefined()
    expect(screen.getByText(/2 messages/)).toBeDefined()
  })

  it('calls onSelect when clicking the row body', () => {
    render(<ConversationRow {...base} />)
    const body = screen.getByText('Test Chat').closest('[role="button"]')!
    fireEvent.click(body)
    expect(base.onSelect).toHaveBeenCalledOnce()
  })

  it('calls onToggleSelect when clicking checkbox', () => {
    render(<ConversationRow {...base} />)
    const checkboxes = screen.getAllByRole('checkbox')
    fireEvent.click(checkboxes[0])
    expect(base.onToggleSelect).toHaveBeenCalledOnce()
  })

  it('shows starred icon when starred', () => {
    render(<ConversationRow {...base} conversation={{ ...base.conversation, starred: true, pinned: false, archived: false }} />)
    const star = document.querySelector('.text-warning')
    expect(star).toBeDefined()
  })

  it('shows pinned icon when pinned', () => {
    render(<ConversationRow {...base} conversation={{ ...base.conversation, pinned: true, starred: false, archived: false }} />)
    const pin = document.querySelector('.text-primary')
    expect(pin).toBeDefined()
  })

  it('shows archived badge when archived', () => {
    render(<ConversationRow {...base} conversation={{ ...base.conversation, archived: true, pinned: false, starred: false }} />)
    expect(screen.getByText('Archived')).toBeDefined()
  })

  it('renders dropdown menu trigger button', () => {
    render(<ConversationRow {...base} />)
    expect(screen.getByLabelText('More options')).toBeDefined()
  })
})
