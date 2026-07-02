// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

import ConversationSection from './ConversationSection'
import type { Conversation } from '@/lib/session-controller'

describe('ConversationSection', () => {
  afterEach(cleanup)

  const mkConv = (id: string, name: string) => ({
    id, name, session_id: `s${id}`, created_at: `2026-01-0${id}T00:00:00Z`,
    messages: id === '1' ? [{ id: 'm1', role: 'user', content: 'hi', timestamp: '2026-01-01T00:00:00Z' }] : [],
    updated_at: `2026-01-0${id}T00:00:00Z`, pinned: false, starred: false, message_count: id === '1' ? 1 : 0,
  } as Conversation)
  const convs = [mkConv('1', 'Chat 1'), mkConv('2', 'Chat 2')]
  const handlers = {
    selectedIds: new Set<string>(),
    onToggleSelect: vi.fn(),
    onSelect: vi.fn(),
    onPin: vi.fn(),
    onStar: vi.fn(),
    onArchive: vi.fn(),
    onDelete: vi.fn(),
    onRename: vi.fn(),
    onExport: vi.fn(),
  }

  it('renders section label', () => {
    render(<ConversationSection label="Recent" conversations={convs} {...handlers} />)
    expect(screen.getByText('Recent')).toBeDefined()
  })

  it('renders conversation rows', () => {
    render(<ConversationSection label="Recent" conversations={convs} {...handlers} />)
    expect(screen.getByText('Chat 1')).toBeDefined()
    expect(screen.getByText('Chat 2')).toBeDefined()
  })

  it('returns null when empty', () => {
    const { container } = render(<ConversationSection label="Empty" conversations={[]} {...handlers} />)
    expect(container.innerHTML).toBe('')
  })
})
