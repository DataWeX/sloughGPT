// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('next/link', () => ({ default: ({ children, href, className, onClick }: any) => <a href={href} className={className} onClick={onClick}>{children}</a> }))

import { ConversationsDropdown } from './ConversationsDropdown'
import { ChatToolbarProvider } from '@/contexts/ChatToolbarContext'
import type { ChatToolbarContextValue } from '@/contexts/ChatToolbarContext'

const convs = [
  { id: 'c1', name: 'Chat A', session_id: 'c1', created_at: '2026-06-21T10:00:00Z', messages: [{ role: 'user' as const, content: 'Hello', id: 'm1', timestamp: '2026-06-21T10:00:00Z' }], updated_at: '2026-06-21T10:00:00Z', starred: false, pinned: false, message_count: 1 },
  { id: 'c2', name: 'Chat B', session_id: 'c2', created_at: '2026-06-20T10:00:00Z', messages: [{ role: 'user' as const, content: 'World', id: 'm2', timestamp: '2026-06-20T10:00:00Z' }], updated_at: '2026-06-20T10:00:00Z', starred: true, pinned: false, message_count: 1 },
  { id: 'c3', name: 'Chat C', session_id: 'c3', created_at: '2026-06-19T10:00:00Z', messages: [{ role: 'user' as const, content: 'Test', id: 'm3', timestamp: '2026-06-19T10:00:00Z' }], updated_at: '2026-06-19T10:00:00Z', starred: false, pinned: true, message_count: 1 },
]

function makeCtx(overrides: { conversations?: typeof convs; onLoad?: ReturnType<typeof vi.fn>; onNewChat?: ReturnType<typeof vi.fn> } = {}): ChatToolbarContextValue {
  const c = overrides.conversations ?? convs
  const onLoad = overrides.onLoad ?? vi.fn()
  const onNew = overrides.onNewChat ?? vi.fn()
  const sessionIdRef = { current: 's1' } as React.MutableRefObject<string>
  return {
    conversations: { conversations: c, sessionIdRef, onLoad, onStar: vi.fn(), onPin: vi.fn(), onNewChat: onNew },
    search: { query: '', onChange: vi.fn(), onClear: vi.fn(), matchIndex: 0, matchCount: 0, matchIds: [], onPrevMatch: vi.fn(), onNextMatch: vi.fn(), showMobile: false, setShowMobile: vi.fn() },
    model: { availableModels: [], current: '', loading: null, generating: false, infoMap: {}, downloadProgress: {}, onSelect: vi.fn() },
    soul: { souls: [], current: null, onSelect: vi.fn() },
    knowledge: { showing: false, count: 0, context: '', onToggle: vi.fn() },
    agent: { agents: [], current: null, onSelect: vi.fn() },
    localEngine: { modelUrl: '', useLocal: false, loading: false, archInfo: null, onToggle: vi.fn() },
    actions: { onVoiceMode: vi.fn(), onToggleTools: vi.fn(), onExportMarkdown: vi.fn(), onSystemPrompt: vi.fn(), hasMessages: false, messageCount: 0 },
    health: { status: 'ok', summary: '', modelLoaded: false, modelType: '' },
    sidebar: { open: false, onToggle: vi.fn(), onClose: vi.fn() },
  }
}

function renderWithCtx(overrides: Parameters<typeof makeCtx>[0] = {}) {
  return render(<ChatToolbarProvider value={makeCtx(overrides)}><ConversationsDropdown /></ChatToolbarProvider>)
}

describe('ConversationsDropdown', () => {
  afterEach(cleanup)

  it('renders trigger button with conversation count', () => {
    renderWithCtx()
    expect(screen.getByText('3')).toBeDefined()
  })

  it('shows dropdown content when trigger clicked', () => {
    renderWithCtx()
    fireEvent.click(screen.getByLabelText('Conversations'))
    const matches = screen.getAllByText('Conversations')
    expect(matches.length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('New')).toBeDefined()
  })

  it('shows Starred section for starred conversations', () => {
    renderWithCtx()
    fireEvent.click(screen.getByLabelText('Conversations'))
    expect(screen.getByText('Starred')).toBeDefined()
    expect(screen.getByText('Chat B')).toBeDefined()
  })

  it('shows Recent section for non-starred conversations', () => {
    renderWithCtx()
    fireEvent.click(screen.getByLabelText('Conversations'))
    expect(screen.getByText('Recent')).toBeDefined()
  })

  it('shows empty state when no conversations', () => {
    renderWithCtx({ conversations: [] })
    fireEvent.click(screen.getByLabelText('Conversations'))
    expect(screen.getByText('No conversations yet')).toBeDefined()
  })

  it('calls onLoad when a conversation row is clicked', () => {
    const onLoad = vi.fn()
    renderWithCtx({ onLoad })
    fireEvent.click(screen.getByLabelText('Conversations'))
    fireEvent.click(screen.getByText('Chat A'))
    expect(onLoad).toHaveBeenCalledWith('c1')
  })

  it('calls onNewChat when New button clicked', () => {
    const onNewChat = vi.fn()
    renderWithCtx({ onNewChat })
    fireEvent.click(screen.getByLabelText('Conversations'))
    fireEvent.click(screen.getByText('New'))
    expect(onNewChat).toHaveBeenCalled()
  })

  it('has View all messages link', () => {
    renderWithCtx()
    fireEvent.click(screen.getByLabelText('Conversations'))
    expect(screen.getByText('View all messages')).toBeDefined()
  })

  it('shows badge count when conversations exist', () => {
    renderWithCtx()
    const btn = screen.getByLabelText('Conversations')
    expect(btn.textContent).toContain('3')
  })
})
