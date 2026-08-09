// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import React from 'react'
import { ChatSessionStatsCard } from './ChatSessionStatsCard'
import { chatDB } from '@/lib/db'

afterEach(() => { cleanup() })

vi.mock('@/lib/db', () => ({
  chatDB: {
    loadSessions: vi.fn(),
  },
}))

const mockLoadSessions = vi.mocked(chatDB.loadSessions)

describe('ChatSessionStatsCard', () => {
  beforeEach(() => {
    mockLoadSessions.mockReset()
  })

  it('returns null for no sessions', async () => {
    mockLoadSessions.mockResolvedValue([])
    const { container } = render(<ChatSessionStatsCard sessionId="s1" />)
    await waitFor(() => {
      expect(container.querySelector('[data-testid="chat-session-stats"]')).toBeNull()
    })
  })

  it('renders card with sessions', async () => {
    mockLoadSessions.mockResolvedValue([
      { id: '1', name: 'Chat 1', messages: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'hello' }], createdAt: Date.now(), updatedAt: Date.now() },
      { id: '2', name: 'Chat 2', messages: [{ role: 'user', content: 'hey' }], createdAt: Date.now(), updatedAt: Date.now() },
    ] as any)
    render(<ChatSessionStatsCard sessionId="s1" />)
    await waitFor(() => {
      expect(screen.getAllByTestId('chat-session-stats').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getAllByText('Session Stats').length).toBeGreaterThanOrEqual(1)
  })

  it('shows session count', async () => {
    mockLoadSessions.mockResolvedValue([
      { id: '1', name: 'Chat 1', messages: [{ role: 'user', content: 'hi' }], createdAt: Date.now(), updatedAt: Date.now() },
      { id: '2', name: 'Chat 2', messages: [{ role: 'user', content: 'hey' }], createdAt: Date.now(), updatedAt: Date.now() },
    ] as any)
    render(<ChatSessionStatsCard sessionId="s1" />)
    await waitFor(() => {
      expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows message count', async () => {
    mockLoadSessions.mockResolvedValue([
      { id: '1', name: 'Chat 1', messages: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'hello' }, { role: 'user', content: 'bye' }], createdAt: Date.now(), updatedAt: Date.now() },
    ] as any)
    render(<ChatSessionStatsCard sessionId="s1" />)
    await waitFor(() => {
      expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows avg messages per session', async () => {
    mockLoadSessions.mockResolvedValue([
      { id: '1', name: 'Chat 1', messages: [{ role: 'user', content: 'a' }, { role: 'user', content: 'b' }], createdAt: Date.now(), updatedAt: Date.now() },
      { id: '2', name: 'Chat 2', messages: [{ role: 'user', content: 'c' }], createdAt: Date.now(), updatedAt: Date.now() },
    ] as any)
    render(<ChatSessionStatsCard sessionId="s1" />)
    await waitFor(() => {
      expect(screen.getAllByText('1.5').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('handles load error gracefully', async () => {
    mockLoadSessions.mockRejectedValue(new Error('fail'))
    const { container } = render(<ChatSessionStatsCard sessionId="s1" />)
    await waitFor(() => {
      expect(container.querySelector('[data-testid="chat-session-stats"]')).toBeNull()
    })
  })
})
