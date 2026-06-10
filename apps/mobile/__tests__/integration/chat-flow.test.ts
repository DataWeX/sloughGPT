import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useChatStore } from '@/stores/chat-store'

// Mock dependencies
vi.mock('@/lib/api-client', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
}))

vi.mock('@/lib/sse-client', () => ({
  streamSSE: vi.fn(),
  createAbortController: () => ({
    abort: vi.fn(),
    signal: {},
  }),
}))

import { apiPost } from '@/lib/api-client'
import { streamSSE } from '@/lib/sse-client'

describe('Chat Flow Integration', () => {
  beforeEach(() => {
    useChatStore.setState({
      sessions: [],
      activeSessionId: null,
      messages: [],
      streaming: false,
      streamBuffer: '',
      error: null,
    })
    vi.clearAllMocks()
  })

  it('should create session and send message in sequence', async () => {
    // Mock session creation
    vi.mocked(apiPost).mockResolvedValueOnce({ session_id: 'new-session' })
    vi.mocked(apiPost).mockResolvedValueOnce({ status: 'stored' })

    // Mock SSE stream
    const mockStream = async function* () {
      yield { token: 'Response', done: false }
      yield { token: '', done: true }
    }
    vi.mocked(streamSSE).mockReturnValue(mockStream())

    // Create session
    await useChatStore.getState().createSession()
    expect(useChatStore.getState().activeSessionId).toBe('new-session')

    // Send message
    await useChatStore.getState().sendMessage('Hello')

    const state = useChatStore.getState()
    expect(state.messages.length).toBe(2)
    expect(state.messages[0].content).toBe('Hello')
    expect(state.messages[1].content).toBe('Response')
  })

  it('should handle rapid message sends', async () => {
    useChatStore.setState({ activeSessionId: 'session-123' })
    vi.mocked(apiPost).mockResolvedValue({ status: 'stored' })

    const mockStream = async function* () {
      yield { token: 'Reply', done: false }
      yield { token: '', done: true }
    }
    vi.mocked(streamSSE).mockReturnValue(mockStream())

    await useChatStore.getState().sendMessage('Message 1')
    await useChatStore.getState().sendMessage('Message 2')

    expect(useChatStore.getState().messages.length).toBe(4)
  })

  it('should maintain message history across session switches', async () => {
    // Load first session
    vi.mocked(apiPost).mockResolvedValue({ status: 'ok' })
    const mockStream1 = async function* () {
      yield { token: 'Session 1', done: false }
      yield { token: '', done: true }
    }
    vi.mocked(streamSSE).mockReturnValue(mockStream1())

    useChatStore.setState({ activeSessionId: 'session-1' })
    await useChatStore.getState().sendMessage('Hello')

    const session1Messages = [...useChatStore.getState().messages]

    // Switch to second session
    useChatStore.setState({ activeSessionId: 'session-2', messages: [] })
    const mockStream2 = async function* () {
      yield { token: 'Session 2', done: false }
      yield { token: '', done: true }
    }
    vi.mocked(streamSSE).mockReturnValue(mockStream2())

    await useChatStore.getState().sendMessage('Hi')

    expect(useChatStore.getState().messages.length).toBe(2)
    expect(useChatStore.getState().messages[1].content).toBe('Session 2')
    expect(session1Messages[1].content).toBe('Session 1')
  })
})
