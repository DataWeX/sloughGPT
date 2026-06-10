import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useChatStore } from '@/stores/chat-store'

// Mock API client and SSE
vi.mock('@/lib/api-client', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
  apiPut: vi.fn(),
  apiDelete: vi.fn(),
}))

vi.mock('@/lib/sse-client', () => ({
  streamSSE: vi.fn(),
  createAbortController: () => ({
    abort: vi.fn(),
    signal: {},
  }),
}))

import { apiGet, apiPost, apiDelete } from '@/lib/api-client'
import { streamSSE } from '@/lib/sse-client'

describe('Chat Store', () => {
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

  it('should initialize with default state', () => {
    const state = useChatStore.getState()
    expect(state.sessions).toEqual([])
    expect(state.activeSessionId).toBeNull()
    expect(state.messages).toEqual([])
    expect(state.streaming).toBe(false)
  })

  it('should refresh sessions', async () => {
    const mockSessions = {
      sessions: [
        { id: '1', title: 'Chat 1', messages: [], updated_at: '2024-01-01' },
        { id: '2', title: 'Chat 2', messages: [], updated_at: '2024-01-02' },
      ],
    }
    vi.mocked(apiGet).mockResolvedValue(mockSessions)

    await useChatStore.getState().refreshSessions()

    expect(useChatStore.getState().sessions).toEqual(mockSessions.sessions)
  })

  it('should create session', async () => {
    vi.mocked(apiPost).mockResolvedValue({ session_id: 'new-session-123' })
    vi.mocked(apiGet).mockResolvedValue({ sessions: [] })

    const sessionId = await useChatStore.getState().createSession()

    expect(sessionId).toBe('new-session-123')
    expect(useChatStore.getState().activeSessionId).toBe('new-session-123')
    expect(useChatStore.getState().messages).toEqual([])
  })

  it('should load session', async () => {
    const mockMessages = [
      { id: '1', role: 'user', content: 'Hello', timestamp: '2024-01-01' },
      { id: '2', role: 'assistant', content: 'Hi!', timestamp: '2024-01-01' },
    ]
    vi.mocked(apiGet).mockResolvedValue({ messages: mockMessages })

    await useChatStore.getState().loadSession('session-123')

    expect(useChatStore.getState().activeSessionId).toBe('session-123')
    expect(useChatStore.getState().messages).toEqual(mockMessages)
  })

  it('should delete session', async () => {
    useChatStore.setState({ activeSessionId: 'session-123' })
    vi.mocked(apiDelete).mockResolvedValue({ status: 'deleted' })
    vi.mocked(apiGet).mockResolvedValue({ sessions: [] })

    await useChatStore.getState().deleteSession('session-123')

    expect(apiDelete).toHaveBeenCalledWith('/chat/sessions/session-123')
    expect(useChatStore.getState().activeSessionId).toBeNull()
  })

  it('should send message and stream response', async () => {
    useChatStore.setState({ activeSessionId: 'session-123' })
    vi.mocked(apiPost).mockResolvedValue({ status: 'stored' })

    // Mock SSE stream
    const mockStream = async function* () {
      yield { token: 'Hello', done: false }
      yield { token: ' world', done: false }
      yield { token: '', done: true }
    }
    vi.mocked(streamSSE).mockReturnValue(mockStream())

    await useChatStore.getState().sendMessage('Hi there')

    const state = useChatStore.getState()
    expect(state.messages.length).toBe(2) // user + assistant
    expect(state.messages[0].role).toBe('user')
    expect(state.messages[0].content).toBe('Hi there')
    expect(state.messages[1].role).toBe('assistant')
    expect(state.messages[1].content).toBe('Hello world')
    expect(state.streaming).toBe(false)
  })

  it('should handle streaming error', async () => {
    useChatStore.setState({ activeSessionId: 'session-123' })

    const mockStream = async function* () {
      yield { token: '', done: true, error: 'Stream failed' }
    }
    vi.mocked(streamSSE).mockReturnValue(mockStream())

    await useChatStore.getState().sendMessage('Test')

    expect(useChatStore.getState().error).toBe('Stream failed')
    expect(useChatStore.getState().streaming).toBe(false)
  })

  it('should record feedback', async () => {
    useChatStore.setState({ activeSessionId: 'session-123' })
    vi.mocked(apiPost).mockResolvedValue({ status: 'recorded' })

    await useChatStore.getState().recordFeedback('msg-123', true)

    expect(apiPost).toHaveBeenCalledWith('/feedback/workflow-record', {
      session_id: 'session-123',
      message_id: 'msg-123',
      positive: true,
    })
  })

  it('should clear error', () => {
    useChatStore.setState({ error: 'Test error' })
    useChatStore.getState().clearError()
    expect(useChatStore.getState().error).toBeNull()
  })
})
