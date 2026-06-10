import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useModelStore } from '@/stores/model-store'
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

import { apiGet, apiPost } from '@/lib/api-client'
import { streamSSE } from '@/lib/sse-client'

describe('Model and Chat Integration', () => {
  beforeEach(() => {
    useModelStore.setState({
      models: [],
      currentModel: null,
      souls: [],
      currentSoul: null,
      loading: false,
      error: null,
    })
    useChatStore.setState({
      sessions: [],
      activeSessionId: null,
      messages: [],
      streaming: false,
      error: null,
    })
    vi.clearAllMocks()
  })

  it('should load model and start chat', async () => {
    // Mock model loading
    vi.mocked(apiPost).mockResolvedValueOnce({ status: 'loaded' })
    vi.mocked(apiGet).mockResolvedValue([])

    // Mock chat
    vi.mocked(apiPost).mockResolvedValueOnce({ session_id: 'session-123' })
    const mockStream = async function* () {
      yield { token: 'Hello from model', done: false }
      yield { token: '', done: true }
    }
    vi.mocked(streamSSE).mockReturnValue(mockStream())

    // Load model
    await useModelStore.getState().loadModel('gpt-2')

    // Create session and send message
    await useChatStore.getState().createSession()
    await useChatStore.getState().sendMessage('Test')

    expect(useChatStore.getState().messages.length).toBe(2)
    expect(useChatStore.getState().messages[1].content).toBe('Hello from model')
  })

  it('should switch soul mid-conversation', async () => {
    useChatStore.setState({ activeSessionId: 'session-123' })

    // Mock soul switch
    vi.mocked(apiPost).mockResolvedValueOnce({ status: 'ok' })
    vi.mocked(apiGet).mockResolvedValue([])

    // Mock chat responses
    const mockStream1 = async function* () {
      yield { token: 'Default response', done: false }
      yield { token: '', done: true }
    }
    const mockStream2 = async function* () {
      yield { token: 'Creative response', done: false }
      yield { token: '', done: true }
    }
    vi.mocked(streamSSE)
      .mockReturnValueOnce(mockStream1())
      .mockReturnValueOnce(mockStream2())

    // Send message with default soul
    await useChatStore.getState().sendMessage('Hello')

    // Switch soul
    await useModelStore.getState().switchSoul('Creative')

    // Send message with new soul
    vi.mocked(apiPost).mockResolvedValueOnce({ status: 'stored' })
    await useChatStore.getState().sendMessage('Hello again')

    expect(useChatStore.getState().messages.length).toBe(4)
    expect(useChatStore.getState().messages[1].content).toBe('Default response')
    expect(useChatStore.getState().messages[3].content).toBe('Creative response')
  })
})
