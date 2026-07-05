import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

// ── Set up hoisted mocks first ──────────────────────────────────────────────
const { mockGetErrorInfo, mockGenerateSessionId, mockGetOrCreateUserId, mockStreamChatResponse } = vi.hoisted(() => ({
  mockGetErrorInfo: vi.fn((): any => null),
  mockGenerateSessionId: vi.fn(() => 'test-session-id'),
  mockGetOrCreateUserId: vi.fn(() => 'test-user'),
  mockStreamChatResponse: vi.fn(),
}))

vi.mock('@/components/chat/ErrorBanner', () => ({ getErrorInfo: mockGetErrorInfo }))

vi.mock('@/lib/chat-utils', () => ({
  cleanStreamedContent: (s: string) => s,
  stripAssistantPrefix: (s: string) => s,
  getOrCreateUserId: mockGetOrCreateUserId,
  generateSessionId: mockGenerateSessionId,
  CURRENT_SESSION_KEY: 'man_current_session',
  buildLocalPrompt: vi.fn(() => 'prompt'),
  exportConversationAsMarkdown: vi.fn(),
  copyConversationAsMarkdown: vi.fn(),
}))

vi.mock('@/lib/stream-chat-response', () => ({
  streamChatResponse: mockStreamChatResponse,
}))

vi.mock('@/lib/chat-controller', () => ({
  chatController: { saveSessionContext: vi.fn(), regenerateStream: vi.fn() },
}))

vi.mock('@/lib/knowledge-controller', () => ({
  knowledgeController: { context: vi.fn().mockResolvedValue({ count: 0, context: '' }) },
}))

vi.mock('@/lib/multimodal-controller', () => ({
  multimodalController: { trainImage: vi.fn().mockResolvedValue({ caption: 'test' }), getCapabilities: vi.fn().mockResolvedValue({}), getTrainingReport: vi.fn().mockResolvedValue({}) },
}))

vi.mock('@/lib/db', () => ({ chatDB: { loadSessions: vi.fn().mockResolvedValue([]), saveSession: vi.fn().mockResolvedValue(undefined), loadSession: vi.fn().mockResolvedValue(null) } }))

vi.mock('@/lib/error-store', () => ({ useErrorStore: { getState: vi.fn(() => ({ addError: vi.fn() })) } }))

vi.mock('@/lib/dev-log', () => ({ devDebug: vi.fn() }))



import { useChatMessages } from './useChatMessages'

function makeConfig(overrides = {}) {
  return {
    model: 'gpt2', temperature: 0.8, maxTokens: 200,
    currentSoul: null, currentAgent: null,
    useLocalEngine: false,
    engineRef: { current: null },
    engineLoadingRef: { current: false },
    initLocalEngine: vi.fn().mockResolvedValue(false),
    showToast: vi.fn(),
    recordFeedback: vi.fn().mockResolvedValue(true),
    fetchStats: vi.fn(),
    fetchAdapterStats: vi.fn(),
    onVisionUpdate: vi.fn(),
    onKnowledgeUpdate: vi.fn(),
    ...overrides,
  }
}

describe('useChatMessages', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    mockGenerateSessionId.mockReturnValue('test-session-id')
  })

  afterEach(() => {
    localStorage.clear()
  })

  it('returns default state', () => {
    const { result } = renderHook(() => useChatMessages(makeConfig()))
    expect(result.current.messages).toEqual([])
    expect(result.current.input).toBe('')
    expect(result.current.loading).toBe(false)
  })

  it('initializes session ID from localStorage', () => {
    localStorage.setItem('man_current_session', 'existing-session')
    const { result } = renderHook(() => useChatMessages(makeConfig()))
    expect(result.current.sessionIdRef.current).toBe('existing-session')
  })

  it('generates new session ID when none exists', () => {
    const { result } = renderHook(() => useChatMessages(makeConfig()))
    expect(result.current.sessionIdRef.current).toBe('test-session-id')
    expect(localStorage.getItem('man_current_session')).toBe('test-session-id')
  })

  it('newChat resets state', () => {
    const { result } = renderHook(() => useChatMessages(makeConfig()))
    act(() => { result.current.setMessages([{ id: '1', role: 'user', content: 'hi', timestamp: new Date() }]) })
    act(() => { result.current.setInput('hello') })
    act(() => { result.current.newChat() })
    expect(result.current.messages).toEqual([])
    expect(result.current.input).toBe('')
  })

  it('handleCopy shows toast', async () => {
    const showToast = vi.fn()
    const { result } = renderHook(() => useChatMessages(makeConfig({ showToast })))
    act(() => { result.current.handleCopy() })
    expect(showToast).toHaveBeenCalledWith('Copied to clipboard')
  })

  it('sendMessage does nothing when no text and no images', async () => {
    const { result } = renderHook(() => useChatMessages(makeConfig()))
    await act(async () => { await result.current.sendMessage() })
    expect(result.current.messages).toEqual([])
  })

  it('sendMessage adds user and assistant messages', async () => {
    const showToast = vi.fn()
    mockStreamChatResponse.mockImplementation(({ onComplete }) => {
      onComplete?.()
    })
    const { result } = renderHook(() => useChatMessages(makeConfig({ showToast })))
    act(() => { result.current.setInput('Hello') })
    await act(async () => { await result.current.sendMessage() })
    expect(result.current.messages.length).toBe(2)
    expect(result.current.messages[0].role).toBe('user')
    expect(result.current.messages[0].content).toBe('Hello')
    expect(result.current.messages[1].role).toBe('assistant')
  })

  it('handleSuggestionClick sends suggestion', async () => {
    mockStreamChatResponse.mockImplementation(({ onComplete }) => { onComplete?.() })
    const { result } = renderHook(() => useChatMessages(makeConfig()))
    await act(async () => { result.current.handleSuggestionClick('Tell me a joke') })
    expect(result.current.messages.length).toBe(2)
    expect(result.current.messages[0].content).toBe('Tell me a joke')
  })

  it('handleRetry resends last user message', async () => {
    mockStreamChatResponse.mockImplementation(({ onComplete }) => { onComplete?.() })
    const { result } = renderHook(() => useChatMessages(makeConfig()))
    act(() => { result.current.setMessages([
      { id: '1', role: 'user', content: 'Hi', timestamp: new Date() },
      { id: '2', role: 'assistant', content: 'Hello!', timestamp: new Date() },
    ])})
    await act(async () => { result.current.handleRetry() })
    expect(result.current.messages.length).toBe(4)
    expect(result.current.messages[2].content).toBe('Hi')
  })

  it('handleEditMessage replaces content from that point', async () => {
    mockStreamChatResponse.mockImplementation(({ onComplete }) => { onComplete?.() })
    const { result } = renderHook(() => useChatMessages(makeConfig()))
    act(() => { result.current.setMessages([
      { id: '1', role: 'user', content: 'Hi', timestamp: new Date() },
      { id: '2', role: 'assistant', content: 'Hello', timestamp: new Date() },
    ])})
    await act(async () => { result.current.handleEditMessage('2', 'Edited') })
    await new Promise(r => setTimeout(r, 50))
    expect(result.current.messages.length).toBeLessThan(4)
    expect(result.current.messages[0].content).toBe('Hi')
  })

  it('handleThumbsUp records feedback', async () => {
    const recordFeedback = vi.fn().mockResolvedValue(true)
    const showToast = vi.fn()
    const { result } = renderHook(() => useChatMessages(makeConfig({ recordFeedback, showToast })))
    act(() => { result.current.setMessages([
      { id: '1', role: 'user', content: 'Hi', timestamp: new Date() },
      { id: '2', role: 'assistant', content: 'Hello', timestamp: new Date() },
    ])})
    await act(async () => { result.current.handleThumbsUp('2') })
    expect(recordFeedback).toHaveBeenCalledWith(expect.objectContaining({ rating: 'thumbs_up', userMessage: 'Hi', assistantResponse: 'Hello' }))
    expect(showToast).toHaveBeenCalledWith('Thanks for the feedback!', 'success')
  })

  it('handleThumbsDown records feedback', async () => {
    const recordFeedback = vi.fn().mockResolvedValue(true)
    const showToast = vi.fn()
    const { result } = renderHook(() => useChatMessages(makeConfig({ recordFeedback, showToast })))
    act(() => { result.current.setMessages([
      { id: '1', role: 'user', content: 'Bad', timestamp: new Date() },
      { id: '2', role: 'assistant', content: 'Sorry', timestamp: new Date() },
    ])})
    await act(async () => { result.current.handleThumbsDown('2') })
    expect(recordFeedback).toHaveBeenCalledWith(expect.objectContaining({ rating: 'thumbs_down', userMessage: 'Bad', assistantResponse: 'Sorry' }))
  })

  it('handleAddImage calls multimodalController', async () => {
    const { result } = renderHook(() => useChatMessages(makeConfig()))
    await act(async () => { result.current.handleAddImage('data:image/png;base64,abc') })
    expect(result.current.images).toHaveLength(1)
  })

  it('handleRemoveImage removes by id', async () => {
    const { result } = renderHook(() => useChatMessages(makeConfig()))
    await act(async () => { result.current.handleAddImage('data:image/png;base64,abc') })
    expect(result.current.images).toHaveLength(1)
    act(() => { result.current.handleRemoveImage(result.current.images[0].id) })
    expect(result.current.images).toEqual([])
  })

  it('handles stream error', async () => {
    mockGetErrorInfo.mockReturnValue({ type: 'error', message: 'Server error' })
    mockStreamChatResponse.mockImplementation(({ onError }) => { onError?.(500, 'Server error') })
    const { result } = renderHook(() => useChatMessages(makeConfig()))
    act(() => { result.current.setInput('Hi') })
    await act(async () => { await result.current.sendMessage() })
    expect(result.current.currentError).toBeTruthy()
  })
})
