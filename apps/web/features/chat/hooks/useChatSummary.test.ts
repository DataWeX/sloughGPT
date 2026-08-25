import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, cleanup } from '@testing-library/react'
import { useChatSummary } from './useChatSummary'
import { chatController } from '@/lib/chat-controller'

afterEach(cleanup)

vi.mock('@/lib/chat-controller', () => ({
  chatController: {
    send: vi.fn(),
  },
}))

const mockMessages = [
  { id: '1', role: 'user', content: 'Hello', timestamp: Date.now() },
  { id: '2', role: 'assistant', content: 'Hi there!', timestamp: Date.now() },
]

describe('useChatSummary', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('initializes with default state', () => {
    const { result } = renderHook(() => useChatSummary({ model: 'test-model' }))
    expect(result.current.summary).toBeNull()
    expect(result.current.isGenerating).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it('generates summary successfully', async () => {
    vi.mocked(chatController.send).mockResolvedValue({ message: 'Test summary', session_id: 'test', done: true })
    const { result } = renderHook(() => useChatSummary({ model: 'test-model' }))

    await act(async () => {
      await result.current.generateSummary(mockMessages)
    })

    expect(result.current.summary).toBe('Test summary')
    expect(result.current.isGenerating).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it('handles empty messages', async () => {
    const { result } = renderHook(() => useChatSummary({ model: 'test-model' }))

    await act(async () => {
      await result.current.generateSummary([])
    })

    expect(result.current.error).toBe('No messages to summarize')
    expect(result.current.summary).toBeNull()
  })

  it('handles errors', async () => {
    vi.mocked(chatController.send).mockRejectedValue(new Error('API error'))
    const { result } = renderHook(() => useChatSummary({ model: 'test-model' }))

    await act(async () => {
      await result.current.generateSummary(mockMessages)
    })

    expect(result.current.error).toBe('API error')
    expect(result.current.summary).toBeNull()
  })

  it('clears summary', async () => {
    vi.mocked(chatController.send).mockResolvedValue({ message: 'Test summary', session_id: 'test', done: true })
    const { result } = renderHook(() => useChatSummary({ model: 'test-model' }))

    await act(async () => {
      await result.current.generateSummary(mockMessages)
    })

    expect(result.current.summary).toBe('Test summary')

    act(() => {
      result.current.clearSummary()
    })

    expect(result.current.summary).toBeNull()
    expect(result.current.error).toBeNull()
  })
})