import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, cleanup } from '@testing-library/react'
import { useChatSummary } from './useChatSummary'
import { chatController } from '@/lib/chat-controller'

afterEach(cleanup)

vi.mock('@/lib/chat-controller', () => ({
  chatController: {
    stream: vi.fn(),
  },
}))

const mockMessages = [
  { id: '1', role: 'user' as const, content: 'Hello', timestamp: new Date() },
  { id: '2', role: 'assistant' as const, content: 'Hi there!', timestamp: new Date() },
]

function mockStream(...tokens: string[]) {
  vi.mocked(chatController.stream).mockImplementation(async function* () {
    for (const t of tokens) yield t
  })
}

describe('useChatSummary', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('initializes with default state', () => {
    const { result } = renderHook(() => useChatSummary())
    expect(result.current.summary).toBeNull()
    expect(result.current.isGenerating).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it('generates summary by folding the chat stream', async () => {
    mockStream('Test ', 'summary')
    const { result } = renderHook(() => useChatSummary())

    await act(async () => {
      await result.current.generateSummary(mockMessages)
    })

    expect(chatController.stream).toHaveBeenCalledWith(expect.any(String), {
      temperature: 0.3,
      max_tokens: 500,
    })
    expect(result.current.summary).toBe('Test summary')
    expect(result.current.isGenerating).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it('handles empty messages', async () => {
    const { result } = renderHook(() => useChatSummary())

    await act(async () => {
      await result.current.generateSummary([])
    })

    expect(result.current.error).toBe('No messages to summarize')
    expect(result.current.summary).toBeNull()
  })

  it('handles stream exceptions', async () => {
    vi.mocked(chatController.stream).mockImplementation(() => {
      throw new Error('API error')
    })
    const { result } = renderHook(() => useChatSummary())

    await act(async () => {
      await result.current.generateSummary(mockMessages)
    })

    expect(result.current.error).toBe('API error')
    expect(result.current.summary).toBeNull()
  })

  it("treats stream()'s bracketed sentinel as an error, not content", async () => {
    mockStream('[No model loaded]')
    const { result } = renderHook(() => useChatSummary())

    await act(async () => {
      await result.current.generateSummary(mockMessages)
    })

    expect(result.current.error).toBe('No model loaded')
    expect(result.current.summary).toBeNull()
  })

  it('clears summary', async () => {
    mockStream('Test summary')
    const { result } = renderHook(() => useChatSummary())

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
