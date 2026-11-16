import { describe, it, expect, vi } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useChatMode } from './useChatMode'
import type { ChatMessage } from '@/lib/chat-utils'

const { mockAddToast } = vi.hoisted(() => ({
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/notify', () => ({
  notify: { toast: mockAddToast, banner: vi.fn() },
  useToastStore: { getState: () => ({ addToast: mockAddToast }) },
}))

function makeChat(overrides?: {
  input?: string
  setInput?: any
  sendMessage?: any
  setMessages?: any
  setLoading?: any
}) {
  const messages: ChatMessage[] = []
  return {
    input: '',
    setInput: vi.fn((v: string | ((p: string) => string)) => {
      if (typeof v === 'function') v('')
    }),
    sendMessage: vi.fn().mockResolvedValue(undefined),
    setMessages: vi.fn((v: any) => {
      if (typeof v === 'function') v(messages)
    }),
    setLoading: vi.fn(),
    ...overrides,
  }
}

describe('useChatMode', () => {
  it('exposes only handleSend — no mode state leaks into the composer', () => {
    const { result } = renderHook(() => useChatMode({ chat: makeChat() }))
    expect(Object.keys(result.current)).toEqual(['handleSend'])
  })

  describe('handleSend (plain)', () => {
    it('sends the raw input with no prompt transform', async () => {
      const chat = makeChat()
      chat.input = 'Write a sonnet about the sea'
      const { result } = renderHook(() => useChatMode({ chat }))
      await act(async () => {
        await result.current.handleSend()
      })
      // No mode decoration: sendMessage() reads input as-is.
      expect(chat.sendMessage).toHaveBeenCalledWith()
      expect(chat.setInput).not.toHaveBeenCalled()
    })

    it('never rewrites the text into a decorated prompt', async () => {
      const chat = makeChat()
      chat.input = 'hello'
      const { result } = renderHook(() => useChatMode({ chat }))
      await act(async () => {
        await result.current.handleSend()
      })
      const arg = chat.sendMessage.mock.calls[0][0]
      expect(arg).toBeUndefined()
    })
  })

  describe('handleSend (file attach)', () => {
    it('injects file content with the question when a file is attached', async () => {
      const chat = makeChat()
      chat.input = 'Summarize this'
      const { result } = renderHook(() => useChatMode({ chat }))
      await act(async () => {
        await result.current.handleSend({ text: 'File content here', filename: 'doc.txt' })
      })
      expect(chat.sendMessage).toHaveBeenCalledWith(expect.stringContaining('doc.txt'))
      expect(chat.sendMessage).toHaveBeenCalledWith(expect.stringContaining('File content here'))
      expect(chat.sendMessage).toHaveBeenCalledWith(expect.stringContaining('Summarize this'))
      expect(chat.setInput).toHaveBeenCalledWith('')
    })

    it('truncates large files to 12000 chars', async () => {
      const chat = makeChat()
      chat.input = 'q'
      const { result } = renderHook(() => useChatMode({ chat }))
      await act(async () => {
        await result.current.handleSend({ text: 'x'.repeat(50000), filename: 'big.txt' })
      })
      const sent = chat.sendMessage.mock.calls[0][0] as string
      expect(sent).toContain('x'.repeat(12000))
      expect(sent).not.toContain('x'.repeat(12001))
    })

    it('toasts when a file is attached but no question was typed', async () => {
      const chat = makeChat()
      chat.input = '   '
      const { result } = renderHook(() => useChatMode({ chat }))
      await act(async () => {
        await result.current.handleSend({ text: 'content', filename: 'doc.txt' })
      })
      expect(mockAddToast).toHaveBeenCalledWith('Ask a question about your file', { type: 'info' })
      expect(chat.sendMessage).not.toHaveBeenCalled()
    })
  })
})
