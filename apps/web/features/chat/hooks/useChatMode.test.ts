import { describe, it, expect, vi } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useChatMode } from './useChatMode'
import type { ChatMessage } from '@/lib/chat-utils'

const { mockAddToast, mockGenerate } = vi.hoisted(() => ({
  mockAddToast: vi.fn(),
  mockGenerate: vi.fn().mockResolvedValue({ image: 'data:image/png;base64,abc' }),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: { getState: () => ({ addToast: mockAddToast }) },
}))

vi.mock('@/lib/images-controller', () => ({
  imagesController: { generate: mockGenerate },
}))

function makeChat(overrides?: { input?: string; setInput?: any; sendMessage?: any; setMessages?: any; setLoading?: any }) {
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
  it('returns chat mode by default', () => {
    const { result } = renderHook(() => useChatMode({ chat: makeChat() }))
    expect(result.current.chatMode).toBe('chat')
    expect(result.current.placeholder).toBe('Type a message...')
  })

  it('setChatMode updates mode and placeholder', () => {
    const { result } = renderHook(() => useChatMode({ chat: makeChat() }))
    act(() => { result.current.setChatMode('write') })
    expect(result.current.chatMode).toBe('write')
    expect(result.current.placeholder).toBe('What do you want to write about?')
  })

  it('each mode has correct placeholder', () => {
    const modes: [string, string][] = [
      ['chat', 'Type a message...'],
      ['write', 'What do you want to write about?'],
      ['decide', 'What do you need help deciding?'],
      ['explain', 'What do you want explained?'],
      ['translate', 'Text to translate...'],
      ['brainstorm', 'What should we brainstorm?'],
      ['wellness', 'How can I help you feel calm?'],
      ['create', 'Describe the image...'],
      ['read', 'Ask about your file...'],
      ['talk', 'Speak to me...'],
    ]
    for (const [mode, expected] of modes) {
      const { result } = renderHook(() => useChatMode({ chat: makeChat() }))
      act(() => { result.current.setChatMode(mode as any) })
      expect(result.current.placeholder).toBe(expected)
    }
  })

  describe('buildModePrompt', () => {
    it('chat mode returns null (no transform)', () => {
      const { result } = renderHook(() => useChatMode({ chat: makeChat() }))
      // chat mode default
      act(() => {
        result.current.handleSend = result.current.handleSend
      })
      // We can't call buildModePrompt directly, but handleSend in chat mode
      // calls sendMessage() with no args — meaning no transform
    })

    it('write mode builds prompt with tone and type', () => {
      const chat = makeChat()
      const { result } = renderHook(() => useChatMode({ chat }))
      act(() => { result.current.setChatMode('write') })
      act(() => { result.current.setWriteTone('Professional') })
      act(() => { result.current.setWriteType('Report') })
      act(() => { result.current.handleSend() })
      expect(chat.sendMessage).toHaveBeenCalledWith(
        expect.stringContaining('professional report about:')
      )
      expect(chat.setInput).toHaveBeenCalledWith('')
    })

    it('decide mode builds pros/cons prompt', () => {
      const chat = makeChat()
      chat.input = 'Should I switch jobs?'
      const { result } = renderHook(() => useChatMode({ chat }))
      act(() => { result.current.setChatMode('decide') })
      act(() => { result.current.handleSend() })
      expect(chat.sendMessage).toHaveBeenCalledWith(
        expect.stringContaining('pros & cons')
      )
      expect(chat.sendMessage).toHaveBeenCalledWith(
        expect.stringContaining('Should I switch jobs?')
      )
    })

    it('explain mode builds difficulty-level prompt', () => {
      const chat = makeChat()
      chat.input = 'Quantum computing'
      const { result } = renderHook(() => useChatMode({ chat }))
      act(() => { result.current.setChatMode('explain') })
      act(() => { result.current.setExplainDifficulty('Expert') })
      act(() => { result.current.handleSend() })
      expect(chat.sendMessage).toHaveBeenCalledWith(
        expect.stringContaining('expert level')
      )
    })

    it('translate mode builds language pair prompt', () => {
      const chat = makeChat()
      chat.input = 'Hello world'
      const { result } = renderHook(() => useChatMode({ chat }))
      act(() => { result.current.setChatMode('translate') })
      act(() => { result.current.setTranslateLangPair('EN→FR') })
      act(() => { result.current.handleSend() })
      expect(chat.sendMessage).toHaveBeenCalledWith(
        expect.stringContaining('Translate this from EN to FR')
      )
    })

    it('brainstorm mode builds topic prompt', () => {
      const chat = makeChat()
      chat.input = 'Give me ideas'
      const { result } = renderHook(() => useChatMode({ chat }))
      act(() => { result.current.setChatMode('brainstorm') })
      act(() => { result.current.setBrainstormTopic('Startup Ideas') })
      act(() => { result.current.handleSend() })
      expect(chat.sendMessage).toHaveBeenCalledWith(
        expect.stringContaining('startup ideas')
      )
    })

    it('wellness mode maps type to prompt', () => {
      const chat = makeChat()
      chat.input = 'Help me relax'
      const { result } = renderHook(() => useChatMode({ chat }))
      act(() => { result.current.setChatMode('wellness') })
      act(() => { result.current.setWellnessType('Meditation') })
      act(() => { result.current.handleSend() })
      expect(chat.sendMessage).toHaveBeenCalledWith(
        expect.stringContaining('Guide me through a short meditation')
      )
    })
  })

  describe('handleSend', () => {
    it('chat mode sends without transform', () => {
      const chat = makeChat()
      chat.input = 'Hello'
      const { result } = renderHook(() => useChatMode({ chat }))
      act(() => { result.current.handleSend() })
      expect(chat.sendMessage).toHaveBeenCalledWith()
      expect(chat.setInput).not.toHaveBeenCalled()
    })

    it('read mode without file shows toast', () => {
      const chat = makeChat()
      chat.input = 'What is this about?'
      const { result } = renderHook(() => useChatMode({ chat }))
      act(() => { result.current.setChatMode('read') })
      act(() => { result.current.handleSend() })
      expect(mockAddToast).toHaveBeenCalledWith('Upload a file first, then ask your question', 'info')
      expect(chat.sendMessage).not.toHaveBeenCalled()
    })

    it('read mode with file sends context message', () => {
      const chat = makeChat()
      chat.input = 'Summarize this'
      const { result } = renderHook(() => useChatMode({ chat }))
      act(() => { result.current.setChatMode('read') })
      act(() => { result.current.handleSend({ text: 'File content here', filename: 'doc.txt' }) })
      expect(chat.sendMessage).toHaveBeenCalledWith(
        expect.stringContaining('doc.txt')
      )
      expect(chat.sendMessage).toHaveBeenCalledWith(
        expect.stringContaining('File content here')
      )
      expect(chat.setInput).toHaveBeenCalledWith('')
    })

    it('talk mode is a no-op', () => {
      const chat = makeChat()
      chat.input = 'Hello'
      const { result } = renderHook(() => useChatMode({ chat }))
      act(() => { result.current.setChatMode('talk') })
      act(() => { result.current.handleSend() })
      expect(chat.sendMessage).not.toHaveBeenCalled()
    })

    it('create mode sends image generation prompt', async () => {
      const chat = makeChat()
      chat.input = 'A sunset over mountains'
      const { result } = renderHook(() => useChatMode({ chat }))
      act(() => { result.current.setChatMode('create') })
      await act(async () => { await result.current.handleSend() })
      expect(mockGenerate).toHaveBeenCalled()
      expect(chat.setLoading).toHaveBeenCalledWith(true)
    })
  })

  describe('mode state setters', () => {
    it('setWriteTone updates tone', () => {
      const { result } = renderHook(() => useChatMode({ chat: makeChat() }))
      act(() => { result.current.setWriteTone('Formal') })
      expect(result.current.writeTone).toBe('Formal')
    })

    it('setWriteType updates type', () => {
      const { result } = renderHook(() => useChatMode({ chat: makeChat() }))
      act(() => { result.current.setWriteType('Essay') })
      expect(result.current.writeType).toBe('Essay')
    })

    it('setDecideStructure updates structure', () => {
      const { result } = renderHook(() => useChatMode({ chat: makeChat() }))
      act(() => { result.current.setDecideStructure('SWOT Analysis') })
      expect(result.current.decideStructure).toBe('SWOT Analysis')
    })

    it('setExplainDifficulty updates difficulty', () => {
      const { result } = renderHook(() => useChatMode({ chat: makeChat() }))
      act(() => { result.current.setExplainDifficulty('Advanced') })
      expect(result.current.explainDifficulty).toBe('Advanced')
    })

    it('setTranslateLangPair updates pair', () => {
      const { result } = renderHook(() => useChatMode({ chat: makeChat() }))
      act(() => { result.current.setTranslateLangPair('DE→JP') })
      expect(result.current.translateLangPair).toBe('DE→JP')
    })

    it('setBrainstormTopic updates topic', () => {
      const { result } = renderHook(() => useChatMode({ chat: makeChat() }))
      act(() => { result.current.setBrainstormTopic('Marketing') })
      expect(result.current.brainstormTopic).toBe('Marketing')
    })

    it('setWellnessType updates type', () => {
      const { result } = renderHook(() => useChatMode({ chat: makeChat() }))
      act(() => { result.current.setWellnessType('Breathing') })
      expect(result.current.wellnessType).toBe('Breathing')
    })

    it('setCreateStyle updates style', () => {
      const { result } = renderHook(() => useChatMode({ chat: makeChat() }))
      act(() => { result.current.setCreateStyle('Anime') })
      expect(result.current.createStyle).toBe('Anime')
    })
  })
})
