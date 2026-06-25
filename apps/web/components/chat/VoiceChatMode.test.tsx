// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

vi.mock('@/components/ui/button', () => ({
  Button: ({ children, onClick, ...rest }: any) => (
    <button onClick={onClick} {...rest}>{children}</button>
  ),
}))

vi.mock('@/components/ui', () => ({
  IconX: () => <span data-testid="icon-x">x</span>,
  IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
}))

import { VoiceChatMode } from './VoiceChatMode'

function makeMockRecognition() {
  const handlers: Record<string, ((...args: any[]) => void) | null> = {
    start: null,
    end: null,
    error: null,
    result: null,
  }
  const instance = {
    continuous: false,
    interimResults: false,
    lang: '',
    start: vi.fn(() => {
      setTimeout(() => handlers.start?.())
    }),
    stop: vi.fn(() => {
      setTimeout(() => handlers.end?.())
    }),
    abort: vi.fn(),
    get onstart() { return handlers.start },
    set onstart(fn) { handlers.start = fn },
    get onend() { return handlers.end },
    set onend(fn) { handlers.end = fn },
    get onerror() { return handlers.error },
    set onerror(fn) { handlers.error = fn },
    get onresult() { return handlers.result },
    set onresult(fn) { handlers.result = fn },
  }
  return { instance, handlers }
}

function makeResultEvent(resultIndex = 0, results: Array<{ transcript: string; isFinal: boolean }>) {
  return {
    resultIndex,
    results: results.map((r, i) => ({
      isFinal: r.isFinal,
      0: { transcript: r.transcript, confidence: 0.9 },
      [i === 0 ? 'length' : '']: undefined,
    })),
    length: results.length,
  } as any
}

describe('VoiceChatMode', () => {
  const onMessage = vi.fn()
  const onClose = vi.fn()
  let mockRecognition: ReturnType<typeof makeMockRecognition>

  function setupSpeechRecognition() {
    mockRecognition = makeMockRecognition()
    const Ctor = vi.fn(() => mockRecognition.instance)
    ;(window as any).SpeechRecognition = Ctor
    return Ctor
  }

  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
    delete (window as any).SpeechRecognition
    delete (window as any).webkitSpeechRecognition
  })

  afterEach(() => {
    vi.useRealTimers()
    cleanup()
    delete (window as any).SpeechRecognition
    delete (window as any).webkitSpeechRecognition
  })

  it('shows error when SpeechRecognition not available', () => {
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    act(() => { vi.runAllTimers() })
    expect(screen.getByText(/Speech recognition not supported/)).toBeDefined()
  })

  it('renders close button', () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    expect(screen.getByLabelText('Exit voice mode')).toBeDefined()
  })

  it('shows listening state after recognition starts', () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    act(() => { vi.runAllTimers() })
    expect(screen.getByText('Listening...')).toBeDefined()
  })

  it('calls onClose when close button clicked', () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    fireEvent.click(screen.getByLabelText('Exit voice mode'))
    expect(onClose).toHaveBeenCalled()
  })

  it('shows interim text during recognition', () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    act(() => { vi.runAllTimers() })

    act(() => {
      mockRecognition.handlers.result?.(makeResultEvent(0, [
        { transcript: 'hello ', isFinal: false },
      ]))
    })

    expect(screen.getByText('hello')).toBeDefined()
  })

  it('accumulates final text', () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    act(() => { vi.runAllTimers() })

    act(() => {
      mockRecognition.handlers.result?.(makeResultEvent(0, [
        { transcript: 'hello world', isFinal: true },
      ]))
    })

    expect(screen.getByText('hello world')).toBeDefined()
  })

  it('stops listening when toggle button clicked while listening', () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    act(() => { vi.runAllTimers() })

    fireEvent.click(screen.getByLabelText('Tap to stop listening'))
    act(() => { vi.runAllTimers() })

    expect(mockRecognition.instance.stop).toHaveBeenCalled()
  })

  it('shows error for not-allowed', () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    act(() => { vi.runAllTimers() })

    act(() => {
      mockRecognition.handlers.error?.({ error: 'not-allowed' } as any)
    })

    expect(screen.getByText('Microphone access denied')).toBeDefined()
  })

  it('shows error for speech recognition errors', () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    act(() => { vi.runAllTimers() })

    act(() => {
      mockRecognition.handlers.error?.({ error: 'no-speech' } as any)
    })

    expect(screen.getByText('Speech error: no-speech')).toBeDefined()
  })

  it('shows instructions text', () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    expect(screen.getByText(/auto-sends after 2s of silence/)).toBeDefined()
  })
})
