import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => {
  function SelectTrigger({ children, ...props }: any) {
    return <button {...props}>{children}</button>
  }
  return {
    Button: ({ children, onClick, ...rest }: any) => (
      <button onClick={onClick} {...rest}>{children}</button>
    ),
    IconX: () => <span data-testid="icon-x">x</span>,
    IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
    IconSettings: () => <span data-testid="icon-settings">settings</span>,
    Slider: ({ value, onValueChange, min, max, step }: any) => (
      <input type="range" value={value?.[0]} onChange={(e) => onValueChange?.([Number(e.target.value)])} min={min} max={max} step={step} data-testid="slider" />
    ),
    Select: ({ children }: any) => <div data-testid="select">{children}</div>,
    SelectTrigger,
    SelectValue: ({ placeholder }: any) => <span data-testid="select-value">{placeholder}</span>,
    SelectContent: ({ children }: any) => <div data-testid="select-content">{children}</div>,
    SelectItem: ({ children, value }: any) => <button data-testid="select-item" data-value={value}>{children}</button>,
  }
})

import { VoiceChatMode } from './VoiceChatMode'

function makeMockRecognition() {
  const handlers: Record<string, ((...args: any[]) => void) | null> = {
    start: null, end: null, error: null, result: null,
  }
  const instance = {
    continuous: false, interimResults: false, lang: '',
    start: vi.fn(() => setTimeout(() => handlers.start?.())),
    stop: vi.fn(() => setTimeout(() => handlers.end?.())),
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
    // Mock getUserMedia for audio level monitoring
    ;(navigator as any).mediaDevices = {
      getUserMedia: vi.fn().mockResolvedValue({
        getTracks: () => [{ stop: vi.fn() }],
      }),
    }
    // Mock AudioContext for waveform
    const mockDisconnect = vi.fn()
    ;(window as any).AudioContext = vi.fn(() => ({
      createAnalyser: () => ({
        fftSize: 256, smoothingTimeConstant: 0.4,
        frequencyBinCount: 128,
        getByteFrequencyData: vi.fn(),
        connect: vi.fn(),
      }),
      createMediaStreamSource: () => ({ connect: vi.fn(), disconnect: mockDisconnect }),
      close: vi.fn(),
    }))
  })

  afterEach(() => {
    vi.useRealTimers()
    cleanup()
    delete (window as any).SpeechRecognition
    delete (window as any).webkitSpeechRecognition
    delete (navigator as any).mediaDevices
    delete (window as any).AudioContext
  })

  it('shows error when SpeechRecognition not available', async () => {
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })
    expect(screen.getByText(/Speech recognition not supported/)).toBeDefined()
  })

  it('renders close button', () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    expect(screen.getByLabelText('Exit voice mode')).toBeDefined()
  })

  it('shows listening state after recognition starts', async () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    // Header shows "Listening" (no dots), body shows instructions
    expect(screen.getByText('Listening')).toBeDefined()
  })

  it('calls onClose when close button clicked', () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    fireEvent.click(screen.getByLabelText('Exit voice mode'))
    expect(onClose).toHaveBeenCalled()
  })

  it('shows interim text during recognition', async () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })

    act(() => {
      mockRecognition.handlers.result?.(makeResultEvent(0, [
        { transcript: 'hello ', isFinal: false },
      ]))
    })

    expect(screen.getByText('hello')).toBeDefined()
  })

  it('accumulates final text', async () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })

    act(() => {
      mockRecognition.handlers.result?.(makeResultEvent(0, [
        { transcript: 'hello world', isFinal: true },
      ]))
    })

    expect(screen.getByText('hello world')).toBeDefined()
  })

  it('stops listening when toggle button clicked while listening', async () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })

    fireEvent.click(screen.getByLabelText('Tap to stop listening'))
    await act(async () => { await vi.advanceTimersByTimeAsync(100) })

    expect(mockRecognition.instance.stop).toHaveBeenCalled()
  })

  it('shows error for not-allowed', async () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })

    act(() => {
      mockRecognition.handlers.error?.({ error: 'not-allowed' } as any)
    })

    expect(screen.getByText('Microphone access denied')).toBeDefined()
  })

  it('shows error for speech recognition errors', async () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })

    act(() => {
      mockRecognition.handlers.error?.({ error: 'no-speech' } as any)
    })

    expect(screen.getByText('Speech error: no-speech')).toBeDefined()
  })

  it('shows instructions text when listening', async () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })
    expect(screen.getByText(/Speak naturally/)).toBeDefined()
  })

  it('renders settings button', () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    expect(screen.getByLabelText('Voice settings')).toBeDefined()
  })

  it('opens settings panel when settings button clicked', async () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })
    fireEvent.click(screen.getByLabelText('Voice settings'))
    expect(screen.getByText('Speech Rate')).toBeDefined()
    expect(screen.getByText('Interrupt Sensitivity')).toBeDefined()
    expect(screen.getByText('Auto-resume Listening')).toBeDefined()
  })

  it('toggles transcript visibility', async () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })
    const toggleBtn = screen.getByText('Show transcript')
    fireEvent.click(toggleBtn)
    expect(screen.getByText('Hide transcript')).toBeDefined()
  })

  it('shows exchange count in footer', async () => {
    setupSpeechRecognition()
    render(<VoiceChatMode onMessage={onMessage} onClose={onClose} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })
    expect(screen.getByText(/No conversation yet/)).toBeDefined()
  })
})
