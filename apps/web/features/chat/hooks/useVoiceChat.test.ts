import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { renderHook, act, cleanup } from '@testing-library/react'
import { useVoiceChat, VoiceSettings } from './useVoiceChat'

// Mock the controllers
vi.mock('@/lib/chat-controller', () => ({
  chatController: {
    stream: vi.fn(async function* () {
      yield 'test response'
    }),
  },
}))

vi.mock('@/lib/voice-controller', () => ({
  voiceController: {
    tts: vi.fn().mockRejectedValue(new Error('no server')),
    playAudio: vi.fn().mockResolvedValue(undefined),
  },
}))

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
    start: vi.fn(() => setTimeout(() => handlers.start?.())),
    stop: vi.fn(() => setTimeout(() => handlers.end?.())),
    abort: vi.fn(),
    get onstart() {
      return handlers.start
    },
    set onstart(fn) {
      handlers.start = fn
    },
    get onend() {
      return handlers.end
    },
    set onend(fn) {
      handlers.end = fn
    },
    get onerror() {
      return handlers.error
    },
    set onerror(fn) {
      handlers.error = fn
    },
    get onresult() {
      return handlers.result
    },
    set onresult(fn) {
      handlers.result = fn
    },
  }
  return { instance, handlers }
}

function setupMocks() {
  mockRec = makeMockRecognition()
  ;(window as any).SpeechRecognition = vi.fn(() => mockRec.instance)
  ;(navigator as any).mediaDevices = {
    getUserMedia: vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }],
    }),
  }
  const mockDisconnect = vi.fn()
  ;(window as any).AudioContext = vi.fn((opts?: { sampleRate?: number }) => ({
    sampleRate: opts?.sampleRate ?? 48000,
    createAnalyser: () => ({
      fftSize: 1024,
      smoothingTimeConstant: 0.4,
      frequencyBinCount: 512,
      getByteFrequencyData: vi.fn(),
      getByteTimeDomainData: vi.fn(),
      connect: vi.fn(),
    }),
    createMediaStreamSource: () => ({ connect: vi.fn(), disconnect: mockDisconnect }),
    createScriptProcessor: () => ({
      connect: vi.fn(),
      disconnect: vi.fn(),
      onaudioprocess: null,
    }),
    createGain: () => ({ gain: { value: 0 }, connect: vi.fn(), disconnect: vi.fn() }),
    close: vi.fn(),
  }))
}

let mockRec: ReturnType<typeof makeMockRecognition>

describe('useVoiceChat', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
    setupMocks()
  })

  afterEach(() => {
    vi.useRealTimers()
    cleanup()
    delete (window as any).SpeechRecognition
    delete (window as any).webkitSpeechRecognition
    delete (navigator as any).mediaDevices
    delete (window as any).AudioContext
  })

  it('starts in idle state', () => {
    const { result } = renderHook(() => useVoiceChat({ onMessage: vi.fn() }))
    expect(result.current.state).toBe('idle')
  })

  it('starts listening when startListening called', async () => {
    const { result } = renderHook(() => useVoiceChat({ onMessage: vi.fn() }))
    await act(async () => {
      result.current.startListening()
      await vi.advanceTimersByTimeAsync(100)
    })
    expect(result.current.state).toBe('listening')
    expect(result.current.isListening).toBe(true)
  })

  it('falls back to server transcription when SpeechRecognition unavailable', async () => {
    delete (window as any).SpeechRecognition
    delete (window as any).webkitSpeechRecognition
    const { result } = renderHook(() => useVoiceChat({ onMessage: vi.fn() }))
    await act(async () => {
      result.current.startListening()
      await vi.advanceTimersByTimeAsync(100)
    })
    // Server mode engaged (OpenCode pane / Android WebView path): mic opened,
    // capture wired, listening started — no dead-end error state.
    expect((navigator as any).mediaDevices.getUserMedia).toHaveBeenCalled()
    expect(result.current.state).toBe('listening')
    expect(result.current.errorMessage).toBeNull()
  })

  it('errors only when neither recognizer nor mic is available', async () => {
    delete (window as any).SpeechRecognition
    delete (window as any).webkitSpeechRecognition
    ;(navigator as any).mediaDevices.getUserMedia = vi.fn().mockRejectedValue(new Error('denied'))
    const { result } = renderHook(() => useVoiceChat({ onMessage: vi.fn() }))
    await act(async () => {
      result.current.startListening()
      await vi.advanceTimersByTimeAsync(100)
    })
    expect(result.current.state).toBe('error')
    expect(result.current.errorMessage).toContain('Microphone')
  })

  it('returns default settings', () => {
    const { result } = renderHook(() => useVoiceChat({ onMessage: vi.fn() }))
    expect(result.current.settings.rate).toBe(0.95)
    expect(result.current.settings.pitch).toBe(1.0)
    expect(result.current.settings.interruptThreshold).toBe(0.15)
    expect(result.current.settings.autoResume).toBe(true)
  })

  it('updates settings', () => {
    const { result } = renderHook(() => useVoiceChat({ onMessage: vi.fn() }))
    act(() => {
      result.current.updateSettings({ rate: 1.5 })
    })
    expect(result.current.settings.rate).toBe(1.5)
    expect(result.current.settings.pitch).toBe(1.0) // unchanged
  })

  it('accumulates conversation exchanges', async () => {
    const onMessage = vi.fn()
    const { result } = renderHook(() => useVoiceChat({ onMessage }))

    // Start listening
    await act(async () => {
      result.current.startListening()
      await vi.advanceTimersByTimeAsync(100)
    })

    // Simulate a final transcript
    act(() => {
      mockRec.handlers.result?.({
        resultIndex: 0,
        results: [{ isFinal: true, 0: { transcript: 'hello', confidence: 0.9 }, length: 1 }],
      } as any)
    })

    // Trigger silence timeout → submit
    await act(async () => {
      vi.advanceTimersByTime(2500)
      await vi.advanceTimersByTimeAsync(500)
    })

    // onMessage should have been called with the user text
    expect(onMessage).toHaveBeenCalledWith('hello')
  })

  it('toggles between listening and idle', async () => {
    const { result } = renderHook(() => useVoiceChat({ onMessage: vi.fn() }))

    // Start listening
    await act(async () => {
      result.current.handleToggle()
      await vi.advanceTimersByTimeAsync(100)
    })
    expect(result.current.state).toBe('listening')

    // Stop listening
    act(() => {
      result.current.handleToggle()
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(100)
    })
    expect(result.current.state).toBe('idle')
  })

  it('stopListening resets to idle', async () => {
    const { result } = renderHook(() => useVoiceChat({ onMessage: vi.fn() }))
    await act(async () => {
      result.current.startListening()
      await vi.advanceTimersByTimeAsync(100)
    })
    expect(result.current.state).toBe('listening')

    act(() => {
      result.current.stopListening()
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(100)
    })
    expect(result.current.state).toBe('idle')
  })
})
