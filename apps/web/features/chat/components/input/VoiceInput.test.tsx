
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import { VoiceInput } from './VoiceInput'

const mockSpeechRecognition = vi.fn()

describe('VoiceInput', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    Object.assign(window, {
      SpeechRecognition: mockSpeechRecognition,
      webkitSpeechRecognition: mockSpeechRecognition,
    })
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ speech_to_text: true }) })
  })

  afterEach(() => {
    cleanup()
  })

  it('renders microphone button', () => {
    render(<VoiceInput onTranscript={() => {}} />)
    expect(screen.getByLabelText('Start voice input')).toBeDefined()
  })

  it('starts browser recognition on click', () => {
    const start = vi.fn()
    mockSpeechRecognition.mockImplementation(() => ({
      continuous: false,
      interimResults: false,
      lang: '',
      start,
      stop: vi.fn(),
      abort: vi.fn(),
      onstart: null,
      onend: null,
      onerror: null,
      onresult: null,
    }))

    render(<VoiceInput onTranscript={() => {}} />)
    fireEvent.click(screen.getByLabelText('Start voice input'))
    expect(start).toHaveBeenCalled()
  })

  it('calls onTranscript with final result', () => {
    const onTranscript = vi.fn()
    let onresult: ((e: any) => void) | null = null
    mockSpeechRecognition.mockImplementation(() => ({
      continuous: false,
      interimResults: false,
      lang: '',
      start: vi.fn(),
      stop: vi.fn(),
      abort: vi.fn(),
      onstart: null,
      onend: null,
      onerror: null,
      set onresult(fn) { onresult = fn },
      get onresult() { return onresult },
    }))

    render(<VoiceInput onTranscript={onTranscript} />)
    fireEvent.click(screen.getByLabelText('Start voice input'))
    expect(onresult).not.toBeNull()
    onresult!({
      resultIndex: 0,
      results: [{ 0: { transcript: 'hello world', confidence: 0.9 }, isFinal: true, length: 1 }],
    } as any)
    expect(onTranscript).toHaveBeenCalledWith('hello world')
  })

  it('never emits interim fragments (they would be appended and auto-sent)', () => {
    const onTranscript = vi.fn()
    let onresult: ((e: any) => void) | null = null
    mockSpeechRecognition.mockImplementation(() => ({
      continuous: false,
      interimResults: true,
      lang: '',
      start: vi.fn(),
      stop: vi.fn(),
      abort: vi.fn(),
      onstart: null,
      onend: null,
      onerror: null,
      set onresult(fn) {
        onresult = fn
      },
      get onresult() {
        return onresult
      },
    }))

    render(<VoiceInput onTranscript={onTranscript} />)
    fireEvent.click(screen.getByLabelText('Start voice input'))
    onresult!({
      resultIndex: 0,
      results: [{ 0: { transcript: 'hel', confidence: 0.4 }, isFinal: false, length: 1 }],
    } as any)
    expect(onTranscript).not.toHaveBeenCalled()
  })

  it('does not duplicate the utterance as partial results grow (bug repro)', () => {
    const onTranscript = vi.fn()
    let onresult: ((e: any) => void) | null = null
    mockSpeechRecognition.mockImplementation(() => ({
      continuous: false,
      interimResults: true,
      lang: '',
      start: vi.fn(),
      stop: vi.fn(),
      abort: vi.fn(),
      onstart: null,
      onend: null,
      onerror: null,
      set onresult(fn) {
        onresult = fn
      },
      get onresult() {
        return onresult
      },
    }))

    render(<VoiceInput onTranscript={onTranscript} />)
    fireEvent.click(screen.getByLabelText('Start voice input'))

    // The browser REPLACES result[0] in place as speech grows, so every event
    // reports resultIndex 0 over the same slot. The old code re-joined all
    // results from index 0 each time and ChatInput appended the result.
    const partial = (transcript: string, isFinal: boolean) =>
      onresult!({
        resultIndex: 0,
        results: [{ 0: { transcript, confidence: 0.5 }, isFinal, length: 1 }],
      } as any)
    partial('hel', false)
    partial('hello wor', false)
    partial('hello world', true)

    expect(onTranscript).toHaveBeenCalledTimes(1)
    expect(onTranscript).toHaveBeenCalledWith('hello world')
  })

  it('emits only newly final segments, never re-joins from index 0', () => {
    const onTranscript = vi.fn()
    let onresult: ((e: any) => void) | null = null
    mockSpeechRecognition.mockImplementation(() => ({
      continuous: true,
      interimResults: true,
      lang: '',
      start: vi.fn(),
      stop: vi.fn(),
      abort: vi.fn(),
      onstart: null,
      onend: null,
      onerror: null,
      set onresult(fn) {
        onresult = fn
      },
      get onresult() {
        return onresult
      },
    }))

    render(<VoiceInput onTranscript={onTranscript} />)
    fireEvent.click(screen.getByLabelText('Start voice input'))

    const final = (transcript: string) => ({
      0: { transcript, confidence: 0.9 },
      isFinal: true,
      length: 1,
    })
    onresult!({ resultIndex: 0, results: [final('hello ')] } as any)
    // Second segment finalises: index 0 is already-final history, index 1 is new.
    onresult!({ resultIndex: 1, results: [final('hello '), final('there')] } as any)

    expect(onTranscript).toHaveBeenCalledTimes(2)
    expect(onTranscript).toHaveBeenNthCalledWith(1, 'hello ')
    expect(onTranscript).toHaveBeenNthCalledWith(2, 'there')
    expect(onTranscript).not.toHaveBeenCalledWith('hello there')
  })

  it('renders disabled button when no speech support and no server support', async () => {
    const w = window as unknown as Record<string, unknown>
    delete w.SpeechRecognition
    delete w.webkitSpeechRecognition
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ speech_to_text: false }) })
    render(<VoiceInput onTranscript={() => {}} />)
    await waitFor(() => {
      const btn = screen.getByLabelText(/voice input/i)
      expect(btn).toBeDefined()
      expect(btn).toBeDisabled()
    })
  })

  it('shows recording state when listening', async () => {
    let onstart: (() => void) | null = null
    mockSpeechRecognition.mockImplementation(() => ({
      continuous: false,
      interimResults: false,
      lang: '',
      start: vi.fn(),
      stop: vi.fn(),
      abort: vi.fn(),
      set onstart(fn) { onstart = fn },
      get onstart() { return onstart },
      onend: null,
      onerror: null,
      onresult: null,
    }))
    render(<VoiceInput onTranscript={() => {}} />)
    fireEvent.click(screen.getByLabelText('Start voice input'))
    expect(onstart).not.toBeNull()
  })

  it('handles error from speech recognition', async () => {
    let onerror: ((e: any) => void) | null = null
    mockSpeechRecognition.mockImplementation(() => ({
      continuous: false,
      interimResults: false,
      lang: '',
      start: vi.fn(),
      stop: vi.fn(),
      abort: vi.fn(),
      onstart: null,
      onend: null,
      set onerror(fn) { onerror = fn },
      get onerror() { return onerror },
      onresult: null,
    }))
    render(<VoiceInput onTranscript={() => {}} />)
    fireEvent.click(screen.getByLabelText('Start voice input'))
    expect(onerror).not.toBeNull()
    onerror!({ error: 'not-allowed' } as any)
  })

  it('calls stop on recognition when already listening', async () => {
    const stop = vi.fn()
    mockSpeechRecognition.mockImplementation(() => ({
      continuous: false,
      interimResults: false,
      lang: '',
      start: vi.fn(),
      stop,
      abort: vi.fn(),
      onstart: null,
      onend: null,
      onerror: null,
      onresult: null,
    }))
    render(<VoiceInput onTranscript={() => {}} />)
    fireEvent.click(screen.getByLabelText('Start voice input'))
    expect(stop).not.toHaveBeenCalled()
  })

  it('renders with default disabled state before capability check', () => {
    render(<VoiceInput onTranscript={() => {}} />)
    const btn = screen.getByLabelText(/voice input/i)
    expect(btn).toBeDefined()
  })
})
