// @vitest-environment jsdom

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
    onresult!({ results: [{ 0: { transcript: 'hello world', confidence: 0.9 }, isFinal: true, length: 1 }] } as any)
    expect(onTranscript).toHaveBeenCalledWith('hello world')
  })

  it('does not render when no speech support and no server support', async () => {
    const w = window as unknown as Record<string, unknown>
    delete w.SpeechRecognition
    delete w.webkitSpeechRecognition
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ speech_to_text: false }) })
    render(<VoiceInput onTranscript={() => {}} />)
    await waitFor(() => expect(screen.queryByLabelText(/voice input/i)).toBeNull())
  })
})
