// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import React from 'react'

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

interface MockSR extends EventTarget {
  continuous: boolean
  interimResults: boolean
  lang: string
  start: ReturnType<typeof vi.fn>
  stop: ReturnType<typeof vi.fn>
  abort: ReturnType<typeof vi.fn>
  onstart: (() => void) | null
  onend: (() => void) | null
  onerror: ((e: Event) => void) | null
  onresult: ((e: any) => void) | null
}

let mockRecognition: MockSR

beforeEach(() => {
  mockRecognition = {
    continuous: false,
    interimResults: false,
    lang: '',
    start: vi.fn(),
    stop: vi.fn(),
    abort: vi.fn(),
    onstart: null,
    onend: null,
    onerror: null,
    onresult: null,
  } as any
  window.SpeechRecognition = vi.fn(() => mockRecognition) as any
  window.webkitSpeechRecognition = undefined as any
  mockFetch.mockResolvedValue({ json: () => Promise.resolve({ speech_to_text: true }) })
})

import { VoiceInput } from './VoiceInput'

function triggerStart() {
  act(() => { if (mockRecognition.onstart) mockRecognition.onstart() })
}

function triggerEnd() {
  act(() => { if (mockRecognition.onend) mockRecognition.onend() })
}

function triggerError() {
  act(() => { if (mockRecognition.onerror) mockRecognition.onerror(new Event('error')) })
}

function triggerResult(transcript: string) {
  act(() => {
    if (mockRecognition.onresult) {
      mockRecognition.onresult({ results: [[{ transcript, confidence: 0.9 }]] } as any)
    }
  })
}

describe('VoiceInput', () => {
  afterEach(cleanup)

  it('renders start listening button when browser supported', () => {
    render(<VoiceInput onTranscript={vi.fn()} />)
    expect(screen.getByLabelText('Start voice input')).toBeDefined()
  })

  it('changes label to stop listening when active', () => {
    render(<VoiceInput onTranscript={vi.fn()} />)
    fireEvent.click(screen.getByLabelText('Start voice input'))
    triggerStart()
    expect(screen.getByLabelText('Stop listening')).toBeDefined()
  })

  it('sets aria-pressed when listening', () => {
    render(<VoiceInput onTranscript={vi.fn()} />)
    fireEvent.click(screen.getByLabelText('Start voice input'))
    triggerStart()
    expect(screen.getByLabelText('Stop listening').getAttribute('aria-pressed')).toBe('true')
  })

  it('calls onTranscript when speech recognition fires result', () => {
    const onTranscript = vi.fn()
    render(<VoiceInput onTranscript={onTranscript} />)
    fireEvent.click(screen.getByLabelText('Start voice input'))
    expect(mockRecognition.start).toHaveBeenCalled()
    triggerResult('hello world')
    expect(onTranscript).toHaveBeenCalledWith('hello world')
  })

  it('stops listening when recognition ends', () => {
    render(<VoiceInput onTranscript={vi.fn()} />)
    fireEvent.click(screen.getByLabelText('Start voice input'))
    triggerStart()
    expect(screen.getByLabelText('Stop listening')).toBeDefined()
    triggerEnd()
    expect(screen.getByLabelText('Start voice input')).toBeDefined()
  })

  it('stops listening when recognition errors', () => {
    render(<VoiceInput onTranscript={vi.fn()} />)
    fireEvent.click(screen.getByLabelText('Start voice input'))
    triggerStart()
    triggerError()
    expect(screen.getByLabelText('Start voice input')).toBeDefined()
  })

  it('has assertive live region for screen readers', () => {
    render(<VoiceInput onTranscript={vi.fn()} />)
    const live = document.querySelector('[aria-live="assertive"]')
    expect(live).toBeDefined()
  })
})

describe('VoiceInput unsupported', () => {
  beforeEach(() => {
    delete (window as any).SpeechRecognition
    delete (window as any).webkitSpeechRecognition
    mockFetch.mockResolvedValue({ json: () => Promise.resolve({ speech_to_text: false }) })
  })

  afterEach(cleanup)

  it('renders nothing when neither browser nor server supported', () => {
    const { container } = render(<VoiceInput onTranscript={vi.fn()} />)
    expect(container.innerHTML).toBe('')
  })
})
