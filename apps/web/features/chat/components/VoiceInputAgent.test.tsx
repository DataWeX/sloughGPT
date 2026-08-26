import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import { VoiceInputAgent } from './VoiceInputAgent'

afterEach(cleanup)

// Mock getUserMedia
Object.defineProperty(navigator, 'mediaDevices', {
  value: {
    getUserMedia: vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }],
    }),
  },
})

describe('VoiceInputAgent', () => {
  it('renders voice input title', () => {
    render(<VoiceInputAgent onTranscript={vi.fn()} />)
    expect(screen.getByText('Voice Input')).toBeInTheDocument()
  })

  it('renders model badge', () => {
    render(<VoiceInputAgent onTranscript={vi.fn()} model="whisper-1" />)
    expect(screen.getByText('whisper-1')).toBeInTheDocument()
  })

  it('renders timer', () => {
    render(<VoiceInputAgent onTranscript={vi.fn()} />)
    expect(screen.getByText('0:00')).toBeInTheDocument()
  })

  it('renders recording instruction', () => {
    render(<VoiceInputAgent onTranscript={vi.fn()} />)
    expect(screen.getByText(/Click mic to start/)).toBeInTheDocument()
  })

  it('starts recording on mic click', async () => {
    render(<VoiceInputAgent onTranscript={vi.fn()} />)
    const micButton = screen.getByRole('button', { name: /start recording/i })
    await act(async () => {
      fireEvent.click(micButton)
    })
    expect(screen.getByText('Recording')).toBeInTheDocument()
  })

  it('shows error on mic denied', async () => {
    vi.mocked(navigator.mediaDevices.getUserMedia).mockRejectedValue(new Error('denied'))
    render(<VoiceInputAgent onTranscript={vi.fn()} />)
    const micButton = screen.getByRole('button', { name: /start recording/i })
    await act(async () => {
      fireEvent.click(micButton)
    })
    expect(screen.getByText('Microphone access denied')).toBeInTheDocument()
  })

  it('renders custom model', () => {
    render(<VoiceInputAgent onTranscript={vi.fn()} model="gpt-4" />)
    expect(screen.getByText('gpt-4')).toBeInTheDocument()
  })
})