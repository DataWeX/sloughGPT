import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

const mockGetStatus = vi.fn()
const mockTts = vi.fn()

vi.mock('@/lib/voice-controller', () => ({
  voiceController: {
    getStatus: (...args: unknown[]) => mockGetStatus(...args),
    tts: (...args: unknown[]) => mockTts(...args),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

import VoicePage from './page'

describe('VoicePage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockGetStatus.mockResolvedValue({ server_tts: true, model: 'bark', error: null })
  })

  it('renders page header', async () => {
    render(<VoicePage />)
    expect(screen.getAllByText('Voice').length).toBeGreaterThanOrEqual(1)
  })

  it('shows TTS backend status', async () => {
    render(<VoicePage />)
    await screen.findByText('Available')
    expect(screen.getAllByText('bark').length).toBeGreaterThanOrEqual(1)
  })

  it('renders generate button', async () => {
    render(<VoicePage />)
    await screen.findAllByText(/available|unavailable/i)
    expect(screen.getAllByText(/generate/i).length).toBeGreaterThanOrEqual(1)
  })

  it('renders TTS textarea', async () => {
    render(<VoicePage />)
    await screen.findAllByText(/available|unavailable/i)
    expect(screen.getAllByPlaceholderText(/enter text/i).length).toBeGreaterThanOrEqual(1)
  })

  it('shows unavailable when server_tts is false', async () => {
    mockGetStatus.mockResolvedValue({ server_tts: false, model: null, error: null })
    render(<VoicePage />)
    await screen.findByText('Unavailable')
  })
})
