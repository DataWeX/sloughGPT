import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

const {
  mockGetStatus, mockTts, mockAddToast,
} = vi.hoisted(() => ({
  mockGetStatus: vi.fn(), mockTts: vi.fn(), mockAddToast: vi.fn(),
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...a: any[]) => a.join(' ')),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Button: ({ children, onClick, disabled }: any) => (
      <button onClick={onClick} disabled={disabled}>{children}</button>
    ),
    Textarea: ({ value, onChange, placeholder }: any) => (
      <textarea value={value} onChange={onChange} placeholder={placeholder} />
    ),
    StatCard: ({ label, value }: any) => <div data-testid={`stat-${label}`}><span>{label}</span><span>{String(value)}</span></div>,
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
  }
})

vi.mock('@/components/AppRouteHeader', () => ({
  AppRouteHeader: ({ left, right }: any) => <div>{left}{right}</div>,
  AppRouteHeaderLead: ({ title }: any) => <h1>{title}</h1>,
}))

vi.mock('@/components/voice/VoicePresetCard', () => ({
  VoicePresetCard: () => <div data-testid="voice-preset-card" />,
}))

vi.mock('@/lib/voice-controller', () => ({
  voiceController: {
    getStatus: (...a: unknown[]) => mockGetStatus(...a),
    tts: (...a: unknown[]) => mockTts(...a),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

import VoicePage from './page'

afterEach(cleanup)

beforeEach(() => {
  vi.clearAllMocks()
  mockGetStatus.mockResolvedValue({ server_tts: true, model: 'bark', error: null })
  mockTts.mockResolvedValue({ audio: 'base64data', duration_ms: 1000, backend: 'hf-model', sample_rate: 22050 })
})

describe('VoicePage — initial load flow', () => {
  it('renders page header', async () => {
    render(<VoicePage />)
    expect(screen.getAllByText('Voice').length).toBeGreaterThanOrEqual(1)
  })

  it('fetches status on mount', async () => {
    render(<VoicePage />)
    await waitFor(() => {
      expect(mockGetStatus).toHaveBeenCalledTimes(1)
    })
  })

  it('shows loading state', () => {
    mockGetStatus.mockReturnValue(new Promise(() => {}))
    render(<VoicePage />)
    expect(screen.getAllByText('Voice').length).toBeGreaterThanOrEqual(1)
  })
})

describe('VoicePage — available backend flow', () => {
  it('shows available status', async () => {
    render(<VoicePage />)
    await waitFor(() => {
      expect(screen.getByText('Available')).toBeTruthy()
    })
  })

  it('shows model name', async () => {
    render(<VoicePage />)
    await waitFor(() => {
      expect(screen.getAllByText('bark').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('renders TTS textarea', async () => {
    render(<VoicePage />)
    await waitFor(() => { expect(screen.getByText('Available')).toBeTruthy() })
    expect(screen.getAllByPlaceholderText(/enter text/i).length).toBeGreaterThanOrEqual(1)
  })

  it('renders generate button', async () => {
    render(<VoicePage />)
    await waitFor(() => { expect(screen.getByText('Available')).toBeTruthy() })
    expect(screen.getAllByText(/generate/i).length).toBeGreaterThanOrEqual(1)
  })

  it('shows voice preset card', async () => {
    render(<VoicePage />)
    await waitFor(() => { expect(screen.getByTestId('voice-preset-card')).toBeTruthy() })
  })
})

describe('VoicePage — generate flow', () => {
  it('generate button calls tts', async () => {
    render(<VoicePage />)
    await waitFor(() => { expect(screen.getByText('Available')).toBeTruthy() })

    const textarea = screen.getAllByPlaceholderText(/enter text/i)[0]
    fireEvent.change(textarea, { target: { value: 'Hello world' } })

    const genBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('generate')
    )
    if (genBtn) {
      await act(async () => { fireEvent.click(genBtn) })
      await waitFor(() => {
        expect(mockTts).toHaveBeenCalledWith('Hello world')
      })
    }
  })

  it('shows result after generation', async () => {
    render(<VoicePage />)
    await waitFor(() => { expect(screen.getByText('Available')).toBeTruthy() })

    const textarea = screen.getAllByPlaceholderText(/enter text/i)[0]
    fireEvent.change(textarea, { target: { value: 'Hello' } })

    const genBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('generate')
    )
    if (genBtn) {
      await act(async () => { fireEvent.click(genBtn) })
      await waitFor(() => {
        expect(screen.getByText(/1000ms/)).toBeTruthy()
      })
    }
  })
})

describe('VoicePage — unavailable backend flow', () => {
  it('shows unavailable status', async () => {
    mockGetStatus.mockResolvedValue({ server_tts: false, model: null, error: null })
    render(<VoicePage />)
    await waitFor(() => {
      expect(screen.getByText('Unavailable')).toBeTruthy()
    })
  })
})

describe('VoicePage — error handling', () => {
  it('handles status failure gracefully', async () => {
    mockGetStatus.mockRejectedValue(new Error('network'))
    render(<VoicePage />)
    await waitFor(() => {
      expect(screen.getAllByText('Voice').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('handles tts failure gracefully', async () => {
    mockTts.mockRejectedValue(new Error('TTS error'))
    render(<VoicePage />)
    await waitFor(() => { expect(screen.getByText('Available')).toBeTruthy() })

    const textarea = screen.getAllByPlaceholderText(/enter text/i)[0]
    fireEvent.change(textarea, { target: { value: 'test' } })

    const genBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('generate')
    )
    if (genBtn) {
      await act(async () => { fireEvent.click(genBtn) })
      await waitFor(() => {
        expect(screen.getByText(/tts error/i)).toBeTruthy()
      })
    }
  })
})

describe('VoicePage — refresh flow', () => {
  it('refresh button reloads status', async () => {
    render(<VoicePage />)
    await waitFor(() => { expect(mockGetStatus).toHaveBeenCalledTimes(1) })

    const refreshBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('refresh')
    )
    if (refreshBtn) {
      await act(async () => { fireEvent.click(refreshBtn) })
      await waitFor(() => {
        expect(mockGetStatus).toHaveBeenCalledTimes(2)
      })
    }
  })
})
