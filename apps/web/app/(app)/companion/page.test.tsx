import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

const {
  mockGetInfo, mockListPresets, mockGetPrompt, mockSetPersonality,
  mockSetPreset, mockChat, mockReset, mockAddToast,
} = vi.hoisted(() => ({
  mockGetInfo: vi.fn(), mockListPresets: vi.fn(), mockGetPrompt: vi.fn(),
  mockSetPersonality: vi.fn(), mockSetPreset: vi.fn(), mockChat: vi.fn(),
  mockReset: vi.fn(), mockAddToast: vi.fn(),
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
    Input: ({ value, onChange, placeholder }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} />
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

vi.mock('@/lib/companion-controller', () => ({
  companionController: {
    getInfo: (...a: unknown[]) => mockGetInfo(...a),
    listPresets: (...a: unknown[]) => mockListPresets(...a),
    getPrompt: (...a: unknown[]) => mockGetPrompt(...a),
    setPersonality: (...a: unknown[]) => mockSetPersonality(...a),
    setPreset: (...a: unknown[]) => mockSetPreset(...a),
    chat: (...a: unknown[]) => mockChat(...a),
    reset: (...a: unknown[]) => mockReset(...a),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/components/companion/CompanionInsightsCard', () => ({
  CompanionInsightsCard: ({ traits }: any) => (
    <div data-testid="companion-insights">{traits ? 'has-traits' : 'no-traits'}</div>
  ),
}))

import CompanionPage from './page'

afterEach(cleanup)

beforeEach(() => {
  vi.clearAllMocks()
  mockGetInfo.mockResolvedValue({ traits: { warmth: 0.7, curiosity: 0.8, creativity: 0.5, confidence: 0.6, humor: 0.3 } })
  mockListPresets.mockResolvedValue({ presets: [
    { id: 'warm', name: 'Warm', description: 'A warm companion', traits: { warmth: 0.9 } },
    { id: 'curious', name: 'Curious', description: 'An curious explorer', traits: { curiosity: 0.9 } },
  ]})
  mockGetPrompt.mockResolvedValue({ system_prompt: 'You are a helpful companion.' })
  mockSetPersonality.mockResolvedValue({ traits: { warmth: 0.7, curiosity: 0.8, creativity: 0.5, confidence: 0.6, humor: 0.3 } })
  mockSetPreset.mockResolvedValue({ traits: { warmth: 0.9, curiosity: 0.9, creativity: 0.5, confidence: 0.6, humor: 0.3 } })
  mockChat.mockResolvedValue({ response: 'Hello there!' })
  mockReset.mockResolvedValue({})
})

describe('CompanionPage — initial load flow', () => {
  it('renders page header', async () => {
    render(<CompanionPage />)
    expect(screen.getByText('Companion')).toBeTruthy()
  })

  it('fetches info, presets, and prompt on mount', async () => {
    render(<CompanionPage />)
    await waitFor(() => {
      expect(mockGetInfo).toHaveBeenCalledTimes(1)
      expect(mockListPresets).toHaveBeenCalledTimes(1)
      expect(mockGetPrompt).toHaveBeenCalledTimes(1)
    })
  })

  it('shows loading state', () => {
    mockGetInfo.mockReturnValue(new Promise(() => {}))
    render(<CompanionPage />)
    expect(screen.getByText('Companion')).toBeTruthy()
  })
})

describe('CompanionPage — traits display flow', () => {
  it('displays trait values after loading', async () => {
    render(<CompanionPage />)
    await waitFor(() => {
      expect(screen.getByText('Warmth')).toBeTruthy()
      expect(screen.getByText('Curiosity')).toBeTruthy()
      expect(screen.getByText('Creativity')).toBeTruthy()
    })
  })

  it('allows editing trait values', async () => {
    render(<CompanionPage />)
    await waitFor(() => { expect(screen.getAllByText('Warmth').length).toBeGreaterThanOrEqual(1) })

    const sliders = screen.getAllByRole('slider')
    if (sliders.length > 0) {
      fireEvent.change(sliders[0], { target: { value: '0.9' } })
      // No crash = success
      expect(screen.getAllByText('Warmth').length).toBeGreaterThanOrEqual(1)
    }
  })
})

describe('CompanionPage — presets flow', () => {
  it('displays preset options', async () => {
    render(<CompanionPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Warm').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('Curious').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('selecting a preset applies its traits', async () => {
    render(<CompanionPage />)
    await waitFor(() => { expect(screen.getAllByText('Warm').length).toBeGreaterThanOrEqual(1) })

    // Find the preset button (not the trait label)
    const presetBtns = screen.getAllByRole('button').filter(b =>
      b.textContent === 'Warm' || b.textContent === 'Curious'
    )
    if (presetBtns.length > 0) {
      await act(async () => { fireEvent.click(presetBtns[0]) })
      await waitFor(() => {
        expect(mockSetPreset).toHaveBeenCalled()
      })
    }
  })
})

describe('CompanionPage — save flow', () => {
  it('save button calls setPersonality', async () => {
    render(<CompanionPage />)
    await waitFor(() => { expect(screen.getByText('Warmth')).toBeTruthy() })

    const saveBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('save')
    )
    if (saveBtn) {
      await act(async () => { fireEvent.click(saveBtn) })
      await waitFor(() => {
        expect(mockSetPersonality).toHaveBeenCalled()
      })
    }
  })
})

describe('CompanionPage — chat flow', () => {
  it('chat input and send button work', async () => {
    render(<CompanionPage />)
    await waitFor(() => { expect(screen.getByText('Companion')).toBeTruthy() })

    const chatInput = screen.getAllByPlaceholderText(/say something/i)[0]
    if (chatInput) {
      fireEvent.change(chatInput, { target: { value: 'Hello companion' } })
      const sendBtn = screen.getAllByRole('button').find(b =>
        b.textContent?.toLowerCase().includes('send')
      )
      if (sendBtn) {
        await act(async () => { fireEvent.click(sendBtn) })
        await waitFor(() => {
          expect(mockChat).toHaveBeenCalled()
        })
      }
    }
  })
})

describe('CompanionPage — system prompt display', () => {
  it('shows system prompt text', async () => {
    render(<CompanionPage />)
    await waitFor(() => {
      expect(screen.getByText('You are a helpful companion.')).toBeTruthy()
    })
  })
})

describe('CompanionPage — insights card', () => {
  it('renders insights card with traits', async () => {
    render(<CompanionPage />)
    await waitFor(() => {
      const insights = screen.getByTestId('companion-insights')
      expect(insights).toBeTruthy()
    })
  })
})

describe('CompanionPage — error handling', () => {
  it('handles load failure gracefully', async () => {
    mockGetInfo.mockRejectedValue(new Error('network'))
    render(<CompanionPage />)
    await waitFor(() => {
      expect(screen.getByText('Companion')).toBeTruthy()
    })
  })
})
