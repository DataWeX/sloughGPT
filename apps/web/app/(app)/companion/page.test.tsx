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
    Input: ({ value, onChange, placeholder, onKeyDown }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} onKeyDown={onKeyDown} />
    ),
    StatCard: ({ label, value }: any) => <div data-testid={`stat-${label}`}><span>{label}</span><span>{String(value)}</span></div>,
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
  }
})

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

  it('shows loading skeleton placeholders', () => {
    mockGetInfo.mockReturnValue(new Promise(() => {}))
    mockListPresets.mockReturnValue(new Promise(() => {}))
    mockGetPrompt.mockReturnValue(new Promise(() => {}))
    render(<CompanionPage />)
    expect(screen.getByText('Companion')).toBeTruthy()
    expect(screen.getByTestId('stat-Active Preset')).toBeTruthy()
    expect(screen.getByTestId('stat-Warmth')).toBeTruthy()
    expect(screen.getByTestId('stat-Curiosity')).toBeTruthy()
    expect(screen.getByTestId('stat-Creativity')).toBeTruthy()
  })
})

describe('CompanionPage — traits display flow', () => {
  it('displays all five trait labels after loading', async () => {
    render(<CompanionPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Warmth').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('Curiosity').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('Creativity').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('Confidence').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('Humor').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('displays trait values as numeric text', async () => {
    render(<CompanionPage />)
    await waitFor(() => {
      expect(screen.getByText('0.70')).toBeTruthy()
      expect(screen.getByText('0.80')).toBeTruthy()
      expect(screen.getByText('0.50')).toBeTruthy()
    })
  })

  it('allows editing trait values via sliders', async () => {
    render(<CompanionPage />)
    await waitFor(() => { expect(screen.getAllByText('Warmth').length).toBeGreaterThanOrEqual(1) })

    const sliders = screen.getAllByRole('slider')
    expect(sliders.length).toBeGreaterThanOrEqual(1)
    fireEvent.change(sliders[0], { target: { value: '0.9' } })
    expect(screen.getAllByText('Warmth').length).toBeGreaterThanOrEqual(1)
  })
})

describe('CompanionPage — presets flow', () => {
  it('displays preset buttons with names', async () => {
    render(<CompanionPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Warm').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('Curious').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('preset buttons have titles showing descriptions', async () => {
    render(<CompanionPage />)
    await waitFor(() => {
      const warmBtn = screen.getAllByRole('button').find(b => b.textContent === 'Warm')
      expect(warmBtn).toBeDefined()
      expect(warmBtn?.getAttribute('title')).toBe('A warm companion')
    })
  })

  it('selecting a preset calls setPreset with correct id', async () => {
    render(<CompanionPage />)
    await waitFor(() => { expect(screen.getAllByText('Warm').length).toBeGreaterThanOrEqual(1) })

    const presetBtns = screen.getAllByRole('button').filter(b =>
      b.textContent === 'Warm' || b.textContent === 'Curious'
    )
    await act(async () => { fireEvent.click(presetBtns[0]) })
    await waitFor(() => {
      expect(mockSetPreset).toHaveBeenCalledWith('warm')
    })
  })

  it('shows empty preset list gracefully', async () => {
    mockListPresets.mockResolvedValue({ presets: [] })
    render(<CompanionPage />)
    await waitFor(() => {
      expect(mockListPresets).toHaveBeenCalled()
    })
    expect(screen.getByText('Companion')).toBeTruthy()
  })

  it('displays Active Preset stat card', async () => {
    render(<CompanionPage />)
    await waitFor(() => {
      const stat = screen.getByTestId('stat-Active Preset')
      expect(stat).toBeTruthy()
      expect(stat.textContent).toContain('Warm')
    })
  })
})

describe('CompanionPage — save flow', () => {
  it('save button calls setPersonality', async () => {
    render(<CompanionPage />)
    await waitFor(() => { expect(screen.getByText('Warmth')).toBeTruthy() })

    const saveBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('save')
    )
    expect(saveBtn).toBeDefined()
    await act(async () => { fireEvent.click(saveBtn!) })
    await waitFor(() => {
      expect(mockSetPersonality).toHaveBeenCalled()
    })
  })

  it('save button text changes to "Saving..." while saving', async () => {
    let resolveSave: (v: any) => void
    mockSetPersonality.mockReturnValue(new Promise(r => { resolveSave = r }))

    render(<CompanionPage />)
    await waitFor(() => { expect(screen.getByText('Warmth')).toBeTruthy() })

    const saveBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('save')
    )
    await act(async () => { fireEvent.click(saveBtn!) })

    expect(screen.getByText('Saving...')).toBeTruthy()
    expect(saveBtn!).toHaveAttribute('disabled')

    await act(async () => { resolveSave!({ traits: { warmth: 0.7 } }) })
  })
})

describe('CompanionPage — chat flow', () => {
  it('chat input and send button work', async () => {
    render(<CompanionPage />)
    await waitFor(() => { expect(screen.getByText('Companion')).toBeTruthy() })

    const chatInput = screen.getAllByPlaceholderText(/type a message/i)[0]
    fireEvent.change(chatInput, { target: { value: 'Hello companion' } })
    const sendBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('send')
    )
    await act(async () => { fireEvent.click(sendBtn!) })
    await waitFor(() => {
      expect(mockChat).toHaveBeenCalledWith('Hello companion')
    })
  })

  it('chat response is displayed after send', async () => {
    render(<CompanionPage />)
    await waitFor(() => { expect(screen.getByText('Companion')).toBeTruthy() })

    const chatInput = screen.getAllByPlaceholderText(/type a message/i)[0]
    fireEvent.change(chatInput, { target: { value: 'Hi' } })
    const sendBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('send')
    )
    await act(async () => { fireEvent.click(sendBtn!) })
    await waitFor(() => {
      expect(screen.getByText('Hello there!')).toBeTruthy()
    })
  })

  it('Enter key in chat input triggers send', async () => {
    render(<CompanionPage />)
    await waitFor(() => { expect(screen.getByText('Companion')).toBeTruthy() })

    const chatInput = screen.getAllByPlaceholderText(/type a message/i)[0]
    fireEvent.change(chatInput, { target: { value: 'Hello' } })
    await act(async () => {
      fireEvent.keyDown(chatInput, { key: 'Enter' })
    })
    await waitFor(() => {
      expect(mockChat).toHaveBeenCalled()
    })
  })

  it('send button is disabled when input is empty', async () => {
    render(<CompanionPage />)
    await waitFor(() => { expect(screen.getByText('Companion')).toBeTruthy() })

    const sendBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('send')
    )
    expect(sendBtn).toBeDefined()
    expect(sendBtn).toHaveAttribute('disabled')
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
      expect(insights.textContent).toBe('has-traits')
    })
  })
})

describe('CompanionPage — reset flow', () => {
  it('reset button calls controller.reset', async () => {
    render(<CompanionPage />)
    await waitFor(() => { expect(screen.getByText('Warmth')).toBeTruthy() })

    const resetBtn = screen.getByTestId('icon-refresh').closest('button')
    expect(resetBtn).toBeDefined()
    await act(async () => { fireEvent.click(resetBtn!) })
    await waitFor(() => {
      expect(mockReset).toHaveBeenCalled()
    })
  })
})

describe('CompanionPage — error handling', () => {
  it('shows error message when load fails', async () => {
    mockGetInfo.mockRejectedValue(new Error('network'))
    mockListPresets.mockRejectedValue(new Error('network'))
    mockGetPrompt.mockRejectedValue(new Error('network'))
    render(<CompanionPage />)
    await waitFor(() => {
      expect(screen.getByText('network')).toBeTruthy()
    })
  })

  it('error state shows retry button', async () => {
    mockGetInfo.mockRejectedValue(new Error('network'))
    mockListPresets.mockRejectedValue(new Error('network'))
    mockGetPrompt.mockRejectedValue(new Error('network'))
    render(<CompanionPage />)
    await waitFor(() => {
      const retryBtn = screen.getAllByRole('button').find(b =>
        b.textContent?.toLowerCase().includes('retry')
      )
      expect(retryBtn).toBeDefined()
    })
  })

  it('save failure shows error toast', async () => {
    mockSetPersonality.mockRejectedValue(new Error('fail'))
    render(<CompanionPage />)
    await waitFor(() => { expect(screen.getByText('Warmth')).toBeTruthy() })

    const saveBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('save')
    )
    await act(async () => { fireEvent.click(saveBtn!) })
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('fail', 'error')
    })
  })

  it('preset failure shows error toast', async () => {
    mockSetPreset.mockRejectedValue(new Error('fail'))
    render(<CompanionPage />)
    await waitFor(() => { expect(screen.getAllByText('Warm').length).toBeGreaterThanOrEqual(1) })

    const presetBtn = screen.getAllByRole('button').find(b => b.textContent === 'Warm')
    await act(async () => { fireEvent.click(presetBtn!) })
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('fail', 'error')
    })
  })
})

describe('CompanionPage — personality traits section', () => {
  it('shows Personality Traits card title', async () => {
    render(<CompanionPage />)
    await waitFor(() => {
      expect(screen.getByText('Personality Traits')).toBeTruthy()
    })
  })

  it('shows Save Personality button', async () => {
    render(<CompanionPage />)
    await waitFor(() => {
      expect(screen.getByText('Save Personality')).toBeTruthy()
    })
  })

  it('renders SVG radar chart', async () => {
    const { container } = render(<CompanionPage />)
    await waitFor(() => {
      expect(screen.getByText('Warmth')).toBeTruthy()
    })
    const svg = container.querySelector('svg')
    expect(svg).toBeTruthy()
    expect(svg?.getAttribute('viewBox')).toBe('0 0 200 200')
  })

  it('trait card is hidden when traits are null', async () => {
    mockGetInfo.mockResolvedValue({ traits: null })
    render(<CompanionPage />)
    await waitFor(() => { expect(screen.getByText('Companion')).toBeTruthy() })
    expect(screen.queryByText('Personality Traits')).toBeNull()
  })
})
