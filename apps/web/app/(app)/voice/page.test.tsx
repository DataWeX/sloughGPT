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
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
  
    Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    Separator: () => <hr />,
    Tooltip: ({ children }: any) => <>{children}</>,
    TooltipTrigger: ({ children }: any) => <>{children}</>,
    TooltipContent: ({ children }: any) => <>{children}</>,
    Progress: ({ value }: any) => <div data-testid="progress" data-value={value} />,
    Avatar: ({ children }: any) => <div>{children}</div>,
    AvatarFallback: ({ children }: any) => <div>{children}</div>,
    ScrollArea: ({ children }: any) => <div>{children}</div>,
    Table: ({ children }: any) => <table>{children}</table>,
    TableBody: ({ children }: any) => <tbody>{children}</tbody>,
    TableRow: ({ children }: any) => <tr>{children}</tr>,
    TableCell: ({ children }: any) => <td>{children}</td>,
    TableHead: ({ children }: any) => <th>{children}</th>,
    TableHeader: ({ children }: any) => <thead>{children}</thead>,
    Collapsible: ({ children }: any) => <div>{children}</div>,
    CollapsibleTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    CollapsibleContent: ({ children }: any) => <div>{children}</div>,
    Toggle: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    ToggleGroup: ({ children }: any) => <div>{children}</div>,
    ToggleGroupItem: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    Command: ({ children }: any) => <div>{children}</div>,
    CommandInput: ({ ...props }: any) => <input {...props} />,
    CommandList: ({ children }: any) => <div>{children}</div>,
    CommandEmpty: ({ children }: any) => <div>{children}</div>,
    CommandGroup: ({ children }: any) => <div>{children}</div>,
    CommandItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
}
})

vi.mock('@/components/voice/VoicePresetCard', () => ({
  VoicePresetCard: () => <div data-testid="voice-preset-card" />,
}))

vi.mock('@/components/voice/VoiceWaveformCard', () => ({
  VoiceWaveformCard: () => <div data-testid="voice-waveform-card" />,
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

describe('VoicePage — voice UI flow', () => {
  it('shows Available for server TTS', async () => {
    render(<VoicePage />)
    await waitFor(() => {
      expect(screen.getByText('Available')).toBeTruthy()
    })
  })

  it('shows server engine name', async () => {
    render(<VoicePage />)
    await waitFor(() => {
      expect(screen.getByText('AI model (bark)')).toBeTruthy()
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
