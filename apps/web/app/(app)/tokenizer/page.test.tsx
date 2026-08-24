import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@sloughgpt/strui', () => {
  const iconMock = (name: string) => {
    const C = () => <span data-testid={`icon-${name}`}>{name}</span>
    C.displayName = `Icon${name}`
    return C
  }
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    Card: passthrough,
    CardHeader: passthrough,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    CardContent: passthrough,
    Button: ({ children, onClick, disabled, variant, className, 'aria-label': ariaLabel }: any) => (
      <button onClick={onClick} disabled={disabled} className={className} data-variant={variant} aria-label={ariaLabel}>{children}</button>
    ),
    Input: ({ value, onChange, type, placeholder, className, min, max, 'aria-label': ariaLabel }: any) => (
      <input value={value} onChange={onChange} type={type} placeholder={placeholder} className={className} min={min} max={max} aria-label={ariaLabel} />
    ),
    Textarea: ({ value, onChange, placeholder, rows }: any) => (
      <textarea value={value} onChange={onChange} placeholder={placeholder} rows={rows} />
    ),
    StatCard: ({ label, value }: any) => (
      <div data-testid={`stat-${label}`}>
        <span>{label}</span>
        <span>{value}</span>
      </div>
    ),
    KpiGrid: ({ children }: any) => <div data-testid="kpi-grid">{children}</div>,
    IconRefresh: iconMock('refresh'),
    IconSearch: iconMock('search'),
    IconChevronRight: iconMock('chevron-right'),
    IconChevronLeft: iconMock('chevron-left'),
    IconActivity: iconMock('activity'),
    Chip: ({ label }: any) => <span>{label}</span>,
    Skeleton: () => <div data-testid="skeleton" />,
  }
})

const { mockGetStats, mockTokenize, mockGetVocab, mockGetSamples, mockTrain, mockAddToast } = vi.hoisted(() => ({
  mockGetStats: vi.fn(),
  mockTokenize: vi.fn(),
  mockGetVocab: vi.fn(),
  mockGetSamples: vi.fn(),
  mockTrain: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/tokenizer-controller', () => ({
  tokenizerController: {
    getStats: mockGetStats,
    tokenize: mockTokenize,
    getVocab: mockGetVocab,
    getSamples: mockGetSamples,
    train: mockTrain,
  },
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))
vi.mock('@/components/tokenizer/TokenizerEfficiencyCard', () => ({
  TokenizerEfficiencyCard: ({ stats }: any) => (stats ? <div data-testid="tokenizer-efficiency-card" /> : null),
}))

import TokenizerPage from './page'

const stats = {
  vocab_size: 100,
  base_chars: 80,
  special_tokens: 3,
  total_merges: 20,
  merged_subwords: 20,
  trained: true,
}

afterEach(() => { cleanup() })
beforeEach(() => {
  vi.clearAllMocks()
  mockGetStats.mockResolvedValue(stats)
})

async function renderLoaded() {
  render(<TokenizerPage />)
  await waitFor(() => { expect(screen.getByText('Vocab: 100 · Merges: 20')).toBeTruthy() })
}

describe('TokenizerPage', () => {
  it('shows loading skeleton and calls getStats on mount', async () => {
    let resolveStats: (v: typeof stats) => void
    mockGetStats.mockReturnValue(new Promise(r => { resolveStats = r }))
    render(<TokenizerPage />)
    expect(screen.queryByText('Tokenizer')).toBeNull()
    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThan(0)
    expect(mockGetStats).toHaveBeenCalled()
    await act(async () => { resolveStats!(stats) })
    expect(screen.getByText('Vocab: 100 · Merges: 20')).toBeTruthy()
  })

  it('displays stats in header and stat cards after load', async () => {
    render(<TokenizerPage />)
    await waitFor(() => { expect(screen.getByText('Vocab: 100 · Merges: 20')).toBeTruthy() })
    expect(screen.getByText('Vocab Size')).toBeTruthy()
    expect(screen.getByText('Base Chars')).toBeTruthy()
    expect(screen.getByText('Merges')).toBeTruthy()
    expect(screen.getByText('Special Tokens')).toBeTruthy()
    expect(screen.getAllByText('100').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('80').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('20').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByTestId('tokenizer-efficiency-card')).toBeTruthy()
  })

  it('falls back to plain subtitle when stats fail', async () => {
    mockGetStats.mockRejectedValueOnce(new Error('boom'))
    render(<TokenizerPage />)
    await waitFor(() => { expect(screen.getByText('BPE tokenizer')).toBeTruthy() })
    expect(screen.queryByText('Vocab Size')).toBeFalsy()
  })

  it('keeps Tokenize disabled and does not call tokenize for empty input', async () => {
    await renderLoaded()
    const tokenizeBtn = screen.getByText('Tokenize') as HTMLButtonElement
    expect(tokenizeBtn.disabled).toBe(true)
    await act(async () => { tokenizeBtn.click() })
    expect(mockTokenize).not.toHaveBeenCalled()
  })

  it('tokenizes text and renders tokens and ids', async () => {
    mockTokenize.mockResolvedValue({ tokens: ['hello', 'world'], ids: [100, 200] })
    await renderLoaded()
    const ta = screen.getByPlaceholderText('Enter text to tokenize...')
    await act(async () => { fireEvent.change(ta, { target: { value: 'hello world' } }) })
    expect((screen.getByText('Tokenize') as HTMLButtonElement).disabled).toBe(false)
    await act(async () => { screen.getByText('Tokenize').click() })
    await waitFor(() => { expect(mockTokenize).toHaveBeenCalledWith('hello world') })
    expect(screen.getByText('Tokens (2)')).toBeTruthy()
    expect(screen.getByText('hello')).toBeTruthy()
    expect(screen.getByText('world')).toBeTruthy()
    expect(screen.getByText('[100, 200]')).toBeTruthy()
  })

  it('shows error toast when tokenization fails', async () => {
    mockTokenize.mockRejectedValue(new Error('fail'))
    await renderLoaded()
    const ta = screen.getByPlaceholderText('Enter text to tokenize...')
    await act(async () => { fireEvent.change(ta, { target: { value: 'hello' } }) })
    await act(async () => { screen.getByText('Tokenize').click() })
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Tokenization failed', 'error') })
  })

  it('loads vocab entries on Vocab tab with pagination disabled at start', async () => {
    mockGetVocab.mockResolvedValue({
      entries: [
        { id: 0, token: '<pad>', is_special: true },
        { id: 1, token: 'hello', is_special: false },
      ],
      total: 2,
    })
    await renderLoaded()
    await act(async () => { screen.getByText('Vocab').click() })
    await waitFor(() => { expect(mockGetVocab).toHaveBeenCalledWith(50, 0) })
    expect(screen.getByText('Vocabulary (2)')).toBeTruthy()
    expect(screen.getByText('<pad>')).toBeTruthy()
    expect(screen.getByText('hello')).toBeTruthy()
    expect((screen.getByText('Prev') as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByText('Next') as HTMLButtonElement).disabled).toBe(true)
  })

  it('paginates through vocabulary with Prev and Next', async () => {
    mockGetVocab.mockResolvedValue({
      entries: [{ id: 0, token: 'a', is_special: false }, { id: 1, token: 'b', is_special: false }],
      total: 200,
    })
    await renderLoaded()
    await act(async () => { screen.getByText('Vocab').click() })
    await waitFor(() => { expect(mockGetVocab).toHaveBeenCalledWith(50, 0) })
    expect((screen.getByText('Prev') as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByText('Next') as HTMLButtonElement).disabled).toBe(false)
    await act(async () => { screen.getByText('Next').click() })
    await waitFor(() => { expect(mockGetVocab).toHaveBeenCalledWith(50, 50) })
    expect((screen.getByText('Prev') as HTMLButtonElement).disabled).toBe(false)
    await act(async () => { screen.getByText('Prev').click() })
    await waitFor(() => { expect(mockGetVocab).toHaveBeenCalledWith(50, 0) })
  })

  it('shows error toast when vocab load fails', async () => {
    mockGetVocab.mockRejectedValue(new Error('boom'))
    await renderLoaded()
    await act(async () => { screen.getByText('Vocab').click() })
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Could not load vocabulary', 'error') })
  })

  it('loads and renders tokenization samples', async () => {
    mockGetSamples.mockResolvedValue({
      samples: [{ word: 'hello', tokens: ['he', 'llo'], ids: [1, 2], count: 3 }],
    })
    await renderLoaded()
    await act(async () => { screen.getByText('Samples').click() })
    await waitFor(() => { expect(mockGetSamples).toHaveBeenCalled() })
    expect(screen.getByText('Tokenization Samples')).toBeTruthy()
    expect(screen.getByText('hello')).toBeTruthy()
    expect(screen.getByText('he')).toBeTruthy()
    expect(screen.getByText('llo')).toBeTruthy()
    expect(screen.getByText('3 tokens')).toBeTruthy()
  })

  it('shows Load Samples button and loads samples on click when empty', async () => {
    mockGetSamples.mockResolvedValue({ samples: [] })
    await renderLoaded()
    await act(async () => { screen.getByText('Samples').click() })
    await waitFor(() => { expect(mockGetSamples).toHaveBeenCalledTimes(1) })
    expect(screen.getByText('Load Samples')).toBeTruthy()
    await act(async () => { screen.getByText('Load Samples').click() })
    expect(mockGetSamples).toHaveBeenCalledTimes(2)
  })

  it('shows error toast when samples load fails', async () => {
    mockGetSamples.mockRejectedValue(new Error('boom'))
    await renderLoaded()
    await act(async () => { screen.getByText('Samples').click() })
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Could not load samples', 'error') })
  })

  it('trains tokenizer with configured vocab size and shows result', async () => {
    mockTrain.mockResolvedValue({
      status: 'ok',
      corpus_size: 1000,
      stats: { vocab_size: 512, base_chars: 120, special_tokens: 5, total_merges: 100, merged_subwords: 100, trained: true },
    })
    await renderLoaded()
    await act(async () => { screen.getByText('Train').click() })
    expect(screen.getByText('Train Tokenizer')).toBeTruthy()
    const input = screen.getByRole('spinbutton', { name: 'Vocab size' })
    await act(async () => { fireEvent.change(input, { target: { value: '768' } }) })
    await act(async () => { screen.getByText('Train on Shakespeare').click() })
    await waitFor(() => { expect(mockTrain).toHaveBeenCalledWith({ vocab_size: 768 }) })
    expect(screen.getByText('Trained on 1000 lines. Vocab: 512')).toBeTruthy()
  })

  it('shows error message when training fails', async () => {
    mockTrain.mockRejectedValue(new Error('boom'))
    await renderLoaded()
    await act(async () => { screen.getByText('Train').click() })
    await act(async () => { screen.getByText('Train on Shakespeare').click() })
    await waitFor(() => { expect(screen.getByText('boom')).toBeTruthy() })
  })

  it('shows Training... and disables button while training is pending', async () => {
    let resolveTrain: (v: any) => void
    mockTrain.mockReturnValue(new Promise(r => { resolveTrain = r }))
    await renderLoaded()
    await act(async () => { screen.getByText('Train').click() })
    await act(async () => { screen.getByText('Train on Shakespeare').click() })
    await waitFor(() => { expect(screen.getByText('Training...')).toBeTruthy() })
    expect((screen.getByText('Training...') as HTMLButtonElement).disabled).toBe(true)
    await act(async () => {
      resolveTrain!({ status: 'ok', corpus_size: 10, stats })
    })
    await waitFor(() => { expect(screen.getByText('Train on Shakespeare')).toBeTruthy() })
  })
})
