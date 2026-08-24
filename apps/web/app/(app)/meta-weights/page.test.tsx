import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'

const mockGetStats = vi.fn()
const mockGetWeights = vi.fn()
const mockAddToast = vi.fn()

vi.mock('@/lib/meta-weights-controller', () => ({
  metaWeightsController: {
    getStats: (...args: unknown[]) => mockGetStats(...args),
    getWeights: (...args: unknown[]) => mockGetWeights(...args),
    ping: vi.fn().mockResolvedValue({ status: 'ok' }),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => mockAddToast,
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...a: any[]) => a.join(' '),
    Button: ({ children, onClick, disabled }: any) => <button onClick={onClick} disabled={disabled}>{children}</button>,
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Input: ({ value, onChange, placeholder, onKeyDown }: any) => <input value={value} onChange={onChange} placeholder={placeholder} onKeyDown={onKeyDown} />,
    StatCard: ({ label, value }: any) => <div data-testid={`stat-${label}`}><span>{label}</span><span>{String(value)}</span></div>,
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    IconRefresh: () => <span data-testid="icon-refresh" />,
  }
})

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title, subtitle }: any) => (
    <div data-testid="page-container"><h1>{title}</h1><p>{subtitle}</p>{children}</div>
  ),
}))

import MetaWeightsPage from './page'

describe('MetaWeightsPage', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders page title and subtitle', async () => {
    mockGetStats.mockResolvedValue({ history_length: 0 })
    render(<MetaWeightsPage />)
    expect(screen.getByText('Meta-Weights')).toBeInTheDocument()
    expect(screen.getByText('Feedback-driven inference tuning')).toBeInTheDocument()
  })

  it('fetches stats on mount and shows Active status', async () => {
    mockGetStats.mockResolvedValue({ history_length: 10, avg_temperature: 0.75, avg_top_p: 0.9 })
    render(<MetaWeightsPage />)
    await waitFor(() => {
      expect(screen.getByTestId('stat-Status')).toHaveTextContent('Active')
    }, { timeout: 5000 })
    expect(mockGetStats).toHaveBeenCalled()
  })

  it('shows Unavailable when stats fetch fails', async () => {
    mockGetStats.mockRejectedValue(new Error('fail'))
    render(<MetaWeightsPage />)
    await waitFor(() => {
      expect(screen.getByTestId('stat-Status')).toHaveTextContent('Unavailable')
    }, { timeout: 5000 })
  })

  it('computes weights when test message entered', async () => {
    mockGetStats.mockResolvedValue({ history_length: 5 })
    mockGetWeights.mockResolvedValue({ temperature: 0.8, top_p: 0.9, repetition_penalty: 1.1, top_k: 40, style_bias: 0.5, confidence_boost: 0.6, based_on_samples: 5 })
    render(<MetaWeightsPage />)
    await waitFor(() => {
      expect(screen.getByTestId('stat-Status')).toHaveTextContent('Active')
    }, { timeout: 5000 })

    const input = screen.getByPlaceholderText('Type a message to test...')
    fireEvent.change(input, { target: { value: 'hello' } })
    fireEvent.click(screen.getByText('Compute'))

    await waitFor(() => {
      expect(screen.getAllByText('Based on 5 feedback samples').length).toBeGreaterThanOrEqual(1)
    }, { timeout: 5000 })
    expect(screen.getByText('40')).toBeInTheDocument()
  })

  it('shows How It Works section', async () => {
    mockGetStats.mockResolvedValue({ history_length: 0 })
    render(<MetaWeightsPage />)
    expect(screen.getByText('How It Works')).toBeInTheDocument()
    expect(screen.getByText(/Meta-weights adjust inference parameters/)).toBeInTheDocument()
  })
})
