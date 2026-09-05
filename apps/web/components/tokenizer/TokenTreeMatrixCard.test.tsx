import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mocks = vi.hoisted(() => ({
  mockGetMatrixSummary: vi.fn(),
}))

vi.mock('@/lib/token-tree-controller', () => ({
  tokenTreeController: {
    getMatrixSummary: mocks.mockGetMatrixSummary,
  },
}))

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  ActionCard: ({ title, actions, children, ...p }: any) => <div data-testid="action-card" {...p}>{title}{actions}{children}</div>,
  CardHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Chip: ({ label }: { label: string }) => <span>{label}</span>,
  Skeleton: ({ className }: { className?: string }) => <div data-testid="skeleton" className={className} />,
  IconRefresh: () => <svg data-testid="icon-refresh" />,
}))

import { TokenTreeMatrixCard } from './TokenTreeMatrixCard'

const SUMMARY = {
  matrix: [128, 16],
  norm_min: 0.5,
  norm_mean: 0.8,
  norm_max: 1,
  dead_tokens: 2,
  live_tokens: 126,
  most_energetic: [
    ['quick</w>', 12, 1],
    ['brown</w>', 20, 0.98],
  ],
  least_energetic: [['the</w>', 3, 0.55]],
}

describe('TokenTreeMatrixCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(cleanup)

  it('renders the card title', () => {
    render(<TokenTreeMatrixCard />)
    expect(screen.getByText('Embedding Matrix Overview')).toBeDefined()
  })

  it('fetches the matrix summary on mount', async () => {
    mocks.mockGetMatrixSummary.mockResolvedValue(SUMMARY)
    render(<TokenTreeMatrixCard />)
    await waitFor(() => expect(mocks.mockGetMatrixSummary).toHaveBeenCalledWith(8))
    expect(screen.getByText('128 x 16 matrix')).toBeDefined()
    expect(screen.getByText('norm 0.500–1.000')).toBeDefined()
    expect(screen.getByText('mean 0.800')).toBeDefined()
    expect(screen.getByText('126 live / 2 dead')).toBeDefined()
  })

  it('shows most and least energetic tokens', async () => {
    mocks.mockGetMatrixSummary.mockResolvedValue(SUMMARY)
    render(<TokenTreeMatrixCard />)
    await waitFor(() => expect(screen.getByText('Most energetic')).toBeDefined())
    expect(screen.getByText(/quick/)).toBeDefined()
    expect(screen.getByText(/brown/)).toBeDefined()
    expect(screen.getByText('Least energetic')).toBeDefined()
    expect(screen.getAllByText(/the/).length).toBeGreaterThan(0)
    expect(screen.getByText('1.0000')).toBeDefined()
    expect(screen.getByText('0.5500')).toBeDefined()
  })

  it('shows the disabled state when matrix is null', async () => {
    mocks.mockGetMatrixSummary.mockResolvedValue({ ...SUMMARY, matrix: null })
    render(<TokenTreeMatrixCard />)
    await waitFor(() =>
      expect(screen.getByText(/Embeddings are disabled for this tree/)).toBeDefined(),
    )
  })

  it('refreshes when the refresh button is clicked', async () => {
    mocks.mockGetMatrixSummary.mockResolvedValue(SUMMARY)
    render(<TokenTreeMatrixCard />)
    await waitFor(() => expect(mocks.mockGetMatrixSummary).toHaveBeenCalledTimes(1))
    fireEvent.click(screen.getByRole('button', { name: /Refresh matrix overview/ }))
    await waitFor(() => expect(mocks.mockGetMatrixSummary).toHaveBeenCalledTimes(2))
  })

  it('shows an error box when the fetch fails', async () => {
    mocks.mockGetMatrixSummary.mockRejectedValue(new Error('boom'))
    render(<TokenTreeMatrixCard />)
    await waitFor(() =>
      expect(screen.getByText(/Could not load the embedding matrix overview/)).toBeDefined(),
    )
  })

  it('shows a skeleton while loading', async () => {
    let resolvePromise: any
    mocks.mockGetMatrixSummary.mockImplementation(() => new Promise(r => { resolvePromise = r }))
    render(<TokenTreeMatrixCard />)
    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThan(0)
    resolvePromise(SUMMARY)
    await waitFor(() => expect(screen.getByText('128 x 16 matrix')).toBeDefined())
  })
})
