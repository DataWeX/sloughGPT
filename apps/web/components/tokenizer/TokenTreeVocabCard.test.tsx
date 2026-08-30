import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mocks = vi.hoisted(() => ({
  mockGetStats: vi.fn(),
  mockGetVocab: vi.fn(),
  mockLineage: vi.fn(),
}))

vi.mock('@/lib/token-tree-controller', () => ({
  tokenTreeController: {
    getStats: mocks.mockGetStats,
    getVocab: mocks.mockGetVocab,
    lineage: mocks.mockLineage,
  },
}))

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => (
    <button {...props}>{children}</button>
  ),
  Skeleton: ({ className }: { className: string }) => <div data-testid="skeleton" className={className} />,
  Chip: ({ label }: { label: string }) => <span>{label}</span>,
  IconRefresh: () => <span />,
}))

import { TokenTreeVocabCard } from './TokenTreeVocabCard'

const STATS = {
  trained: true,
  vocab_size: 53,
  num_merges: 20,
  num_base_tokens: 29,
  embedding_points: 53,
  embedding_compression_ratio: 2.0,
  embed_dim: 16,
}

const PAGE = {
  total: 3,
  entries: [
    { id: 0, token: '<pad>', freq: 0, is_special: true, is_merged: false },
    { id: 1, token: 'the</w>', freq: 7, is_special: false, is_merged: true },
    { id: 2, token: 'quick</w>', freq: 4, is_special: false, is_merged: false },
  ],
}

describe('TokenTreeVocabCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.mockGetStats.mockResolvedValue(STATS)
    mocks.mockGetVocab.mockResolvedValue(PAGE)
  })

  afterEach(cleanup)

  it('fetches stats and renders a paged vocabulary with badges', async () => {
    mocks.mockGetVocab.mockResolvedValue(PAGE)
    render(<TokenTreeVocabCard />)

    await waitFor(() => expect(mocks.mockGetStats).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(mocks.mockGetVocab).toHaveBeenCalledWith(50, 0))

    expect(screen.getByText('<pad>')).toBeDefined()
    expect(screen.getByText('the')).toBeDefined()
    expect(screen.getByText('quick')).toBeDefined()
    expect(screen.getAllByText('Special')).toHaveLength(1)
    expect(screen.getAllByText('Merged')).toHaveLength(1)
    expect(screen.getByText(/Showing 1–3 of 3/)).toBeDefined()
  })

  it('renders stat chips from stats', async () => {
    mocks.mockGetVocab.mockResolvedValue(PAGE)
    render(<TokenTreeVocabCard />)
    await waitFor(() => expect(screen.getByText('Vocab 53')).toBeDefined())
    expect(screen.getByText('Merged 20')).toBeDefined()
    expect(screen.getByText('Base 29')).toBeDefined()
  })

  it('shows an empty state when the tree is not trained', async () => {
    mocks.mockGetStats.mockResolvedValue({ ...STATS, trained: false })
    render(<TokenTreeVocabCard />)
    await waitFor(() => expect(screen.getByText(/not trained/)).toBeDefined())
    expect(mocks.mockGetVocab).not.toHaveBeenCalled()
  })

  it('shows an empty state when stats fail to load', async () => {
    mocks.mockGetStats.mockRejectedValueOnce(new Error('boom'))
    render(<TokenTreeVocabCard />)
    await waitFor(() => expect(screen.getByText(/Could not load the token tree/)).toBeDefined())
  })

  it('shows an empty state on vocab fetch failure', async () => {
    mocks.mockGetStats.mockResolvedValueOnce(STATS)
    mocks.mockGetVocab.mockRejectedValueOnce(new Error('boom'))
    render(<TokenTreeVocabCard />)
    await waitFor(() => expect(screen.getByText(/Could not load the vocabulary/)).toBeDefined())
  })

  it('paginates forward and backward', async () => {
    mocks.mockGetVocab.mockResolvedValue({ total: 103, entries: PAGE.entries })
    render(<TokenTreeVocabCard />)
    await waitFor(() => expect(screen.getByText('<pad>')).toBeDefined())

    mocks.mockGetVocab.mockResolvedValue({
      total: 103,
      entries: [{ id: 50, token: 'brown</w>', freq: 2, is_special: false, is_merged: true }],
    })
    fireEvent.click(screen.getByRole('button', { name: /Next/i }))
    await waitFor(() => expect(mocks.mockGetVocab).toHaveBeenLastCalledWith(50, 50))
    expect(screen.getByText('brown')).toBeDefined()

    mocks.mockGetVocab.mockResolvedValue({ total: 103, entries: PAGE.entries })
    fireEvent.click(screen.getByRole('button', { name: /Prev/i }))
    await waitFor(() => expect(mocks.mockGetVocab).toHaveBeenLastCalledWith(50, 0))
  })

  it('disables Prev on the first page', async () => {
    mocks.mockGetVocab.mockResolvedValue(PAGE)
    render(<TokenTreeVocabCard />)
    await waitFor(() => expect(screen.getByText('<pad>')).toBeDefined())
    expect((screen.getByRole('button', { name: /Prev/i }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('expands an entry to show its lineage and toggles closed', async () => {
    mocks.mockGetVocab.mockResolvedValue(PAGE)
    mocks.mockLineage.mockResolvedValueOnce({
      token: 'the</w>',
      leaves: ['t', 'h', 'e'],
      tree: 'the',
    })
    render(<TokenTreeVocabCard />)
    await waitFor(() => expect(screen.getByText('the')).toBeDefined())

    fireEvent.click(screen.getByLabelText('Toggle lineage for the'))
    await waitFor(() => expect(mocks.mockLineage).toHaveBeenCalledWith('the</w>'))
    expect(screen.getByText(/Merge lineage of/)).toBeDefined()

    fireEvent.click(screen.getByLabelText('Toggle lineage for the'))
    await waitFor(() => expect(screen.queryByText(/Merge lineage of/)).toBeNull())
    expect(mocks.mockLineage).toHaveBeenCalledTimes(1)
  })

  it('refreshes via the button', async () => {
    mocks.mockGetVocab.mockResolvedValue(PAGE)
    render(<TokenTreeVocabCard />)
    await waitFor(() => expect(mocks.mockGetVocab).toHaveBeenCalledTimes(1))

    mocks.mockGetVocab.mockResolvedValue(PAGE)
    fireEvent.click(screen.getByLabelText('Refresh vocabulary'))
    await waitFor(() => expect(mocks.mockGetVocab).toHaveBeenCalledTimes(2))
  })

  it('refetches when refreshKey changes', async () => {
    mocks.mockGetVocab.mockResolvedValue(PAGE)
    const { rerender } = render(<TokenTreeVocabCard refreshKey={0} />)
    await waitFor(() => expect(mocks.mockGetVocab).toHaveBeenCalledTimes(1))

    mocks.mockGetVocab.mockResolvedValue(PAGE)
    rerender(<TokenTreeVocabCard refreshKey={1} />)
    await waitFor(() => expect(mocks.mockGetVocab).toHaveBeenCalledTimes(2))
  })
})
