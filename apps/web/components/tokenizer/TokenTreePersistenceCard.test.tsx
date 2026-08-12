import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mocks = vi.hoisted(() => ({
  mockGetStats: vi.fn(),
  mockListSaved: vi.fn(),
  mockSaveTree: vi.fn(),
  mockLoadTree: vi.fn(),
  mockDeleteSavedTree: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/token-tree-controller', () => ({
  tokenTreeController: {
    getStats: mocks.mockGetStats,
    listSaved: mocks.mockListSaved,
    saveTree: mocks.mockSaveTree,
    loadTree: mocks.mockLoadTree,
    deleteSavedTree: mocks.mockDeleteSavedTree,
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: typeof mocks.mockAddToast }) => unknown) =>
    selector({ addToast: mocks.mockAddToast }),
}))

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Input: (props: any) => <input {...props} />,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Skeleton: ({ className }: { className: string }) => <div data-testid="skeleton" className={className} />,
  Chip: ({ label }: { label: string }) => <span>{label}</span>,
  IconRefresh: () => <span />,
}))

import { TokenTreePersistenceCard } from './TokenTreePersistenceCard'

const STATS = {
  trained: true,
  vocab_size: 53,
  num_merges: 20,
  num_base_tokens: 29,
  embedding_points: 53,
  embedding_compression_ratio: 2.0,
  embed_dim: 16,
}

const TREES = [
  {
    name: 'the-default',
    path: '/data/token_trees/the-default',
    vocab_size: 53,
    num_merges: 20,
    trained: true,
    saved_at: 1000,
  },
]

describe('TokenTreePersistenceCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.mockGetStats.mockResolvedValue(STATS)
    mocks.mockListSaved.mockResolvedValue(TREES)
  })

  afterEach(cleanup)

  it('fetches stats and saved trees on mount', async () => {
    render(<TokenTreePersistenceCard />)
    await waitFor(() => expect(mocks.mockGetStats).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(mocks.mockListSaved).toHaveBeenCalledTimes(1))
    expect(await screen.findByText('the-default')).toBeDefined()
    expect(screen.getByText('Vocab 53')).toBeDefined()
  })

  it('disables the save button when the tree is not trained', async () => {
    mocks.mockGetStats.mockResolvedValue({ ...STATS, trained: false })
    render(<TokenTreePersistenceCard />)
    await waitFor(() => expect(screen.getByText('the-default')).toBeDefined())
    expect((screen.getByRole('button', { name: /^Save$/ }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('disables the save button while the name is blank', async () => {
    render(<TokenTreePersistenceCard />)
    await waitFor(() => expect(screen.getByText('the-default')).toBeDefined())
    expect((screen.getByRole('button', { name: /^Save$/ }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('saves a named tree and refreshes the list', async () => {
    mocks.mockSaveTree.mockResolvedValue(TREES[0])
    render(<TokenTreePersistenceCard />)
    await waitFor(() => expect(screen.getByText('the-default')).toBeDefined())

    fireEvent.change(screen.getByLabelText('Saved tree name'), { target: { value: 'shakespeare' } })
    fireEvent.click(screen.getByRole('button', { name: /^Save$/ }))
    await waitFor(() => expect(mocks.mockSaveTree).toHaveBeenCalledWith('shakespeare'))
    await waitFor(() => expect(mocks.mockListSaved).toHaveBeenCalledTimes(2))
    expect(mocks.mockAddToast).toHaveBeenCalledWith('Saved token tree "shakespeare"', 'success')
    expect((screen.getByLabelText('Saved tree name') as HTMLInputElement).value).toBe('')
  })

  it('shows an error toast when saving fails', async () => {
    mocks.mockSaveTree.mockRejectedValue(new Error('nope'))
    render(<TokenTreePersistenceCard />)
    await waitFor(() => expect(screen.getByText('the-default')).toBeDefined())

    fireEvent.change(screen.getByLabelText('Saved tree name'), { target: { value: 'shakespeare' } })
    fireEvent.click(screen.getByRole('button', { name: /^Save$/ }))
    await waitFor(() => expect(mocks.mockAddToast).toHaveBeenCalledWith('nope', 'error'))
  })

  it('loads a saved tree and notifies the page', async () => {
    mocks.mockLoadTree.mockResolvedValue(TREES[0])
    const onLoaded = vi.fn()
    render(<TokenTreePersistenceCard onLoaded={onLoaded} />)
    await waitFor(() => expect(screen.getByText('the-default')).toBeDefined())

    fireEvent.click(screen.getByRole('button', { name: /Load the-default/ }))
    await waitFor(() => expect(mocks.mockLoadTree).toHaveBeenCalledWith('the-default'))
    expect(mocks.mockAddToast).toHaveBeenCalledWith('Loaded token tree "the-default"', 'success')
    expect(onLoaded).toHaveBeenCalled()
  })

  it('deletes a saved tree and refreshes the list', async () => {
    mocks.mockDeleteSavedTree.mockResolvedValue({ deleted: true })
    render(<TokenTreePersistenceCard />)
    await waitFor(() => expect(screen.getByText('the-default')).toBeDefined())

    fireEvent.click(screen.getByRole('button', { name: /Delete the-default/ }))
    await waitFor(() => expect(mocks.mockDeleteSavedTree).toHaveBeenCalledWith('the-default'))
    await waitFor(() => expect(mocks.mockListSaved).toHaveBeenCalledTimes(2))
    expect(mocks.mockAddToast).toHaveBeenCalledWith('Deleted token tree "the-default"', 'success')
  })

  it('shows an empty state when no trees are saved', async () => {
    mocks.mockListSaved.mockResolvedValue([])
    render(<TokenTreePersistenceCard />)
    await waitFor(() => expect(screen.getByText(/No saved trees yet/)).toBeDefined())
  })

  it('shows a retry state when the fetch fails', async () => {
    mocks.mockGetStats.mockRejectedValueOnce(new Error('boom'))
    mocks.mockListSaved.mockRejectedValueOnce(new Error('boom'))
    render(<TokenTreePersistenceCard />)
    await waitFor(() => expect(screen.getByText(/Could not load saved trees/)).toBeDefined())

    mocks.mockGetStats.mockResolvedValue(STATS)
    mocks.mockListSaved.mockResolvedValue(TREES)
    fireEvent.click(screen.getByText(/Retry/))
    await waitFor(() => expect(screen.getByText('the-default')).toBeDefined())
  })

  it('refetches when refreshKey changes', async () => {
    const { rerender } = render(<TokenTreePersistenceCard refreshKey={0} />)
    await waitFor(() => expect(mocks.mockListSaved).toHaveBeenCalledTimes(1))

    rerender(<TokenTreePersistenceCard refreshKey={1} />)
    await waitFor(() => expect(mocks.mockListSaved).toHaveBeenCalledTimes(2))
  })
})
