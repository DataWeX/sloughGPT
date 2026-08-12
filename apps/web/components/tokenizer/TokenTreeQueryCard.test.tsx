import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'

const mocks = vi.hoisted(() => ({
  mockGetStats: vi.fn(),
  mockSimilar: vi.fn(),
  mockLineage: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/token-tree-controller', () => ({
  tokenTreeController: {
    getStats: mocks.mockGetStats,
    similar: mocks.mockSimilar,
    lineage: mocks.mockLineage,
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
  Button: ({ children, onClick, disabled }: any) => (
    <button onClick={onClick} disabled={disabled}>{children}</button>
  ),
  Chip: ({ label }: { label: string }) => <span>{label}</span>,
  Skeleton: ({ className }: { className?: string }) => (
    <div data-testid="skeleton" className={className} />
  ),
  IconSearch: () => <span data-testid="icon-search" />,
}))

import { TokenTreeQueryCard } from './TokenTreeQueryCard'

const STATS = {
  trained: true,
  vocab_size: 211,
  num_merges: 20,
  num_base_tokens: 191,
  embedding_points: 211,
  embedding_compression_ratio: 2.0,
  embed_dim: 16,
}

const SIMILAR = {
  query: 'quick</w>',
  neighbors: [
    { id: 20, token: 'brown</w>', score: 0.85 },
    { id: 33, token: 'lazy</w>', score: 0.71 },
  ],
}

const LINEAGE = {
  token: 'brown</w>',
  leaves: ['b', 'r', 'o', 'w', 'n', '</w>'],
  tree: 'brown</w>\n  br\n    brown',
}

describe('TokenTreeQueryCard', () => {
  beforeEach(() => {
    mocks.mockGetStats.mockResolvedValue(STATS)
    mocks.mockSimilar.mockResolvedValue(SIMILAR)
    mocks.mockLineage.mockResolvedValue(LINEAGE)
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders stat chips from getStats on mount', async () => {
    render(<TokenTreeQueryCard />)
    expect(await screen.findByText('Vocab 211')).toBeDefined()
    expect(screen.getByText('Embed dim 16')).toBeDefined()
    expect(screen.getByText('Points 211')).toBeDefined()
    expect(screen.getByText('Compression 2x')).toBeDefined()
    expect(mocks.mockGetStats).toHaveBeenCalledWith()
  })

  it('shows skeletons while stats are loading', () => {
    mocks.mockGetStats.mockImplementation(() => new Promise(() => {}))
    render(<TokenTreeQueryCard />)
    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThan(0)
  })

  it('searches neighbors on button click', async () => {
    render(<TokenTreeQueryCard />)
    await screen.findByText('Vocab 211')
    fireEvent.change(screen.getByLabelText('Token to query'), { target: { value: 'quick' } })
    fireEvent.click(screen.getByText('Find neighbors'))
    await waitFor(() => expect(mocks.mockSimilar).toHaveBeenCalledWith('quick', 5))
    expect(await screen.findByText('brown')).toBeDefined()
    expect(screen.getByText('lazy')).toBeDefined()
    expect(screen.getByText('0.850')).toBeDefined()
    expect(screen.getByText(/Nearest neighbors of/)).toBeDefined()
  })

  it('triggers search on Enter key', async () => {
    render(<TokenTreeQueryCard />)
    await screen.findByText('Vocab 211')
    const input = screen.getByLabelText('Token to query')
    fireEvent.change(input, { target: { value: 'fox' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    await waitFor(() => expect(mocks.mockSimilar).toHaveBeenCalledWith('fox', 5))
  })

  it('shows a toast when the token is not found', async () => {
    mocks.mockSimilar.mockRejectedValue(new Error('not found'))
    render(<TokenTreeQueryCard />)
    await screen.findByText('Vocab 211')
    fireEvent.change(screen.getByLabelText('Token to query'), { target: { value: 'zzz' } })
    fireEvent.click(screen.getByText('Find neighbors'))
    await waitFor(() => expect(mocks.mockAddToast).toHaveBeenCalledWith('Token not found: zzz', 'error'))
  })

  it('renders empty state when no neighbors are returned', async () => {
    mocks.mockSimilar.mockResolvedValue({ query: 'x</w>', neighbors: [] })
    render(<TokenTreeQueryCard />)
    await screen.findByText('Vocab 211')
    fireEvent.change(screen.getByLabelText('Token to query'), { target: { value: 'x' } })
    fireEvent.click(screen.getByText('Find neighbors'))
    expect(await screen.findByText('No neighbors found.')).toBeDefined()
  })

  it('fetches and shows real lineage when a neighbor is expanded', async () => {
    render(<TokenTreeQueryCard />)
    await screen.findByText('Vocab 211')
    fireEvent.click(screen.getByText('Find neighbors'))
    const neighbor = await screen.findByLabelText('Toggle lineage for brown')
    fireEvent.click(neighbor)
    await waitFor(() => expect(mocks.mockLineage).toHaveBeenCalledWith('20'))
    expect(await screen.findByText(/brown<\/w>/)).toBeDefined()
    expect(screen.getAllByText(/brown<\/w>/).length).toBeGreaterThan(0)
    expect(mocks.mockLineage).toHaveBeenCalledWith('20')
  })

  it('toggles lineage off on second click', async () => {
    render(<TokenTreeQueryCard />)
    await screen.findByText('Vocab 211')
    fireEvent.click(screen.getByText('Find neighbors'))
    const neighbor = await screen.findByLabelText('Toggle lineage for brown')
    fireEvent.click(neighbor)
    await waitFor(() => expect(mocks.mockLineage).toHaveBeenCalledWith('20'))
    expect(await screen.findByText(/brown<\/w>/)).toBeDefined()
    fireEvent.click(screen.getByLabelText('Toggle lineage for brown'))
    await waitFor(() => expect(screen.queryAllByText(/brown<\/w>/).length).toBe(0))
  })

  it('renders "Lineage unavailable" when lineage fetch fails', async () => {
    mocks.mockLineage.mockRejectedValue(new Error('boom'))
    render(<TokenTreeQueryCard />)
    await screen.findByText('Vocab 211')
    fireEvent.click(screen.getByText('Find neighbors'))
    const neighbor = await screen.findByLabelText('Toggle lineage for brown')
    fireEvent.click(neighbor)
    expect(await screen.findByText('Lineage unavailable.')).toBeDefined()
  })

  it('disables the search button when the query is empty', async () => {
    render(<TokenTreeQueryCard />)
    await screen.findByText('Vocab 211')
    fireEvent.change(screen.getByLabelText('Token to query'), { target: { value: '  ' } })
    const button = screen.getByText('Find neighbors').closest('button')
    expect(button).not.toBeNull()
    expect(button?.disabled).toBe(true)
  })
})
