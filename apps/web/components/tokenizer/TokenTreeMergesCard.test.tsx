import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mocks = vi.hoisted(() => ({
  mockGetMerges: vi.fn(),
  mockLineage: vi.fn(),
}))

vi.mock('@/lib/token-tree-controller', () => ({
  tokenTreeController: { getMerges: mocks.mockGetMerges, lineage: mocks.mockLineage },
}))

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => (
    <button {...props}>{children}</button>
  ),
  Input: (props: any) => <input {...props} />,
  Skeleton: ({ className }: { className: string }) => <div data-testid="skeleton" className={className} />,
  Chip: ({ label }: { label: string }) => <span>{label}</span>,
  IconRefresh: () => <span />,
  IconSearch: () => <span />,
}))

import { TokenTreeMergesCard } from './TokenTreeMergesCard'

const RULES = [
  { rank: 1, left: 'th', right: 'e', token: 'the</w>', count: 42 },
  { rank: 2, left: 'qu', right: 'ic', token: 'quic</w>', count: 30 },
  { rank: 3, left: 'quic', right: 'k', token: 'quick</w>', count: 18 },
]

describe('TokenTreeMergesCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(cleanup)

  it('fetches and renders ranked merge rules', async () => {
    mocks.mockGetMerges.mockResolvedValueOnce(RULES)
    render(<TokenTreeMergesCard />)

    await waitFor(() => expect(mocks.mockGetMerges).toHaveBeenCalledWith(20, ''))
    expect(screen.getByText('the')).toBeDefined()
    expect(screen.getAllByText('quic')).toHaveLength(2)
    expect(screen.getByText('quick')).toBeDefined()
    expect(screen.getAllByText('42')).toHaveLength(1)
  })

  it('searches with the typed query', async () => {
    mocks.mockGetMerges.mockResolvedValueOnce(RULES)
    render(<TokenTreeMergesCard />)
    await waitFor(() => expect(mocks.mockGetMerges).toHaveBeenCalledWith(20, ''))

    mocks.mockGetMerges.mockResolvedValueOnce([RULES[0]])
    fireEvent.change(screen.getByLabelText('Filter merge rules'), { target: { value: 'the' } })
    fireEvent.click(screen.getByRole('button', { name: /^Search$/i }))

    await waitFor(() => expect(mocks.mockGetMerges).toHaveBeenLastCalledWith(20, 'the'))
  })

  it('shows an empty state when no rules match the search', async () => {
    mocks.mockGetMerges.mockResolvedValueOnce(RULES)
    render(<TokenTreeMergesCard />)
    await waitFor(() => expect(mocks.mockGetMerges).toHaveBeenCalled())

    mocks.mockGetMerges.mockResolvedValueOnce([])
    fireEvent.change(screen.getByLabelText('Filter merge rules'), { target: { value: 'zzz' } })
    fireEvent.click(screen.getByRole('button', { name: /^Search$/i }))

    await waitFor(() => expect(screen.getByText(/No merge rules match "zzz"/)).toBeDefined())
  })

  it('shows an empty state when no rules exist', async () => {
    mocks.mockGetMerges.mockResolvedValueOnce([])
    render(<TokenTreeMergesCard />)
    await waitFor(() => expect(screen.getByText(/No merge rules yet/)).toBeDefined())
  })

  it('shows an empty state on fetch failure', async () => {
    mocks.mockGetMerges.mockRejectedValueOnce(new Error('boom'))
    render(<TokenTreeMergesCard />)
    await waitFor(() => expect(screen.getByText(/No merge rules yet/)).toBeDefined())
  })

  it('shows skeletons while loading', async () => {
    let resolveMerges!: (v: typeof RULES) => void
    mocks.mockGetMerges.mockReturnValue(new Promise(r => { resolveMerges = r }))
    render(<TokenTreeMergesCard />)
    expect(screen.getAllByTestId('skeleton')).toHaveLength(3)
    resolveMerges(RULES)
    await waitFor(() => expect(screen.queryAllByTestId('skeleton')).toHaveLength(0))
  })

  it('expands a rule to show its merge lineage', async () => {
    mocks.mockGetMerges.mockResolvedValueOnce(RULES)
    mocks.mockLineage.mockResolvedValueOnce({
      token: 'the</w>',
      leaves: ['t', 'h', 'e'],
      tree: 'the\n├─ th\n│  ├─ t\n│  └─ h\n└─ e',
    })
    render(<TokenTreeMergesCard />)
    await waitFor(() => expect(screen.getByText('the')).toBeDefined())

    fireEvent.click(screen.getByLabelText('Toggle lineage for the'))

    await waitFor(() => expect(mocks.mockLineage).toHaveBeenCalledWith('the</w>'))
    expect(screen.getAllByText('t')).toHaveLength(1)
    expect(screen.getAllByText('h')).toHaveLength(1)
    expect(screen.getAllByText('e')).toHaveLength(2)
    expect(screen.getByText(/Merge lineage of/)).toBeDefined()
  })

  it('toggles a rule lineage closed on second click', async () => {
    mocks.mockGetMerges.mockResolvedValueOnce(RULES)
    mocks.mockLineage.mockResolvedValueOnce({
      token: 'the</w>',
      leaves: ['t', 'h', 'e'],
      tree: 'the',
    })
    render(<TokenTreeMergesCard />)
    await waitFor(() => expect(screen.getByText('the')).toBeDefined())

    fireEvent.click(screen.getByLabelText('Toggle lineage for the'))
    await waitFor(() => expect(mocks.mockLineage).toHaveBeenCalledTimes(1))

    fireEvent.click(screen.getByLabelText('Toggle lineage for the'))
    await waitFor(() => expect(screen.queryByText(/Merge lineage of/)).toBeNull())
    expect(mocks.mockLineage).toHaveBeenCalledTimes(1)
  })

  it('loads more rules in page-size increments', async () => {
    mocks.mockGetMerges.mockResolvedValueOnce(RULES)
    render(<TokenTreeMergesCard />)
    await waitFor(() => expect(screen.getByText('the')).toBeDefined())

    mocks.mockGetMerges.mockResolvedValueOnce(RULES)
    fireEvent.click(screen.getByRole('button', { name: /Show 20 more/i }))

    await waitFor(() => expect(mocks.mockGetMerges).toHaveBeenLastCalledWith(40, ''))
  })

  it('refetches when refreshKey changes', async () => {
    mocks.mockGetMerges.mockResolvedValueOnce(RULES)
    const { rerender } = render(<TokenTreeMergesCard refreshKey={0} />)
    await waitFor(() => expect(mocks.mockGetMerges).toHaveBeenCalledTimes(1))

    mocks.mockGetMerges.mockResolvedValueOnce(RULES)
    rerender(<TokenTreeMergesCard refreshKey={1} />)
    await waitFor(() => expect(mocks.mockGetMerges).toHaveBeenCalledTimes(2))
  })

  it('refreshes via the button', async () => {
    mocks.mockGetMerges.mockResolvedValueOnce(RULES)
    render(<TokenTreeMergesCard />)
    await waitFor(() => expect(mocks.mockGetMerges).toHaveBeenCalledTimes(1))

    mocks.mockGetMerges.mockResolvedValueOnce(RULES)
    fireEvent.click(screen.getByLabelText('Refresh merge rules'))
    await waitFor(() => expect(mocks.mockGetMerges).toHaveBeenCalledTimes(2))
  })
})
