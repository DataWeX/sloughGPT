import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mocks = vi.hoisted(() => ({
  mockListSaved: vi.fn(),
  mockCompare: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/token-tree-controller', () => ({
  tokenTreeController: {
    listSaved: mocks.mockListSaved,
    compare: mocks.mockCompare,
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: typeof mocks.mockAddToast }) => unknown) =>
    selector({ addToast: mocks.mockAddToast }),
}))

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Skeleton: ({ className }: { className: string }) => <div data-testid="skeleton" className={className} />,
  Chip: ({ label }: { label: string }) => <span>{label}</span>,
  Select: ({ value, onValueChange, children }: any) => (
    <select
      value={value ?? ''}
      onChange={e => onValueChange?.(e.target.value)}
      data-testid="select"
    >
      {children}
    </select>
  ),
  SelectTrigger: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  SelectContent: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  SelectItem: ({ value, children }: any) => <option value={value}>{children}</option>,
  SelectValue: () => null,
}))

import { TokenTreeCompareCard } from './TokenTreeCompareCard'

const TREES = [
  { name: 'alpha', path: '/x', vocab_size: 53, num_merges: 20, trained: true, saved_at: 1000 },
  { name: 'beta', path: '/y', vocab_size: 61, num_merges: 27, trained: true, saved_at: 2000 },
]

const RESULT = {
  a: { name: 'alpha', stats: { vocab_size: 53 }, vocab: {} },
  b: { name: 'beta', stats: { vocab_size: 61 }, vocab: {} },
  shared_tokens: 40,
  only_a_tokens: 13,
  only_b_tokens: 21,
  shared_merges: 15,
  only_a_merges: 5,
  only_b_merges: 12,
  shared_examples: [['the', 100], ['quick', 60]],
  only_a_examples: [['fox', 30]],
  only_b_examples: [['jumps', 25]],
}

describe('TokenTreeCompareCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.mockListSaved.mockResolvedValue(TREES)
  })

  afterEach(cleanup)

  const selectTree = async (index: number, value: string) => {
    const selects = screen.getAllByTestId('select')
    fireEvent.change(selects[index], { target: { value } })
    await waitFor(() => expect((selects[index] as HTMLSelectElement).value).toBe(value))
  }

  it('fetches saved trees on mount', async () => {
    render(<TokenTreeCompareCard />)
    await waitFor(() => expect(mocks.mockListSaved).toHaveBeenCalledTimes(1))
  })

  it('disables Compare until two distinct trees are picked', async () => {
    render(<TokenTreeCompareCard />)
    await waitFor(() => expect(screen.getAllByTestId('select').length).toBe(2))
    const btn = screen.getByRole('button', { name: /Compare/ }) as HTMLButtonElement
    expect(btn.disabled).toBe(true)

    await selectTree(0, 'alpha')
    expect(btn.disabled).toBe(true)

    await selectTree(1, 'alpha')
    expect(btn.disabled).toBe(true)
    expect(screen.getByText(/Choose two different trees/)).toBeDefined()

    await selectTree(1, 'beta')
    expect(btn.disabled).toBe(false)
  })

  it('calls compare with both trees and top_k 10', async () => {
    mocks.mockCompare.mockResolvedValue(RESULT)
    render(<TokenTreeCompareCard />)
    await waitFor(() => expect(screen.getAllByTestId('select').length).toBe(2))
    await selectTree(0, 'alpha')
    await selectTree(1, 'beta')

    fireEvent.click(screen.getByRole('button', { name: /Compare/ }))
    await waitFor(() => expect(mocks.mockCompare).toHaveBeenCalledWith('alpha', 'beta', 10))
  })

  it('renders the comparison summary and examples', async () => {
    mocks.mockCompare.mockResolvedValue(RESULT)
    render(<TokenTreeCompareCard />)
    await waitFor(() => expect(screen.getAllByTestId('select').length).toBe(2))
    await selectTree(0, 'alpha')
    await selectTree(1, 'beta')

    fireEvent.click(screen.getByRole('button', { name: /Compare/ }))
    expect(await screen.findByText('40')).toBeDefined()
    expect(screen.getByText('13')).toBeDefined()
    expect(screen.getByText('21')).toBeDefined()
    expect(screen.getByText('15 shared merges')).toBeDefined()
    expect(screen.getByText('5 merges in A')).toBeDefined()
    expect(screen.getByText('12 merges in B')).toBeDefined()
    expect(screen.getByText(/Shared token examples/)).toBeDefined()
    expect(screen.getByText(/Only in alpha/)).toBeDefined()
    expect(screen.getByText(/Only in beta/)).toBeDefined()
    const hasText = (token: string, freq: number) => (content: string, element: Element | null): boolean => {
      const text = element?.textContent ?? content
      return text.includes(token) && text.includes(`×${freq}`)
    }
    expect(screen.getAllByText(hasText('the', 100)).length).toBeGreaterThan(0)
    expect(screen.getAllByText(hasText('quick', 60)).length).toBeGreaterThan(0)
    expect(screen.getAllByText(hasText('fox', 30)).length).toBeGreaterThan(0)
    expect(screen.getAllByText(hasText('jumps', 25)).length).toBeGreaterThan(0)
  })

  it('shows an error toast when compare fails', async () => {
    mocks.mockCompare.mockRejectedValue(new Error('boom'))
    render(<TokenTreeCompareCard />)
    await waitFor(() => expect(screen.getAllByTestId('select').length).toBe(2))
    await selectTree(0, 'alpha')
    await selectTree(1, 'beta')

    fireEvent.click(screen.getByRole('button', { name: /Compare/ }))
    await waitFor(() => expect(mocks.mockAddToast).toHaveBeenCalledWith('boom', 'error'))
  })

  it('shows an empty state when fewer than two trees exist', async () => {
    mocks.mockListSaved.mockResolvedValue([TREES[0]])
    render(<TokenTreeCompareCard />)
    await waitFor(() => expect(screen.getByText(/save at least two trees/)).toBeDefined())
  })

  it('shows a retry state when the fetch fails', async () => {
    mocks.mockListSaved.mockRejectedValueOnce(new Error('boom'))
    render(<TokenTreeCompareCard />)
    await waitFor(() => expect(screen.getByText(/Could not load saved trees/)).toBeDefined())

    mocks.mockListSaved.mockResolvedValue(TREES)
    fireEvent.click(screen.getByText(/Retry/))
    await waitFor(() => expect(screen.getAllByTestId('select').length).toBe(2))
  })

  it('refetches when refreshKey changes', async () => {
    const { rerender } = render(<TokenTreeCompareCard refreshKey={0} />)
    await waitFor(() => expect(mocks.mockListSaved).toHaveBeenCalledTimes(1))

    rerender(<TokenTreeCompareCard refreshKey={1} />)
    await waitFor(() => expect(mocks.mockListSaved).toHaveBeenCalledTimes(2))
  })
})
