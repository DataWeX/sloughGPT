import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'

const mocks = vi.hoisted(() => ({
  mockLineage: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/token-tree-controller', () => ({
  tokenTreeController: {
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
  IconActivity: () => <span data-testid="icon-activity" />,
}))

import { TokenTreeLineageCard } from './TokenTreeLineageCard'

const LINEAGE = {
  token: 'quick</w>',
  leaves: ['q', 'u', 'i', 'c', 'k', '</w>'],
  tree: 'quick</w>\n  qu\n    ick',
}

describe('TokenTreeLineageCard', () => {
  beforeEach(() => {
    mocks.mockLineage.mockResolvedValue(LINEAGE)
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('fetches and renders a token lineage on click', async () => {
    render(<TokenTreeLineageCard />)
    fireEvent.change(screen.getByLabelText('Token to inspect'), { target: { value: 'quick' } })
    fireEvent.click(screen.getByText('Show lineage'))
    await waitFor(() => expect(mocks.mockLineage).toHaveBeenCalledWith('quick'))
    expect(await screen.findByText(/Merge lineage of/)).toBeDefined()
    expect(screen.getByText('q')).toBeDefined()
    expect(screen.getByText('</w>')).toBeDefined()
    expect(screen.getByText(/quick<\/w>/)).toBeDefined()
  })

  it('shows a skeleton while loading', async () => {
    mocks.mockLineage.mockImplementation(() => new Promise(() => {}))
    render(<TokenTreeLineageCard />)
    fireEvent.change(screen.getByLabelText('Token to inspect'), { target: { value: 'quick' } })
    fireEvent.click(screen.getByText('Show lineage'))
    expect(await screen.findByTestId('skeleton')).toBeDefined()
  })

  it('triggers lookup on Enter key', async () => {
    render(<TokenTreeLineageCard />)
    const input = screen.getByLabelText('Token to inspect')
    fireEvent.change(input, { target: { value: 'fox' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    await waitFor(() => expect(mocks.mockLineage).toHaveBeenCalledWith('fox'))
  })

  it('shows a toast when the token is not in the vocabulary', async () => {
    mocks.mockLineage.mockRejectedValue(new Error('not found'))
    render(<TokenTreeLineageCard />)
    fireEvent.change(screen.getByLabelText('Token to inspect'), { target: { value: 'zzz' } })
    fireEvent.click(screen.getByText('Show lineage'))
    await waitFor(() => expect(mocks.mockAddToast).toHaveBeenCalledWith('Token not in vocabulary: zzz', 'error'))
    expect(screen.queryByText(/Merge lineage of/)).toBeNull()
  })

  it('disables the button when the input is empty', async () => {
    render(<TokenTreeLineageCard />)
    fireEvent.change(screen.getByLabelText('Token to inspect'), { target: { value: '   ' } })
    const button = screen.getByText('Show lineage').closest('button')
    expect(button).not.toBeNull()
    expect(button?.disabled).toBe(true)
    expect(mocks.mockLineage).not.toHaveBeenCalled()
  })

  it('displays the merge tree as preformatted text', async () => {
    render(<TokenTreeLineageCard />)
    fireEvent.change(screen.getByLabelText('Token to inspect'), { target: { value: 'quick' } })
    fireEvent.click(screen.getByText('Show lineage'))
    await waitFor(() => {
      expect(screen.getByText('q')).toBeDefined()
    })
    expect(document.querySelector('pre')?.textContent).toContain('quick')
  })

  it('shows character leaf count', async () => {
    render(<TokenTreeLineageCard />)
    fireEvent.change(screen.getByLabelText('Token to inspect'), { target: { value: 'quick' } })
    fireEvent.click(screen.getByText('Show lineage'))
    await waitFor(() => {
      expect(screen.getByText('q')).toBeDefined()
    })
    expect(screen.getByText(/6 character leaves/)).toBeDefined()
  })

  it('renders the CardTitle', () => {
    render(<TokenTreeLineageCard />)
    expect(screen.getByText('Merge Lineage Explorer')).toBeDefined()
  })
})
