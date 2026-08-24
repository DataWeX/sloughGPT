// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'

vi.mock('@/lib/knowledge-controller', () => ({
  knowledgeController: {
    getDueReviews: vi.fn(),
    list: vi.fn(),
    scheduleReview: vi.fn(),
  },
}))
vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, ...p }: any) => <div data-testid="card" {...p}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...p }: any) => <div data-testid="card-title" {...p}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, ...p }: any) => <button onClick={onClick} disabled={disabled} {...p}>{children}</button>,
  Skeleton: (p: any) => <div data-testid="skeleton" {...p} />,
}))

import { SpacedReviewCard } from './SpacedReviewCard'
import { knowledgeController } from '@/lib/knowledge-controller'

const mockToast = vi.fn()
const mockItem = { id: 'k1', content: 'The capital of France is Paris.', source: 'test' }

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(knowledgeController.getDueReviews).mockResolvedValue({ due_ids: ['k1', 'k2'], stats: { due_count: 2, total_scheduled: 2 } })
  vi.mocked(knowledgeController.list).mockResolvedValue([mockItem] as any)
  vi.mocked(knowledgeController.scheduleReview).mockResolvedValue(undefined as any)
})

afterEach(() => cleanup())

describe('SpacedReviewCard', () => {
  it('renders title when reviews available', async () => {
    render(<SpacedReviewCard addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getByText('Quick review')).toBeDefined()
    })
  })

  it('returns null when no due reviews', async () => {
    vi.mocked(knowledgeController.getDueReviews).mockResolvedValue({ due_ids: [], stats: { due_count: 0, total_scheduled: 0 } })
    const { container } = render(<SpacedReviewCard addToast={mockToast} />)
    await waitFor(() => {
      expect(container.innerHTML).toBe('')
    })
  })

  it('shows loading skeletons initially', () => {
    vi.mocked(knowledgeController.getDueReviews).mockReturnValue(new Promise(() => {}))
    render(<SpacedReviewCard addToast={mockToast} />)
    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThan(0)
  })

  it('displays current review item content', async () => {
    render(<SpacedReviewCard addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getByText('The capital of France is Paris.')).toBeDefined()
    })
  })

  it('shows Show answer button', async () => {
    render(<SpacedReviewCard addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getByText('Show answer')).toBeDefined()
    })
  })

  it('reveals rating buttons on Show answer', async () => {
    render(<SpacedReviewCard addToast={mockToast} />)
    await waitFor(() => { expect(screen.getByText('Show answer')).toBeDefined() })
    fireEvent.click(screen.getByText('Show answer'))
    expect(screen.getByText('No')).toBeDefined()
    expect(screen.getByText('Kind of')).toBeDefined()
    expect(screen.getByText('Yes')).toBeDefined()
    expect(screen.getByText('Easily')).toBeDefined()
  })

  it('submits review with correct performance score', async () => {
    render(<SpacedReviewCard addToast={mockToast} />)
    await waitFor(() => { expect(screen.getByText('Show answer')).toBeDefined() })
    fireEvent.click(screen.getByText('Show answer'))
    fireEvent.click(screen.getByText('Yes'))
    await waitFor(() => {
      expect(knowledgeController.scheduleReview).toHaveBeenCalledWith('k1', 0.8)
    })
  })

  it('moves to next item after review', async () => {
    const item2 = { id: 'k2', content: 'Second item', source: 'test' }
    vi.mocked(knowledgeController.list)
      .mockResolvedValueOnce([mockItem] as any)
      .mockResolvedValueOnce([item2] as any)
    render(<SpacedReviewCard addToast={mockToast} />)
    await waitFor(() => { expect(screen.getByText('The capital of France is Paris.')).toBeDefined() })
    fireEvent.click(screen.getByText('Show answer'))
    fireEvent.click(screen.getByText('Easily'))
    await waitFor(() => {
      expect(screen.getByText('Second item')).toBeDefined()
    })
  })
})
