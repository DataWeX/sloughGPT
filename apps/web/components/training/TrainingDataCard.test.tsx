// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

const mockAddToast = vi.fn()
const mockToastStore = { addToast: mockAddToast, toasts: [], dismissToast: vi.fn(), clearToasts: vi.fn() }

vi.mock('@/lib/controllers', () => ({
  trainingController: {
    getTrainingStats: vi.fn(),
    getPendingPairs: vi.fn(),
    deletePair: vi.fn(),
    deleteSyncedPairs: vi.fn(),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel?: any) => sel ? sel(mockToastStore) : mockToastStore,
}))

import { TrainingDataCard } from './TrainingDataCard'
import { trainingController } from '@/lib/controllers'

const mockGetStats = vi.mocked(trainingController.getTrainingStats)
const mockGetPairs = vi.mocked(trainingController.getPendingPairs)
const mockDeletePair = vi.mocked(trainingController.deletePair)
const mockDeleteSynced = vi.mocked(trainingController.deleteSyncedPairs)

beforeEach(() => {
  mockAddToast.mockClear()
  mockGetStats.mockResolvedValue({
    total: 42,
    pending: 30,
    synced: 10,
    used: 2,
    by_quality: { '0': 20, '3': 15, '5': 7 },
  })
  mockGetPairs.mockResolvedValue({
    pairs: [
      { id: 'p1', user_msg: 'Hello', assistant_msg: 'Hi there!', quality: 3.5, session_id: 's1', timestamp: 123 },
      { id: 'p2', user_msg: 'Bye', assistant_msg: 'Goodbye!', quality: 4.0, session_id: 's1', timestamp: 124 },
    ],
    count: 2,
  })
})

afterEach(cleanup)

describe('TrainingDataCard', () => {
  it('renders loading state', () => {
    render(<TrainingDataCard />)
    expect(screen.getByText('Training data')).toBeDefined()
  })

  it('fetches stats and pairs on mount', async () => {
    render(<TrainingDataCard />)
    await waitFor(() => {
      expect(mockGetStats).toHaveBeenCalled()
      expect(mockGetPairs).toHaveBeenCalledWith(20)
    })
  })

  it('shows stats', async () => {
    render(<TrainingDataCard />)
    await waitFor(() => {
      expect(screen.getByText(/Total: 42/)).toBeDefined()
      expect(screen.getByText(/Pending: 30/)).toBeDefined()
      expect(screen.getByText(/Synced: 10/)).toBeDefined()
      expect(screen.getByText(/Used: 2/)).toBeDefined()
    })
  })

  it('shows quality breakdown', async () => {
    render(<TrainingDataCard />)
    await waitFor(() => {
      expect(screen.getByText(/0\.0: 20/)).toBeDefined()
      expect(screen.getByText(/3\.0: 15/)).toBeDefined()
      expect(screen.getByText(/5\.0: 7/)).toBeDefined()
    })
  })

  it('toggles pairs list on click', async () => {
    const user = userEvent.setup()
    render(<TrainingDataCard />)

    await waitFor(() => {
      expect(screen.getByText(/Show recent pairs/)).toBeDefined()
    })

    await user.click(screen.getByText(/Show recent pairs/))

    await waitFor(() => {
      expect(screen.getByText('Hello')).toBeDefined()
      expect(screen.getByText('Hi there!')).toBeDefined()
    })
  })

  it('deletes a pair', async () => {
    mockDeletePair.mockResolvedValue({ status: 'deleted' })
    const user = userEvent.setup()
    render(<TrainingDataCard />)

    await waitFor(() => {
      expect(screen.getByText(/Show recent pairs/)).toBeDefined()
    })

    await user.click(screen.getByText(/Show recent pairs/))

    await waitFor(() => {
      expect(screen.getByText('Hello')).toBeDefined()
    })

    const deleteButtons = screen.getAllByLabelText('Delete pair')
    await user.click(deleteButtons[0])

    await waitFor(() => {
      expect(mockDeletePair).toHaveBeenCalledWith('p1')
      expect(mockAddToast).toHaveBeenCalledWith('Pair deleted', 'success')
    })
  })
})
