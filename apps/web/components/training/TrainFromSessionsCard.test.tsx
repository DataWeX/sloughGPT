// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

const mockAddToast = vi.fn()
const mockToastStore = { addToast: mockAddToast, toasts: [], dismissToast: vi.fn(), clearToasts: vi.fn() }

vi.mock('@/lib/controllers', () => ({
  trainingController: {
    getAutoTrainStatus: vi.fn(),
    trainFromSessions: vi.fn(),
    streamTrainFromSessions: vi.fn(),
    listChatSessions: vi.fn(),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel?: any) => sel ? sel(mockToastStore) : mockToastStore,
}))

import { TrainFromSessionsCard } from './TrainFromSessionsCard'
import { trainingController } from '@/lib/controllers'

const mockGetAutoTrainStatus = vi.mocked(trainingController.getAutoTrainStatus)
const mockTrainFromSessions = vi.mocked(trainingController.trainFromSessions)
const mockStreamTrainFromSessions = vi.mocked(trainingController.streamTrainFromSessions)
const mockListSessions = vi.mocked(trainingController.listChatSessions)

beforeEach(() => {
  mockAddToast.mockClear()
  mockGetAutoTrainStatus.mockResolvedValue({
    enabled: true,
    threshold: 10,
    interval_s: 300,
    pending_conversations: 5,
    total_trains: 3,
    last_loss: 2.1,
    last_checkpoint: 'sessions_1234567890',
    last_train: '2026-07-13T10:00:00Z',
    session_count: 42,
    response_log_count: 7,
  })
  mockListSessions.mockResolvedValue([
    { id: 's1', name: 'First chat', updated_at: '2026-07-13T10:00:00Z', messages: [{ role: 'user', content: 'Hello' }, { role: 'assistant', content: 'Hi' }] },
    { id: 's2', name: 'Second chat', updated_at: '2026-07-13T09:00:00Z', messages: [{ role: 'user', content: 'Bye' }] },
  ])
})

afterEach(cleanup)

describe('TrainFromSessionsCard', () => {
  it('renders loading state', () => {
    render(<TrainFromSessionsCard />)
    expect(screen.getByText('From conversations')).toBeDefined()
  })

  it('fetches auto-train status on mount', async () => {
    render(<TrainFromSessionsCard />)
    await waitFor(() => {
      expect(mockGetAutoTrainStatus).toHaveBeenCalled()
    })
  })

  it('shows auto-training enabled when enabled', async () => {
    render(<TrainFromSessionsCard />)
    await waitFor(() => {
      expect(screen.getByText('Auto-training on')).toBeDefined()
    })
  })

  it('shows training stats when enabled', async () => {
    render(<TrainFromSessionsCard />)
    await waitFor(() => {
      expect(screen.getByText(/Next train: 5 \/ 10 conversations/)).toBeDefined()
      expect(screen.getByText(/42 conversations/)).toBeDefined()
      expect(screen.getByText(/7 log files/)).toBeDefined()
    })
  })

  it('shows session and log counts', async () => {
    render(<TrainFromSessionsCard />)
    await waitFor(() => {
      expect(screen.getByText(/42 conversations/)).toBeDefined()
      expect(screen.getByText(/7 log files/)).toBeDefined()
    })
  })

  it('shows progress toward next train', async () => {
    render(<TrainFromSessionsCard />)
    await waitFor(() => {
      expect(screen.getByText(/Next train: 5 \/ 10 conversations/)).toBeDefined()
      expect(screen.getByText('50%')).toBeDefined()
    })
  })

  it('shows last train time', async () => {
    render(<TrainFromSessionsCard />)
    await waitFor(() => {
      expect(screen.getByText(/Last trained:/)).toBeDefined()
    })
  })

  it('shows last loss in progress section', async () => {
    render(<TrainFromSessionsCard />)
    await waitFor(() => {
      expect(screen.getByText(/loss 2.1000/)).toBeDefined()
    })
  })

  it('calls streamTrainFromSessions on button click', async () => {
    async function* mockStream() {
      yield { stream: 'training', phase: 'GENERATE_DATA', status: 'working', data: { pairs: 10 }, meta: {}, message: 'Extracted 10 pairs' }
      yield { stream: 'training', phase: 'TRAIN', status: 'working', data: { step: 5, loss: 1.5, epoch: 0.5, progress_pct: 50, total_steps: 10 }, meta: { elapsed_ms: 2000 }, message: '' }
      yield { stream: 'training', phase: 'COMPLETE', status: 'complete', data: { checkpoint_name: 'sessions_999', loss: 1.5, steps: 10, elapsed_ms: 5000 }, meta: { elapsed_ms: 5000 }, message: 'Done' }
    }
    mockStreamTrainFromSessions.mockReturnValue(mockStream())

    const user = userEvent.setup()
    render(<TrainFromSessionsCard />)

    await waitFor(() => {
      expect(screen.getByText('Train from conversations')).toBeDefined()
    })

    await user.click(screen.getByText('Train from conversations'))

    await waitFor(() => {
      expect(mockStreamTrainFromSessions).toHaveBeenCalledWith({ limit: 50, min_length: 5, session_ids: undefined })
      expect(mockAddToast).toHaveBeenCalledWith(
        expect.stringContaining('Trained from all sessions'),
        'success'
      )
    })
  })

  it('shows error toast on training failure', async () => {
    async function* mockStream() {
      yield { stream: 'training', phase: 'GENERATE_DATA', status: 'error', data: { error: 'Not enough data' }, meta: {}, message: 'Error: Not enough data' }
    }
    mockStreamTrainFromSessions.mockReturnValue(mockStream())

    const user = userEvent.setup()
    render(<TrainFromSessionsCard />)

    await waitFor(() => {
      expect(screen.getByText('Train from conversations')).toBeDefined()
    })

    await user.click(screen.getByText('Train from conversations'))

    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Not enough data', 'error')
    })
  })
})
