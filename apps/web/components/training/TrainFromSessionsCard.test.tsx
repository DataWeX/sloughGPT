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
    startFromSessionsSloNet: vi.fn(),
    streamFromSessionsSloNet: vi.fn(),
    cancelFromSessionsSloNet: vi.fn(),
    loadCheckpoint: vi.fn(),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel?: any) => sel ? sel(mockToastStore) : mockToastStore,
}))

vi.mock('@/lib/dev-log', () => ({
  logger: { debug: vi.fn(), info: vi.fn(), warning: vi.fn(), error: vi.fn() },
}))

import { TrainFromSessionsCard } from './TrainFromSessionsCard'
import { trainingController } from '@/lib/controllers'

const mockGetAutoTrainStatus = vi.mocked(trainingController.getAutoTrainStatus)
const mockStartSloNet = vi.mocked(trainingController.startFromSessionsSloNet)
const mockStreamSloNet = vi.mocked(trainingController.streamFromSessionsSloNet)
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
    captured_count: 16,
  })
  mockListSessions.mockResolvedValue([
    { id: 's1', name: 'First chat', updated_at: '2026-07-13T10:00:00Z', messages: [{ role: 'user', content: 'Hello' }, { role: 'assistant', content: 'Hi' }] },
    { id: 's2', name: 'Second chat', updated_at: '2026-07-13T09:00:00Z', messages: [{ role: 'user', content: 'Bye' }] },
  ])
  mockStartSloNet.mockResolvedValue(undefined)
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
      expect(screen.getByText(/16 captured pairs/)).toBeDefined()
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

  it('calls startFromSessionsSloNet and stream on button click', async () => {
    async function* mockStream() {
      yield { stream: 'auto-train', phase: 'PAIRS', status: 'working', data: {}, meta: {}, message: 'Extracting pairs...' }
      yield { stream: 'auto-train', phase: 'TRAIN', status: 'working', data: { step: 5, loss: 1.5 }, meta: { epoch: 0, total_epochs: 5 }, message: '' }
      yield { stream: 'auto-train', phase: 'COMPLETE', status: 'complete', data: { checkpoint: '/tmp/test.soul', final_loss: 1.5, num_pairs: 10 }, meta: {}, message: 'Done' }
    }
    mockStreamSloNet.mockReturnValue(mockStream())

    const user = userEvent.setup()
    render(<TrainFromSessionsCard />)

    await waitFor(() => {
      expect(screen.getByText('Train from conversations')).toBeDefined()
    })

    await user.click(screen.getByText('Train from conversations'))

    await waitFor(() => {
      expect(mockStartSloNet).toHaveBeenCalled()
      expect(mockStreamSloNet).toHaveBeenCalled()
      expect(mockAddToast).toHaveBeenCalledWith(
        expect.stringContaining('Training complete'),
        'success'
      )
    })
  })

  it('shows error toast on training failure', async () => {
    async function* mockStream() {
      yield { stream: 'auto-train', phase: 'TRAIN', status: 'error', data: { error: 'Not enough data' }, meta: {}, message: 'Error: Not enough data' }
    }
    mockStreamSloNet.mockReturnValue(mockStream())

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
