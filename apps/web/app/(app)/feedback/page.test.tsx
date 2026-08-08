import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

const mockGetFeedbackStats = vi.fn()
const mockGetWorkflowStatus = vi.fn()
const mockGetTrainingStats = vi.fn()

vi.mock('@/lib/feedback-controller', () => ({
  feedbackController: {
    getFeedbackStats: (...args: unknown[]) => mockGetFeedbackStats(...args),
    getWorkflowStatus: (...args: unknown[]) => mockGetWorkflowStatus(...args),
    getTrainingStats: (...args: unknown[]) => mockGetTrainingStats(...args),
  },
}))

vi.mock('@/lib/feedback-conversations-controller', () => ({
  feedbackConversationsController: {
    list: vi.fn().mockResolvedValue([]),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

vi.mock('@/lib/format-bytes', () => ({
  todayDateString: () => '2026-08-07',
}))

import FeedbackPage from './page'

describe('FeedbackPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockGetFeedbackStats.mockResolvedValue({ total: 0, positive: 0, negative: 0 })
    mockGetWorkflowStatus.mockResolvedValue({ running: false })
    mockGetTrainingStats.mockResolvedValue({ total: 0, sessions: 0 })
  })

  it('renders page header', async () => {
    render(<FeedbackPage />)
    expect(screen.getAllByText('Feedback').length).toBeGreaterThanOrEqual(1)
  })

  it('shows stats tabs after loading', async () => {
    render(<FeedbackPage />)
    await screen.findAllByText('Stats')
    expect(screen.getAllByText('Stats').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Conversations').length).toBeGreaterThanOrEqual(1)
  })
})
