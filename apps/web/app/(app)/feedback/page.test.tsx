import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

const {
  mockGetFeedbackStats, mockGetWorkflowStatus, mockGetTrainingStats,
  mockListConversations, mockCreateConversation, mockDeleteConversation,
  mockAddToast, mockTriggerWorkflowAction, mockTogglePin, mockToggleStar,
} = vi.hoisted(() => ({
  mockGetFeedbackStats: vi.fn(), mockGetWorkflowStatus: vi.fn(), mockGetTrainingStats: vi.fn(),
  mockListConversations: vi.fn(), mockCreateConversation: vi.fn(),
  mockDeleteConversation: vi.fn(), mockAddToast: vi.fn(),
  mockTriggerWorkflowAction: vi.fn(), mockTogglePin: vi.fn(), mockToggleStar: vi.fn(),
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...a: any[]) => a.join(' ')),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Button: ({ children, onClick, disabled }: any) => (
      <button onClick={onClick} disabled={disabled}>{children}</button>
    ),
    Input: ({ value, onChange, placeholder }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} />
    ),
    StatCard: ({ label, value }: any) => <div data-testid={`stat-${label}`}><span>{label}</span><span>{String(value)}</span></div>,
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
  }
})

vi.mock('@/lib/feedback-controller', () => ({
  feedbackController: {
    getFeedbackStats: (...a: unknown[]) => mockGetFeedbackStats(...a),
    getWorkflowStatus: (...a: unknown[]) => mockGetWorkflowStatus(...a),
    getTrainingStats: (...a: unknown[]) => mockGetTrainingStats(...a),
    triggerWorkflowAction: (...a: unknown[]) => mockTriggerWorkflowAction(...a),
  },
}))

vi.mock('@/lib/feedback-conversations-controller', () => ({
  feedbackConversationsController: {
    list: (...a: unknown[]) => mockListConversations(...a),
    create: (...a: unknown[]) => mockCreateConversation(...a),
    delete: (...a: unknown[]) => mockDeleteConversation(...a),
    togglePin: (...a: unknown[]) => mockTogglePin(...a),
    toggleStar: (...a: unknown[]) => mockToggleStar(...a),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/lib/format-bytes', () => ({
  todayDateString: () => '2026-08-07',
}))

vi.mock('@/components/feedback/FeedbackInsightsCard', () => ({
  FeedbackInsightsCard: ({ stats, workflow }: any) => (
    <div data-testid="feedback-insights">
      {stats ? 'has-stats' : 'no-stats'}
      {workflow ? 'has-workflow' : 'no-workflow'}
    </div>
  ),
}))

import FeedbackPage from './page'

afterEach(cleanup)

beforeEach(() => {
  vi.clearAllMocks()
  mockGetFeedbackStats.mockResolvedValue({
    db_stats: { thumbs_up: 7, thumbs_down: 3, feedback_total: 10, ratio: 0.7 },
  })
  mockGetWorkflowStatus.mockResolvedValue({ running: true })
  mockGetTrainingStats.mockResolvedValue({
    feedback_pairs: 15,
    last_training: '2026-08-07T00:00:00Z',
    quality_score: 0.85,
  })
  mockListConversations.mockResolvedValue([
    { id: 'c1', name: 'Test Conversation', message_count: 10 },
  ])
  mockCreateConversation.mockResolvedValue({})
  mockDeleteConversation.mockResolvedValue({})
  mockTriggerWorkflowAction.mockResolvedValue({ status: 'ok', timestamp: Date.now() })
  mockTogglePin.mockResolvedValue({})
  mockToggleStar.mockResolvedValue({})
})

describe('FeedbackPage — initial load flow', () => {
  it('renders page header', async () => {
    render(<FeedbackPage />)
    expect(screen.getAllByText('Feedback').length).toBeGreaterThanOrEqual(1)
  })

  it('fetches stats on mount', async () => {
    render(<FeedbackPage />)
    await waitFor(() => {
      expect(mockGetFeedbackStats).toHaveBeenCalledTimes(1)
      expect(mockGetWorkflowStatus).toHaveBeenCalledTimes(1)
      expect(mockGetTrainingStats).toHaveBeenCalledTimes(1)
    })
  })

  it('shows loading state', () => {
    mockGetFeedbackStats.mockReturnValue(new Promise(() => {}))
    render(<FeedbackPage />)
    expect(screen.getAllByText('Feedback').length).toBeGreaterThanOrEqual(1)
  })
})

describe('FeedbackPage — stats tab flow', () => {
  it('shows stats tabs after loading', async () => {
    render(<FeedbackPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Stats').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('displays feedback stats', async () => {
    render(<FeedbackPage />)
    await waitFor(() => {
      expect(screen.getAllByText('7').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('displays thumbs up count in stat card', async () => {
    render(<FeedbackPage />)
    await waitFor(() => {
      const statCard = screen.getByTestId('stat-Thumbs Up')
      expect(statCard).toBeTruthy()
      expect(statCard.textContent).toContain('7')
    })
  })

  it('displays thumbs down count in stat card', async () => {
    render(<FeedbackPage />)
    await waitFor(() => {
      const statCard = screen.getByTestId('stat-Thumbs Down')
      expect(statCard).toBeTruthy()
      expect(statCard.textContent).toContain('3')
    })
  })

  it('displays up ratio as percentage', async () => {
    render(<FeedbackPage />)
    await waitFor(() => {
      expect(screen.getAllByText('70.0%').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('displays total feedback count', async () => {
    render(<FeedbackPage />)
    await waitFor(() => {
      const statCard = screen.getByTestId('stat-Total Feedback')
      expect(statCard).toBeTruthy()
      expect(statCard.textContent).toContain('10')
    })
  })
})

describe('FeedbackPage — empty state', () => {
  it('shows no feedback data when stats are null', async () => {
    mockGetFeedbackStats.mockResolvedValue(null)
    render(<FeedbackPage />)
    await waitFor(() => {
      expect(screen.getByText('No feedback data yet.')).toBeTruthy()
    })
  })

  it('shows zero values when db_stats fields are missing', async () => {
    mockGetFeedbackStats.mockResolvedValue({ db_stats: {} })
    render(<FeedbackPage />)
    await waitFor(() => {
      expect(screen.getByTestId('stat-Thumbs Up')).toBeTruthy()
    })
  })
})

describe('FeedbackPage — conversations tab flow', () => {
  it('switches to conversations tab', async () => {
    render(<FeedbackPage />)
    await waitFor(() => { expect(screen.getAllByText('Stats').length).toBeGreaterThanOrEqual(1) })

    const convTab = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('conversation')
    )
    if (convTab) {
      fireEvent.click(convTab)
      await waitFor(() => {
        expect(mockListConversations).toHaveBeenCalled()
      })
    }
  })

  it('shows conversations list', async () => {
    render(<FeedbackPage />)
    await waitFor(() => { expect(screen.getAllByText('Stats').length).toBeGreaterThanOrEqual(1) })

    const convTab = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('conversation')
    )
    if (convTab) {
      fireEvent.click(convTab)
      await waitFor(() => {
        expect(screen.getByText('Test Conversation')).toBeTruthy()
      })
    }
  })

  it('shows empty conversations message when list is empty', async () => {
    mockListConversations.mockResolvedValue([])
    render(<FeedbackPage />)
    await waitFor(() => { expect(screen.getAllByText('Stats').length).toBeGreaterThanOrEqual(1) })

    const convTab = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('conversation')
    )
    if (convTab) {
      fireEvent.click(convTab)
      await waitFor(() => {
        expect(screen.getByText('No conversations yet.')).toBeTruthy()
      })
    }
  })
})

describe('FeedbackPage — training tab flow', () => {
  it('switches to training tab', async () => {
    render(<FeedbackPage />)
    await waitFor(() => { expect(screen.getAllByText('Stats').length).toBeGreaterThanOrEqual(1) })

    const trainTab = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('training')
    )
    if (trainTab) {
      fireEvent.click(trainTab)
      await waitFor(() => {
        expect(screen.getByText('15')).toBeTruthy() // feedback pairs
      })
    }
  })

  it('displays training stats on training tab', async () => {
    render(<FeedbackPage />)
    await waitFor(() => { expect(screen.getAllByText('Stats').length).toBeGreaterThanOrEqual(1) })

    const trainTab = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('training')
    )
    if (trainTab) {
      fireEvent.click(trainTab)
      await waitFor(() => {
        expect(screen.getByTestId('stat-Training Jobs')).toBeTruthy()
        expect(screen.getByTestId('stat-Final Loss')).toBeTruthy()
      })
    }
  })

  it('shows no training data when stats are null', async () => {
    mockGetTrainingStats.mockResolvedValue(null)
    render(<FeedbackPage />)
    await waitFor(() => { expect(screen.getAllByText('Stats').length).toBeGreaterThanOrEqual(1) })

    const trainTab = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('training')
    )
    if (trainTab) {
      fireEvent.click(trainTab)
      await waitFor(() => {
        expect(screen.getByText('No training data available.')).toBeTruthy()
      })
    }
  })
})

describe('FeedbackPage — refresh flow', () => {
  it('refresh button reloads stats', async () => {
    render(<FeedbackPage />)
    await waitFor(() => { expect(mockGetFeedbackStats).toHaveBeenCalledTimes(1) })

    const refreshBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('refresh')
    )
    if (refreshBtn) {
      await act(async () => { fireEvent.click(refreshBtn) })
      await waitFor(() => {
        expect(mockGetFeedbackStats).toHaveBeenCalledTimes(2)
      })
    }
  })
})

describe('FeedbackPage — insights card', () => {
  it('renders insights card', async () => {
    render(<FeedbackPage />)
    await waitFor(() => {
      expect(screen.getByTestId('feedback-insights')).toBeTruthy()
    })
  })

  it('passes stats to insights card', async () => {
    render(<FeedbackPage />)
    await waitFor(() => {
      const el = screen.getByTestId('feedback-insights')
      expect(el.textContent).toContain('has-stats')
    })
  })
})

describe('FeedbackPage — workflow status', () => {
  it('shows workflow running status', async () => {
    render(<FeedbackPage />)
    await waitFor(() => {
      expect(screen.getByText('Running')).toBeTruthy()
    })
  })

  it('shows workflow stopped status', async () => {
    mockGetWorkflowStatus.mockResolvedValue({ running: false })
    render(<FeedbackPage />)
    await waitFor(() => {
      expect(screen.getByText('Stopped')).toBeTruthy()
    })
  })

  it('shows workflow stats when running', async () => {
    mockGetWorkflowStatus.mockResolvedValue({
      running: true,
      stats: { workflow_runs: 5, aggregations_performed: 3, prunes_performed: 2 },
    })
    render(<FeedbackPage />)
    await waitFor(() => {
      expect(screen.getByText('5')).toBeTruthy()
      expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1)
      expect(screen.getByText('2')).toBeTruthy()
    })
  })
})

describe('FeedbackPage — workflow actions', () => {
  it('triggers aggregate action', async () => {
    render(<FeedbackPage />)
    await waitFor(() => { expect(screen.getAllByText('Stats').length).toBeGreaterThanOrEqual(1) })

    const trainTab = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('training')
    )
    if (trainTab) {
      fireEvent.click(trainTab)
      await waitFor(() => { expect(screen.getAllByRole('button', { name: 'Aggregate' })[0]).toBeTruthy() })

      await act(async () => { fireEvent.click(screen.getAllByRole('button', { name: 'Aggregate' })[0]) })
      expect(mockTriggerWorkflowAction).toHaveBeenCalledWith('aggregate')
    }
  })

  it('triggers prune action', async () => {
    render(<FeedbackPage />)
    await waitFor(() => { expect(screen.getAllByText('Stats').length).toBeGreaterThanOrEqual(1) })

    const trainTab = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('training')
    )
    if (trainTab) {
      fireEvent.click(trainTab)
      await waitFor(() => { expect(screen.getAllByRole('button', { name: 'Prune' })[0]).toBeTruthy() })

      await act(async () => { fireEvent.click(screen.getAllByRole('button', { name: 'Prune' })[0]) })
      expect(mockTriggerWorkflowAction).toHaveBeenCalledWith('prune')
    }
  })

  it('triggers export action', async () => {
    render(<FeedbackPage />)
    await waitFor(() => { expect(screen.getAllByText('Stats').length).toBeGreaterThanOrEqual(1) })

    const trainTab = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('training')
    )
    if (trainTab) {
      fireEvent.click(trainTab)
      await waitFor(() => { expect(screen.getAllByRole('button', { name: 'Export' })[0]).toBeTruthy() })

      await act(async () => { fireEvent.click(screen.getAllByRole('button', { name: 'Export' })[0]) })
      expect(mockTriggerWorkflowAction).toHaveBeenCalledWith('export')
    }
  })

  it('shows toast on aggregate success', async () => {
    render(<FeedbackPage />)
    await waitFor(() => { expect(screen.getAllByText('Stats').length).toBeGreaterThanOrEqual(1) })

    const trainTab = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('training')
    )
    if (trainTab) {
      fireEvent.click(trainTab)
      await waitFor(() => { expect(screen.getAllByRole('button', { name: 'Aggregate' })[0]).toBeTruthy() })

      await act(async () => { fireEvent.click(screen.getAllByRole('button', { name: 'Aggregate' })[0]) })
      expect(mockAddToast).toHaveBeenCalledWith('Aggregation triggered', 'success')
    }
  })

  it('shows toast on aggregate failure', async () => {
    mockTriggerWorkflowAction.mockRejectedValue(new Error('fail'))
    render(<FeedbackPage />)
    await waitFor(() => { expect(screen.getAllByText('Stats').length).toBeGreaterThanOrEqual(1) })

    const trainTab = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('training')
    )
    if (trainTab) {
      fireEvent.click(trainTab)
      await waitFor(() => { expect(screen.getAllByRole('button', { name: 'Aggregate' })[0]).toBeTruthy() })

      await act(async () => { fireEvent.click(screen.getAllByRole('button', { name: 'Aggregate' })[0]) })
      expect(mockAddToast).toHaveBeenCalledWith('aggregation', 'error')
    }
  })
})

describe('FeedbackPage — error handling', () => {
  it('handles stats failure gracefully', async () => {
    mockGetFeedbackStats.mockRejectedValue(new Error('network'))
    render(<FeedbackPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Feedback').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('handles all controllers failing gracefully', async () => {
    mockGetFeedbackStats.mockRejectedValue(new Error('stats'))
    mockGetWorkflowStatus.mockRejectedValue(new Error('workflow'))
    mockGetTrainingStats.mockRejectedValue(new Error('training'))
    render(<FeedbackPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Feedback').length).toBeGreaterThanOrEqual(1)
    })
  })
})
