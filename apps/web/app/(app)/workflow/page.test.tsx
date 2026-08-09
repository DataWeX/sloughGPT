import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

const mockStatus = vi.fn()
const mockStart = vi.fn()
const mockStop = vi.fn()
const mockTrigger = vi.fn()

vi.mock('@/lib/workflow-controller', () => ({
  workflowController: {
    status: (...args: unknown[]) => mockStatus(...args),
    start: (...args: unknown[]) => mockStart(...args),
    stop: (...args: unknown[]) => mockStop(...args),
    trigger: (...args: unknown[]) => mockTrigger(...args),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

import WorkflowPage from './page'

describe('WorkflowPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockStatus.mockResolvedValue({ running: false, last_run: null, total_runs: 0 })
  })

  it('renders page header', async () => {
    render(<WorkflowPage />)
    expect(screen.getAllByText('Workflow').length).toBeGreaterThanOrEqual(1)
  })

  it('shows workflow status', async () => {
    render(<WorkflowPage />)
    await vi.waitFor(() => expect(mockStatus).toHaveBeenCalled())
  })

  it('shows running status', async () => {
    mockStatus.mockResolvedValue({ running: true, stats: { feedback_records: 10, adapters_count: 2 } })
    render(<WorkflowPage />)
    await vi.waitFor(() => expect(mockStatus).toHaveBeenCalled())
  })

  it('renders toggle button after loading', async () => {
    render(<WorkflowPage />)
    await screen.findAllByText('Stopped')
    expect(screen.getAllByText('Start').length).toBeGreaterThanOrEqual(1)
  })
})
