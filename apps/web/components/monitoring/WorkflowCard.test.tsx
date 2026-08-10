/** @vitest-environment jsdom */
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { vi, describe, it, expect, beforeEach } from 'vitest'
import { WorkflowCard } from './WorkflowCard'
import type { WorkflowStatus } from '@/lib/workflow-controller'

vi.mock('@/lib/workflow-controller', () => ({
  workflowController: {
    status: vi.fn(),
    start: vi.fn().mockResolvedValue({ status: 'started' }),
    stop: vi.fn().mockResolvedValue({ status: 'stopped' }),
  },
}))

function makeStatus(overrides: Partial<WorkflowStatus> = {}): WorkflowStatus {
  return {
    running: false,
    stats: {
      workflow_runs: 10,
      aggregations_performed: 3,
      prunes_performed: 2,
      exports_performed: 1,
      feedback_recorded: 12,
      auto_train_steps: 3,
      dpo_train_steps: 0,
      dpo_train_rejected: 0,
      user_adapter_trained: 2,
      user_adapter_rejected: 1,
      start_time: null,
    },
    pending_thumbs_up: 0,
    auto_train_threshold: 5,
    config: {
      aggregate_interval_minutes: 60,
      prune_interval_minutes: 1440,
      export_interval_hours: 24,
      auto_dpo_interval_minutes: 120,
      health_check_interval_seconds: 300,
      background_training_interval_seconds: 600,
      background_training_enabled: true,
    },
    last_runs: { aggregate: 0, prune: 0, export: 0, dpo: 0, health_check: 0, last_rollback: 0, background_training: 0 },
    systems: {},
    ...overrides,
  }
}

describe('WorkflowCard', () => {
  beforeEach(async () => {
    vi.clearAllMocks()
    const { workflowController } = await import('@/lib/workflow-controller')
    vi.mocked(workflowController.status).mockResolvedValue(makeStatus())
  })

  it('renders loading skeleton initially', () => {
    const { container } = render(<WorkflowCard />)
    expect(container.querySelector('.animate-pulse')).toBeTruthy()
  })

  it('shows stopped status when workflow is not running', async () => {
    render(<WorkflowCard />)
    await waitFor(() => expect(screen.getAllByText('Stopped').length).toBeGreaterThanOrEqual(1))
    expect(screen.getAllByText('Background Training').length).toBeGreaterThanOrEqual(1)
  })

  it('shows running status dot', async () => {
    const { workflowController } = await import('@/lib/workflow-controller')
    vi.mocked(workflowController.status).mockResolvedValue(makeStatus({ running: true }))
    render(<WorkflowCard />)
    await waitFor(() => expect(screen.getAllByText('Running').length).toBeGreaterThanOrEqual(1))
  })

  it('displays stats grid', async () => {
    render(<WorkflowCard />)
    await waitFor(() => expect(screen.getAllByText('Stopped').length).toBeGreaterThanOrEqual(1))
    expect(screen.getAllByText('Auto-trains:').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Rejected:').length).toBeGreaterThanOrEqual(1)
  })

  it('shows start button when stopped', async () => {
    render(<WorkflowCard />)
    await waitFor(() => expect(screen.getAllByText('Start').length).toBeGreaterThanOrEqual(1))
  })

  it('shows stop button when running', async () => {
    const { workflowController } = await import('@/lib/workflow-controller')
    vi.mocked(workflowController.status).mockResolvedValue(makeStatus({ running: true }))
    render(<WorkflowCard />)
    await waitFor(() => expect(screen.getAllByText('Stop').length).toBeGreaterThanOrEqual(1))
  })

  it('start button triggers start action', async () => {
    render(<WorkflowCard />)
    await waitFor(() => expect(screen.getAllByText('Start').length).toBeGreaterThanOrEqual(1))
    const { workflowController } = await import('@/lib/workflow-controller')
    fireEvent.click(screen.getAllByText('Start')[0])
    await waitFor(() => expect(vi.mocked(workflowController.start)).toHaveBeenCalled())
  })

  it('stop button triggers stop action', async () => {
    const { workflowController } = await import('@/lib/workflow-controller')
    vi.mocked(workflowController.status).mockResolvedValue(makeStatus({ running: true }))
    render(<WorkflowCard />)
    await waitFor(() => expect(screen.getAllByText('Stop').length).toBeGreaterThanOrEqual(1))
    fireEvent.click(screen.getAllByText('Stop')[0])
    await waitFor(() => expect(vi.mocked(workflowController.stop)).toHaveBeenCalled())
  })

  it('renders config values', async () => {
    render(<WorkflowCard />)
    await waitFor(() => expect(screen.getAllByText('Stopped').length).toBeGreaterThanOrEqual(1))
    expect(screen.getAllByText('Background interval:').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('600s').length).toBeGreaterThanOrEqual(1)
  })

  it('shows Never for zero timestamps', async () => {
    render(<WorkflowCard />)
    await waitFor(() => expect(screen.getAllByText('Stopped').length).toBeGreaterThanOrEqual(1))
    const nevers = screen.getAllByText('Never')
    expect(nevers.length).toBeGreaterThanOrEqual(1)
  })
})
