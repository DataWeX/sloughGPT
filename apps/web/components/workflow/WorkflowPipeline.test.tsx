// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { WorkflowPipeline } from './WorkflowPipeline'
import type { WorkflowStatus } from '@/lib/workflow-controller'

beforeEach(() => { vi.clearAllMocks() })
afterEach(() => { cleanup() })

const runningStatus: WorkflowStatus = {
  running: true,
  config: {
    aggregate_interval_minutes: 30,
    prune_interval_minutes: 60,
    export_interval_hours: 24,
    auto_dpo_interval_minutes: 60,
    health_check_interval_seconds: 30,
    background_training_interval_seconds: 300,
    background_training_enabled: true,
  },
  stats: {
    workflow_runs: 10,
    aggregations_performed: 3,
    prunes_performed: 2,
    exports_performed: 1,
    feedback_recorded: 150,
    auto_train_steps: 0,
    dpo_train_steps: 0,
    dpo_train_rejected: 0,
    user_adapter_trained: 3,
    user_adapter_rejected: 0,
    start_time: null,
  },
  pending_thumbs_up: 0,
  auto_train_threshold: 3,
  last_runs: {
    aggregate: Math.floor(Date.now() / 1000) - 600,
    prune: Math.floor(Date.now() / 1000) - 3600,
    export: 0,
    dpo: 0,
    health_check: 0,
    last_rollback: 0,
    background_training: 0,
  },
  systems: {},
}

const stoppedStatus: WorkflowStatus = {
  running: false,
  config: runningStatus.config,
  stats: runningStatus.stats,
  pending_thumbs_up: 0,
  auto_train_threshold: 3,
  last_runs: runningStatus.last_runs,
  systems: {},
}

describe('WorkflowPipeline', () => {
  it('shows not configured message when no status', () => {
    render(<WorkflowPipeline status={null} />)
    expect(screen.getByText('Pipeline not configured')).toBeInTheDocument()
  })

  it('returns null when no config', () => {
    const noConfigStatus = { ...runningStatus, config: undefined as unknown as WorkflowStatus['config'] }
    const { container } = render(<WorkflowPipeline status={noConfigStatus} />)
    expect(container.querySelector('[data-testid="workflow-pipeline"]')).toBeNull()
  })

  it('renders pipeline card when status has config', () => {
    render(<WorkflowPipeline status={runningStatus} />)
    expect(screen.getAllByTestId('workflow-pipeline').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Pipeline').length).toBeGreaterThanOrEqual(1)
  })

  it('shows all 5 pipeline steps', () => {
    render(<WorkflowPipeline status={runningStatus} />)
    expect(screen.getAllByText('Feedback').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Aggregate').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Train').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Prune').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Export').length).toBeGreaterThanOrEqual(1)
  })

  it('shows step numbers 1-5', () => {
    render(<WorkflowPipeline status={runningStatus} />)
    expect(screen.getAllByText('1').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('5').length).toBeGreaterThanOrEqual(1)
  })

  it('shows interval labels', () => {
    render(<WorkflowPipeline status={runningStatus} />)
    expect(screen.getAllByText('continuous').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('30m').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('60m').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('24h').length).toBeGreaterThanOrEqual(1)
  })

  it('shows feedback record count', () => {
    render(<WorkflowPipeline status={runningStatus} />)
    expect(screen.getAllByText('150 feedback records').length).toBeGreaterThanOrEqual(1)
  })

  it('shows Active and Idle legend', () => {
    render(<WorkflowPipeline status={runningStatus} />)
    expect(screen.getAllByText('Active').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Idle').length).toBeGreaterThanOrEqual(1)
  })

  it('shows last run times for steps with data', () => {
    render(<WorkflowPipeline status={runningStatus} />)
    expect(screen.getAllByText(/ago/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows never for steps without last run', () => {
    const statusNoRuns: WorkflowStatus = {
      running: true,
      config: runningStatus.config,
      stats: {
        workflow_runs: 0,
        aggregations_performed: 0,
        prunes_performed: 0,
        exports_performed: 0,
        feedback_recorded: 0,
        auto_train_steps: 0,
        dpo_train_steps: 0,
        dpo_train_rejected: 0,
        user_adapter_trained: 0,
        user_adapter_rejected: 0,
        start_time: null,
      },
      pending_thumbs_up: 0,
      auto_train_threshold: 3,
      last_runs: { aggregate: 0, prune: 0, export: 0, dpo: 0, health_check: 0, last_rollback: 0, background_training: 0 },
      systems: {},
    }
    render(<WorkflowPipeline status={statusNoRuns} />)
    expect(screen.getAllByText('never').length).toBeGreaterThanOrEqual(1)
  })
})
