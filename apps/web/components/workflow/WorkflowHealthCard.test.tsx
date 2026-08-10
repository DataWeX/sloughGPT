// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { WorkflowHealthCard } from './WorkflowHealthCard'
import type { WorkflowStatus } from '@/lib/workflow-controller'

afterEach(() => { cleanup() })

function makeStatus(overrides: Partial<WorkflowStatus> = {}): WorkflowStatus {
  return {
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
      feedback_recorded: 100,
      auto_train_steps: 0,
      dpo_train_steps: 0,
      dpo_train_rejected: 0,
      user_adapter_trained: 5,
      user_adapter_rejected: 0,
      start_time: null,
    },
    pending_thumbs_up: 0,
    auto_train_threshold: 3,
    last_runs: {
      aggregate: Math.floor(Date.now() / 1000) - 600,
      prune: Math.floor(Date.now() / 1000) - 1200,
      export: Math.floor(Date.now() / 1000) - 1800,
      dpo: 0,
      health_check: 0,
      last_rollback: 0,
      background_training: 0,
    },
    systems: {},
    ...overrides,
  }
}

describe('WorkflowHealthCard', () => {
  it('returns null when no stats', () => {
    const { container } = render(<WorkflowHealthCard status={{ running: true } as unknown as WorkflowStatus} />)
    expect(container.querySelector('[data-testid="workflow-health"]')).toBeNull()
  })

  it('renders health card', () => {
    render(<WorkflowHealthCard status={makeStatus()} />)
    expect(screen.getAllByTestId('workflow-health').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Workflow Health').length).toBeGreaterThanOrEqual(1)
  })

  it('shows healthy status when running and recent ops', () => {
    render(<WorkflowHealthCard status={makeStatus()} />)
    expect(screen.getAllByText('Healthy').length).toBeGreaterThanOrEqual(1)
  })

  it('shows stopped status when not running', () => {
    render(<WorkflowHealthCard status={makeStatus({ running: false })} />)
    expect(screen.getAllByText('Stopped').length).toBeGreaterThanOrEqual(1)
  })

  it('shows feedback count', () => {
    render(<WorkflowHealthCard status={makeStatus()} />)
    expect(screen.getAllByText('100').length).toBeGreaterThanOrEqual(1)
  })

  it('shows fb/adapter ratio', () => {
    render(<WorkflowHealthCard status={makeStatus()} />)
    expect(screen.getAllByText('20.0').length).toBeGreaterThanOrEqual(1)
  })

  it('shows last operations', () => {
    render(<WorkflowHealthCard status={makeStatus()} />)
    expect(screen.getAllByText('Aggregate').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Prune').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Export').length).toBeGreaterThanOrEqual(1)
  })

  it('shows never for missing last operation', () => {
    render(<WorkflowHealthCard status={makeStatus({ last_runs: { aggregate: 0, prune: 0, export: 0, dpo: 0, health_check: 0, last_rollback: 0, background_training: 0 } })} />)
    expect(screen.getAllByText('never').length).toBeGreaterThanOrEqual(1)
  })
})
