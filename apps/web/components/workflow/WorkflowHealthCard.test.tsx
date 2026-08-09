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
      health_check_interval_seconds: 30,
    },
    stats: {
      feedback_records: 100,
      adapters_count: 5,
      last_aggregate: new Date(Date.now() - 600000).toISOString(),
      last_prune: new Date(Date.now() - 1200000).toISOString(),
      last_export: new Date(Date.now() - 1800000).toISOString(),
    },
    ...overrides,
  }
}

describe('WorkflowHealthCard', () => {
  it('returns null when no stats', () => {
    const { container } = render(<WorkflowHealthCard status={{ running: true }} />)
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
    render(<WorkflowHealthCard status={makeStatus({ stats: {
      feedback_records: 10,
      adapters_count: 2,
    }})} />)
    expect(screen.getAllByText('never').length).toBeGreaterThanOrEqual(1)
  })
})
